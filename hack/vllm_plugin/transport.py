"""TCP transport for staged KV data.

The prefill instance stages the KV of a finished prefill in pinned host memory and serves
it from a listener thread. The decode instance pulls it by request id into pinned host
memory. Payloads are sent in chunks; an optional sender-side rate limit, shared by all
transfers of the instance, emulates a network link of a given bandwidth.

Wire format: a 4-byte big-endian length followed by a JSON message in both directions;
the reply to a found request is followed by the raw bytes of all segments, and the client
acknowledges with one byte.
"""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass, field

import torch

CHUNK_BYTES = 8 << 20
_LENGTH = struct.Struct(">I")


@dataclass
class Segment:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass
class StagedTransfer:
    request_id: str
    segments: list[Segment]
    payload: torch.Tensor
    info: dict = field(default_factory=dict)
    payload_buffer: torch.Tensor | None = None
    created: float = field(default_factory=time.monotonic)

    @property
    def nbytes(self) -> int:
        return int(self.payload.numel())


@dataclass
class TransferRecord:
    request_id: str
    role: str
    nbytes: int
    seconds: float
    started: float


class RateLimiter:
    """Paces the bytes sent by all connections of one instance to `gbps` gigabits per second."""

    CATCH_UP_SECONDS = 0.05  # a sender that fell behind its schedule by less than this catches up

    def __init__(self, gbps: float | None):
        self.bytes_per_second = gbps * 1e9 / 8 if gbps else None
        self._lock = threading.Lock()
        self._next_free = time.monotonic()

    def acquire(self, nbytes: int) -> None:
        if self.bytes_per_second is None:
            return
        with self._lock:
            now = time.monotonic()
            start = self._next_free if now - self._next_free <= self.CATCH_UP_SECONDS else now
            self._next_free = start + nbytes / self.bytes_per_second
            wake = self._next_free
        delay = wake - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class PinnedBufferPool:
    """Reuses pinned host buffers: a request gets the smallest free buffer that is large enough.

    Pinning memory is slow, so a new buffer is made as large as the largest one so far and serves later requests.
    """

    GRANULARITY = 64 << 20

    def __init__(self, pin: bool = True):
        self.pin = pin
        self._free: dict[int, list[torch.Tensor]] = {}
        self._largest = 0
        self._lock = threading.Lock()

    def take(self, nbytes: int) -> torch.Tensor:
        capacity = max(self.GRANULARITY, -(-nbytes // self.GRANULARITY) * self.GRANULARITY)
        with self._lock:
            sizes = [size for size, bucket in self._free.items() if size >= capacity and bucket]
            if sizes:
                return self._free[min(sizes)].pop()
            capacity = self._largest = max(capacity, self._largest)
        return torch.empty(capacity, dtype=torch.uint8, pin_memory=self.pin)

    def give(self, buffer: torch.Tensor) -> None:
        with self._lock:
            self._free.setdefault(buffer.numel(), []).append(buffer)


def _send_message(sock: socket.socket, message: dict) -> None:
    data = json.dumps(message).encode()
    sock.sendall(_LENGTH.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, view: memoryview) -> None:
    received = 0
    while received < len(view):
        count = sock.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("connection closed during a KV transfer")
        received += count


def _recv_message(sock: socket.socket) -> dict:
    header = bytearray(_LENGTH.size)
    _recv_exact(sock, memoryview(header))
    body = bytearray(_LENGTH.unpack(header)[0])
    _recv_exact(sock, memoryview(body))
    return json.loads(body)


class TransferServer:
    def __init__(self, host: str, port: int, gbps: float | None, buffers: PinnedBufferPool, ttl_seconds: float = 600.0):
        self.buffers = buffers
        self.limiter = RateLimiter(gbps)
        self.ttl_seconds = ttl_seconds
        self.records: list[TransferRecord] = []
        self._staged: dict[str, StagedTransfer] = {}
        self._lock = threading.Lock()
        self._listener = socket.create_server((host, port), reuse_port=False)
        self.port = self._listener.getsockname()[1]
        self._closed = False
        threading.Thread(target=self._accept_loop, name="kv-transfer-server", daemon=True).start()

    def stage(self, transfer: StagedTransfer) -> None:
        with self._lock:
            self._staged[transfer.request_id] = transfer
            self._drop_expired()

    def _drop_expired(self) -> None:
        deadline = time.monotonic() - self.ttl_seconds
        for request_id in [r for r, t in self._staged.items() if t.created < deadline]:
            self.buffers.give(self._staged.pop(request_id).payload_buffer)

    def take_records(self) -> list[TransferRecord]:
        with self._lock:
            records, self.records = self.records, []
        return records

    def close(self) -> None:
        self._closed = True
        self._listener.close()

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            request_id = _recv_message(connection)["request_id"]
            with self._lock:
                transfer = self._staged.pop(request_id, None)
            if transfer is None:
                _send_message(connection, {"found": False})
                return
            started = time.monotonic()
            _send_message(
                connection,
                {
                    "found": True,
                    "nbytes": transfer.nbytes,
                    "info": transfer.info,
                    "segments": [[s.name, list(s.shape), s.dtype, s.nbytes] for s in transfer.segments],
                },
            )
            view = memoryview(transfer.payload.numpy())
            for offset in range(0, transfer.nbytes, CHUNK_BYTES):
                chunk = view[offset : offset + CHUNK_BYTES]
                self.limiter.acquire(len(chunk))
                connection.sendall(chunk)
            connection.recv(1)
            record = TransferRecord(request_id, "send", transfer.nbytes, time.monotonic() - started, time.time())
            with self._lock:
                self.records.append(record)
            self.buffers.give(transfer.payload_buffer)


def fetch(
    host: str, port: int, request_id: str, buffers: PinnedBufferPool
) -> tuple[StagedTransfer | None, TransferRecord]:
    """Pull the staged KV of `request_id`; returns None when the server does not hold it."""
    started_wall, started = time.time(), time.monotonic()
    with socket.create_connection((host, port)) as connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        _send_message(connection, {"request_id": request_id})
        reply = _recv_message(connection)
        if not reply["found"]:
            return None, TransferRecord(request_id, "recv", 0, time.monotonic() - started, started_wall)
        nbytes = reply["nbytes"]
        buffer = buffers.take(nbytes)
        _recv_exact(connection, memoryview(buffer.numpy())[:nbytes])
        connection.sendall(b"\x01")
    segments = [Segment(name, tuple(shape), dtype, size) for name, shape, dtype, size in reply["segments"]]
    transfer = StagedTransfer(request_id, segments, buffer[:nbytes], reply["info"], payload_buffer=buffer)
    return transfer, TransferRecord(request_id, "recv", nbytes, time.monotonic() - started, started_wall)

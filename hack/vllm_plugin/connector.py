"""KV connector for disaggregated prefill and decode.

Prefill instance (`kv_role="kv_producer"`): at the end of the step that completes a prompt,
the KV data of the request is copied to pinned host memory and offered on a TCP port.
The response of the prefill request carries the address in `kv_transfer_params`.

Decode instance (`kv_role="kv_consumer"`): a request that arrives with these parameters
gets its blocks allocated first, pulls the staged data in a background thread and is
scheduled once the data sits in its blocks. The request must contain at least one token
after the transferred ones (the first generated token), which the decode instance computes.

Pages are moved as opaque bytes, so the connector serves any attention backend whose
per-layer cache tensor has the block dimension first.

`kv_connector_extra_config` keys: `transfer_host`, `transfer_port` (producer listen address),
`bandwidth_gbps` (sender-side rate limit), `stats_path` (JSON lines, one per transfer).
"""

import json
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from hack.vllm_plugin.runtime import HackRuntime, get_runtime
from hack.vllm_plugin.transport import (
    PinnedBufferPool,
    Segment,
    StagedTransfer,
    TransferRecord,
    TransferServer,
    fetch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class SaveTask:
    request_id: str
    block_ids: list[int]
    num_tokens: int


@dataclass
class LoadTask:
    request_id: str
    remote_request_id: str
    remote_host: str
    remote_port: int
    block_ids: list[int]
    skip_blocks: int = 0
    submitted: float = 0.0


@dataclass
class HackConnectorMetadata(KVConnectorMetadata):
    saves: list[SaveTask] = field(default_factory=list)
    loads: list[LoadTask] = field(default_factory=list)
    started: list[str] = field(default_factory=list)


class EventLog:
    """Appends one JSON line per request event: {"request_id", "event", "time", ...}."""

    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, request_id: str, event: str, **extra: Any) -> None:
        if self.path is None:
            return
        line = json.dumps({"request_id": request_id, "event": event, "time": time.time(), **extra})
        with self._lock, open(self.path, "a") as stream:
            stream.write(line + "\n")

    def write_record(self, record: TransferRecord) -> None:
        self.write(record.request_id, record.role, nbytes=record.nbytes, seconds=record.seconds)


def _transfer_params(request_or_data: Any) -> dict[str, Any]:
    sampling_params = getattr(request_or_data, "sampling_params", None)
    extra = getattr(sampling_params, "extra_args", None) or {}
    return extra.get("kv_transfer_params") or {}


class HackKVConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        config = self._kv_transfer_config
        self.host = config.get_from_extra_config("transfer_host", config.kv_ip)
        self.port = int(config.get_from_extra_config("transfer_port", config.kv_port))
        events = EventLog(config.get_from_extra_config("stats_path", None))
        block_size = vllm_config.cache_config.block_size
        is_scheduler = role == KVConnectorRole.SCHEDULER
        self.scheduler = _SchedulerSide(self.host, self.port, block_size, events) if is_scheduler else None
        self.worker = None if is_scheduler else _WorkerSide(vllm_config, self.host, self.port, events)

    # ------------------------------------------------------------ scheduler side

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int | None, bool]:
        return self.scheduler.matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        self.scheduler.after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> KVConnectorMetadata:
        return self.scheduler.build_meta(scheduler_output)

    def on_new_request(self, request: "Request") -> None:
        self.scheduler.events.write(request.request_id, "arrive", prompt_tokens=request.num_prompt_tokens)

    def request_finished(self, request: "Request", block_ids: list[int]) -> tuple[bool, dict[str, Any] | None]:
        return self.scheduler.finished(request)

    def request_finished_all_groups(
        self, request: "Request", block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.scheduler.finished(request)

    # --------------------------------------------------------------- worker side

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.worker.register(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HackConnectorMetadata)
        self.worker.start_loads(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        return

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, HackConnectorMetadata)
        self.worker.save(metadata.saves)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        return None, self.worker.finish_loads(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        return self.worker.take_failed_blocks()

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()


class _SchedulerSide:
    def __init__(self, host: str, port: int, block_size: int, events: EventLog):
        self.host = host
        self.port = port
        self.block_size = block_size
        self.events = events
        self._producer_blocks: dict[str, list[int]] = {}
        self._prompt_lens: dict[str, int] = {}
        self._saved: set[str] = set()
        self._pending_loads: list[LoadTask] = []
        self._loaded: set[str] = set()

    def matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return 0, False
        remote = int(params["remote_num_tokens"])
        if remote >= request.num_tokens:
            logger.warning("request %s has no token after the transferred ones; computing locally", request.request_id)
            params["do_remote_prefill"] = False
            return 0, False
        count = remote - num_computed_tokens
        return (count, True) if count > 0 else (0, False)

    def after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int) -> None:
        params = request.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        params["do_remote_prefill"] = False
        if num_external_tokens <= 0:
            return
        groups = blocks.get_block_ids()
        if len(groups) != 1:
            raise ValueError("the connector supports models with a single KV-cache group")
        self._pending_loads.append(
            LoadTask(
                request_id=request.request_id,
                remote_request_id=params["remote_request_id"],
                remote_host=params["remote_host"],
                remote_port=int(params["remote_port"]),
                block_ids=list(groups[0]),
                skip_blocks=(int(params["remote_num_tokens"]) - num_external_tokens) // self.block_size,
            )
        )
        self._loaded.add(request.request_id)

    def build_meta(self, scheduler_output: "SchedulerOutput") -> HackConnectorMetadata:
        meta = HackConnectorMetadata(loads=self._pending_loads)
        self._pending_loads = []
        scheduled = scheduler_output.num_scheduled_tokens
        for new in scheduler_output.scheduled_new_reqs:
            self.events.write(new.req_id, "scheduled", computed_tokens=new.num_computed_tokens)
            if _transfer_params(new).get("do_remote_decode"):
                self._producer_blocks[new.req_id] = list(new.block_ids[0])
                self._maybe_save(meta, new.req_id, len(new.prompt_token_ids), new.num_computed_tokens, scheduled)
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in self._producer_blocks and req_id not in self._saved:
                new_blocks = cached.new_block_ids[i]
                if new_blocks is not None:
                    if req_id in cached.resumed_req_ids:
                        self._producer_blocks[req_id] = []
                    self._producer_blocks[req_id].extend(new_blocks[0])
                prompt_len = self._prompt_lens[req_id]
                self._maybe_save(meta, req_id, prompt_len, cached.num_computed_tokens[i], scheduled)
        for req_id in scheduled:
            if req_id in self._loaded:
                self._loaded.discard(req_id)
                meta.started.append(req_id)
        return meta

    def _maybe_save(self, meta: HackConnectorMetadata, req_id: str, prompt_len: int, computed: int, scheduled) -> None:
        self._prompt_lens[req_id] = prompt_len
        if computed + scheduled.get(req_id, 0) >= prompt_len:
            meta.saves.append(SaveTask(req_id, list(self._producer_blocks[req_id]), prompt_len))
            self._saved.add(req_id)

    def finished(self, request: "Request") -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        self.events.write(req_id, "finished", output_tokens=request.num_output_tokens)
        self._loaded.discard(req_id)
        self._prompt_lens.pop(req_id, None)
        was_producer = self._producer_blocks.pop(req_id, None) is not None
        if not was_producer or req_id not in self._saved:
            return False, None
        self._saved.discard(req_id)
        return False, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_request_id": req_id,
            "remote_host": self.host,
            "remote_port": self.port,
            "remote_num_tokens": request.num_prompt_tokens,
        }


SEGMENT_ALIGNMENT = 16


def _aligned(offset: int) -> int:
    return -(-offset // SEGMENT_ALIGNMENT) * SEGMENT_ALIGNMENT


class _WorkerSide:
    def __init__(self, vllm_config: "VllmConfig", host: str, port: int, events: EventLog):
        config = vllm_config.kv_transfer_config
        self.vllm_config = vllm_config
        self.events = events
        self.buffers = PinnedBufferPool()
        self.kv_caches: dict[str, torch.Tensor] = {}
        self.stores: dict[str, Any] = {}
        self.runtime: HackRuntime | None = None
        self.max_pinned = 0
        self.server: TransferServer | None = None
        if config.is_kv_producer:
            gbps = config.get_from_extra_config("bandwidth_gbps", None)
            self.server = TransferServer(host, port, float(gbps) if gbps else None, self.buffers)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="kv-transfer-recv")
        self._received: queue.Queue = queue.Queue()
        self._deferred: list[tuple[LoadTask, StagedTransfer | None]] = []
        self._in_flight: list[tuple[torch.cuda.Event, torch.Tensor]] = []
        self._failed_blocks: set[int] = set()
        self._loading: set[str] = set()
        self._cancelled: set[str] = set()

    def register(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self.kv_caches = dict(kv_caches)
        self.runtime = get_runtime()
        first = next(iter(kv_caches.values()))
        self.events.write(
            "",
            "cache",
            num_blocks=first.shape[0],
            bytes_per_block=sum(c[0].numel() * c.element_size() for c in kv_caches.values()),
            side_pool_bytes=self.runtime.side_pool_bytes() if self.runtime is not None else 0,
        )
        if self.runtime is None:
            return
        layers = self.vllm_config.compilation_config.static_forward_context
        for name, cache in kv_caches.items():
            store = layers[name].impl.store
            store.bind(cache)
            self.stores[name] = store
        self.runtime.bind_blocks(next(iter(kv_caches.values())).shape[0])
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.max_pinned = max(1, self.runtime.num_side_slots - max_num_seqs)

    # ---------------------------------------------------------------- producer

    def save(self, tasks: list[SaveTask]) -> None:
        if self.server is None:
            return
        for task in tasks:
            started = time.monotonic()
            transfer = self._stage(task)
            self.server.stage(transfer)
            self.events.write(task.request_id, "staged", nbytes=transfer.nbytes, seconds=time.monotonic() - started)
        for record in self.server.take_records():
            self.events.write_record(record)

    def _side_tensors(self, task: SaveTask, blocks: torch.Tensor) -> dict[str, torch.Tensor]:
        """16-bit data of the request outside its pages: open block and sink tokens, per layer."""
        if self.runtime is None:
            return {}
        cfg = self.runtime.settings.config
        sinks = min(cfg.sink_tokens, task.num_tokens)
        after_sinks = task.num_tokens - sinks
        open_tokens = after_sinks % cfg.partition_size
        result = {}
        if open_tokens:
            slot = self.runtime.open_slots.lookup(blocks[after_sinks // cfg.partition_size].reshape(1))
            for name, store in self.stores.items():
                result[f"{name}/open_k"] = store.open_k[slot, :, :open_tokens]
                result[f"{name}/open_v"] = store.open_v[slot, :, :open_tokens]
        if sinks:
            slot = self.runtime.sink_slots.lookup(blocks[:1])
            for name, store in self.stores.items():
                result[f"{name}/sink_k"] = store.sink_k[slot, :, :sinks]
                result[f"{name}/sink_v"] = store.sink_v[slot, :, :sinks]
        return result

    def _stage(self, task: SaveTask) -> StagedTransfer:
        device = next(iter(self.kv_caches.values())).device
        blocks = torch.tensor(task.block_ids, dtype=torch.int64, device=device)
        sources = [(f"{name}/pages", cache, blocks) for name, cache in self.kv_caches.items()]
        sources += [(name, tensor, None) for name, tensor in self._side_tensors(task, blocks).items()]
        segments, total = [], 0
        for name, tensor, rows in sources:
            shape = (len(task.block_ids), *tensor.shape[1:]) if rows is not None else tuple(tensor.shape)
            nbytes = math.prod(shape) * tensor.element_size()
            segments.append(Segment(name, shape, str(tensor.dtype).removeprefix("torch."), nbytes))
            total = _aligned(total) + nbytes
        buffer = self.buffers.take(total)
        offset = 0
        for segment, (_, tensor, rows) in zip(segments, sources):
            offset = _aligned(offset)
            values = tensor.index_select(0, rows) if rows is not None else tensor.contiguous()
            buffer[offset : offset + segment.nbytes].copy_(values.view(torch.uint8).reshape(-1), non_blocking=True)
            offset += segment.nbytes
        torch.cuda.current_stream().synchronize()
        info = {"num_tokens": task.num_tokens, "num_blocks": len(task.block_ids)}
        return StagedTransfer(task.request_id, segments, buffer[:total], info, payload_buffer=buffer)

    # ---------------------------------------------------------------- consumer

    def start_loads(self, metadata: HackConnectorMetadata) -> None:
        for task in metadata.loads:
            task.submitted = time.monotonic()
            self._loading.add(task.request_id)
            self._pool.submit(self._pull, task)
        if self.runtime is not None:
            for req_id in metadata.started:
                self._unpin(req_id)

    def _pull(self, task: LoadTask) -> None:
        transfer = None
        try:
            transfer, record = fetch(task.remote_host, task.remote_port, task.remote_request_id, self.buffers)
            record.request_id = task.request_id
            self.events.write_record(record)
        except OSError:
            logger.exception("KV transfer of request %s failed", task.request_id)
        self._received.put((task, transfer))

    def finish_loads(self, finished_req_ids: set[str]) -> set[str] | None:
        self._cancelled.update(finished_req_ids & self._loading)
        if self.runtime is not None:
            for req_id in finished_req_ids:
                self._unpin(req_id)
        self._release_copied_buffers()
        arrived, self._deferred = self._deferred, []
        while True:
            try:
                arrived.append(self._received.get_nowait())
            except queue.Empty:
                break
        done: set[str] = set()
        for task, transfer in arrived:
            if task.request_id in self._cancelled:
                if transfer is not None:
                    self.buffers.give(transfer.payload_buffer)
            elif transfer is None:
                self._failed_blocks.update(task.block_ids)
            elif self.runtime is not None and self.runtime.open_slots.num_pinned >= self.max_pinned:
                self._deferred.append((task, transfer))
                continue
            else:
                self._install(task, transfer)
                seconds = time.monotonic() - task.submitted
                self.events.write(task.request_id, "loaded", nbytes=transfer.nbytes, seconds=seconds)
            self._loading.discard(task.request_id)
            self._cancelled.discard(task.request_id)
            done.add(task.request_id)
        return done or None

    def _release_copied_buffers(self) -> None:
        pending = []
        for event, buffer in self._in_flight:
            if event.query():
                self.buffers.give(buffer)
            else:
                pending.append((event, buffer))
        self._in_flight = pending

    def _install(self, task: LoadTask, transfer: StagedTransfer) -> None:
        device = next(iter(self.kv_caches.values())).device
        if transfer.info["num_blocks"] != len(task.block_ids):
            raise RuntimeError("prefill and decode instances disagree on the number of blocks of a request")
        kept = slice(task.skip_blocks, len(task.block_ids))
        blocks = torch.tensor(task.block_ids, dtype=torch.int64, device=device)
        offset, side = 0, {}
        for segment in transfer.segments:
            offset = _aligned(offset)
            data = transfer.payload[offset : offset + segment.nbytes].to(device, non_blocking=True)
            offset += segment.nbytes
            name, kind = segment.name.rsplit("/", 1)
            if kind == "pages":
                cache = self.kv_caches[name]
                pages = data.view(cache.dtype).view(len(task.block_ids), *cache.shape[1:])
                cache.index_copy_(0, blocks[kept], pages[kept])
            else:
                side[(name, kind)] = data.view(getattr(torch, segment.dtype)).view(segment.shape)
        if side:
            self._install_side(task, transfer.info["num_tokens"], blocks, side)
        event = torch.cuda.Event()
        event.record()
        self._in_flight.append((event, transfer.payload_buffer))

    def _install_side(self, task: LoadTask, num_tokens: int, blocks: torch.Tensor, side: dict) -> None:
        cfg = self.runtime.settings.config
        sinks = min(cfg.sink_tokens, num_tokens)
        after_sinks = num_tokens - sinks
        open_tokens = after_sinks % cfg.partition_size
        if open_tokens:
            slot = self.runtime.open_slots.allocate(blocks[after_sinks // cfg.partition_size].reshape(1))
            self.runtime.open_slots.pin(task.request_id, slot)
            for name, store in self.stores.items():
                store.open_k[slot, :, :open_tokens] = side[(name, "open_k")]
                store.open_v[slot, :, :open_tokens] = side[(name, "open_v")]
        if sinks:
            slot = self.runtime.sink_slots.allocate(blocks[:1])
            self.runtime.sink_slots.pin(task.request_id, slot)
            for name, store in self.stores.items():
                store.sink_k[slot, :, :sinks] = side[(name, "sink_k")]
                store.sink_v[slot, :, :sinks] = side[(name, "sink_v")]

    def _unpin(self, req_id: str) -> None:
        self.runtime.open_slots.unpin(req_id)
        if self.runtime.sink_slots is not None:
            self.runtime.sink_slots.unpin(req_id)

    def take_failed_blocks(self) -> set[int]:
        failed, self._failed_blocks = self._failed_blocks, set()
        return failed

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        if self.server is not None:
            for record in self.server.take_records():
                self.events.write_record(record)
            self.server.close()

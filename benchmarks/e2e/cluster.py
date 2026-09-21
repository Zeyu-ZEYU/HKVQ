"""Start and stop a disaggregated serving cluster: prefill instances, decode instances, proxy.

    python benchmarks/e2e/cluster.py --method hack --model Qwen/Qwen3-8B \
        --prefill-gpus 0 --decode-gpus 1 --run-dir runs/demo

The command blocks until it is interrupted; all processes are stopped on exit. Methods are
listed in `METHODS`; a new storage-only quantization baseline is added by registering its
attention backend and one more entry there.
"""

import argparse
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CONNECTOR = {"kv_connector": "HackKVConnector", "kv_connector_module_path": "hack.vllm_plugin.connector"}
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _die_with_parent() -> None:
    """Runs in the child before exec: the child gets SIGTERM when the process that started it dies."""
    _LIBC.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


@dataclass(frozen=True)
class Method:
    """How a serving method is selected on the vLLM command line."""

    attention_backend: str | None
    block_size: int | None = None
    environment: dict[str, str] = field(default_factory=dict)


METHODS: dict[str, Method] = {
    "baseline": Method(attention_backend=None),
    "baseline_triton": Method(attention_backend="TRITON_ATTN"),
    "hack": Method(attention_backend="CUSTOM"),
}


@dataclass
class ClusterSpec:
    method: str
    model: str
    run_dir: Path
    prefill_gpus: list[str]
    decode_gpus: list[str]
    max_model_len: int = 8192
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.9
    bandwidth_gbps: float | None = None
    hack_options: dict[str, str] = field(default_factory=dict)
    base_port: int = 8100
    proxy_port: int = 8000
    extra_args: list[str] = field(default_factory=list)


@dataclass
class Instance:
    role: str
    index: int
    gpu: str
    port: int
    transfer_port: int
    events_path: Path
    log_path: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class Cluster:
    def __init__(self, spec: ClusterSpec):
        self.spec = spec
        self.processes: list[subprocess.Popen] = []
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        self.prefill = [self._instance("prefill", i, gpu) for i, gpu in enumerate(spec.prefill_gpus)]
        self.decode = [self._instance("decode", i, gpu) for i, gpu in enumerate(spec.decode_gpus)]
        self.proxy_log = spec.run_dir / "proxy.jsonl"

    @property
    def proxy_url(self) -> str:
        return f"http://127.0.0.1:{self.spec.proxy_port}"

    def _instance(self, role: str, index: int, gpu: str) -> Instance:
        offset = index if role == "prefill" else 50 + index
        run_dir = self.spec.run_dir
        return Instance(
            role=role,
            index=index,
            gpu=gpu,
            port=self.spec.base_port + offset,
            transfer_port=self.spec.base_port + 1000 + offset,
            events_path=run_dir / f"{role}{index}_events.jsonl",
            log_path=run_dir / f"{role}{index}.log",
        )

    def _serve_command(self, instance: Instance) -> list[str]:
        spec, method = self.spec, METHODS[self.spec.method]
        extra = {"transfer_host": "127.0.0.1", "transfer_port": instance.transfer_port,
                 "stats_path": str(instance.events_path)}  # fmt: skip
        if instance.role == "prefill" and spec.bandwidth_gbps:
            extra["bandwidth_gbps"] = spec.bandwidth_gbps
        transfer = dict(CONNECTOR, kv_role="kv_producer" if instance.role == "prefill" else "kv_consumer",
                        kv_connector_extra_config=extra)  # fmt: skip
        command = [
            sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", spec.model,
            "--host", "127.0.0.1", "--port", str(instance.port),
            "--dtype", "bfloat16", "--enforce-eager",
            "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
            "--max-model-len", str(spec.max_model_len),
            "--max-num-batched-tokens", str(spec.max_model_len),
            "--max-num-seqs", str(spec.max_num_seqs),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            "--kv-transfer-config", json.dumps(transfer),
        ]  # fmt: skip
        if method.attention_backend:
            command += ["--attention-backend", method.attention_backend]
        if method.block_size:
            command += ["--block-size", str(method.block_size)]
        if spec.hack_options and method.attention_backend == "CUSTOM":
            command += ["--additional-config", json.dumps({"hack": spec.hack_options})]
        return command + spec.extra_args

    def _environment(self, instance: Instance) -> dict[str, str]:
        env = dict(os.environ)
        env.update(METHODS[self.spec.method].environment)
        env["CUDA_VISIBLE_DEVICES"] = instance.gpu
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        return env

    def start(self, timeout: float = 1800.0) -> None:
        ports = [self.spec.proxy_port] + [p for i in self.prefill + self.decode for p in (i.port, i.transfer_port)]
        busy = [port for port in ports if _port_in_use(port)]
        if busy:
            raise RuntimeError(
                f"ports {busy} are in use. If an earlier run was interrupted, stop its servers first: "
                "pkill -f 'vllm.entrypoints|hack.vllm_plugin.proxy'"
            )
        for instance in self.prefill + self.decode:
            for path in (instance.events_path, instance.log_path):
                path.unlink(missing_ok=True)
            log = open(instance.log_path, "w")
            self.processes.append(
                subprocess.Popen(
                    self._serve_command(instance), env=self._environment(instance), stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent,
                )  # fmt: skip
            )
        self.proxy_log.unlink(missing_ok=True)
        proxy = [sys.executable, "-m", "hack.vllm_plugin.proxy", "--port", str(self.spec.proxy_port),
                 "--log-path", str(self.proxy_log)]  # fmt: skip
        for instance in self.prefill:
            proxy += ["--prefill", instance.url]
        for instance in self.decode:
            proxy += ["--decode", instance.url]
        log = open(self.spec.run_dir / "proxy.log", "w")
        self.processes.append(
            subprocess.Popen(
                proxy, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_die_with_parent
            )
        )
        self._wait_until_healthy(timeout)

    def _wait_until_healthy(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        pending = [i.url + "/health" for i in self.prefill + self.decode] + [self.proxy_url + "/health"]
        while pending:
            if time.monotonic() > deadline:
                raise TimeoutError(f"not healthy in time: {pending}")
            for process in self.processes:
                if process.poll() is not None:
                    code, logs = process.returncode, self.spec.run_dir
                    raise RuntimeError(f"a cluster process exited with code {code}; see the logs in {logs}")
            pending = [url for url in pending if not _is_up(url)]
            time.sleep(2.0)

    def stop(self) -> None:
        for process in self.processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        for process in self.processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        self.processes = []

    def __enter__(self) -> "Cluster":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def stop_on_signals() -> None:
    """Make SIGTERM and SIGHUP (a closed terminal) end the program like Ctrl-C, so that the cluster is stopped."""

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for stop_signal in STOP_SIGNALS[1:]:
        signal.signal(stop_signal, interrupt)


def _is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except OSError:
        return False


def add_cluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--prefill-gpus", nargs="+", default=["0"], help="one prefill instance per entry")
    parser.add_argument("--decode-gpus", nargs="+", default=["1"], help="one decode instance per entry")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--bandwidth-gbps", type=float, default=None, help="emulated prefill-to-decode link")
    parser.add_argument("--hack-option", action="append", default=[], metavar="KEY=VALUE",
                        help="HACK setting, e.g. partition_size=64, kv_bits=2, summation_elimination=0")  # fmt: skip
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--proxy-port", type=int, default=8000)
    parser.add_argument("--vllm-arg", action="append", default=[], help="extra argument passed to `vllm serve`")


def spec_from_arguments(args: argparse.Namespace) -> ClusterSpec:
    return ClusterSpec(
        method=args.method,
        model=args.model,
        run_dir=args.run_dir,
        prefill_gpus=args.prefill_gpus,
        decode_gpus=args.decode_gpus,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        bandwidth_gbps=args.bandwidth_gbps,
        hack_options=dict(option.split("=", 1) for option in args.hack_option),
        base_port=args.base_port,
        proxy_port=args.proxy_port,
        extra_args=args.vllm_arg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cluster_arguments(parser)
    stop_on_signals()
    with Cluster(spec_from_arguments(parser.parse_args())) as cluster:
        print(f"cluster is ready: {cluster.proxy_url}", flush=True)
        signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        signal.sigwait(STOP_SIGNALS)


if __name__ == "__main__":
    main()

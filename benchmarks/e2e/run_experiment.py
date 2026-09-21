"""Run one end-to-end experiment: start the cluster, replay a workload, collect the metrics.

    python benchmarks/e2e/run_experiment.py --method hack --model Qwen/Qwen3-8B \
        --prefill-gpus 0 --decode-gpus 1 --run-dir runs/hack_10gbps --bandwidth-gbps 10 \
        --workload synthetic.jsonl --rate 0.5

Results: `<run-dir>/summary.json` and `<run-dir>/per_request.csv`.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import Cluster, add_cluster_arguments, spec_from_arguments, stop_on_signals  # noqa: E402
from loadgen import run_load  # noqa: E402
from metrics import summarize  # noqa: E402
from workload import read_workload  # noqa: E402

KV_USAGE_METRIC = "vllm:kv_cache_usage_perc"


def _gpu_memory_mib(gpu: str) -> tuple[float, float]:
    query = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", gpu]
    used, total = subprocess.run(query, capture_output=True, text=True, check=True).stdout.strip().split(",")
    return float(used), float(total)


def _kv_usage(url: str) -> float:
    with urllib.request.urlopen(url + "/metrics", timeout=5) as response:
        for line in response.read().decode().splitlines():
            if line.startswith(KV_USAGE_METRIC):
                return float(line.rsplit(" ", 1)[1])
    return 0.0


class MemoryMonitor(threading.Thread):
    """Samples GPU memory and KV-cache usage of the first decode instance."""

    def __init__(self, gpu: str, url: str, output: Path, period: float = 0.5):
        super().__init__(daemon=True)
        self.gpu, self.url, self.output, self.period = gpu.split(",")[0], url, output, period
        self._stop_event = threading.Event()

    def run(self) -> None:
        with open(self.output, "w") as stream:
            while not self._stop_event.is_set():
                try:
                    used, total = _gpu_memory_mib(self.gpu)
                    sample = {"time": time.time(), "gpu_used_mib": used, "gpu_total_mib": total,
                              "kv_usage": _kv_usage(self.url)}  # fmt: skip
                    stream.write(json.dumps(sample) + "\n")
                    stream.flush()
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                self._stop_event.wait(self.period)

    def stop(self) -> None:
        self._stop_event.set()
        self.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cluster_arguments(parser)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True, help="requests per second (Poisson arrivals)")
    parser.add_argument("--num-requests", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=2, help="requests replayed before the measurement")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = spec_from_arguments(args)
    items = read_workload(args.workload)[: args.num_requests]
    stop_on_signals()
    with Cluster(spec) as cluster:
        if args.warmup_requests:
            warmup = items[: args.warmup_requests]
            asyncio.run(run_load(cluster.proxy_url, spec.model, warmup, 0.0, args.seed, spec.run_dir / "warmup.jsonl"))
            for path in [cluster.proxy_log] + [i.events_path for i in cluster.prefill + cluster.decode]:
                _keep_only_cache_events(path)
        monitor = MemoryMonitor(cluster.decode[0].gpu, cluster.decode[0].url, spec.run_dir / "memory.jsonl")
        monitor.start()
        output = spec.run_dir / "requests.jsonl"
        asyncio.run(run_load(cluster.proxy_url, spec.model, items, args.rate, args.seed, output))
        monitor.stop()
    config = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    (spec.run_dir / "config.json").write_text(json.dumps(config, indent=2))
    summary = summarize(spec.run_dir)
    print(json.dumps(summary, indent=2))


def _keep_only_cache_events(path: Path) -> None:
    if not path.exists():
        return
    kept = [line for line in path.read_text().splitlines() if '"event": "cache"' in line]
    path.write_text("".join(line + "\n" for line in kept))


if __name__ == "__main__":
    main()

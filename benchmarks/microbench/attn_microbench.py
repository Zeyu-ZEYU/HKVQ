"""Decode attention-kernel microbenchmark.

One attention layer with the head configuration of Llama-3.1 70B (64 query heads, 8 KV
heads, head dimension 128), batch 1, one query token. A measurement decodes `--steps`
consecutive tokens, so the KV length grows from L + 1 to L + steps. For every method and
KV length the script reports, per decode step,

  T  the latency of the attention kernel,
  D  the time of a load-only kernel that reads exactly what the attention kernel reads,
  C  the time of the same arithmetic when all reads are wrapped onto a few resident blocks,

together with the number of KV bytes that one step reads. Kernels are replayed as CUDA
graphs after a warm-up; every number is the median of several rounds.
"""

import argparse
import csv
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import torch
import triton

from hack import HackConfig
from hack.cache import HackLayerCache
from hack.kernels import decode, fa2
from hack.kernels.common import ATTENTION, LOAD_ONLY, RESIDENT
from hack.kernels.decode import hack_decode
from hack.kernels.fa2 import bf16_source, fa2_decode_step, quantized_source
from hack.kernels.quantized_kv import QuantizedKV
from hack.kernels.tuning import LaunchConfig, candidates, graph_time, load, lookup, record, save, search

QUERY_HEADS, KV_HEADS, HEAD_DIM = 64, 8, 128
MODES = {"T": ATTENTION, "D": LOAD_ONLY, "C": RESIDENT}
Step = Callable[[int, LaunchConfig], torch.Tensor]


@dataclass
class Workload:
    """The decode steps of one method, the KV bytes one step reads, and the key of its launch configuration."""

    steps: list[Step]
    kv_bytes: float
    key: Callable[[int], tuple]


def snapshot(cache: HackLayerCache) -> HackLayerCache:
    """A cache that shares the tensors of `cache` and keeps its current length."""
    view = HackLayerCache(cache.config, cache.meta_dtype)
    view.load_state_dict(cache.tensors(), cache.num_tokens, HEAD_DIM)
    return view


def hack_workload(k, v, queries, args) -> Workload:
    config = HackConfig(partition_size=args.partition_size, kv_bits=2, stochastic=args.rounding == "stochastic")
    cache = HackLayerCache(config)
    cache.append(k[:, :, : args.length], v[:, :, : args.length])
    caches = []
    for i in range(args.steps):
        token = slice(args.length + i, args.length + i + 1)
        cache.append(k[:, :, token], v[:, :, token])
        caches.append(snapshot(cache))
    scaling = 1.0 / math.sqrt(HEAD_DIM)
    count = range(args.steps)
    steps = [lambda mode, cfg, i=i: hack_decode(queries[i], caches[i], scaling, 1 + i, mode, cfg) for i in count]
    key = lambda mode: decode.decode_key(caches[-1], QUERY_HEADS, HEAD_DIM, mode)  # noqa: E731
    return Workload(steps, statistics.mean(c.nbytes() for c in caches), key)


def bf16_workload(k, v, queries, args) -> Workload:
    sources = [bf16_source(k[:, :, : args.length + i + 1], v[:, :, : args.length + i + 1]) for i in range(args.steps)]
    scaling = 1.0 / math.sqrt(HEAD_DIM)
    count = range(args.steps)
    steps = [lambda mode, cfg, i=i: fa2_decode_step(queries[i], sources[i], scaling, mode, cfg) for i in count]
    kv_bytes = statistics.mean(2 * s.num_tokens * KV_HEADS * HEAD_DIM * k.element_size() for s in sources)
    return Workload(steps, kv_bytes, lambda mode: fa2.decode_key(sources[-1], QUERY_HEADS, HEAD_DIM, mode))


def dequant_workload(k, v, queries, args) -> Workload:
    qkv = QuantizedKV.from_tensors(k, v, bits=2, group_size=args.partition_size)
    views = [qkv.narrow(args.length + i + 1) for i in range(args.steps)]
    sources = [quantized_source(view, skip_dequant=False) for view in views]
    scaling = 1.0 / math.sqrt(HEAD_DIM)
    count = range(args.steps)
    steps = [lambda mode, cfg, i=i: fa2_decode_step(queries[i], sources[i], scaling, mode, cfg) for i in count]
    kv_bytes = statistics.mean(view.nbytes() for view in views)
    return Workload(steps, kv_bytes, lambda mode: fa2.decode_key(sources[-1], QUERY_HEADS, HEAD_DIM, mode))


WORKLOADS = {"hack": hack_workload, "bf16": bf16_workload, "dequant": dequant_workload}
TILES = {"hack": (0,), "bf16": (64, 128), "dequant": (64, 128)}
DEFAULTS = {"hack": decode.DEFAULT_LAUNCH, "bf16": fa2.DEFAULT_LAUNCH, "dequant": fa2.DEFAULT_LAUNCH}


def tune(steps: list[Step], mode: int, method: str, length: int) -> LaunchConfig:
    """Coordinate search: sequence splits, then compiler options and tile, then splits again."""

    def run(config: LaunchConfig):
        return steps[-1](mode, config)

    space = candidates(length, TILES[method])
    best = replace(DEFAULTS[method], splits=min(64, max(length // 64, 1)), tile=TILES[method][0])
    for same in (("num_warps", "num_stages", "tile"), ("splits",), ("num_warps", "num_stages", "tile")):
        subset = [c for c in space if all(getattr(c, name) == getattr(best, name) for name in same)]
        best, _ = search(run, subset or [best])
    return best


def measure(steps: list[Step], mode: int, config: LaunchConfig, args) -> float:
    """Median latency of one decode step in microseconds."""

    def run():
        for step in steps:
            step(mode, config)

    seconds = graph_time(run, warmup=args.warmup, replays=args.replays, rounds=args.rounds)
    return seconds / len(steps) * 1e6


def benchmark(method: str, args, device: torch.device) -> dict:
    generator = torch.Generator(device=device).manual_seed(args.length)
    noise = dict(device=device, dtype=torch.bfloat16, generator=generator)
    k = torch.randn(1, KV_HEADS, args.length + args.steps, HEAD_DIM, **noise)
    v = torch.randn(1, KV_HEADS, args.length + args.steps, HEAD_DIM, **noise)
    queries = [torch.randn(1, QUERY_HEADS, 1, HEAD_DIM, **noise) for _ in range(args.steps)]
    workload = WORKLOADS[method](k, v, queries, args)
    row = {"method": method, "kv_length": args.length, "kv_bytes_per_step": round(workload.kv_bytes)}
    for name, mode in MODES.items():
        config = lookup(workload.key(mode), None if args.tune else DEFAULTS[method])
        if config is None:
            config = tune(workload.steps, mode, method, args.length)
            record(workload.key(mode), config)
        row[f"{name}_us"] = measure(workload.steps, mode, config, args)
        row[f"{name}_config"] = asdict(config)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768, 65536, 131072])
    parser.add_argument("--methods", nargs="+", choices=sorted(WORKLOADS), default=["hack", "bf16"])
    parser.add_argument("--steps", type=int, default=8, help="output tokens per measurement")
    parser.add_argument("--partition-size", type=int, default=64)
    rounding = "stochastic" if HackConfig().stochastic else "nearest"
    parser.add_argument("--rounding", choices=["stochastic", "nearest"], default=rounding)
    parser.add_argument("--no-tune", dest="tune", action="store_false", help="use the default launch configuration")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", default="attn_microbench.json")
    parser.add_argument("--tuning-file", default="attn_microbench_tuning.json", help="launch configurations to reuse")
    args = parser.parse_args()
    if args.tune and os.path.exists(args.tuning_file):
        load(args.tuning_file)

    device = torch.device("cuda")
    rows = []
    for args.length in args.lengths:
        for method in args.methods:
            rows.append(benchmark(method, args, device))
            row = rows[-1]
            print(
                f"{method:8s} L={args.length:7d}  T={row['T_us']:8.1f} us  D={row['D_us']:8.1f} us  "
                f"C={row['C_us']:8.1f} us  KV bytes/step={row['kv_bytes_per_step'] / 2**20:8.1f} MiB",
                flush=True,
            )
    environment = {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "triton": triton.__version__,
        "query_heads": QUERY_HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "steps": args.steps,
        "partition_size": args.partition_size, "rounding": args.rounding, "sink_tokens": HackConfig().sink_tokens,
        "tuned": args.tune,
    }  # fmt: skip
    if args.tune:
        save(args.tuning_file)
    with open(args.output, "w") as file:
        json.dump({"environment": environment, "results": rows}, file, indent=1)
    with open(args.output.rsplit(".", 1)[0] + ".csv", "w", newline="") as file:
        columns = ["method", "kv_length", "T_us", "D_us", "C_us", "kv_bytes_per_step"]
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

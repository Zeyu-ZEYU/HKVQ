"""Launch configurations of the attention kernels and a small search over them."""

import functools
import json
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

MAX_SPLITS = 256


@dataclass(frozen=True)
class LaunchConfig:
    """splits: sequence-level parallelism of the decode kernels (0 = derive it from the length).
    num_warps, num_stages, num_ctas, maxnreg: compiler options of the Triton kernel (0 = compiler default).
    tile: tokens per KV tile where the kernel is free to choose it (0 = kernel default)."""

    splits: int = 0
    num_warps: int = 4
    num_stages: int = 3
    num_ctas: int = 1
    tile: int = 0
    maxnreg: int = 0

    def options(self) -> dict[str, int]:
        options = {"num_warps": self.num_warps, "num_stages": self.num_stages}
        if self.num_ctas > 1:
            options["num_ctas"] = self.num_ctas
        if self.maxnreg:
            options["maxnreg"] = self.maxnreg
        return options

    def resolve_splits(self, tiles: int) -> int:
        if self.splits:
            return max(1, min(self.splits, tiles, MAX_SPLITS))
        return default_splits(tiles)


def default_splits(tiles: int) -> int:
    """Largest power of two that leaves at least two tiles per split, at most 128."""
    splits = 1
    while splits < 128 and 4 * splits <= tiles:
        splits *= 2
    return splits


_registry: dict[tuple, LaunchConfig] = {}


@functools.lru_cache(maxsize=None)
def _device_name(index: int) -> str:
    return torch.cuda.get_device_name(index)


def device_name(device: torch.device | None = None) -> str:
    """Name of a CUDA device (default: the current one), as used in the keys of the launch configurations."""
    index = device.index if device is not None and device.index is not None else torch.cuda.current_device()
    return _device_name(index)


def length_bucket(tiles: int) -> int:
    return max(tiles, 1).bit_length()


def lookup(key: tuple, default: LaunchConfig | None) -> LaunchConfig | None:
    return _registry.get(key, default)


def record(key: tuple, config: LaunchConfig) -> None:
    _registry[key] = config


def candidates(tokens: int, tiles: tuple[int, ...] = (0,)) -> list[LaunchConfig]:
    """Launch configurations worth measuring for a decode step over `tokens` tokens."""
    limit = max(tokens // 64, 1)
    splits = sorted({min(s, limit, MAX_SPLITS) for s in (1, 8, 32, 64, 128, 256)})
    return [LaunchConfig(s, w, n, 1, t) for t in tiles for s in splits for w in (1, 2, 4, 8) for n in (1, 3, 4)]


def graph_time(run: Callable[[], object], warmup: int = 10, replays: int = 50, rounds: int = 3) -> float:
    """Median time in seconds of one CUDA-graph replay of `run`."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    times = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(replays):
            graph.replay()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop) / replays / 1e3)
    return statistics.median(times)


def search(run: Callable[[LaunchConfig], object], configs: Iterable[LaunchConfig]) -> tuple[LaunchConfig, float]:
    """Return the fastest of `configs` for `run` and its time; configurations the device cannot run are skipped."""
    best, best_time = LaunchConfig(), float("inf")
    for config in configs:
        try:
            elapsed = graph_time(lambda: run(config), warmup=3, replays=30, rounds=3)
        except (CompilationError, OutOfResources, RuntimeError, ValueError):
            continue
        if elapsed < best_time:
            best, best_time = config, elapsed
    return best, best_time


def save(path: str) -> None:
    with open(path, "w") as file:
        json.dump([{"key": list(key), "config": asdict(config)} for key, config in _registry.items()], file, indent=1)


def _as_key(value):
    return tuple(_as_key(item) for item in value) if isinstance(value, list) else value


def load(path: str) -> None:
    with open(path) as file:
        for entry in json.load(file):
            record(_as_key(entry["key"]), LaunchConfig(**entry["config"]))

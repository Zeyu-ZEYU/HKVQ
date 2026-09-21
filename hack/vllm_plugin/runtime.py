"""State shared by all attention layers of one worker process."""

from collections.abc import Callable

import torch

from hack.vllm_plugin.layout import PageLayout
from hack.vllm_plugin.plan import StepPlanner
from hack.vllm_plugin.settings import PluginSettings
from hack.vllm_plugin.slots import SlotPool
from hack.vllm_plugin.store import LayerStore


class HackRuntime:
    def __init__(self, settings: PluginSettings, max_num_seqs: int, device: torch.device):
        self.settings = settings
        self.device = device
        self.num_side_slots = settings.num_side_slots(max_num_seqs)
        self.open_slots = SlotPool(self.num_side_slots, device, settings.debug_checks)
        self.sink_slots = (
            SlotPool(self.num_side_slots, device, settings.debug_checks) if settings.config.sink_tokens else None
        )
        self.planner = StepPlanner(settings.config, self.open_slots, self.sink_slots, device)
        self.stores: list[LayerStore] = []
        self._layouts: dict[tuple[int, int], PageLayout] = {}

    def layout(self, num_kv_heads: int, head_dim: int) -> PageLayout:
        key = (num_kv_heads, head_dim)
        if key not in self._layouts:
            self._layouts[key] = PageLayout(self.settings.config, num_kv_heads, head_dim)
        return self._layouts[key]

    def new_store(self, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> LayerStore:
        store = LayerStore(self.layout(num_kv_heads, head_dim), self.num_side_slots, dtype, self.device)
        self.stores.append(store)
        return store

    def bind_blocks(self, num_blocks: int) -> None:
        self.open_slots.bind(num_blocks)
        if self.sink_slots is not None:
            self.sink_slots.bind(num_blocks)

    def side_pool_bytes(self) -> int:
        return sum(store.side_pool_bytes() for store in self.stores)


_runtime: HackRuntime | None = None


def get_runtime() -> HackRuntime | None:
    return _runtime


def ensure_runtime(settings: PluginSettings, max_num_seqs: int, device: torch.device) -> HackRuntime:
    global _runtime
    if _runtime is None:
        _runtime = HackRuntime(settings, max_num_seqs, device)
    return _runtime


def reset_runtime() -> None:
    global _runtime
    _runtime = None


def resolve_attention(name: str) -> Callable[..., torch.Tensor]:
    """Attention over a contiguous `HackLayerCache`: the Triton kernels or the PyTorch reference."""
    if name == "kernels":
        from hack.kernels import hack_attention
    else:
        from hack.attention_ref import hack_attention
    return hack_attention


def resolve_paged_decode(name: str) -> Callable[..., torch.Tensor]:
    """Batched decode attention that reads the pages in place: the Triton kernel or the PyTorch reference."""
    if name == "kernels":
        from hack.kernels.paged import paged_decode_attention
    else:
        from hack.vllm_plugin.decode_ref import paged_decode_attention
    return paged_decode_attention

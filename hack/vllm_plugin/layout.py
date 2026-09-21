"""Byte layout of one KV-cache page."""

from dataclasses import dataclass
from math import prod

import torch

from hack.cache import channel_partition_size
from hack.config import HackConfig
from hack.quant import sum_dtype

FIELD_ALIGNMENT = 8


@dataclass(frozen=True)
class PageField:
    name: str
    offset: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def nbytes(self) -> int:
        return prod(self.shape) * self.dtype.itemsize


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class PageLayout:
    def __init__(self, config: HackConfig, num_kv_heads: int, head_dim: int, meta_dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.meta_dtype = meta_dtype
        self.block_size = config.partition_size
        self.channel_part = channel_partition_size(head_dim, config.partition_size)
        self.channel_partitions = head_dim // self.channel_part
        self.packed_dim = head_dim * config.kv_bits // 8

        tokens, heads, parts = self.block_size, num_kv_heads, self.channel_partitions
        specs: list[tuple[str, tuple[int, ...], torch.dtype]] = [
            ("k_codes", (heads, tokens, self.packed_dim), torch.uint8),
            ("k_scale", (heads, head_dim), meta_dtype),
            ("k_min", (heads, head_dim), meta_dtype),
        ]
        if config.summation_elimination:
            specs.append(("k_sums", (heads, tokens, parts), sum_dtype(self.channel_part, config.kv_bits)))
        specs += [
            ("v_codes", (heads, tokens, self.packed_dim), torch.uint8),
            ("v_scale", (heads, tokens, parts), meta_dtype),
            ("v_min", (heads, tokens, parts), meta_dtype),
        ]
        if config.summation_elimination:
            specs.append(("v_sums", (heads, head_dim), sum_dtype(self.block_size, config.kv_bits)))

        self.fields: dict[str, PageField] = {}
        offset = 0
        for name, shape, dtype in specs:
            offset = _round_up(offset, FIELD_ALIGNMENT)
            self.fields[name] = PageField(name, offset, shape, dtype)
            offset += self.fields[name].nbytes
        self.used_bytes = offset
        self.page_bytes = _round_up(offset, 4 * self.block_size)

    @property
    def bytes_per_token_slot(self) -> int:
        return self.page_bytes // self.block_size

    def full_precision_page_bytes(self, dtype: torch.dtype = torch.bfloat16) -> int:
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * dtype.itemsize

    def as_pages(self, kv_cache: torch.Tensor) -> torch.Tensor:
        """View the per-layer cache tensor of vLLM as [num_blocks, page_bytes] uint8."""
        if kv_cache.dtype != torch.uint8:
            kv_cache = kv_cache.view(torch.uint8)
        num_blocks = kv_cache.shape[0]
        if prod(kv_cache.shape[1:]) != self.page_bytes:
            raise ValueError(f"cache tensor {tuple(kv_cache.shape)} does not hold pages of {self.page_bytes} bytes")
        return kv_cache.as_strided((num_blocks, self.page_bytes), (kv_cache.stride(0), 1))

    def views(self, pages: torch.Tensor) -> dict[str, torch.Tensor]:
        """Typed views [num_blocks, *field.shape] that alias the bytes of `pages`."""
        result = {}
        for field in self.fields.values():
            raw = pages[:, field.offset : field.offset + field.nbytes]
            typed = raw if field.dtype == torch.uint8 else raw.view(field.dtype)
            result[field.name] = typed.unflatten(-1, field.shape)
        return result

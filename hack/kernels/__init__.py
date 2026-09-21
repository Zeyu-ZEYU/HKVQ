"""Triton kernels: attention on the quantized KV cache of HACK and the baselines it is compared with."""

import itertools
import math

import torch

from hack.attention_ref import hack_attention as _reference_attention
from hack.cache import HackLayerCache
from hack.kernels.decode import hack_decode
from hack.kernels.fa2 import bf16_source, fa2_attention, quantized_source
from hack.kernels.prefill import hack_prefill
from hack.kernels.quantized_kv import QuantizedKV

__all__ = ["QuantizedKV", "bf16_attention", "dequant_attention", "hack_attention"]

_calls = itertools.count(1)


def _power_of_two(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def _has_kernel(q: torch.Tensor, cache: HackLayerCache) -> bool:
    sizes_fit = _power_of_two(q.shape[-1]) and q.shape[-1] >= 32 and _power_of_two(cache.config.partition_size)
    return q.is_cuda and sizes_fit and cache.config.partition_size >= 32


def hack_attention(
    q: torch.Tensor, cache: HackLayerCache, scaling: float | None = None, causal: bool = True
) -> torch.Tensor:
    """Attention of the queries `q` [batch, heads, q_len, head_dim], the last q_len tokens, over `cache`.

    Both matrix multiplications over the closed blocks run on integer codes in Triton: one
    query token uses `attn_decode`, several use `attn_prefill`. Sizes without a kernel use
    the PyTorch reference.
    """
    if not _has_kernel(q, cache):
        return _reference_attention(q, cache, scaling, causal)
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])
    seed = (next(_calls) * 0x9E3779B1) & 0x7FFFFFFF
    if q.shape[2] == 1:
        return hack_decode(q, cache, scaling, seed)
    return hack_prefill(q, cache, scaling, seed, causal)


def bf16_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scaling: float | None = None, causal: bool = True
) -> torch.Tensor:
    """FlashAttention-2-style BF16 attention of `q`, the last q_len tokens, over `k`, `v` [batch, kv_heads, T, D]."""
    return fa2_attention(q, bf16_source(k, v), scaling, causal)


def dequant_attention(
    q: torch.Tensor, qkv: QuantizedKV, scaling: float | None = None, causal: bool = True, skip_dequant: bool = False
) -> torch.Tensor:
    """Attention over storage-only codes that the kernel dequantizes to BF16 tile by tile at every call.

    `skip_dequant` runs the same kernel without the dequantization arithmetic; its output is
    meaningless and only its run time is of interest.
    """
    return fa2_attention(q, quantized_source(qkv, skip_dequant), scaling, causal)

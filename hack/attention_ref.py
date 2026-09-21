"""Reference implementation of attention on quantized K and V (PyTorch)."""

import math

import torch

from hack.cache import HackLayerCache, quantize_partitions
from hack.config import HackConfig
from hack.homomorphic import homomorphic_matmul


def _repeat_kv(x: torch.Tensor | None, groups: int) -> torch.Tensor | None:
    return x if x is None or groups == 1 else x.repeat_interleave(groups, dim=1)


def _quantized_scores(q, cache: HackLayerCache, groups: int) -> torch.Tensor:
    """Q K^T over the quantized blocks: [batch, heads, q_len, quantized tokens]."""
    cfg = cache.config
    batch, heads, q_len, head_dim = q.shape
    part, block = cache.channel_part, cfg.partition_size
    blocks = cache.k_scale.length
    k_scale = _repeat_kv(cache.k_scale.view(), groups).float()
    k_min = _repeat_kv(cache.k_min.view(), groups).float()
    k_codes = _repeat_kv(cache.k_unpacked(), groups).reshape(batch, heads, blocks, block, head_dim).transpose(-2, -1)
    k_sums = None
    if cfg.summation_elimination:
        k_sums = _repeat_kv(cache.k_sums.view(), groups).reshape(batch, heads, blocks, block, -1).transpose(-2, -1)

    scaled_q = q.unsqueeze(2) * k_scale.unsqueeze(3)
    q_codes, q_scale, q_min = quantize_partitions(scaled_q, cfg.qp_bits, -1, part, cfg.stochastic, torch.float32)
    ones = torch.ones(batch, heads, blocks, head_dim // part, block, device=q.device)
    scores = homomorphic_matmul(q_codes, q_scale, q_min, k_codes, ones, torch.zeros_like(ones), part, b_sums=k_sums)
    scores = scores + (q @ k_min.transpose(-2, -1)).transpose(-2, -1).unsqueeze(-1)
    return scores.transpose(2, 3).reshape(batch, heads, q_len, blocks * block)


def _quantized_values(probs, cache: HackLayerCache, groups: int) -> torch.Tensor:
    """P V over the quantized blocks: [batch, heads, q_len, head_dim]."""
    cfg = cache.config
    batch, heads, q_len, tokens = probs.shape
    part, block = cache.channel_part, cfg.partition_size
    v_scale = _repeat_kv(cache.v_scale.view(), groups).float()
    v_min = _repeat_kv(cache.v_min.view(), groups).float()
    parts = v_scale.shape[-1]
    v_codes = _repeat_kv(cache.v_unpacked(), groups).reshape(batch, heads, tokens, parts, part).transpose(2, 3)
    v_sums = None
    if cfg.summation_elimination:
        v_sums = _repeat_kv(cache.v_sums.view(), groups).reshape(batch, heads, tokens // block, parts, part).transpose(2, 3)

    scaled_p = probs.unsqueeze(2) * v_scale.permute(0, 1, 3, 2).unsqueeze(3)
    p_codes, p_scale, p_min = quantize_partitions(scaled_p, cfg.qp_bits, -1, block, cfg.stochastic, torch.float32)
    ones = torch.ones(batch, heads, parts, tokens // block, part, device=probs.device)
    values = homomorphic_matmul(p_codes, p_scale, p_min, v_codes, ones, torch.zeros_like(ones), block, b_sums=v_sums)
    values = values + (probs @ v_min).transpose(2, 3).unsqueeze(-1)
    return values.transpose(2, 3).reshape(batch, heads, q_len, parts * part)


def hack_attention(
    q: torch.Tensor,
    cache: HackLayerCache,
    scaling: float | None = None,
    causal: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Attention of the queries `q` [batch, heads, q_len, head_dim] over everything in `cache`.

    The queries are taken to be the last `q_len` tokens of the cached sequence.
    """
    cfg: HackConfig = cache.config
    batch, heads, q_len, head_dim = q.shape
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(head_dim)
    kv_heads = next(t for t in (cache.k_scale.data, cache.sink_k, cache.k_open) if t is not None).shape[1]
    groups = heads // kv_heads
    total, sinks, quantized = cache.num_tokens, cache.num_sink_tokens, cache.num_quantized_tokens
    sink_k, sink_v = _repeat_kv(cache.sink_k, groups), _repeat_kv(cache.sink_v, groups)
    k_open, v_open = _repeat_kv(cache.k_open, groups), _repeat_kv(cache.v_open, groups)
    if chunk_size is None:
        chunk_size = max(16, min(256, (1 << 22) // max(1, quantized)))

    out = torch.empty(batch, heads, q_len, head_dim, dtype=q.dtype, device=q.device)
    positions = torch.arange(total, device=q.device)
    for start in range(0, q_len, chunk_size):
        stop = min(start + chunk_size, q_len)
        q_chunk = q[:, :, start:stop].float()
        pieces = []
        if sinks:
            pieces.append(q_chunk @ sink_k.float().transpose(-2, -1))
        if quantized:
            pieces.append(_quantized_scores(q_chunk, cache, groups))
        if k_open is not None:
            pieces.append(q_chunk @ k_open.float().transpose(-2, -1))
        scores = torch.cat(pieces, dim=-1) * scaling
        if causal:
            query_positions = positions[total - q_len + start : total - q_len + stop]
            scores = scores.masked_fill(positions[None, :] > query_positions[:, None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)

        result = torch.zeros(batch, heads, stop - start, head_dim, dtype=torch.float32, device=q.device)
        if sinks:
            result += probs[..., :sinks] @ sink_v.float()
        if quantized:
            result += _quantized_values(probs[..., sinks : sinks + quantized], cache, groups)
        if v_open is not None:
            result += probs[..., sinks + quantized :] @ v_open.float()
        out[:, :, start:stop] = result.to(q.dtype)
    return out

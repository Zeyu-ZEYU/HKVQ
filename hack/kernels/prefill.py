"""`attn_prefill`: attention of a run of new tokens over a `HackLayerCache` (FlashAttention-2-style tiling)."""

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from hack.cache import HackLayerCache
from hack.kernels.common import from_field_order, rows_to_field_order, softmax_step
from hack.kernels.decode import DENSE_TILE, block_values, dense_segment, kernel_tensors, tile_scores

QUERY_ROWS = 64


@triton.jit(do_not_specialize=["num_blocks", "num_sinks", "num_open", "q_len", "seed"])
def attn_prefill(
    q_ptr, out_ptr,
    k_words_ptr, k_sums_ptr, k_scale_ptr, k_min_ptr, v_words_ptr, v_scale_ptr, v_min_ptr, v_sums_ptr,
    sink_k_ptr, sink_v_ptr, open_k_ptr, open_v_ptr,
    stride_qb, stride_qh, stride_ql, stride_ob, stride_oh, stride_ol,
    stride_wb, stride_wh, stride_tb, stride_th, stride_cb, stride_ch, stride_sb, stride_sh, stride_nb, stride_nh,
    kv_heads, num_blocks, num_sinks, num_open, q_len, scaling, seed,
    G: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, PI: tl.constexpr, BITS: tl.constexpr, QT: tl.constexpr,
    STOCHASTIC: tl.constexpr, USE_SUMS: tl.constexpr, CAUSAL: tl.constexpr,
):  # fmt: skip
    ROWS: tl.constexpr = G * QT
    WORDS: tl.constexpr = D * BITS // 32
    DT: tl.constexpr = min(PI, DENSE_TILE)
    batch = (tl.program_id(0) // kv_heads).to(tl.int64)
    head = (tl.program_id(0) % kv_heads).to(tl.int64)
    words = batch * stride_wb + head * stride_wh
    tokens = batch * stride_tb + head * stride_th
    channels = batch * stride_cb + head * stride_ch
    sinks = batch * stride_sb + head * stride_sh
    opened = batch * stride_nb + head * stride_nh
    num_tokens = num_sinks + num_blocks * PI + num_open

    heads = head * G + tl.arange(0, G)
    q_rows = tl.program_id(1) * QT + tl.arange(0, QT)
    dims = tl.arange(0, D)
    row_valid = tl.reshape(tl.broadcast_to(q_rows[None, :] < q_len, (G, QT)), (ROWS,))
    positions = tl.reshape(tl.broadcast_to(num_tokens - q_len + q_rows[None, :], (G, QT)), (ROWS,))
    q_offsets = heads[:, None, None] * stride_qh + q_rows[None, :, None] * stride_ql + dims[None, None, :]
    q_offsets += batch * stride_qb
    q = tl.load(tl.reshape(q_ptr + q_offsets, (ROWS, D)), mask=row_valid[:, None], other=0.0).to(tl.float32)
    seed ^= (tl.program_id(0) * 7919 + tl.program_id(1)) * 104729

    running_max = tl.full((ROWS,), float("-inf"), tl.float32)
    running_norm = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    limit = num_tokens
    if CAUSAL:
        limit = tl.minimum(num_tokens, num_tokens - q_len + (tl.program_id(1) + 1) * QT)
    running_max, running_norm, acc = dense_segment(
        q, sink_k_ptr + sinks, sink_v_ptr + sinks, num_sinks, 0, positions, scaling,
        running_max, running_norm, acc, DT, D, CAUSAL,
    )  # fmt: skip
    if BITS == 2:
        acc = rows_to_field_order(acc, ROWS, D)
    for block in range(0, tl.minimum(num_blocks, tl.maximum(limit - num_sinks + PI - 1, 0) // PI)):
        k_words, v_words = k_words_ptr + words + block * (PI * WORDS), v_words_ptr + words + block * (PI * WORDS)
        token_meta, channel_meta = tokens + block * (PI * NP), channels + block * D
        scores = tile_scores(
            q, k_words, k_sums_ptr + token_meta, k_scale_ptr + channel_meta, k_min_ptr + channel_meta,
            seed + block, scaling, PI, D, NP, BITS, STOCHASTIC, USE_SUMS,
        )  # fmt: skip
        if CAUSAL:
            visible = (num_sinks + block * PI + tl.arange(0, PI))[None, :] <= positions[:, None]
            scores = tl.where(visible, scores, float("-inf"))
        weights, decay, running_max, running_norm = softmax_step(scores, running_max, running_norm)
        acc = acc * decay[:, None] + block_values(
            weights, v_words, v_scale_ptr + token_meta, v_min_ptr + token_meta, v_sums_ptr + channel_meta,
            seed + block, PI, D, NP, BITS, STOCHASTIC, USE_SUMS,
        )  # fmt: skip
    if BITS == 2:
        acc = from_field_order(acc, ROWS, D)

    running_max, running_norm, acc = dense_segment(
        q, open_k_ptr + opened, open_v_ptr + opened, num_open, num_sinks + num_blocks * PI, positions, scaling,
        running_max, running_norm, acc, DT, D, CAUSAL,
    )  # fmt: skip

    out_offsets = heads[:, None, None] * stride_oh + q_rows[None, :, None] * stride_ol + dims[None, None, :]
    out_offsets += batch * stride_ob
    tl.store(tl.reshape(out_ptr + out_offsets, (ROWS, D)), acc / running_norm[:, None], mask=row_valid[:, None])


_fitting_rows: dict[tuple, int] = {}


def hack_prefill(q: torch.Tensor, cache: HackLayerCache, scaling: float, seed: int, causal: bool) -> torch.Tensor:
    """Attention of `q` [batch, heads, q_len, head_dim], the last q_len tokens of the sequence, over `cache`.

    A program owns as many query rows as the shared memory of the device allows, at most QUERY_ROWS.
    """
    cfg = cache.config
    batch, heads, q_len, head_dim = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    t = kernel_tensors(cache, q)
    kv_heads = next(x for x in (cache.k_scale.data, cache.sink_k, cache.k_open) if x is not None).shape[1]
    groups = heads // kv_heads
    out = torch.empty_like(q)
    strides = [x.stride(i) for x in (q, out) for i in (0, 1, 2)]
    strides += [t[name][0].stride(i) for name in ("words", "tokens", "channels", "sinks", "opened") for i in (0, 1)]
    k_words, v_words = t["words"]
    k_sums, v_scale, v_min = t["tokens"]
    k_scale, k_min, v_sums = t["channels"]
    sizes = (groups, head_dim, cfg.partition_size, cfg.kv_bits)
    key = (q.device, sizes, cfg.stochastic, cfg.summation_elimination, causal)
    rows = _fitting_rows.get(key, QUERY_ROWS)
    while True:
        query_tile = max(1, rows // triton.next_power_of_2(groups))
        try:
            attn_prefill[(batch * kv_heads, triton.cdiv(q_len, query_tile))](
                q, out, k_words, k_sums, k_scale, k_min, v_words, v_scale, v_min, v_sums,
                *t["sinks"], *t["opened"], *strides,
                kv_heads, cache.k_scale.length, cache.num_sink_tokens, cache.num_open_tokens, q_len, scaling, seed,
                G=groups, D=head_dim, NP=head_dim // cache.channel_part, PI=cfg.partition_size, BITS=cfg.kv_bits,
                QT=query_tile, STOCHASTIC=cfg.stochastic, USE_SUMS=cfg.summation_elimination, CAUSAL=causal,
            )  # fmt: skip
            break
        except OutOfResources:
            if query_tile == 1:
                raise
            rows //= 2
    _fitting_rows[key] = rows
    return out

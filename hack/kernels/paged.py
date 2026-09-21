"""Decode attention of a batch of sequences of different lengths on a paged cache."""

import itertools
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from hack.config import HackConfig
from hack.kernels.common import combine_splits, from_field_order, softmax_step
from hack.kernels.decode import DEFAULT_LAUNCH, DENSE_TILE, block_values, dense_segment, tile_scores
from hack.kernels.tuning import LaunchConfig, device_name, length_bucket, lookup

CODE_FIELDS = ("k_codes", "v_codes")
FIELDS = ("k_codes", "k_sums", "k_scale", "k_min", "v_codes", "v_scale", "v_min", "v_sums")
_calls = itertools.count(1)


@dataclass
class PagedCache:
    """Where the kernel finds the cache of a batch of decode rows.

    fields: per-field views of the page pool, [pages, kv_heads, ...] as in `HackLayerCache`
        with one closed block per page; `k_sums` and `v_sums` are absent without summation elimination.
    sink_k, sink_v: [slots, kv_heads, sink_tokens, head_dim] or None; open_k, open_v: [slots, kv_heads, Pi, head_dim].
    block_table: [rows, blocks]; totals: [rows] tokens of every row; sink_slot, open_slot: [rows].
    max_closed: upper bound of the closed blocks of a row.
    """

    config: HackConfig
    channel_part: int
    fields: dict[str, torch.Tensor]
    sink_k: torch.Tensor | None
    sink_v: torch.Tensor | None
    open_k: torch.Tensor
    open_v: torch.Tensor
    block_table: torch.Tensor
    totals: torch.Tensor
    sink_slot: torch.Tensor
    open_slot: torch.Tensor
    max_closed: int


@triton.jit(do_not_specialize=["stride_table", "blocks_per_split", "seed", "num_splits"])
def attn_decode_paged(
    q_ptr, out_ptr, part_out_ptr, part_max_ptr, part_norm_ptr,
    k_codes_ptr, k_sums_ptr, k_scale_ptr, k_min_ptr, v_codes_ptr, v_scale_ptr, v_min_ptr, v_sums_ptr,
    sink_k_ptr, sink_v_ptr, open_k_ptr, open_v_ptr, block_table_ptr, totals_ptr, sink_slot_ptr, open_slot_ptr,
    stride_qr, stride_qh, stride_or, stride_oh,
    kc_page, kc_head, ks_page, ks_head, kx_page, kx_head, km_page, km_head,
    vc_page, vc_head, vx_page, vx_head, vm_page, vm_head, vs_page, vs_head,
    stride_sink, stride_sink_head, stride_open, stride_open_head, stride_table,
    kv_heads, blocks_per_split, scaling, seed, num_splits,
    G: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, PI: tl.constexpr, BITS: tl.constexpr, SINKS: tl.constexpr,
    STOCHASTIC: tl.constexpr, USE_SUMS: tl.constexpr, SINGLE: tl.constexpr,
):  # fmt: skip
    DT: tl.constexpr = min(PI, DENSE_TILE)
    row = (tl.program_id(0) // kv_heads).to(tl.int64)
    head = (tl.program_id(0) % kv_heads).to(tl.int64)
    split = tl.program_id(1)
    total = tl.load(totals_ptr + row).to(tl.int32)
    num_sinks = tl.minimum(total, SINKS)
    num_blocks = (total - num_sinks) // PI
    num_open = total - num_sinks - num_blocks * PI

    heads = head * G + tl.arange(0, G)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + row * stride_qr + heads[:, None] * stride_qh + dims[None, :]).to(tl.float32)
    seed ^= tl.program_id(0) * 7919

    running_max = tl.full((G,), float("-inf"), tl.float32)
    running_norm = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, D), tl.float32)
    first_block = split * blocks_per_split
    for block in range(first_block, tl.minimum(first_block + blocks_per_split, num_blocks)):
        page = tl.load(block_table_ptr + row * stride_table + block).to(tl.int64)
        k_words = (k_codes_ptr + page * kc_page + head * kc_head).to(tl.pointer_type(tl.int32))
        v_words = (v_codes_ptr + page * vc_page + head * vc_head).to(tl.pointer_type(tl.int32))
        scores = tile_scores(
            q, k_words, k_sums_ptr + page * ks_page + head * ks_head, k_scale_ptr + page * kx_page + head * kx_head,
            k_min_ptr + page * km_page + head * km_head, seed + block, scaling, PI, D, NP, BITS, STOCHASTIC, USE_SUMS,
        )  # fmt: skip
        weights, decay, running_max, running_norm = softmax_step(scores, running_max, running_norm)
        acc = acc * decay[:, None] + block_values(
            weights, v_words, v_scale_ptr + page * vx_page + head * vx_head,
            v_min_ptr + page * vm_page + head * vm_head, v_sums_ptr + page * vs_page + head * vs_head,
            seed + block, PI, D, NP, BITS, STOCHASTIC, USE_SUMS,
        )  # fmt: skip
    if BITS == 2:
        acc = from_field_order(acc, G, D)

    if split == (0 if SINGLE else num_splits):
        sink_slot = tl.load(sink_slot_ptr + row).to(tl.int64)
        open_slot = tl.load(open_slot_ptr + row).to(tl.int64)
        sinks = sink_slot * stride_sink + head * stride_sink_head
        opened = open_slot * stride_open + head * stride_open_head
        running_max, running_norm, acc = dense_segment(
            q, sink_k_ptr + sinks, sink_v_ptr + sinks, num_sinks, 0, heads, scaling,
            running_max, running_norm, acc, DT, D, False,
        )  # fmt: skip
        running_max, running_norm, acc = dense_segment(
            q, open_k_ptr + opened, open_v_ptr + opened, num_open, 0, heads, scaling,
            running_max, running_norm, acc, DT, D, False,
        )  # fmt: skip

    if SINGLE:
        tl.store(out_ptr + row * stride_or + heads[:, None] * stride_oh + dims[None, :], acc / running_norm[:, None])
    else:
        slot = ((row * kv_heads + head) * (num_splits + 1) + split) * G + tl.arange(0, G)
        tl.store(part_out_ptr + slot[:, None] * D + dims[None, :], acc)
        tl.store(part_max_ptr + slot, running_max)
        tl.store(part_norm_ptr + slot, running_norm)


def _check_layout(cache: PagedCache, head_dim: int) -> None:
    """The kernel reads every (page, head) and every (slot, head) as one contiguous range."""
    for name, field in cache.fields.items():
        inner = field.shape[2:]
        expected = tuple(torch.empty(inner, device="meta").stride())
        if tuple(field.stride()[2:]) != expected:
            raise ValueError(f"field {name} is not contiguous inside a page")
    for pool in (cache.sink_k, cache.sink_v, cache.open_k, cache.open_v):
        if pool is not None and tuple(pool.stride()[2:]) != (head_dim, 1):
            raise ValueError("the 16-bit pools must store the tokens of a head contiguously")
    for name in CODE_FIELDS:
        codes = cache.fields[name]
        if codes.data_ptr() % 4 or codes.stride(0) % 4 or codes.stride(1) % 4:
            raise ValueError(f"{name} must be aligned to 4 bytes")


def paged_decode(
    q: torch.Tensor, cache: PagedCache, scaling: float, seed: int = 1, config: LaunchConfig | None = None,
    check_layout: bool = True,
) -> torch.Tensor:  # fmt: skip
    """Attention of one new token per row, `q` [rows, heads, head_dim], over the paged cache of every row."""
    cfg = cache.config
    rows, heads, head_dim = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    if check_layout:
        _check_layout(cache, head_dim)
    kv_heads = cache.open_k.shape[1]
    groups = heads // kv_heads
    use_sums = "k_sums" in cache.fields
    fields = [cache.fields[name] if name in cache.fields else cache.fields["k_codes"] for name in FIELDS]
    sink_k, sink_v = (cache.sink_k, cache.sink_v) if cache.sink_k is not None else (cache.open_k, cache.open_v)

    key = ("hack_paged_decode", device_name(q.device), heads, head_dim, length_bucket(cache.max_closed))
    config = config or lookup(key, DEFAULT_LAUNCH)
    splits = config.resolve_splits(cache.max_closed)
    slots = splits + (1 if splits > 1 else 0)
    out = torch.empty_like(q)
    part_out = torch.empty(rows * kv_heads * slots * groups, head_dim, dtype=torch.float32, device=q.device)
    part_max = torch.empty(part_out.shape[0], dtype=torch.float32, device=q.device)
    part_norm = torch.empty_like(part_max)
    strides = [x.stride(i) for x in (q, out) for i in (0, 1)]
    strides += [field.stride(i) for field in fields for i in (0, 1)]
    strides += [x.stride(i) for x in (sink_k, cache.open_k) for i in (0, 1)] + [cache.block_table.stride(0)]
    attn_decode_paged[(rows * kv_heads, slots)](
        q, out, part_out, part_max, part_norm, *fields, sink_k, sink_v, cache.open_k, cache.open_v,
        cache.block_table, cache.totals, cache.sink_slot, cache.open_slot, *strides,
        kv_heads, triton.cdiv(max(cache.max_closed, 1), splits), scaling, seed, splits,
        G=groups, D=head_dim, NP=head_dim // cache.channel_part, PI=cfg.partition_size, BITS=cfg.kv_bits,
        SINKS=cfg.sink_tokens, STOCHASTIC=cfg.stochastic, USE_SUMS=use_sums, SINGLE=splits == 1,
        **config.options(),
    )  # fmt: skip
    if splits > 1:
        combine_splits[(rows * heads,)](
            part_out, part_max, part_norm, out, out.stride(0), out.stride(1), kv_heads, slots,
            G=groups, D=head_dim, SPLITS=triton.next_power_of_2(slots),
        )  # fmt: skip
    return out


def paged_decode_attention(q: torch.Tensor, store, batch, scaling: float) -> torch.Tensor:
    """`paged_decode` on a layer store and a decode batch of `hack.vllm_plugin`; `q` is [rows, heads, head_dim]."""
    layout = store.layout
    cache = PagedCache(
        store.config, layout.channel_part, store.fields, store.sink_k, store.sink_v, store.open_k, store.open_v,
        batch.block_table, batch.totals, batch.sink_slot, batch.open_slot, int(batch.max_closed),
    )  # fmt: skip
    seed = (next(_calls) * 0x9E3779B1) & 0x7FFFFFFF
    out = paged_decode(q, cache, scaling, seed=seed, check_layout=not store.kernel_layout_checked)
    store.kernel_layout_checked = True  # the views of a bound page pool do not change
    return out

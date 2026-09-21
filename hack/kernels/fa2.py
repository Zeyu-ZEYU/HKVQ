"""FlashAttention-2-style kernels for K and V kept in BF16 or as storage-only codes.

The same tiling, online softmax and sequence-level splits serve three formats of K and V:
BF16 tensors, packed codes that are dequantized to BF16 tile by tile, and packed codes that
are fed to the matrix multiplications without the dequantization arithmetic.
"""

import math
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

from hack.kernels.common import (
    ATTENTION, IS_LOAD_ONLY, IS_RESIDENT, bit_checksum, bits16, combine_splits, dense_rows, load_range, same_layout,
    softmax_step, widen, words_view,
)  # fmt: skip
from hack.kernels.quantized_kv import TOKEN_GROUPS, QuantizedKV
from hack.kernels.tuning import LaunchConfig, device_name, length_bucket, lookup

BF16 = 0
DEQUANT = 1
SKIP_DEQUANT = 2

IS_BF16 = tl.constexpr(BF16)
IS_SKIP_DEQUANT = tl.constexpr(SKIP_DEQUANT)
RESIDENT_TILES = tl.constexpr(4)
DEFAULT_TILE = 64
QUERY_ROWS = 128
DEFAULT_LAUNCH = LaunchConfig(num_warps=4, num_stages=1)


@triton.jit
def load_meta(
    ptr, first_token, count,
    TILE: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr, BY_TOKEN: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """Scales or zeros of a tile: [TILE * D / GROUP] for channel groups, [groups in the tile * D] for token groups."""
    if BY_TOKEN:
        rows = tl.arange(0, max(1, TILE // GROUP) * D)
        values = load_range(ptr + (first_token // GROUP) * D, rows, ((count + GROUP - 1) // GROUP) * D, PARTIAL)
    else:
        GROUPS: tl.constexpr = D // GROUP
        values = load_range(ptr + first_token * GROUPS, tl.arange(0, TILE * GROUPS), count * GROUPS, PARTIAL)
    return values


@triton.jit
def dequantize_tile(
    words, scale_ptr, zero_ptr, lut_ptr, first_token, count,
    FORMAT: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
    BY_TOKEN: tl.constexpr, USE_LUT: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """Turn the packed words [TILE * D * BITS / 32] of a tile into BF16 values [TILE, D]."""
    WORDS: tl.constexpr = D * BITS // 32
    if FORMAT == IS_SKIP_DEQUANT:
        values = tl.reshape(tl.broadcast_to(words[:, None], (TILE * WORDS, 32 // BITS)), (TILE, D)).to(tl.float32)
    else:
        scale = load_meta(scale_ptr, first_token, count, TILE, D, GROUP, BY_TOKEN, PARTIAL).to(tl.float32)
        zero = load_meta(zero_ptr, first_token, count, TILE, D, GROUP, BY_TOKEN, PARTIAL).to(tl.float32)
        codes = widen(words, BITS, TILE, D, 0)
        if USE_LUT:
            numbers = tl.load(lut_ptr + codes.to(tl.int32)).to(tl.float32)
        else:
            numbers = codes.to(tl.float32)
        if BY_TOKEN:
            SPAN: tl.constexpr = max(1, TILE // GROUP)
            grouped = tl.reshape(numbers, (SPAN, TILE // SPAN, D))
            values = grouped * tl.reshape(scale, (SPAN, 1, D)) + tl.reshape(zero, (SPAN, 1, D))
        else:
            values = tl.reshape(numbers, (TILE * D // GROUP, GROUP)) * scale[:, None] + zero[:, None]
    return tl.reshape(values, (TILE, D)).to(tl.bfloat16)


@triton.jit
def load_tile(
    data_ptr, scale_ptr, zero_ptr, lut_ptr, first_token, count,
    FORMAT: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
    BY_TOKEN: tl.constexpr, USE_LUT: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """TILE consecutive tokens of K or V as BF16 [TILE, D]; with PARTIAL only the first `count` tokens exist."""
    WIDTH: tl.constexpr = D if FORMAT == IS_BF16 else D * BITS // 32
    data = load_range(data_ptr + first_token * WIDTH, tl.arange(0, TILE * WIDTH), count * WIDTH, PARTIAL)
    if FORMAT == IS_BF16:
        tile = tl.reshape(data, (TILE, D)).to(tl.bfloat16)
    else:
        tile = dequantize_tile(
            data, scale_ptr, zero_ptr, lut_ptr, first_token, count,
            FORMAT, TILE, D, BITS, GROUP, BY_TOKEN, USE_LUT, PARTIAL,
        )  # fmt: skip
    return tile


@triton.jit
def tile_bits(
    data_ptr, scale_ptr, zero_ptr, first_token, count,
    FORMAT: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
    BY_TOKEN: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """Read everything that `load_tile` reads; returns the bit patterns of the data and a checksum of the metadata."""
    WIDTH: tl.constexpr = D if FORMAT == IS_BF16 else D * BITS // 32
    data = load_range(data_ptr + first_token * WIDTH, tl.arange(0, TILE * WIDTH), count * WIDTH, PARTIAL)
    data_bits = bits16(data) if FORMAT == IS_BF16 else data
    checksum = tl.zeros((), tl.int32)
    if FORMAT != IS_BF16 and FORMAT != IS_SKIP_DEQUANT:
        checksum ^= bit_checksum(load_meta(scale_ptr, first_token, count, TILE, D, GROUP, BY_TOKEN, PARTIAL))
        checksum ^= bit_checksum(load_meta(zero_ptr, first_token, count, TILE, D, GROUP, BY_TOKEN, PARTIAL))
    return data_bits, checksum


@triton.jit
def attend_tile(
    q, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
    first_token, count, visible, running_max, running_norm, acc,
    FORMAT: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
    K_BY_TOKEN: tl.constexpr, V_BY_TOKEN: tl.constexpr, K_LUT: tl.constexpr, V_LUT: tl.constexpr,
    MASKED: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """One online-softmax step over TILE tokens; with MASKED the scores outside `visible` [rows, TILE] are dropped."""
    k = load_tile(
        k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, first_token, count,
        FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, K_LUT, PARTIAL,
    )  # fmt: skip
    v = load_tile(
        v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr, first_token, count,
        FORMAT, TILE, D, BITS, GROUP, V_BY_TOKEN, V_LUT, PARTIAL,
    )  # fmt: skip
    scores = tl.dot(q, tl.trans(k))
    if MASKED:
        scores = tl.where(visible, scores, float("-inf"))
    weights, decay, running_max, running_norm = softmax_step(scores, running_max, running_norm)
    return running_max, running_norm, acc * decay[:, None] + tl.dot(weights.to(tl.bfloat16), v)


@triton.jit
def read_tile(
    k_ptr, k_scale_ptr, k_zero_ptr, v_ptr, v_scale_ptr, v_zero_ptr, first_token, count, data_bits, checksum,
    FORMAT: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, GROUP: tl.constexpr,
    K_BY_TOKEN: tl.constexpr, V_BY_TOKEN: tl.constexpr, PARTIAL: tl.constexpr,
):  # fmt: skip
    """Read everything that `attend_tile` reads and fold it into the running bit patterns."""
    k_bits, k_sum = tile_bits(
        k_ptr, k_scale_ptr, k_zero_ptr, first_token, count, FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, PARTIAL
    )
    v_bits, v_sum = tile_bits(
        v_ptr, v_scale_ptr, v_zero_ptr, first_token, count, FORMAT, TILE, D, BITS, GROUP, V_BY_TOKEN, PARTIAL
    )
    return data_bits ^ k_bits ^ v_bits, checksum ^ k_sum ^ v_sum


@triton.jit(do_not_specialize=["num_tokens", "num_splits"])
def fa2_decode(
    q_ptr, out_ptr, part_out_ptr, part_max_ptr, part_norm_ptr,
    k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
    stride_qb, stride_qh, stride_ob, stride_oh, stride_kb, stride_kh, stride_vb, stride_vh,
    stride_kmb, stride_kmh, stride_vmb, stride_vmh,
    kv_heads, num_tokens, scaling, num_splits,
    G: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr, FORMAT: tl.constexpr, BITS: tl.constexpr,
    GROUP: tl.constexpr, K_BY_TOKEN: tl.constexpr, V_BY_TOKEN: tl.constexpr, K_LUT: tl.constexpr, V_LUT: tl.constexpr,
    MODE: tl.constexpr, SINGLE: tl.constexpr,
):  # fmt: skip
    WIDTH: tl.constexpr = D if FORMAT == IS_BF16 else D * BITS // 32
    batch = (tl.program_id(0) // kv_heads).to(tl.int64)
    head = (tl.program_id(0) % kv_heads).to(tl.int64)
    split = tl.program_id(1)
    k_ptr += batch * stride_kb + head * stride_kh
    v_ptr += batch * stride_vb + head * stride_vh
    k_scale_ptr += batch * stride_kmb + head * stride_kmh
    k_zero_ptr += batch * stride_kmb + head * stride_kmh
    v_scale_ptr += batch * stride_vmb + head * stride_vmh
    v_zero_ptr += batch * stride_vmb + head * stride_vmh

    rows = head * G + tl.arange(0, G)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + batch * stride_qb + rows[:, None] * stride_qh + dims[None, :])
    q = (q.to(tl.float32) * scaling).to(tl.bfloat16)

    running_max = tl.full((G,), float("-inf"), tl.float32)
    running_norm = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, D), tl.float32)
    data_bits = tl.zeros((TILE * WIDTH,), tl.int16 if FORMAT == IS_BF16 else tl.int32)
    checksum = tl.zeros((), tl.int32)
    full_tiles = num_tokens // TILE
    rest = num_tokens - full_tiles * TILE
    tiles_per_split = (full_tiles + num_splits - 1) // num_splits
    first_tile = split * tiles_per_split
    nothing = tl.full((G, TILE), 1, tl.int1)
    for tile in range(first_tile, tl.minimum(first_tile + tiles_per_split, full_tiles)):
        source = (tile % RESIDENT_TILES if MODE == IS_RESIDENT else tile) * TILE
        if MODE == IS_LOAD_ONLY:
            data_bits, checksum = read_tile(
                k_ptr, k_scale_ptr, k_zero_ptr, v_ptr, v_scale_ptr, v_zero_ptr, source, 0, data_bits, checksum,
                FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, False,
            )  # fmt: skip
        else:
            running_max, running_norm, acc = attend_tile(
                q, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
                source, 0, nothing, running_max, running_norm, acc,
                FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, K_LUT, V_LUT, False, False,
            )  # fmt: skip

    if (rest > 0) & (split == (0 if SINGLE else num_splits)):
        first = full_tiles * TILE
        if MODE == IS_LOAD_ONLY:
            data_bits, checksum = read_tile(
                k_ptr, k_scale_ptr, k_zero_ptr, v_ptr, v_scale_ptr, v_zero_ptr, first, rest, data_bits, checksum,
                FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, True,
            )  # fmt: skip
        else:
            visible = tl.broadcast_to(tl.arange(0, TILE)[None, :] < rest, (G, TILE))
            running_max, running_norm, acc = attend_tile(
                q, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
                first, rest, visible, running_max, running_norm, acc,
                FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, K_LUT, V_LUT, True, True,
            )  # fmt: skip

    if MODE == IS_LOAD_ONLY:
        running_norm += (bit_checksum(data_bits) ^ checksum).to(tl.float32)
    if SINGLE:
        tl.store(out_ptr + batch * stride_ob + rows[:, None] * stride_oh + dims[None, :], acc / running_norm[:, None])
    else:
        slots = num_splits + tl.where(rest > 0, 1, 0)
        slot = ((batch * kv_heads + head) * slots + split) * G + tl.arange(0, G)
        tl.store(part_out_ptr + slot[:, None] * D + dims[None, :], acc)
        tl.store(part_max_ptr + slot, running_max)
        tl.store(part_norm_ptr + slot, running_norm)


@triton.jit(do_not_specialize=["num_tokens", "q_len"])
def fa2_prefill(
    q_ptr, out_ptr, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
    stride_qb, stride_qh, stride_ql, stride_ob, stride_oh, stride_ol,
    stride_kb, stride_kh, stride_vb, stride_vh, stride_kmb, stride_kmh, stride_vmb, stride_vmh,
    kv_heads, num_tokens, q_len, scaling,
    G: tl.constexpr, D: tl.constexpr, QT: tl.constexpr, TILE: tl.constexpr, FORMAT: tl.constexpr,
    BITS: tl.constexpr, GROUP: tl.constexpr, K_BY_TOKEN: tl.constexpr, V_BY_TOKEN: tl.constexpr,
    K_LUT: tl.constexpr, V_LUT: tl.constexpr, CAUSAL: tl.constexpr,
):  # fmt: skip
    ROWS: tl.constexpr = G * QT
    batch = (tl.program_id(0) // kv_heads).to(tl.int64)
    head = (tl.program_id(0) % kv_heads).to(tl.int64)
    k_ptr += batch * stride_kb + head * stride_kh
    v_ptr += batch * stride_vb + head * stride_vh
    k_scale_ptr += batch * stride_kmb + head * stride_kmh
    k_zero_ptr += batch * stride_kmb + head * stride_kmh
    v_scale_ptr += batch * stride_vmb + head * stride_vmh
    v_zero_ptr += batch * stride_vmb + head * stride_vmh

    heads = head * G + tl.arange(0, G)
    q_rows = tl.program_id(1) * QT + tl.arange(0, QT)
    dims = tl.arange(0, D)
    row_valid = tl.reshape(tl.broadcast_to(q_rows[None, :] < q_len, (G, QT)), (ROWS,))
    positions = tl.reshape(tl.broadcast_to(num_tokens - q_len + q_rows[None, :], (G, QT)), (ROWS,))
    q_offsets = heads[:, None, None] * stride_qh + q_rows[None, :, None] * stride_ql + dims[None, None, :]
    q_offsets += batch * stride_qb
    q = tl.load(tl.reshape(q_ptr + q_offsets, (ROWS, D)), mask=row_valid[:, None], other=0.0)
    q = (q.to(tl.float32) * scaling).to(tl.bfloat16)

    running_max = tl.full((ROWS,), float("-inf"), tl.float32)
    running_norm = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    limit = num_tokens
    if CAUSAL:
        limit = tl.minimum(num_tokens, num_tokens - q_len + (tl.program_id(1) + 1) * QT)
    full_tiles = num_tokens // TILE
    for tile in range(0, tl.minimum(full_tiles, (limit + TILE - 1) // TILE)):
        visible = (tile * TILE + tl.arange(0, TILE))[None, :] <= positions[:, None]
        running_max, running_norm, acc = attend_tile(
            q, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
            tile * TILE, 0, visible, running_max, running_norm, acc,
            FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, K_LUT, V_LUT, CAUSAL, False,
        )  # fmt: skip

    if limit > full_tiles * TILE:
        tokens = full_tiles * TILE + tl.arange(0, TILE)
        visible = tl.broadcast_to(tokens[None, :] < num_tokens, (ROWS, TILE))
        if CAUSAL:
            visible = visible & (tokens[None, :] <= positions[:, None])
        running_max, running_norm, acc = attend_tile(
            q, k_ptr, k_scale_ptr, k_zero_ptr, k_lut_ptr, v_ptr, v_scale_ptr, v_zero_ptr, v_lut_ptr,
            full_tiles * TILE, num_tokens - full_tiles * TILE, visible, running_max, running_norm, acc,
            FORMAT, TILE, D, BITS, GROUP, K_BY_TOKEN, V_BY_TOKEN, K_LUT, V_LUT, True, True,
        )  # fmt: skip

    out_offsets = heads[:, None, None] * stride_oh + q_rows[None, :, None] * stride_ol + dims[None, None, :]
    out_offsets += batch * stride_ob
    tl.store(tl.reshape(out_ptr + out_offsets, (ROWS, D)), acc / running_norm[:, None], mask=row_valid[:, None])


@dataclass
class _Source:
    """K and V of one attention call in the layout the kernels read."""

    k: torch.Tensor
    v: torch.Tensor
    k_meta: tuple[torch.Tensor, torch.Tensor]
    v_meta: tuple[torch.Tensor, torch.Tensor]
    k_lut: torch.Tensor | None
    v_lut: torch.Tensor | None
    fmt: int
    bits: int
    group: int
    k_by_token: bool
    v_by_token: bool
    num_tokens: int

    def pointers(self) -> tuple[torch.Tensor, ...]:
        k_lut = self.k_lut if self.k_lut is not None else self.k_meta[0]
        v_lut = self.v_lut if self.v_lut is not None else self.v_meta[0]
        return self.k, *self.k_meta, k_lut, self.v, *self.v_meta, v_lut

    def strides(self) -> list[int]:
        return [t.stride(i) for t in (self.k, self.v, self.k_meta[0], self.v_meta[0]) for i in (0, 1)]

    def constants(self) -> dict:
        return dict(
            FORMAT=self.fmt, BITS=self.bits, GROUP=self.group, K_BY_TOKEN=self.k_by_token, V_BY_TOKEN=self.v_by_token,
            K_LUT=self.k_lut is not None, V_LUT=self.v_lut is not None,
        )  # fmt: skip


def bf16_source(k: torch.Tensor, v: torch.Tensor) -> _Source:
    k, v = dense_rows(k), dense_rows(v)
    return _Source(k, v, (k, k), (v, v), None, None, BF16, 2, 32, False, False, k.shape[2])


def quantized_source(qkv: QuantizedKV, skip_dequant: bool) -> _Source:
    if qkv.group_size & (qkv.group_size - 1) or qkv.group_size < 16:
        raise ValueError("the fused kernel needs a group size that is a power of two and at least 16")
    k_meta = same_layout(dense_rows(qkv.k_scale), dense_rows(qkv.k_zero))
    v_meta = same_layout(dense_rows(qkv.v_scale), dense_rows(qkv.v_zero))
    luts = [None if lut is None else lut.float().contiguous() for lut in (qkv.k_lut, qkv.v_lut)]
    fmt = SKIP_DEQUANT if skip_dequant else DEQUANT
    k, v = dense_rows(words_view(qkv.k_codes)), dense_rows(words_view(qkv.v_codes))
    by_token = (qkv.k_axis == TOKEN_GROUPS, qkv.v_axis == TOKEN_GROUPS)
    return _Source(k, v, k_meta, v_meta, *luts, fmt, qkv.bits, qkv.group_size, *by_token, qkv.num_tokens)


def decode_key(source: _Source, heads: int, head_dim: int, mode: int) -> tuple:
    """Key under which the launch configuration of a decode call is looked up."""
    device = device_name(source.k.device)
    layout = (source.fmt, source.bits, source.k_by_token, source.v_by_token, source.k_lut is not None)
    return ("fa2_decode", device, mode, heads, head_dim, layout, length_bucket(source.num_tokens))


def fa2_decode_step(
    q: torch.Tensor, source: _Source, scaling: float, mode: int = ATTENTION, config: LaunchConfig | None = None
) -> torch.Tensor:
    """Attention of `q` [batch, heads, 1, head_dim] over `source`; `mode` selects the benchmark variants."""
    batch, heads, _, head_dim = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    kv_heads = source.k.shape[1]
    groups = heads // kv_heads
    config = config or lookup(decode_key(source, heads, head_dim, mode), DEFAULT_LAUNCH)
    tile = config.tile or DEFAULT_TILE
    splits = config.resolve_splits(source.num_tokens // tile)
    slots = splits + (1 if source.num_tokens % tile and splits > 1 else 0)

    out = torch.empty_like(q)
    part_out = torch.empty(batch * kv_heads * slots * groups, head_dim, dtype=torch.float32, device=q.device)
    part_max = torch.empty(part_out.shape[0], dtype=torch.float32, device=q.device)
    part_norm = torch.empty_like(part_max)
    fa2_decode[(batch * kv_heads, slots)](
        q, out, part_out, part_max, part_norm, *source.pointers(),
        q.stride(0), q.stride(1), out.stride(0), out.stride(1), *source.strides(),
        kv_heads, source.num_tokens, scaling, splits,
        G=groups, D=head_dim, TILE=tile, MODE=mode, SINGLE=splits == 1, **source.constants(), **config.options(),
    )  # fmt: skip
    if splits > 1:
        combine_splits[(batch * heads,)](
            part_out, part_max, part_norm, out, out.stride(0), out.stride(1), kv_heads, slots,
            G=groups, D=head_dim, SPLITS=triton.next_power_of_2(slots),
        )  # fmt: skip
    return out


_fitting_rows: dict[tuple, int] = {}


def fa2_prefill_step(q: torch.Tensor, source: _Source, scaling: float, causal: bool) -> torch.Tensor:
    """A program owns as many query rows as the shared memory of the device allows, at most QUERY_ROWS."""
    batch, heads, q_len, head_dim = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    kv_heads = source.k.shape[1]
    groups = heads // kv_heads
    out = torch.empty_like(q)
    key = (q.device, groups, head_dim, causal, tuple(source.constants().items()))
    rows = _fitting_rows.get(key, QUERY_ROWS)
    while True:
        query_tile = max(1, rows // triton.next_power_of_2(groups))
        try:
            fa2_prefill[(batch * kv_heads, triton.cdiv(q_len, query_tile))](
                q, out, *source.pointers(),
                q.stride(0), q.stride(1), q.stride(2), out.stride(0), out.stride(1), out.stride(2), *source.strides(),
                kv_heads, source.num_tokens, q_len, scaling,
                G=groups, D=head_dim, QT=query_tile, TILE=DEFAULT_TILE, CAUSAL=causal, **source.constants(),
            )  # fmt: skip
            break
        except OutOfResources:
            if query_tile == 1:
                raise
            rows //= 2
    _fitting_rows[key] = rows
    return out


def fa2_attention(q: torch.Tensor, source: _Source, scaling: float | None, causal: bool) -> torch.Tensor:
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(q.shape[-1])
    if q.shape[2] == 1:
        return fa2_decode_step(q, source, scaling)
    return fa2_prefill_step(q, source, scaling, causal)

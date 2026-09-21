"""Triton helpers shared by the attention kernels."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ATTENTION = 0
LOAD_ONLY = 1
RESIDENT = 2

IS_LOAD_ONLY = tl.constexpr(LOAD_ONLY)
IS_RESIDENT = tl.constexpr(RESIDENT)
CODE_OFFSET = tl.constexpr(128)
RESIDENT_BLOCKS = tl.constexpr(4)


def words_view(codes: torch.Tensor) -> torch.Tensor:
    """Reinterpret packed uint8 codes [..., n] as int32 words [..., n // 4]."""
    if codes.stride(-1) != 1 or codes.shape[-1] % 4 != 0 or codes.storage_offset() % 4 != 0:
        codes = codes.contiguous()
    return codes.view(torch.int32)


def same_layout(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return the tensors unchanged when they share strides, otherwise contiguous copies."""
    if all(t.stride() == tensors[0].stride() for t in tensors):
        return tensors
    return tuple(t.contiguous() for t in tensors)


def dense_rows(x: torch.Tensor) -> torch.Tensor:
    """`x` [batch, heads, n, width] with its last two dimensions stored as one contiguous range."""
    if x.stride(-1) == 1 and x.stride(-2) == x.shape[-1]:
        return x
    return x.contiguous()


@triton.jit
def widen(words, BITS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr, OFFSET: tl.constexpr):
    """Expand int32 words that hold 32 // BITS codes each into the int8 matrix [ROWS, COLS] of (code - OFFSET)."""
    per_word: tl.constexpr = 32 // BITS
    shifts = tl.arange(0, per_word) * BITS
    flat = tl.reshape(words, (ROWS * COLS // per_word, 1))
    codes = ((flat >> shifts[None, :]) & ((1 << BITS) - 1)) - OFFSET
    return tl.reshape(codes.to(tl.int8), (ROWS, COLS))


@triton.jit
def uniform_noise(seed, index):
    """Uniform noise in [0, 1) from an integer hash of `index`."""
    h = (index.to(tl.uint32) * 0x9E3779B1) ^ seed.to(tl.uint32)
    h = (h ^ (h >> 16)) * 0x85EBCA6B
    h = (h ^ (h >> 13)) * 0xC2B2AE35
    h = h ^ (h >> 16)
    return (h >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def round_codes(x, seed, index, STOCHASTIC: tl.constexpr):
    if STOCHASTIC:
        return tl.floor(x + uniform_noise(seed, index))
    return libdevice.rint(x)


@triton.jit
def quantize_int8(x, low, high, seed, index, STOCHASTIC: tl.constexpr):
    """Asymmetric 8-bit min-max quantization of every row of `x` [rows, n] with the row range [low, high].

    Returns the codes shifted into int8 (code - 128), the scale, the offset that belongs
    to the shifted codes (low + 128 * scale) and the sum of the shifted codes of the row.
    """
    scale = (high - low) / 255.0
    scale = tl.where(scale > 0, scale, 1.0)
    inverse = 1.0 / scale
    codes = round_codes((x - low[:, None]) * inverse[:, None], seed, index, STOCHASTIC)
    codes = tl.minimum(tl.maximum(codes, 0.0), 255.0) - 128.0
    return codes.to(tl.int8), scale, low + 128.0 * scale, tl.sum(codes, axis=1)


@triton.jit
def code_fields(data, ROWS: tl.constexpr, D: tl.constexpr):
    """2-bit codes of `data` (uint8 [ROWS * D / 4], four codes per byte) as an int8 matrix [ROWS, D] without shifts.

    Column i * D / 4 + b holds code i of byte b scaled by 4 ** i; the top code (i = 3) is
    stored as 64 * code - 128. `field_weights` gives the factors that undo the scaling.
    """
    field = tl.arange(0, 4)
    masks = (3 << (2 * field)).to(tl.uint8)[None, :, None]
    flips = tl.where(field == 3, 128, 0).to(tl.uint8)[None, :, None]
    fields = (tl.reshape(data, (ROWS, 1, D // 4)) ^ flips) & masks
    return tl.reshape(fields.to(tl.int8, bitcast=True), (ROWS, D))


@triton.jit
def field_weights(D: tl.constexpr):
    """Per column of `code_fields`: 4 ** -i, and the bias 128 of the top field (0 elsewhere)."""
    field = tl.arange(0, 4)
    weight = 1.0 / (1 << (2 * field)).to(tl.float32)
    bias = tl.where(field == 3, 128.0, 0.0)
    weight = tl.reshape(tl.broadcast_to(weight[:, None], (4, D // 4)), (D,))
    return weight, tl.reshape(tl.broadcast_to(bias[:, None], (4, D // 4)), (D,))


@triton.jit
def to_field_order(x, D: tl.constexpr):
    """Reorder the channels of `x` [D] from the stored order to the column order of `code_fields`."""
    return tl.reshape(tl.trans(tl.reshape(x, (D // 4, 4))), (D,))


@triton.jit
def rows_to_field_order(x, ROWS: tl.constexpr, D: tl.constexpr):
    """Reorder the columns of `x` [ROWS, D] from the stored channel order to the column order of `code_fields`."""
    return tl.reshape(tl.permute(tl.reshape(x, (ROWS, D // 4, 4)), (0, 2, 1)), (ROWS, D))


@triton.jit
def from_field_order(x, ROWS: tl.constexpr, D: tl.constexpr):
    """Reorder the columns of `x` [ROWS, D] from the order of `code_fields` to the stored channel order."""
    return tl.reshape(tl.permute(tl.reshape(x, (ROWS, 4, D // 4)), (0, 2, 1)), (ROWS, D))


@triton.jit
def softmax_step(scores, running_max, running_norm):
    """One tile of the online softmax: tile-local weights, the decay of earlier tiles, new max and norm."""
    new_max = tl.maximum(running_max, tl.max(scores, axis=1))
    weights = tl.exp(scores - new_max[:, None])
    decay = tl.exp(running_max - new_max)
    return weights, decay, new_max, running_norm * decay + tl.sum(weights, axis=1)


@triton.jit
def dot_with_half(x, y):
    """`x @ y` for float32 `x` and 16-bit `y`: two 16-bit products, the rounded `x` and what the rounding left."""
    high = x.to(tl.bfloat16)
    low = (x - high.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(high, y) + tl.dot(low, y)


@triton.jit
def load_range(ptr, offsets, limit, PARTIAL: tl.constexpr):
    """Load `ptr[offsets]`; with PARTIAL only offsets below `limit` exist and the others read as zero."""
    if PARTIAL:
        values = tl.load(ptr + offsets, mask=offsets < limit, other=0)
    else:
        values = tl.load(ptr + offsets)
    return values


@triton.jit
def bits16(x):
    """The low 16 bits of the bit pattern of `x`."""
    if x.dtype.is_floating():
        bits = x.to(tl.int16 if x.dtype.primitive_bitwidth == 16 else tl.int32, bitcast=True).to(tl.int16)
    else:
        bits = x.to(tl.int16)
    return bits


@triton.jit
def bit_checksum(x):
    """XOR of the bit patterns of `x`; keeps a load alive without floating-point work."""
    if x.dtype.is_floating():
        bits = x.to(tl.int16 if x.dtype.primitive_bitwidth == 16 else tl.int32, bitcast=True)
    else:
        bits = x
    return tl.xor_sum(bits, axis=None).to(tl.int32)


@triton.jit(do_not_specialize=["num_splits"])
def combine_splits(
    part_out_ptr, part_max_ptr, part_norm_ptr, out_ptr,
    stride_ob, stride_oh, kv_heads, num_splits,
    G: tl.constexpr, D: tl.constexpr, SPLITS: tl.constexpr,
):  # fmt: skip
    """Merge the partial results of the sequence-level splits of one query head."""
    pid = tl.program_id(0)
    head = pid % (kv_heads * G)
    batch = pid // (kv_heads * G)
    slots = tl.arange(0, SPLITS)
    valid = slots < num_splits
    base = ((batch * kv_heads + head // G) * num_splits + slots) * G + head % G
    split_max = tl.load(part_max_ptr + base, mask=valid, other=float("-inf"))
    split_norm = tl.load(part_norm_ptr + base, mask=valid, other=0.0)
    weight = tl.exp(split_max - tl.max(split_max))
    dims = tl.arange(0, D)
    partial = tl.load(part_out_ptr + base[:, None] * D + dims[None, :], mask=valid[:, None], other=0.0)
    out = tl.sum(weight[:, None] * partial, axis=0) / tl.sum(weight * split_norm)
    tl.store(out_ptr + batch * stride_ob + head * stride_oh + dims, out)

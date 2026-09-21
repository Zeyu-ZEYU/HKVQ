"""`attn_decode`: attention of one new token over a `HackLayerCache`."""

import torch
import triton
import triton.language as tl

from hack.cache import HackLayerCache
from hack.kernels.common import (
    ATTENTION, CODE_OFFSET, IS_LOAD_ONLY, IS_RESIDENT, RESIDENT_BLOCKS, bit_checksum, bits16, code_fields,
    combine_splits, dense_rows, dot_with_half, field_weights, from_field_order, quantize_int8, same_layout,
    softmax_step, to_field_order, widen, words_view,
)  # fmt: skip
from hack.kernels.tuning import LaunchConfig, device_name, length_bucket, lookup

DEFAULT_LAUNCH = LaunchConfig(num_warps=1, num_stages=1)
DENSE_TILE = tl.constexpr(64)


@triton.jit
def quantize_queries(q, k_scale, seed, scaling, NP: tl.constexpr, KP: tl.constexpr, STOCHASTIC: tl.constexpr):
    """Multiply the query rows `q` [rows, D] with the channel scales of a K block and quantize them per partition.

    Returns codes [NP * rows, D], where row p * rows + r holds the codes of partition p of
    query r and zeros elsewhere, and the scale, the offset and the code sum of every
    partition, each [NP, rows]; scale and offset contain the softmax scaling.
    """
    rows: tl.constexpr = q.shape[0]
    parts = tl.reshape(q * k_scale[None, :], (rows * NP, KP))
    index = tl.arange(0, rows * NP)[:, None] * KP + tl.arange(0, KP)[None, :]
    low, high = tl.min(parts, axis=1), tl.max(parts, axis=1)
    codes, scale, offset, code_sum = quantize_int8(parts, low, high, seed, index, STOCHASTIC)
    own = tl.arange(0, NP)[:, None, None, None] == tl.arange(0, NP)[None, None, :, None]
    codes = tl.reshape(tl.where(own, tl.reshape(codes, (1, rows, NP, KP)), 0), (NP * rows, NP * KP))
    scale = tl.trans(tl.reshape(scale * scaling, (rows, NP)))
    offset = tl.trans(tl.reshape(offset * scaling, (rows, NP)))
    return codes, scale, offset, tl.trans(tl.reshape(code_sum, (rows, NP)))


@triton.jit
def tile_scores(
    q, k_words_ptr, k_sums_ptr, k_scale_ptr, k_min_ptr, seed, scaling,
    PI: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, BITS: tl.constexpr,
    STOCHASTIC: tl.constexpr, USE_SUMS: tl.constexpr,
):  # fmt: skip
    """Attention scores [rows, PI] of the queries `q` [rows, D] against one closed block of K.

    The pointers address the data of the block: code words [PI, D * BITS / 32], code sums
    [PI, NP], channel scales and minima [D].
    """
    K_OFFSET: tl.constexpr = CODE_OFFSET if BITS == 8 else 0
    WORDS: tl.constexpr = D * BITS // 32
    KP: tl.constexpr = D // NP
    ROWS: tl.constexpr = q.shape[0]
    k_scale = tl.load(k_scale_ptr + tl.arange(0, D)).to(tl.float32)
    k_min = tl.load(k_min_ptr + tl.arange(0, D)).to(tl.float32)
    q_codes, q_scale, q_offset, q_sum = quantize_queries(q, k_scale, seed, scaling, NP, KP, STOCHASTIC)
    codes = widen(tl.load(k_words_ptr + tl.arange(0, PI * WORDS)), BITS, PI, D, K_OFFSET)
    if USE_SUMS:
        k_sums = tl.reshape(tl.load(k_sums_ptr + tl.arange(0, PI * NP)), (PI, NP)).to(tl.float32)
    else:
        k_sums = tl.sum(tl.reshape(codes, (PI, NP, KP)).to(tl.int32), axis=2).to(tl.float32) + K_OFFSET * KP
    dots = tl.reshape(tl.dot(q_codes, tl.trans(codes)), (NP, ROWS, PI)).to(tl.float32)
    if BITS == 8:
        dots += K_OFFSET * q_sum[:, :, None]
    scores = tl.sum(q_scale[:, :, None] * dots + q_offset[:, :, None] * tl.trans(k_sums)[:, None, :], axis=0)
    return scores + (scaling * tl.sum(q * k_min[None, :], axis=1))[:, None]


@triton.jit
def block_values(
    weights, v_words_ptr, v_scale_ptr, v_min_ptr, v_sums_ptr, seed,
    PI: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, BITS: tl.constexpr,
    STOCHASTIC: tl.constexpr, USE_SUMS: tl.constexpr,
):  # fmt: skip
    """Product [rows, D] of tile-local softmax weights [rows, PI] with one closed block of V.

    The pointers address the data of the block: code words [PI, D * BITS / 32], token scales
    and minima [PI, NP], channel code sums [D]. For 2-bit codes the columns of the result
    are in the order of `code_fields`.
    """
    V_OFFSET: tl.constexpr = CODE_OFFSET if BITS == 8 else 0
    WORDS: tl.constexpr = D * BITS // 32
    KP: tl.constexpr = D // NP
    ROWS: tl.constexpr = weights.shape[0]
    v_scale = tl.trans(tl.reshape(tl.load(v_scale_ptr + tl.arange(0, PI * NP)), (PI, NP))).to(tl.float32)
    v_min = tl.trans(tl.reshape(tl.load(v_min_ptr + tl.arange(0, PI * NP)), (PI, NP))).to(tl.float32)
    scaled = tl.reshape(weights[None, :, :] * v_scale[:, None, :], (NP * ROWS, PI))
    index = (tl.arange(0, NP * ROWS) * 40503)[:, None] + tl.arange(0, PI)[None, :]
    low, high = tl.min(scaled, axis=1), tl.max(scaled, axis=1)
    p_codes, p_scale, p_offset, p_sum = quantize_int8(scaled, low, high, seed ^ 0x5BD1E995, index, STOCHASTIC)

    if BITS == 2:
        v_bytes_ptr = v_words_ptr.to(tl.pointer_type(tl.uint8))
        v_codes = code_fields(tl.load(v_bytes_ptr + tl.arange(0, PI * D // 4)), PI, D)
        weight, bias = field_weights(D)
        partition = (tl.arange(0, D) % (D // 4)) // (KP // 4)
    else:
        v_codes = widen(tl.load(v_words_ptr + tl.arange(0, PI * WORDS)), BITS, PI, D, V_OFFSET)
        partition = tl.arange(0, D) // KP
    if USE_SUMS:
        v_sums = tl.load(v_sums_ptr + tl.arange(0, D)).to(tl.float32)
        if BITS == 2:
            v_sums = to_field_order(v_sums, D)
    else:
        v_sums = tl.sum(v_codes.to(tl.int32), axis=0).to(tl.float32) + V_OFFSET * PI
        if BITS == 2:
            v_sums = (v_sums + bias * PI) * weight
    dots = tl.reshape(tl.dot(p_codes, v_codes), (NP, ROWS, D)).to(tl.float32)
    p_scale, p_offset = tl.reshape(p_scale, (NP, ROWS)), tl.reshape(p_offset, (NP, ROWS))
    p_sum = tl.reshape(p_sum, (NP, ROWS))
    if BITS == 2:
        dots = (dots + bias[None, None, :] * p_sum[:, :, None]) * weight[None, None, :]
    if BITS == 8:
        dots += V_OFFSET * p_sum[:, :, None]
    floats = tl.sum(weights[None, :, :] * v_min[:, None, :], axis=2)
    terms = p_scale[:, :, None] * dots + p_offset[:, :, None] * v_sums[None, None, :] + floats[:, :, None]
    own = tl.arange(0, NP)[:, None, None] == partition[None, None, :]
    return tl.sum(tl.where(own, terms, 0.0), axis=0)


@triton.jit
def dense_tile(
    q, k_ptr, v_ptr, first, count, first_position, positions, scaling, running_max, running_norm, acc,
    DT: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):  # fmt: skip
    """One online-softmax step over DT tokens of a segment whose K and V are stored in 16-bit.

    The tile starts at token `first` of the segment, which holds `count` tokens and begins at
    sequence position `first_position`; `positions` [rows] are the positions of the queries.
    """
    cells = first * D + tl.arange(0, DT * D)
    k = tl.reshape(tl.load(k_ptr + cells, mask=cells < count * D, other=0.0), (DT, D)).to(tl.bfloat16)
    v = tl.reshape(tl.load(v_ptr + cells, mask=cells < count * D, other=0.0), (DT, D)).to(tl.bfloat16)
    tokens = first + tl.arange(0, DT)
    visible = tl.broadcast_to((tokens < count)[None, :], (q.shape[0], DT))
    if CAUSAL:
        visible = visible & ((first_position + tokens)[None, :] <= positions[:, None])
    scores = tl.where(visible, dot_with_half(q, tl.trans(k)) * scaling, float("-inf"))
    weights, decay, running_max, running_norm = softmax_step(scores, running_max, running_norm)
    return running_max, running_norm, acc * decay[:, None] + dot_with_half(weights, v)


@triton.jit
def dense_segment(
    q, k_ptr, v_ptr, count, first_position, positions, scaling, running_max, running_norm, acc,
    DT: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):  # fmt: skip
    """Online-softmax steps over all `count` tokens of a 16-bit segment."""
    for first in range(0, count, DT):
        running_max, running_norm, acc = dense_tile(
            q, k_ptr, v_ptr, first, count, first_position, positions, scaling, running_max, running_norm, acc,
            DT, D, CAUSAL,
        )  # fmt: skip
    return running_max, running_norm, acc


@triton.jit
def block_bits(
    k_words_ptr, k_sums_ptr, k_scale_ptr, k_min_ptr, v_words_ptr, v_scale_ptr, v_min_ptr, v_sums_ptr,
    PI: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, BITS: tl.constexpr,
):  # fmt: skip
    """Read everything that attention reads for one closed block; returns bit patterns per kind of data."""
    words = tl.arange(0, PI * D * BITS // 32)
    tokens = tl.arange(0, PI * NP)
    channels = tl.arange(0, D)
    code_bits = tl.load(k_words_ptr + words) ^ tl.load(v_words_ptr + words)
    token_bits = bits16(tl.load(k_sums_ptr + tokens)) ^ bits16(tl.load(v_scale_ptr + tokens))
    token_bits ^= bits16(tl.load(v_min_ptr + tokens))
    channel_bits = bits16(tl.load(k_scale_ptr + channels)) ^ bits16(tl.load(k_min_ptr + channels))
    return code_bits, token_bits, channel_bits ^ bits16(tl.load(v_sums_ptr + channels))


@triton.jit
def dense_bits(k_ptr, v_ptr, count, DT: tl.constexpr, D: tl.constexpr):
    """Read all `count` tokens of a 16-bit segment and fold them into a checksum."""
    checksum = tl.zeros((), tl.int32)
    for first in range(0, count, DT):
        cells = first * D + tl.arange(0, DT * D)
        checksum ^= bit_checksum(tl.load(k_ptr + cells, mask=cells < count * D, other=0.0))
        checksum ^= bit_checksum(tl.load(v_ptr + cells, mask=cells < count * D, other=0.0))
    return checksum


@triton.jit(do_not_specialize=["num_blocks", "num_sinks", "num_open", "seed", "num_splits"])
def attn_decode(
    q_ptr, out_ptr, part_out_ptr, part_max_ptr, part_norm_ptr,
    k_words_ptr, k_sums_ptr, k_scale_ptr, k_min_ptr, v_words_ptr, v_scale_ptr, v_min_ptr, v_sums_ptr,
    sink_k_ptr, sink_v_ptr, open_k_ptr, open_v_ptr,
    stride_qb, stride_qh, stride_ob, stride_oh, stride_wb, stride_wh, stride_tb, stride_th, stride_cb, stride_ch,
    stride_sb, stride_sh, stride_nb, stride_nh,
    kv_heads, num_blocks, num_sinks, num_open, scaling, seed, num_splits,
    G: tl.constexpr, D: tl.constexpr, NP: tl.constexpr, PI: tl.constexpr, BITS: tl.constexpr,
    STOCHASTIC: tl.constexpr, USE_SUMS: tl.constexpr, MODE: tl.constexpr, SINGLE: tl.constexpr,
):  # fmt: skip
    WORDS: tl.constexpr = D * BITS // 32
    DT: tl.constexpr = min(PI, DENSE_TILE)
    batch = (tl.program_id(0) // kv_heads).to(tl.int64)
    head = (tl.program_id(0) % kv_heads).to(tl.int64)
    split = tl.program_id(1)
    words = batch * stride_wb + head * stride_wh
    tokens = batch * stride_tb + head * stride_th
    channels = batch * stride_cb + head * stride_ch
    sinks = batch * stride_sb + head * stride_sh
    opened = batch * stride_nb + head * stride_nh

    rows = head * G + tl.arange(0, G)
    dims = tl.arange(0, D)
    q = tl.load(q_ptr + batch * stride_qb + rows[:, None] * stride_qh + dims[None, :]).to(tl.float32)
    seed ^= bit_checksum(q) + tl.program_id(0) * 7919

    running_max = tl.full((G,), float("-inf"), tl.float32)
    running_norm = tl.zeros((G,), tl.float32)
    acc = tl.zeros((G, D), tl.float32)
    code_bits = tl.zeros((PI * WORDS,), tl.int32)
    token_bits = tl.zeros((PI * NP,), tl.int16)
    channel_bits = tl.zeros((D,), tl.int16)
    blocks_per_split = (num_blocks + num_splits - 1) // num_splits
    first_block = split * blocks_per_split
    for block in range(first_block, tl.minimum(first_block + blocks_per_split, num_blocks)):
        source = block % RESIDENT_BLOCKS if MODE == IS_RESIDENT else block
        word, token, channel = words + source * (PI * WORDS), tokens + source * (PI * NP), channels + source * D
        k_words, k_sums, v_words = k_words_ptr + word, k_sums_ptr + token, v_words_ptr + word
        k_scale, k_min, v_sums = k_scale_ptr + channel, k_min_ptr + channel, v_sums_ptr + channel
        v_scale, v_min = v_scale_ptr + token, v_min_ptr + token
        if MODE == IS_LOAD_ONLY:
            new_code_bits, new_token_bits, new_channel_bits = block_bits(
                k_words, k_sums, k_scale, k_min, v_words, v_scale, v_min, v_sums, PI, D, NP, BITS
            )
            code_bits ^= new_code_bits
            token_bits ^= new_token_bits
            channel_bits ^= new_channel_bits
        else:
            scores = tile_scores(
                q, k_words, k_sums, k_scale, k_min, seed + block, scaling, PI, D, NP, BITS, STOCHASTIC, USE_SUMS
            )
            weights, decay, running_max, running_norm = softmax_step(scores, running_max, running_norm)
            acc = acc * decay[:, None] + block_values(
                weights, v_words, v_scale, v_min, v_sums, seed + block, PI, D, NP, BITS, STOCHASTIC, USE_SUMS
            )
    if BITS == 2:
        acc = from_field_order(acc, G, D)

    if split == (0 if SINGLE else num_splits):
        if MODE == IS_LOAD_ONLY:
            channel_bits ^= dense_bits(sink_k_ptr + sinks, sink_v_ptr + sinks, num_sinks, DT, D).to(tl.int16)
            channel_bits ^= dense_bits(open_k_ptr + opened, open_v_ptr + opened, num_open, DT, D).to(tl.int16)
        else:
            running_max, running_norm, acc = dense_segment(
                q, sink_k_ptr + sinks, sink_v_ptr + sinks, num_sinks, 0, rows, scaling,
                running_max, running_norm, acc, DT, D, False,
            )  # fmt: skip
            running_max, running_norm, acc = dense_segment(
                q, open_k_ptr + opened, open_v_ptr + opened, num_open, 0, rows, scaling,
                running_max, running_norm, acc, DT, D, False,
            )  # fmt: skip

    if MODE == IS_LOAD_ONLY:
        running_norm += (bit_checksum(code_bits) ^ bit_checksum(token_bits) ^ bit_checksum(channel_bits)).to(tl.float32)
    if SINGLE:
        tl.store(out_ptr + batch * stride_ob + rows[:, None] * stride_oh + dims[None, :], acc / running_norm[:, None])
    else:
        slots = num_splits + tl.where(num_sinks + num_open > 0, 1, 0)
        slot = ((batch * kv_heads + head) * slots + split) * G + tl.arange(0, G)
        tl.store(part_out_ptr + slot[:, None] * D + dims[None, :], acc)
        tl.store(part_max_ptr + slot, running_max)
        tl.store(part_norm_ptr + slot, running_norm)


def decode_key(cache: HackLayerCache, heads: int, head_dim: int, mode: int) -> tuple:
    """Key under which the launch configuration of a decode call is looked up."""
    cfg = cache.config
    device = device_name()
    blocks = length_bucket(cache.k_scale.length)
    return ("hack_decode", device, mode, heads, head_dim, cfg.partition_size, cfg.kv_bits, cfg.stochastic, blocks)


_spare: dict[tuple, torch.Tensor] = {}


def _placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A one-element tensor that stands in for a part of the cache that does not exist yet."""
    if (device, dtype) not in _spare:
        _spare[(device, dtype)] = torch.empty(1, 1, 1, 1, dtype=dtype, device=device)
    return _spare[(device, dtype)]


def _dense_rows_of(*fields) -> tuple[torch.Tensor, ...]:
    return same_layout(*(dense_rows(field.view()) for field in fields))


def _segment(k: torch.Tensor | None, v: torch.Tensor | None, like: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if k is None:
        return (_placeholder(like.device, torch.bfloat16),) * 2
    return same_layout(dense_rows(k), dense_rows(v))


def kernel_tensors(cache: HackLayerCache, like: torch.Tensor) -> dict[str, tuple[torch.Tensor, ...]]:
    """The tensors of `cache` grouped by layout, every tile stored as one contiguous range.

    words: K and V codes as int32 words; tokens: k_sums, v_scale, v_min; channels: k_scale,
    k_min, v_sums; sinks and opened: K and V of the 16-bit segments.
    """
    groups = {"sinks": _segment(cache.sink_k, cache.sink_v, like), "opened": _segment(cache.k_open, cache.v_open, like)}
    if cache.k_scale.length:
        groups["words"] = same_layout(*(dense_rows(words_view(f.view())) for f in (cache.k_codes, cache.v_codes)))
        groups["tokens"] = _dense_rows_of(cache.k_sums, cache.v_scale, cache.v_min)
        groups["channels"] = _dense_rows_of(cache.k_scale, cache.k_min, cache.v_sums)
        return groups
    words, sums = _placeholder(like.device, torch.int32), _placeholder(like.device, torch.uint8)
    meta = _placeholder(like.device, torch.bfloat16)
    groups.update(words=(words, words), tokens=(sums, meta, meta), channels=(meta, meta, sums))
    return groups


def hack_decode(
    q: torch.Tensor, cache: HackLayerCache, scaling: float, seed: int, mode: int = ATTENTION,
    config: LaunchConfig | None = None,
) -> torch.Tensor:  # fmt: skip
    """Attention of `q` [batch, heads, 1, head_dim] over `cache`; `mode` selects the benchmark variants."""
    cfg = cache.config
    batch, heads, _, head_dim = q.shape
    q = q if q.stride(-1) == 1 else q.contiguous()
    t = kernel_tensors(cache, q)
    kv_heads = next(x for x in (cache.k_scale.data, cache.sink_k, cache.k_open) if x is not None).shape[1]
    groups, num_blocks = heads // kv_heads, cache.k_scale.length
    dense = cache.num_sink_tokens + cache.num_open_tokens

    config = config or lookup(decode_key(cache, heads, head_dim, mode), DEFAULT_LAUNCH)
    splits = config.resolve_splits(num_blocks)
    slots = splits + (1 if dense and splits > 1 else 0)
    out = torch.empty_like(q)
    part_out = torch.empty(batch * kv_heads * slots * groups, head_dim, dtype=torch.float32, device=q.device)
    part_max = torch.empty(part_out.shape[0], dtype=torch.float32, device=q.device)
    part_norm = torch.empty_like(part_max)
    strides = [x.stride(i) for x in (q, out) for i in (0, 1)]
    strides += [t[name][0].stride(i) for name in ("words", "tokens", "channels", "sinks", "opened") for i in (0, 1)]
    k_words, v_words = t["words"]
    k_sums, v_scale, v_min = t["tokens"]
    k_scale, k_min, v_sums = t["channels"]
    attn_decode[(batch * kv_heads, slots)](
        q, out, part_out, part_max, part_norm, k_words, k_sums, k_scale, k_min, v_words, v_scale, v_min, v_sums,
        *t["sinks"], *t["opened"], *strides,
        kv_heads, num_blocks, cache.num_sink_tokens, cache.num_open_tokens, scaling, seed, splits,
        G=groups, D=head_dim, NP=head_dim // cache.channel_part, PI=cfg.partition_size, BITS=cfg.kv_bits,
        STOCHASTIC=cfg.stochastic, USE_SUMS=cfg.summation_elimination, MODE=mode, SINGLE=splits == 1,
        **config.options(),
    )  # fmt: skip
    if splits > 1:
        combine_splits[(batch * heads,)](
            part_out, part_max, part_norm, out, out.stride(0), out.stride(1), kv_heads, slots,
            G=groups, D=head_dim, SPLITS=triton.next_power_of_2(slots),
        )  # fmt: skip
    return out

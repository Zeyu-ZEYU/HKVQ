"""CacheGen-style KV codec.

Tokens are coded in groups. The first token of a group, the anchor, is quantized on its own; the
other tokens are quantized as differences to the reconstructed anchor. The number of quantization
levels depends on the layer, with more levels for earlier layers. The integer symbols are entropy
coded with symbol statistics per layer and channel.

`encode` turns the K and V of all layers into a byte string, `decode` turns the byte string into
integer codes with their dequantization parameters, and `CacheGenTensor.dequantize` gives BF16.
"""

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hack.baselines import rans
from hack.quant import pack, unpack

META_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class CacheGenConfig:
    """Settings of the CacheGen-style codec.

    group_size: tokens per group; the first token of a group is the anchor.
    anchor_levels: quantization levels of the anchor tokens.
    key_levels, value_levels: quantization levels per group of layers; the layers after the first
        one are split into as many equal groups as there are entries, earliest group first.
    first_layer_levels: quantization levels of the keys and of the values of the first layer.
    key_delta, value_delta: code the tensor as differences to anchors (otherwise every token is
        quantized on its own and the tensor has no anchors).
    chunk_size: tokens that one entropy-coder stream covers; the streams of all chunks are coded in parallel.
    """

    group_size: int = 10
    anchor_levels: int = 255
    key_levels: tuple[int, ...] = (15, 15, 7)
    value_levels: tuple[int, ...] = (5, 5, 5)
    first_layer_levels: tuple[int, int] = (255, 15)
    key_delta: bool = True
    value_delta: bool = False
    chunk_size: int = 1500

    def __post_init__(self):
        for levels in (self.anchor_levels, *self.first_layer_levels, *self.key_levels, *self.value_levels):
            if levels % 2 == 0 or not 3 <= levels <= 255:
                raise ValueError("the number of quantization levels must be odd and between 3 and 255")
        if self.group_size < 2 or self.chunk_size % self.group_size != 0:
            raise ValueError("chunk_size must be a multiple of group_size, and group_size at least 2")

    def levels(self, layer: int, num_layers: int, is_value: bool) -> int:
        if layer == 0:
            return self.first_layer_levels[is_value]
        schedule = self.value_levels if is_value else self.key_levels
        return schedule[(layer - 1) * len(schedule) // max(1, num_layers - 1)]

    def tokens_per_group(self, is_value: bool) -> int:
        """Group size of the tensor, 0 if it is coded without anchors."""
        return self.group_size if (self.value_delta if is_value else self.key_delta) else 0


def code_bits(levels: int) -> int:
    """Width of the stored codes."""
    return 2 if levels <= 4 else 4 if levels <= 16 else 8


def num_anchors(num_tokens: int, group_size: int) -> int:
    return 0 if group_size == 0 else -(-num_tokens // group_size)


def quantize_vectors(x: torch.Tensor, levels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization of every vector along the last dimension: unsigned codes and scale."""
    half = (levels - 1) // 2
    scale = (x.abs().amax(dim=-1, keepdim=True) / half).to(META_DTYPE)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(x / scale.float()).clamp_(-half, half) + half
    return codes.to(torch.uint8), scale


def dequantize_vectors(codes: torch.Tensor, scale: torch.Tensor, levels: int) -> torch.Tensor:
    return (codes.float() - (levels - 1) // 2) * scale.float()


def _no_anchors(batch: int, heads: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor codes and scales of a tensor that is coded without anchors."""
    codes = torch.empty(batch, heads, 0, head_dim, dtype=torch.uint8, device=device)
    return codes, torch.empty(batch, heads, 0, 1, dtype=META_DTYPE, device=device)


@dataclass
class CacheGenTensor:
    """Codes of the keys or the values of one layer, with T tokens of which A are anchors.

    codes: packed unsigned codes of the other tokens [batch, kv_heads, T - A, head_dim * bits / 8]
    scale: [batch, kv_heads, T - A, 1]
    anchor_codes: uint8 [batch, kv_heads, A, head_dim]; anchor_scale: [batch, kv_heads, A, 1]
    """

    codes: torch.Tensor
    scale: torch.Tensor
    anchor_codes: torch.Tensor
    anchor_scale: torch.Tensor
    levels: int
    anchor_levels: int
    group_size: int

    @property
    def num_tokens(self) -> int:
        return self.codes.shape[2] + self.anchor_codes.shape[2]

    @property
    def head_dim(self) -> int:
        return self.anchor_codes.shape[-1]

    def symbols(self) -> torch.Tensor:
        """Unpacked codes of the tokens that are not anchors."""
        return unpack(self.codes, code_bits(self.levels))

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        values = dequantize_vectors(self.symbols(), self.scale, self.levels)
        if self.group_size == 0:
            return values.to(dtype)
        anchors = dequantize_vectors(self.anchor_codes, self.anchor_scale, self.anchor_levels)
        index = torch.arange(values.shape[2], device=values.device)
        group = index // (self.group_size - 1)
        out = torch.empty(*values.shape[:2], self.num_tokens, self.head_dim, device=values.device)
        out.index_copy_(2, torch.arange(anchors.shape[2], device=values.device) * self.group_size, anchors)
        out.index_copy_(2, index + group + 1, anchors[:, :, group] + values)
        return out.to(dtype)

    def prefix(self, num_tokens: int) -> "CacheGenTensor":
        """The codes of the leading `num_tokens` tokens."""
        anchors = num_anchors(num_tokens, self.group_size)
        rest = num_tokens - anchors
        return CacheGenTensor(
            self.codes[:, :, :rest], self.scale[:, :, :rest], self.anchor_codes[:, :, :anchors],
            self.anchor_scale[:, :, :anchors], self.levels, self.anchor_levels, self.group_size,
        )  # fmt: skip

    def nbytes(self) -> int:
        tensors = (self.codes, self.scale, self.anchor_codes, self.anchor_scale)
        return sum(t.numel() * t.element_size() for t in tensors)


def quantize_tokens(
    x: torch.Tensor, start: int, levels: int, anchor_levels: int, group_size: int, anchor: torch.Tensor | None = None
) -> tuple[CacheGenTensor, torch.Tensor | None]:
    """Quantize the tokens `x` [batch, kv_heads, n, head_dim] that start at position `start` of the sequence.

    `anchor` is the reconstructed anchor of the group that contains position `start`; it is needed
    when `start` is not the first token of a group. Returns the codes of the new tokens and the
    reconstructed anchor of the last group.
    """
    x = x.float()
    if group_size == 0:
        codes, scale = quantize_vectors(x, levels)
        anchors = _no_anchors(x.shape[0], x.shape[1], x.shape[3], x.device)
        return CacheGenTensor(pack(codes, code_bits(levels)), scale, *anchors, levels, anchor_levels, 0), None

    positions = torch.arange(start, start + x.shape[2], device=x.device)
    is_anchor = positions % group_size == 0
    anchor_codes, anchor_scale = quantize_vectors(x[:, :, is_anchor], anchor_levels)
    anchors = dequantize_vectors(anchor_codes, anchor_scale, anchor_levels)
    if start % group_size != 0:
        anchors = torch.cat([anchor.float(), anchors], dim=2)
    group = positions // group_size - start // group_size
    codes, scale = quantize_vectors(x[:, :, ~is_anchor] - anchors[:, :, group[~is_anchor]], levels)
    codes = pack(codes, code_bits(levels))
    tensor = CacheGenTensor(codes, scale, anchor_codes, anchor_scale, levels, anchor_levels, group_size)
    return tensor, anchors[:, :, -1:]


@dataclass
class CacheGenProfile:
    """Symbol statistics of a model, collected offline.

    frequencies: normalized frequency tables [tables, levels] for every (layer, is_value, is_anchor):
    one table per channel for the ordinary tokens and one table for the anchors.
    """

    config: CacheGenConfig
    frequencies: dict[tuple[int, bool, bool], torch.Tensor]


@dataclass
class CacheGenPayload:
    """Byte string of an encoded KV cache and the description that is needed to decode it."""

    data: bytes
    meta: dict

    def nbytes(self) -> int:
        return len(self.data) + len(json.dumps(self.meta))


@dataclass
class _Block:
    """The symbols of one layer, tensor and symbol kind, [tokens, batch, channels], and their tables."""

    layer: int
    is_value: bool
    is_anchor: bool
    levels: int
    symbols: torch.Tensor | None = None
    first_table: int = 0
    per_channel: bool = False

    @property
    def key(self) -> tuple[int, bool, bool]:
        return self.layer, self.is_value, self.is_anchor


@dataclass
class _Segment:
    """`chunks` chunks of `length` tokens of a block that start at token `first`, coded `width` channels per stream."""

    block: _Block
    first: int
    chunks: int
    length: int
    width: int

    @property
    def num_streams(self) -> int:
        return self.chunks * self.block.symbols.shape[1] * self.block.symbols.shape[2] // self.width

    @property
    def steps(self) -> int:
        return self.length * self.width

    def layout(self) -> rans.Streams:
        """Every stream starts at the table of its first channel and walks through `width` channels."""
        block, device = self.block, self.block.symbols.device
        stride = int(block.per_channel)
        streams_per_row = block.symbols.shape[2] // self.width
        first_channel = torch.arange(self.num_streams, device=device) % streams_per_row * self.width
        ones = torch.ones(self.num_streams, dtype=torch.int64, device=device)
        first_table = block.first_table + first_channel * stride
        return rans.Streams(ones * self.steps, first_table, ones * self.width, ones * stride)

    def _view(self) -> torch.Tensor:
        return self.block.symbols[self.first : self.first + self.chunks * self.length]

    def streams(self) -> torch.Tensor:
        """Symbols as [streams, length * width]: every stream runs through its chunk token by token."""
        _, batch, channels = self.block.symbols.shape
        x = self._view().reshape(self.chunks, self.length, batch, channels // self.width, self.width)
        return x.permute(0, 2, 3, 1, 4).reshape(self.num_streams, self.steps)

    def store(self, streams: torch.Tensor) -> None:
        _, batch, channels = self.block.symbols.shape
        x = streams.reshape(self.chunks, batch, channels // self.width, self.length, self.width)
        self._view().copy_(x.permute(0, 3, 1, 2, 4).reshape(self.chunks * self.length, batch, channels))


def _interleave(tokens: int, channels: int, max_steps: int) -> int:
    """Number of channels that share a stream, so that a stream has at most `max_steps` symbols."""
    width = 1
    while tokens * width * 2 <= max_steps and channels % (width * 2) == 0:
        width *= 2
    return width


def _serialize(tensors: dict[str, torch.Tensor]) -> tuple[bytes, list]:
    index = [[name, str(t.dtype).removeprefix("torch."), list(t.shape)] for name, t in tensors.items()]
    arrays = [t.contiguous().cpu().view(torch.uint8).numpy().tobytes() for t in tensors.values()]
    return b"".join(arrays), index


def _deserialize(data: bytes, index: list, device: torch.device) -> dict[str, torch.Tensor]:
    tensors, offset = {}, 0
    for name, dtype, shape in index:
        dtype = getattr(torch, dtype)
        size = math.prod(shape) * dtype.itemsize
        raw = torch.from_numpy(np.frombuffer(data, np.uint8, size, offset).copy())
        tensor = raw.view(dtype) if size else torch.empty(0, dtype=dtype)
        tensors[name] = tensor.reshape(shape).to(device)
        offset += size
    return tensors


class CacheGenCodec:
    """Quantizer and entropy coder for the KV cache of a model with `num_layers` layers.

    With a `profile` the symbols are coded with its per-channel statistics. Without one, the
    statistics are gathered per layer from the data and sent along with it.
    """

    def __init__(self, config: CacheGenConfig, num_layers: int, profile: CacheGenProfile | None = None):
        if profile is not None and profile.config != config:
            raise ValueError("the profile was collected with different codec settings")
        self.config = config
        self.num_layers = num_layers
        self.profile = profile

    def quantize(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[CacheGenTensor, CacheGenTensor]:
        """Codes of K and V [batch, kv_heads, tokens, head_dim] of one layer."""
        cfg = self.config
        levels = (cfg.levels(layer, self.num_layers, is_value) for is_value in (False, True))
        groups = (cfg.tokens_per_group(is_value) for is_value in (False, True))
        return tuple(quantize_tokens(x, 0, n, cfg.anchor_levels, g)[0] for x, n, g in zip((k, v), levels, groups))

    def encode(self, kv_per_layer: list[tuple[torch.Tensor, torch.Tensor]]) -> CacheGenPayload:
        return self.pack([self.quantize(layer, k, v) for layer, (k, v) in enumerate(kv_per_layer)])

    def pack(self, layers: list[tuple[CacheGenTensor, CacheGenTensor]]) -> CacheGenPayload:
        """Entropy code the quantized K and V of all layers."""
        first = layers[0][0]
        shape, device = [*first.codes.shape[:2], first.num_tokens, first.head_dim], first.codes.device
        blocks, scales = self._blocks(), []
        for block in blocks:
            tensor = layers[block.layer][block.is_value]
            symbols = tensor.anchor_codes if block.is_anchor else tensor.symbols()
            block.symbols = symbols.permute(2, 0, 1, 3).flatten(start_dim=2).to(device)
            scales.append((tensor.anchor_scale if block.is_anchor else tensor.scale).flatten().to(device))
        tables = self._tables(blocks)

        segments = self._segments(blocks, shape[2])
        rows, steps = sum(s.num_streams for s in segments), max(s.steps for s in segments)
        symbols = torch.zeros(rows, steps, dtype=torch.uint8, device=device)
        row = 0
        for segment in segments:
            symbols[row : row + segment.num_streams, : segment.steps] = segment.streams()
            row += segment.num_streams
        words, states = rans.encode(symbols, rans.Streams.cat([s.layout() for s in segments]), tables)

        sections = {"scale": torch.cat(scales), "words": words, "states": states}
        if self.profile is None:
            sections["tables"] = tables.freq.to(torch.int16)
        data, sections = _serialize(sections)
        return CacheGenPayload(data, {"shape": shape, "config": self._settings(), "sections": sections})

    def decode(self, payload: CacheGenPayload, device: torch.device | str = "cpu") -> list[tuple[CacheGenTensor, ...]]:
        """The codes and dequantization parameters of K and V of every layer."""
        if payload.meta["config"] != self._settings():
            raise ValueError("the payload was encoded with different codec settings")
        batch, heads, num_tokens, head_dim = payload.meta["shape"]
        sections = _deserialize(payload.data, payload.meta["sections"], torch.device(device))
        blocks = self._blocks()
        for block in blocks:
            tokens = self._block_tokens(block, num_tokens)[0]
            block.symbols = torch.empty(tokens, batch, heads * head_dim, dtype=torch.uint8, device=device)
        tables = self._tables(blocks, sections.get("tables"))

        segments = self._segments(blocks, num_tokens)
        layout = rans.Streams.cat([segment.layout() for segment in segments])
        symbols = rans.decode(sections["words"], sections["states"], layout, tables)
        for segment, streams in zip(segments, symbols.split([segment.num_streams for segment in segments])):
            segment.store(streams[:, : segment.steps])
        return self._assemble(blocks, sections["scale"], heads)

    @staticmethod
    def dequantize(layers: list[tuple[CacheGenTensor, ...]]) -> list[tuple[torch.Tensor, ...]]:
        """BF16 K and V of every layer from their codes."""
        return [tuple(tensor.dequantize() for tensor in pair) for pair in layers]

    def _settings(self) -> dict:
        return json.loads(json.dumps(asdict(self.config)))

    def _blocks(self) -> list[_Block]:
        """The symbol blocks of a cache in coding order."""
        cfg, blocks = self.config, []
        for layer in range(self.num_layers):
            for is_value in (False, True):
                blocks.append(_Block(layer, is_value, False, cfg.levels(layer, self.num_layers, is_value)))
                if cfg.tokens_per_group(is_value):
                    blocks.append(_Block(layer, is_value, True, cfg.anchor_levels))
        return blocks

    def _block_tokens(self, block: _Block, num_tokens: int) -> tuple[int, int]:
        """Tokens of the block in a cache of `num_tokens` tokens, and in one chunk."""
        group = self.config.tokens_per_group(block.is_value)
        anchors, chunk_anchors = num_anchors(num_tokens, group), num_anchors(self.config.chunk_size, group)
        if block.is_anchor:
            return anchors, chunk_anchors
        return num_tokens - anchors, self.config.chunk_size - chunk_anchors

    def _tables(self, blocks: list[_Block], received: torch.Tensor | None = None) -> rans.Tables:
        """Frequency tables of all blocks: from the profile, from `received` tables, or from the symbols."""
        groups, first_table, first_entry = [], 0, 0
        for block in blocks:
            if self.profile is not None:
                freq = self.profile.frequencies[block.key].to(block.symbols.device)
            elif received is not None:
                freq = received[first_entry : first_entry + block.levels][None, :].to(torch.int32)
            else:
                counts = torch.bincount(block.symbols.flatten().long(), minlength=block.levels)
                freq = rans.normalize_counts(counts[None, :])
            if freq.shape[0] not in (1, block.symbols.shape[2]):
                raise ValueError("the profile was collected on a model with a different KV cache shape")
            block.first_table, block.per_channel = first_table, freq.shape[0] > 1
            groups.append(freq)
            first_table, first_entry = first_table + freq.shape[0], first_entry + block.levels
        return rans.Tables.from_frequencies(groups)

    def _segments(self, blocks: list[_Block], num_tokens: int) -> list[_Segment]:
        """The full chunks and the remaining tokens of every block."""
        segments = []
        for block in blocks:
            tokens, chunk = self._block_tokens(block, num_tokens)
            for first, chunks, length in ((0, tokens // chunk, chunk), (tokens - tokens % chunk, 1, tokens % chunk)):
                if chunks and length:
                    width = _interleave(length, block.symbols.shape[2], self.config.chunk_size)
                    segments.append(_Segment(block, first, chunks, length, width))
        return segments

    def _assemble(self, blocks: list[_Block], scales: torch.Tensor, heads: int) -> list[tuple[CacheGenTensor, ...]]:
        """Turn the decoded symbol blocks and the scales of all blocks into the tensors of every layer."""
        cfg, fields, offset = self.config, {}, 0
        for block in blocks:
            tokens, batch, channels = block.symbols.shape
            codes = block.symbols.reshape(tokens, batch, heads, channels // heads).permute(1, 2, 0, 3)
            scale = scales[offset : offset + batch * heads * tokens].reshape(batch, heads, tokens, 1)
            offset += scale.numel()
            fields[block.key] = (codes.contiguous() if block.is_anchor else pack(codes, code_bits(block.levels)), scale)

        _, batch, channels = blocks[0].symbols.shape
        no_anchors = _no_anchors(batch, heads, channels // heads, scales.device)
        layers = []
        for layer in range(self.num_layers):
            pair = []
            for is_value in (False, True):
                codes, scale = fields[(layer, is_value, False)]
                anchors = fields.get((layer, is_value, True), no_anchors)
                levels, group = cfg.levels(layer, self.num_layers, is_value), cfg.tokens_per_group(is_value)
                pair.append(CacheGenTensor(codes, scale, *anchors, levels, cfg.anchor_levels, group))
            layers.append(tuple(pair))
        return layers

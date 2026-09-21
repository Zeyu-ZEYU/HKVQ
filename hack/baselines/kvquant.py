"""KVQuant-style KV quantizer.

Keys are quantized per channel and values per token, both to a non-uniform datatype whose levels
are fitted on calibration data. In every vector a small fraction of outliers is kept in 16-bit in
a sparse layout. The outlier thresholds of the keys come from the calibration data, the ones of
the values are found per token at run time.
"""

from dataclasses import dataclass

import torch

from hack.quant import pack, unpack

META_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class KVQuantConfig:
    """Settings of the KVQuant-style quantizer.

    bits: width of the codes.
    outlier_fraction: fraction of every vector that is kept in 16-bit.
    sink_tokens: number of leading tokens whose K and V stay in 16-bit.
    """

    bits: int = 2
    outlier_fraction: float = 0.01
    sink_tokens: int = 1

    def __post_init__(self):
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be 2, 4 or 8")
        if not 0.0 <= self.outlier_fraction < 0.5:
            raise ValueError("outlier_fraction must be in [0, 0.5)")


@dataclass
class LayerCalibration:
    """Calibrated parameters of one layer.

    key_lower, key_upper: outlier thresholds of every key channel [kv_heads, head_dim]
    key_levels, value_levels: the levels of the non-uniform datatype in [-1, 1], ascending [2**bits]
    """

    key_lower: torch.Tensor
    key_upper: torch.Tensor
    key_levels: torch.Tensor
    value_levels: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return self.key_lower, self.key_upper, self.key_levels, self.value_levels

    def to(self, device: torch.device | str) -> "LayerCalibration":
        return LayerCalibration(*(t.to(device) for t in self.tensors()))

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass
class KVQuantCalibration:
    """Calibrated parameters of all layers and the settings they were fitted for."""

    config: KVQuantConfig
    layers: list[LayerCalibration]


@dataclass
class SparseOutliers:
    """Outliers in a compressed-row layout with one row per token and batch entry, token-major.

    counts: outliers per row [tokens, batch]; columns: position in the vector of the token [n]; values: [n]
    """

    counts: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor

    @classmethod
    def extract(cls, x: torch.Tensor, mask: torch.Tensor) -> "SparseOutliers":
        """Outliers of `x` [tokens, batch, width] at the positions where `mask` is set."""
        index_dtype = torch.int16 if x.shape[-1] < 1 << 15 else torch.int32
        columns = mask.nonzero()[:, 2].to(index_dtype)
        return cls(mask.sum(dim=-1).to(index_dtype), columns, x[mask].to(META_DTYPE))

    def scatter_into(self, x: torch.Tensor) -> None:
        """Write the outliers into `x` [tokens, batch, width]."""
        rows = torch.arange(self.counts.numel(), device=x.device).repeat_interleave(self.counts.flatten().long())
        x.view(-1, x.shape[-1])[rows, self.columns.long()] = self.values.to(x.dtype)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.counts, self.columns, self.values))


@dataclass
class KVQuantTensor:
    """Quantized keys or values of one layer.

    codes: packed [batch, kv_heads, tokens, head_dim * bits / 8]
    scale, zero: [1, kv_heads, 1, head_dim] for keys (per channel), [batch, 1, tokens, 1] for values (per token)
    levels: the non-uniform datatype [2**bits]; value = levels[code] * scale + zero
    per_token: the scale and zero belong to the tokens (values) and not to the calibration (keys)
    """

    codes: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    levels: torch.Tensor
    outliers: SparseOutliers
    bits: int
    per_token: bool

    def dense(self) -> torch.Tensor:
        """Values of the codes without the outliers, float32 [batch, kv_heads, tokens, head_dim]."""
        return self.levels[unpack(self.codes, self.bits).long()] * self.scale.float() + self.zero.float()

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        batch, heads, tokens, head_dim = self.codes.shape[:3] + (self.codes.shape[-1] * 8 // self.bits,)
        rows = self.dense().permute(2, 0, 1, 3).reshape(tokens, batch, heads * head_dim)
        self.outliers.scatter_into(rows)
        return rows.reshape(tokens, batch, heads, head_dim).permute(1, 2, 0, 3).to(dtype)

    def nbytes(self) -> int:
        """Bytes of the codes, the per-token parameters and the outliers."""
        parameters = (self.scale.numel() + self.zero.numel()) * self.scale.element_size() if self.per_token else 0
        return self.codes.numel() + parameters + self.outliers.nbytes()


@dataclass
class KVQuantCodes:
    """Quantized K and V of one layer: the leading tokens in 16-bit and the codes of the other tokens."""

    sink_k: torch.Tensor
    sink_v: torch.Tensor
    keys: KVQuantTensor
    values: KVQuantTensor

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.cat([self.sink_k.to(dtype), self.keys.dequantize(dtype)], dim=2)
        return k, torch.cat([self.sink_v.to(dtype), self.values.dequantize(dtype)], dim=2)

    def nbytes(self) -> int:
        sinks = sum(t.numel() * t.element_size() for t in (self.sink_k, self.sink_v))
        return sinks + self.keys.nbytes() + self.values.nbytes()


def _rows(x: torch.Tensor) -> torch.Tensor:
    """[batch, kv_heads, tokens, head_dim] -> float32 [tokens, batch, kv_heads * head_dim]."""
    batch, heads, tokens, head_dim = x.shape
    return x.float().permute(2, 0, 1, 3).reshape(tokens, batch, heads * head_dim)


def _codes(normalized: torch.Tensor, levels: torch.Tensor, shape: torch.Size, bits: int) -> torch.Tensor:
    """Nearest level of `normalized` [tokens, batch, width], packed in the layout of the cache."""
    codes = torch.bucketize(normalized.clamp(-1.0, 1.0), (levels[1:] + levels[:-1]) / 2)
    batch, heads, tokens, head_dim = shape
    return pack(codes.reshape(tokens, batch, heads, head_dim).permute(1, 2, 0, 3).to(torch.uint8), bits)


def _affine(lower: torch.Tensor, upper: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale and zero point that map [lower, upper] to [-1, 1], in the type in which they are stored."""
    scale = ((upper - lower) / 2).to(dtype)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return scale, ((upper + lower) / 2).to(dtype)


def outliers_per_side(width: int, fraction: float) -> int:
    return round(width * fraction / 2)


def token_outliers(rows: torch.Tensor, fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per vector of `rows` [..., width]: the mask of its outliers, and the smallest and largest other value."""
    k = outliers_per_side(rows.shape[-1], fraction)
    top, bottom = torch.topk(rows, k + 1, dim=-1), torch.topk(-rows, k + 1, dim=-1)
    mask = torch.zeros_like(rows, dtype=torch.bool)
    mask.scatter_(-1, top.indices[..., :k], True).scatter_(-1, bottom.indices[..., :k], True)
    return mask, -bottom.values[..., -1:], top.values[..., -1:]


def quantize_keys(k: torch.Tensor, calibration: LayerCalibration, config: KVQuantConfig) -> KVQuantTensor:
    """Per-channel quantization of `k` [batch, kv_heads, tokens, head_dim] with calibrated thresholds."""
    rows = _rows(k)
    lower, upper = calibration.key_lower.float().flatten(), calibration.key_upper.float().flatten()
    scale, zero = _affine(lower, upper, torch.float32)
    outliers = SparseOutliers.extract(rows, (rows < lower) | (rows > upper))
    codes = _codes((rows - zero) / scale, calibration.key_levels, k.shape, config.bits)
    scale, zero = (t.reshape(1, k.shape[1], 1, k.shape[3]) for t in (scale, zero))
    return KVQuantTensor(codes, scale, zero, calibration.key_levels, outliers, config.bits, per_token=False)


def quantize_values(v: torch.Tensor, calibration: LayerCalibration, config: KVQuantConfig) -> KVQuantTensor:
    """Per-token quantization of `v` [batch, kv_heads, tokens, head_dim] with thresholds found per token."""
    rows = _rows(v)
    mask, lower, upper = token_outliers(rows, config.outlier_fraction)
    scale, zero = _affine(lower, upper, META_DTYPE)
    outliers = SparseOutliers.extract(rows, mask)
    codes = _codes((rows - zero.float()) / scale.float(), calibration.value_levels, v.shape, config.bits)
    scale, zero = (t.reshape(v.shape[2], v.shape[0], 1, 1).permute(1, 2, 0, 3) for t in (scale, zero))
    return KVQuantTensor(codes, scale, zero, calibration.value_levels, outliers, config.bits, per_token=True)


class KVQuantQuantizer:
    """Quantizer of all layers of a model."""

    def __init__(self, calibration: KVQuantCalibration):
        self.config = calibration.config
        self.layers = calibration.layers

    def quantize(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> KVQuantCodes:
        """Quantize K and V [batch, kv_heads, tokens, head_dim] of one layer, a sequence from its first token on."""
        calibration, sinks = self.layers[layer].to(k.device), self.config.sink_tokens
        keys = quantize_keys(k[:, :, sinks:], calibration, self.config)
        values = quantize_values(v[:, :, sinks:], calibration, self.config)
        return KVQuantCodes(k[:, :, :sinks], v[:, :, :sinks], keys, values)


def fit_levels(normalized: torch.Tensor, bits: int, iterations: int = 40, max_samples: int = 1 << 20) -> torch.Tensor:
    """Levels of the non-uniform datatype: one-dimensional k-means over `normalized` values in [-1, 1]."""
    samples = normalized.flatten()
    samples = samples[:: max(1, samples.numel() // max_samples)].double().sort().values
    prefix = torch.cat([samples.new_zeros(1), samples.cumsum(dim=0)])
    count = 1 << bits
    levels = samples[((torch.arange(count, device=samples.device) + 0.5) * samples.numel() / count).long()]
    ends = torch.tensor([0, samples.numel()], device=samples.device)
    for _ in range(iterations):
        edges = torch.searchsorted(samples, (levels[1:] + levels[:-1]) / 2)
        bounds = torch.cat([ends[:1], edges, ends[1:]])
        sizes = bounds[1:] - bounds[:-1]
        means = (prefix[bounds[1:]] - prefix[bounds[:-1]]) / sizes.clamp_min(1)
        levels = torch.where(sizes > 0, means, levels)
    return levels.float()


def fit_layer(k: torch.Tensor, v: torch.Tensor, config: KVQuantConfig) -> LayerCalibration:
    """Calibrate one layer on sample keys and values [batch, kv_heads, tokens, head_dim]."""
    keys = k.float().transpose(1, 2).reshape(-1, k.shape[1], k.shape[3])
    side = max(1, round(keys.shape[0] * config.outlier_fraction / 2))
    key_lower = torch.kthvalue(keys, side, dim=0).values
    key_upper = torch.kthvalue(keys, keys.shape[0] - side + 1, dim=0).values
    scale, zero = _affine(key_lower, key_upper, torch.float32)
    normalized_keys = (keys - zero) / scale
    key_levels = fit_levels(normalized_keys[(normalized_keys >= -1) & (normalized_keys <= 1)], config.bits)

    rows = _rows(v)
    mask, lower, upper = token_outliers(rows, config.outlier_fraction)
    scale, zero = _affine(lower, upper, META_DTYPE)
    value_levels = fit_levels(((rows - zero.float()) / scale.float())[~mask].clamp(-1.0, 1.0), config.bits)
    return LayerCalibration(key_lower, key_upper, key_levels, value_levels)

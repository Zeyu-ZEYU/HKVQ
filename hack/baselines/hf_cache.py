"""Hugging Face `Cache` classes of the comparison methods.

The caches keep K and V in the quantized format of the method and hand dequantized BF16 tensors to
the attention implementation of the model at every step.

    cache = make_cache("kvquant", model.config, calibration="kvquant.pt")
    model.generate(**inputs, past_key_values=cache)
"""

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from hack.baselines import calibration as calibration_files
from hack.baselines.cachegen import (
    CacheGenCodec, CacheGenConfig, CacheGenPayload, CacheGenProfile, CacheGenTensor, code_bits, quantize_tokens,
)  # fmt: skip
from hack.baselines.kvquant import (
    KVQuantCalibration, KVQuantCodes, KVQuantConfig, KVQuantTensor, LayerCalibration, SparseOutliers, fit_layer,
    quantize_keys, quantize_values,
)  # fmt: skip
from hack.quant import pack

if TYPE_CHECKING:
    from hack.kernels.quantized_kv import QuantizedKV


class _Buffer:
    """Tensor that grows along one dimension with amortized constant cost."""

    def __init__(self, dim: int):
        self.dim = dim
        self.data: torch.Tensor | None = None
        self.length = 0

    def append(self, x: torch.Tensor) -> None:
        n = x.shape[self.dim]
        if self.data is None or self.length + n > self.data.shape[self.dim]:
            shape = list(x.shape)
            shape[self.dim] = max(2 * (self.length + n), 64)
            grown = torch.empty(shape, dtype=x.dtype, device=x.device)
            if self.data is not None:
                grown.narrow(self.dim, 0, self.length).copy_(self.view())
            self.data = grown
        self.data.narrow(self.dim, self.length, n).copy_(x)
        self.length += n

    def view(self, length: int | None = None) -> torch.Tensor:
        return self.data.narrow(self.dim, 0, self.length if length is None else length)


def _group_size(head_dim: int) -> int:
    """Group size of the kernel container: the largest power of two that divides the head dimension."""
    return head_dim & -head_dim


def _uniform_fields(tensor: CacheGenTensor, bits: int, group: int) -> dict[str, torch.Tensor] | None:
    """Codes, scale and zero of a tensor without anchors, one pair per token and group of channels."""
    if tensor.group_size or tensor.levels > 1 << bits:
        return None
    scale = tensor.scale.float().expand(-1, -1, -1, tensor.head_dim // group)
    return {"codes": pack(tensor.symbols(), bits), "scale": scale, "zero": -((tensor.levels - 1) // 2) * scale}


class StorageOnlyLayer(CacheLayerMixin):
    """Cache layer that stores quantized K and V and dequantizes all of them at every update.

    With `exact_prefill`, the update that fills the empty cache returns the K and V it was given,
    as on a prefill instance that computes attention before it compresses the KV cache.
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, exact_prefill: bool):
        super().__init__()
        self.exact_prefill = exact_prefill
        self.reset()

    def reset(self) -> None:
        self.num_tokens = 0
        self.prompt_tokens = 0
        self.clear()

    def clear(self) -> None:
        """Drop the stored tokens."""
        raise NotImplementedError

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store the K and V of new tokens [batch, kv_heads, new_tokens, head_dim] in quantized form."""
        raise NotImplementedError

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V of all stored tokens in the type of the model."""
        raise NotImplementedError

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        is_prefill = self.num_tokens == 0
        self.append(key_states, value_states)
        self.num_tokens += key_states.shape[2]
        if is_prefill:
            self.prompt_tokens = self.num_tokens
            if self.exact_prefill:
                return key_states, value_states
        return self.dequantize()

    def get_seq_length(self) -> int:
        return self.num_tokens

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.num_tokens + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def offload(self) -> None:
        pass

    def prefetch(self) -> None:
        pass


class _CacheGenStore:
    """Growing CacheGen-style codes of the keys or the values of one layer."""

    FIELDS = ("codes", "scale", "anchor_codes", "anchor_scale")

    def __init__(self, levels: int, anchor_levels: int, group_size: int):
        self.levels, self.anchor_levels, self.group_size = levels, anchor_levels, group_size
        self.buffers = {name: _Buffer(dim=2) for name in self.FIELDS}
        self.anchor: torch.Tensor | None = None
        self.num_tokens = 0

    def append(self, x: torch.Tensor) -> None:
        settings = (self.levels, self.anchor_levels, self.group_size)
        new, self.anchor = quantize_tokens(x, self.num_tokens, *settings, self.anchor)
        for name, buffer in self.buffers.items():
            buffer.append(getattr(new, name))
        self.num_tokens += x.shape[2]

    def tensor(self) -> CacheGenTensor:
        fields = (buffer.view() for buffer in self.buffers.values())
        return CacheGenTensor(*fields, self.levels, self.anchor_levels, self.group_size)


class CacheGenLayer(StorageOnlyLayer):
    """K and V of one layer as CacheGen-style codes."""

    def __init__(self, config: CacheGenConfig, layer_idx: int, num_layers: int, exact_prefill: bool):
        levels = (config.levels(layer_idx, num_layers, is_value) for is_value in (False, True))
        groups = (config.tokens_per_group(is_value) for is_value in (False, True))
        self.settings = [(n, config.anchor_levels, group) for n, group in zip(levels, groups)]
        super().__init__(exact_prefill)

    def clear(self) -> None:
        self.stores = tuple(_CacheGenStore(*settings) for settings in self.settings)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        for store, x in zip(self.stores, (k, v)):
            store.append(x)

    def tensors(self) -> tuple[CacheGenTensor, CacheGenTensor]:
        return tuple(store.tensor() for store in self.stores)

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(tensor.dequantize(self.dtype) for tensor in self.tensors())

    def nbytes(self) -> int:
        return sum(tensor.nbytes() for tensor in self.tensors())

    def to_quantized_kv(self) -> "QuantizedKV":
        """K and V of the layer in the container of the fused dequantization kernel.

        A tensor without anchors keeps its codes and scales. A tensor with anchors is encoded again
        from its dequantized values, per channel in groups of tokens.
        """
        from hack.kernels.quantized_kv import CHANNEL_GROUPS, TOKEN_GROUPS, QuantizedKV

        keys, values = self.tensors()
        bits = min(max(code_bits(keys.levels), code_bits(values.levels)), 4)
        group = _group_size(keys.head_dim)
        kept = {"k": _uniform_fields(keys, bits, group), "v": _uniform_fields(values, bits, group)}
        k, v = keys.dequantize(torch.float32), values.dequantize(torch.float32)
        k_axis = CHANNEL_GROUPS if kept["k"] else TOKEN_GROUPS
        qkv = QuantizedKV.from_tensors(k, v, bits, group, k_axis=k_axis, meta_dtype=torch.float32)
        fields = {f"{name}_{field}": x for name, part in kept.items() if part for field, x in part.items()}
        return replace(qkv, **fields)


class _KVQuantStore:
    """Growing KVQuant-style codes of the keys or the values of one layer."""

    def __init__(self):
        self.dense = {name: _Buffer(dim=2) for name in ("codes", "scale", "zero")}
        self.sparse = {name: _Buffer(dim=0) for name in ("counts", "columns", "values")}
        self.last: KVQuantTensor | None = None

    def append(self, new: KVQuantTensor) -> None:
        self.last = new
        self.dense["codes"].append(new.codes)
        if new.per_token:
            self.dense["scale"].append(new.scale)
            self.dense["zero"].append(new.zero)
        for name, buffer in self.sparse.items():
            buffer.append(getattr(new.outliers, name))

    def tensor(self, num_tokens: int | None = None) -> KVQuantTensor:
        """The codes of all tokens, or of the leading `num_tokens` tokens."""
        last = self.last
        counts = self.sparse["counts"].view(num_tokens)
        entries = None if num_tokens is None else int(counts.sum())
        outliers = SparseOutliers(counts, self.sparse["columns"].view(entries), self.sparse["values"].view(entries))
        scale, zero = last.scale, last.zero
        if last.per_token:
            scale, zero = self.dense["scale"].view(num_tokens), self.dense["zero"].view(num_tokens)
        codes = self.dense["codes"].view(num_tokens)
        return KVQuantTensor(codes, scale, zero, last.levels, outliers, last.bits, last.per_token)


class KVQuantLayer(StorageOnlyLayer):
    """K and V of one layer as KVQuant-style codes; without a calibration the layer is fitted on its first tokens."""

    def __init__(self, config: KVQuantConfig, calibration: LayerCalibration | None, exact_prefill: bool):
        self.config = config
        self.calibration = calibration
        self.calibrated_online = calibration is None
        super().__init__(exact_prefill)

    def clear(self) -> None:
        self.sinks: tuple[torch.Tensor, torch.Tensor] | None = None
        self.stores = (_KVQuantStore(), _KVQuantStore())
        if self.calibrated_online:
            self.calibration = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        kept = 0 if self.sinks is None else self.sinks[0].shape[2]
        take = max(0, min(self.config.sink_tokens - kept, k.shape[2]))
        new = (k[:, :, :take], v[:, :, :take])
        self.sinks = new if self.sinks is None else tuple(torch.cat(pair, dim=2) for pair in zip(self.sinks, new))
        k, v = k[:, :, take:], v[:, :, take:]
        if self.calibration is None and k.shape[2] > 0:
            self.calibration = fit_layer(k, v, self.config)
        if self.calibration is not None:
            self.calibration = self.calibration.to(k.device)
            self.stores[0].append(quantize_keys(k, self.calibration, self.config))
            self.stores[1].append(quantize_values(v, self.calibration, self.config))

    def codes(self, num_tokens: int | None = None) -> KVQuantCodes:
        """The cached K and V, or their leading `num_tokens` tokens."""
        sinks = self.sinks if num_tokens is None else tuple(t[:, :, :num_tokens] for t in self.sinks)
        quantized = None if num_tokens is None else num_tokens - sinks[0].shape[2]
        return KVQuantCodes(*sinks, *(store.tensor(quantized) for store in self.stores))

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sinks if self.calibration is None else self.codes().dequantize(self.dtype)

    def _parameter_nbytes(self) -> int:
        return 0 if self.calibration is None else self.calibration.nbytes()

    def nbytes(self) -> int:
        if self.calibration is None:
            return sum(t.numel() * t.element_size() for t in self.sinks or ())
        return self.codes().nbytes() + self._parameter_nbytes()

    def transfer_nbytes(self) -> int:
        """Bytes of the prompt: sink tokens, codes, per-token parameters and outliers, plus the parameters
        of the layer if they were fitted on the prompt."""
        if self.calibration is None:
            return self.nbytes()
        parameters = self._parameter_nbytes() if self.calibrated_online else 0
        return self.codes(self.prompt_tokens).nbytes() + parameters

    def to_quantized_kv(self) -> "QuantizedKV":
        """K and V of the layer in the container of the fused dequantization kernel.

        The container takes the codes with the tables of the layer, the keys with their per-channel
        parameters for every group of tokens and the values with their per-token parameters. The
        16-bit tokens are quantized like the other tokens, and the outliers stay at their codes.
        """
        from hack.kernels.quantized_kv import TOKEN_GROUPS, QuantizedKV

        k, v = self.dequantize()
        group = _group_size(k.shape[-1])
        if self.calibration is None or self.config.bits > 4:
            return QuantizedKV.from_tensors(k, v, 4, group, k_axis=TOKEN_GROUPS)
        keys = quantize_keys(self.sinks[0], self.calibration, self.config), self.stores[0].tensor()
        values = quantize_values(self.sinks[1], self.calibration, self.config), self.stores[1].tensor()
        key_shape = (k.shape[0], k.shape[1], -(-k.shape[2] // group), k.shape[3])
        value_shape = (*v.shape[:3], v.shape[3] // group)
        return QuantizedKV(
            torch.cat([part.codes for part in keys], dim=2),
            keys[1].scale.expand(key_shape),
            keys[1].zero.expand(key_shape),
            torch.cat([part.codes for part in values], dim=2),
            torch.cat([part.scale for part in values], dim=2).expand(value_shape),
            torch.cat([part.zero for part in values], dim=2).expand(value_shape),
            self.config.bits, group, TOKEN_GROUPS, k_lut=keys[1].levels, v_lut=values[1].levels,
        )  # fmt: skip


class StorageOnlyCache(Cache):
    """Cache of a storage-only quantization method with memory and transfer accounting."""

    def nbytes(self) -> int:
        """Bytes of the cached K and V of all layers."""
        return sum(layer.nbytes() for layer in self.layers)

    def transfer_nbytes(self) -> int:
        """Bytes that the prefill instance sends to the decode instance for the prompt."""
        return sum(layer.transfer_nbytes() for layer in self.layers)

    def to_quantized_kv(self, layer_idx: int) -> "QuantizedKV":
        return self.layers[layer_idx].to_quantized_kv()


def _num_layers(model_config) -> int:
    text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
    return text_config.num_hidden_layers


class CacheGenCache(StorageOnlyCache):
    """CacheGen-style cache; `profile` is a `CacheGenProfile` or the file it was saved to."""

    def __init__(
        self,
        model_config,
        config: CacheGenConfig | None = None,
        profile: CacheGenProfile | str | None = None,
        exact_prefill: bool = False,
    ):
        profile = calibration_files.load(profile) if isinstance(profile, str) else profile
        config = config or (profile.config if profile is not None else CacheGenConfig())
        num_layers = _num_layers(model_config)
        self.codec = CacheGenCodec(config, num_layers, profile)
        super().__init__(layers=[CacheGenLayer(config, i, num_layers, exact_prefill) for i in range(num_layers)])

    def encode_prompt(self) -> CacheGenPayload:
        """The byte string that carries the K and V of the prompt from the prefill to the decode instance."""
        prompt = self.layers[0].prompt_tokens
        return self.codec.pack([tuple(t.prefix(prompt) for t in layer.tensors()) for layer in self.layers])

    def transfer_nbytes(self) -> int:
        """Size of the entropy-coded prompt; every call runs the entropy coder."""
        return self.encode_prompt().nbytes()


class KVQuantCache(StorageOnlyCache):
    """KVQuant-style cache; `calibration` is a `KVQuantCalibration` or the file it was saved to."""

    def __init__(
        self,
        model_config,
        config: KVQuantConfig | None = None,
        calibration: KVQuantCalibration | str | None = None,
        exact_prefill: bool = False,
    ):
        calibration = calibration_files.load(calibration) if isinstance(calibration, str) else calibration
        if calibration is not None and config is not None and calibration.config != config:
            raise ValueError("the calibration was fitted with different quantizer settings")
        config = config or (calibration.config if calibration is not None else KVQuantConfig())
        layers = calibration.layers if calibration is not None else [None] * _num_layers(model_config)
        if len(layers) != _num_layers(model_config):
            raise ValueError("the calibration belongs to a model with a different number of layers")
        super().__init__(layers=[KVQuantLayer(config, layer, exact_prefill) for layer in layers])


def make_cache(method: str, model_config, **options) -> StorageOnlyCache:
    """Cache of the comparison method "cachegen" or "kvquant" for the model of `model_config`.

    Options of "cachegen": config (CacheGenConfig), profile (CacheGenProfile or its file), exact_prefill.
    Options of "kvquant": config (KVQuantConfig), calibration (KVQuantCalibration or its file), exact_prefill.
    """
    if method == "cachegen":
        return CacheGenCache(model_config, **options)
    if method == "kvquant":
        return KVQuantCache(model_config, **options)
    raise ValueError(f"unknown method {method!r}")

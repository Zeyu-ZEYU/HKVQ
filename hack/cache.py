"""Quantized KV cache of one attention layer."""

import torch

from hack.config import HackConfig
from hack.quant import pack, partition_sums, unpack


def channel_partition_size(head_dim: int, partition_size: int) -> int:
    size = min(partition_size, head_dim)
    while size > 0 and head_dim % size != 0:
        size -= 16
    if size <= 0:
        raise ValueError(f"head_dim {head_dim} has no partition size that is a multiple of 16")
    return size


def quantize_partitions(x, bits, dim, partition_size, stochastic, meta_dtype):
    """Quantize `x` in partitions along `dim` against (min, scale) as stored in `meta_dtype`.

    Returns codes with the shape of `x`, and scale and minimum with `dim` reduced to the
    number of partitions.
    """
    dim = dim % x.dim()
    shape = list(x.shape)
    levels = (1 << bits) - 1
    grouped = x.float().reshape(shape[:dim] + [shape[dim] // partition_size, partition_size] + shape[dim + 1 :])
    minimum = grouped.amin(dim=dim + 1, keepdim=True).to(meta_dtype)
    scale = ((grouped.amax(dim=dim + 1, keepdim=True) - minimum.float()) / levels).to(meta_dtype)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (grouped - minimum.float()) / scale.float()
    if stochastic:
        floor = torch.floor(normalized)
        normalized = floor + (torch.rand_like(normalized) < (normalized - floor)).float()
    else:
        normalized = torch.round(normalized)
    codes = normalized.clamp_(0, levels).to(torch.uint8).reshape(shape)
    return codes, scale.squeeze(dim + 1), minimum.squeeze(dim + 1)


class _Growable:
    """Tensor that grows along one dimension with amortized constant cost."""

    def __init__(self, dim: int):
        self.dim = dim
        self.data: torch.Tensor | None = None
        self.length = 0

    def append(self, x: torch.Tensor):
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

    def view(self) -> torch.Tensor:
        return self.data.narrow(self.dim, 0, self.length)

    def set(self, x: torch.Tensor):
        self.data = x
        self.length = x.shape[self.dim]


class HackLayerCache:
    """K and V of one layer for a batch of equally long sequences."""

    FIELDS = ("k_codes", "k_scale", "k_min", "k_sums", "v_codes", "v_scale", "v_min", "v_sums")
    DENSE = ("sink_k", "sink_v", "k_open", "v_open")

    def __init__(self, config: HackConfig, meta_dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.meta_dtype = meta_dtype
        self.channel_part: int | None = None
        for name in self.FIELDS:
            setattr(self, name, _Growable(dim=2))
        for name in self.DENSE:
            setattr(self, name, None)
        self.num_tokens = 0

    @property
    def num_sink_tokens(self) -> int:
        return 0 if self.sink_k is None else self.sink_k.shape[2]

    @property
    def num_quantized_tokens(self) -> int:
        return self.k_scale.length * self.config.partition_size

    @property
    def num_open_tokens(self) -> int:
        return 0 if self.k_open is None else self.k_open.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor):
        """Add the K and V of new tokens, both [batch, kv_heads, new_tokens, head_dim]."""
        cfg = self.config
        self.num_tokens += k.shape[2]
        if self.channel_part is None:
            self.channel_part = channel_partition_size(k.shape[-1], cfg.partition_size)
        missing_sinks = cfg.sink_tokens - self.num_sink_tokens
        if missing_sinks > 0:
            take = min(missing_sinks, k.shape[2])
            self.sink_k = k[:, :, :take] if self.sink_k is None else torch.cat([self.sink_k, k[:, :, :take]], dim=2)
            self.sink_v = v[:, :, :take] if self.sink_v is None else torch.cat([self.sink_v, v[:, :, :take]], dim=2)
            k, v = k[:, :, take:], v[:, :, take:]
            if k.shape[2] == 0:
                return

        k = k if self.k_open is None else torch.cat([self.k_open, k], dim=2)
        v = v if self.v_open is None else torch.cat([self.v_open, v], dim=2)
        full = (k.shape[2] // cfg.partition_size) * cfg.partition_size
        if full:
            self._close_blocks(k[:, :, :full], v[:, :, :full])
        if full == k.shape[2]:
            self.k_open = self.v_open = None
            return
        self.k_open, self.v_open = k[:, :, full:].contiguous(), v[:, :, full:].contiguous()
        if not cfg.requant_elimination:
            self.k_open, self.v_open = self._requantized(self.k_open, 2), self._requantized(self.v_open, -1)

    def _close_blocks(self, k: torch.Tensor, v: torch.Tensor):
        cfg = self.config
        codes, scale, minimum = quantize_partitions(k, cfg.kv_bits, 2, cfg.partition_size, cfg.stochastic, self.meta_dtype)
        self.k_codes.append(pack(codes, cfg.kv_bits))
        self.k_scale.append(scale)
        self.k_min.append(minimum)
        self.k_sums.append(partition_sums(codes, -1, self.channel_part, cfg.kv_bits))

        codes, scale, minimum = quantize_partitions(v, cfg.kv_bits, -1, self.channel_part, cfg.stochastic, self.meta_dtype)
        self.v_codes.append(pack(codes, cfg.kv_bits))
        self.v_scale.append(scale)
        self.v_min.append(minimum)
        self.v_sums.append(partition_sums(codes, 2, cfg.partition_size, cfg.kv_bits))

    def _requantized(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Without requantization elimination the partial block is stored quantized: every
        added token dequantizes the block and quantizes it again with the updated range."""
        cfg = self.config
        size = x.shape[dim] if dim == 2 else self.channel_part
        codes, scale, minimum = quantize_partitions(x, cfg.kv_bits, dim, size, cfg.stochastic, self.meta_dtype)
        if dim != 2:
            scale, minimum = scale.repeat_interleave(size, dim=-1), minimum.repeat_interleave(size, dim=-1)
        return (codes.float() * scale.float() + minimum.float()).to(x.dtype)

    def k_unpacked(self) -> torch.Tensor:
        return unpack(self.k_codes.view(), self.config.kv_bits)

    def v_unpacked(self) -> torch.Tensor:
        return unpack(self.v_codes.view(), self.config.kv_bits)

    def tensors(self) -> dict[str, torch.Tensor]:
        state = {name: getattr(self, name).view() for name in self.FIELDS if getattr(self, name).data is not None}
        state.update({name: getattr(self, name) for name in self.DENSE if getattr(self, name) is not None})
        return state

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors().values())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.contiguous() for name, tensor in self.tensors().items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], num_tokens: int, head_dim: int):
        self.channel_part = channel_partition_size(head_dim, self.config.partition_size)
        for name in self.FIELDS:
            if name in state:
                getattr(self, name).set(state[name])
        for name in self.DENSE:
            setattr(self, name, state.get(name))
        self.num_tokens = num_tokens

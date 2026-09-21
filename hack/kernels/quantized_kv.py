"""Storage-only quantized KV: packed codes that are dequantized inside the attention kernel."""

from dataclasses import dataclass

import torch

from hack.quant import pack, unpack

CHANNEL_GROUPS = "channel"
TOKEN_GROUPS = "token"


def _grouped(x: torch.Tensor, axis: str, group_size: int) -> list[torch.Tensor]:
    """Split `x` [batch, heads, tokens, head_dim] into pieces [..., groups, group, ...] that are quantized alike."""
    if axis == CHANNEL_GROUPS:
        return [x.unflatten(-1, (x.shape[-1] // group_size, group_size))]
    full = (x.shape[2] // group_size) * group_size
    pieces = [x[:, :, :full].unflatten(2, (full // group_size, group_size))] if full else []
    return pieces + ([x[:, :, full:].unsqueeze(2)] if full < x.shape[2] else [])


def _quantize(x: torch.Tensor, bits: int, axis: str, group_size: int, lut: torch.Tensor | None, meta_dtype):
    """Codes of `x` and the (scale, zero) of every group; with a table the codes index its nearest entry."""
    levels = torch.arange(1 << bits, device=x.device, dtype=torch.float32) if lut is None else lut.float()
    dim = -1 if axis == CHANNEL_GROUPS else 3
    codes, scales, zeros = [], [], []
    for piece in _grouped(x.float(), axis, group_size):
        low, high = piece.amin(dim=dim, keepdim=True), piece.amax(dim=dim, keepdim=True)
        scale = ((high - low) / (levels[-1] - levels[0])).to(meta_dtype)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = (low - levels[0] * scale.float()).to(meta_dtype)
        normalized = (piece - zero.float()) / scale.float()
        if lut is None:
            nearest = torch.round(normalized).clamp_(0, levels.numel() - 1).to(torch.uint8)
        else:
            nearest = (normalized.unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.uint8)
        codes.append(nearest.flatten(2, 3) if axis == TOKEN_GROUPS else nearest.flatten(-2, -1))
        scales.append(scale.squeeze(dim))
        zeros.append(zero.squeeze(dim))
    return pack(torch.cat(codes, dim=2), bits), torch.cat(scales, dim=2), torch.cat(zeros, dim=2)


def _dequantize(codes, scale, zero, bits: int, axis: str, group_size: int, lut: torch.Tensor | None) -> torch.Tensor:
    numbers = unpack(codes, bits).float() if lut is None else lut.float()[unpack(codes, bits).long()]
    dim = -1 if axis == CHANNEL_GROUPS else 2
    scale = scale.float().repeat_interleave(group_size, dim=dim)[:, :, : numbers.shape[2]]
    zero = zero.float().repeat_interleave(group_size, dim=dim)[:, :, : numbers.shape[2]]
    return numbers * scale + zero


@dataclass
class QuantizedKV:
    """K and V as `bits`-bit codes with value = level(code) * scale + zero.

    level(code) is the code itself, or `lut[code]` when the tensor has a table of 2 ** bits
    entries. A (scale, zero) pair is shared by a group of `group_size` channels of one token
    (axis "channel", scale and zero [batch, kv_heads, tokens, head_dim / group_size]) or by
    a group of `group_size` tokens of one channel (axis "token", [batch, kv_heads,
    ceil(tokens / group_size), head_dim]). codes: uint8 [batch, kv_heads, tokens, head_dim * bits / 8].
    """

    k_codes: torch.Tensor
    k_scale: torch.Tensor
    k_zero: torch.Tensor
    v_codes: torch.Tensor
    v_scale: torch.Tensor
    v_zero: torch.Tensor
    bits: int
    group_size: int
    k_axis: str = CHANNEL_GROUPS
    v_axis: str = CHANNEL_GROUPS
    k_lut: torch.Tensor | None = None
    v_lut: torch.Tensor | None = None

    @classmethod
    def from_tensors(
        cls, k: torch.Tensor, v: torch.Tensor, bits: int = 2, group_size: int = 64, k_axis: str = CHANNEL_GROUPS,
        v_axis: str = CHANNEL_GROUPS, k_lut: torch.Tensor | None = None, v_lut: torch.Tensor | None = None,
        meta_dtype: torch.dtype = torch.bfloat16,
    ) -> "QuantizedKV":  # fmt: skip
        if bits not in (2, 4):
            raise ValueError("bits must be 2 or 4")
        if any(lut is not None and lut.numel() != 1 << bits for lut in (k_lut, v_lut)):
            raise ValueError("a lookup table needs 2 ** bits entries")
        k_fields = _quantize(k, bits, k_axis, group_size, k_lut, meta_dtype)
        v_fields = _quantize(v, bits, v_axis, group_size, v_lut, meta_dtype)
        return cls(*k_fields, *v_fields, bits, group_size, k_axis, v_axis, k_lut, v_lut)

    @property
    def num_tokens(self) -> int:
        return self.k_codes.shape[2]

    @property
    def head_dim(self) -> int:
        return self.k_codes.shape[-1] * 8 // self.bits

    def narrow(self, tokens: int) -> "QuantizedKV":
        """The first `tokens` tokens, sharing the tensors of this object."""
        rows = {CHANNEL_GROUPS: tokens, TOKEN_GROUPS: -(-tokens // self.group_size)}
        k_codes, v_codes = self.k_codes[:, :, :tokens], self.v_codes[:, :, :tokens]
        k_meta = (self.k_scale[:, :, : rows[self.k_axis]], self.k_zero[:, :, : rows[self.k_axis]])
        v_meta = (self.v_scale[:, :, : rows[self.v_axis]], self.v_zero[:, :, : rows[self.v_axis]])
        settings = (self.bits, self.group_size, self.k_axis, self.v_axis, self.k_lut, self.v_lut)
        return QuantizedKV(k_codes, *k_meta, v_codes, *v_meta, *settings)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
        """K and V as the fused kernel reconstructs them."""
        k = _dequantize(self.k_codes, self.k_scale, self.k_zero, self.bits, self.k_axis, self.group_size, self.k_lut)
        v = _dequantize(self.v_codes, self.v_scale, self.v_zero, self.bits, self.v_axis, self.group_size, self.v_lut)
        return k.to(dtype), v.to(dtype)

    def nbytes(self) -> int:
        tensors = (self.k_codes, self.k_scale, self.k_zero, self.v_codes, self.v_scale, self.v_zero)
        return sum(t.numel() * t.element_size() for t in tensors + (self.k_lut, self.v_lut) if t is not None)

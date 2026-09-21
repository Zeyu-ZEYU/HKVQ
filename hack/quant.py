"""Asymmetric min-max quantization with partitions, and bit packing (PyTorch reference)."""

import torch


def stochastic_round(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    floor = torch.floor(x)
    noise = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return floor + (noise < (x - floor)).to(x.dtype)


def quantize(
    x: torch.Tensor,
    bits: int,
    dim: int,
    partition_size: int,
    stochastic: bool = True,
    generator: torch.Generator | None = None,
):
    """Quantize `x` in partitions of `partition_size` elements along `dim`.

    Returns (codes, scale, minimum). `codes` has the shape of `x` (uint8). `scale` and
    `minimum` have the shape of `x` with `dim` reduced to the number of partitions.
    """
    dim = dim % x.dim()
    size = x.shape[dim]
    if size % partition_size != 0:
        raise ValueError(f"dimension of size {size} is not a multiple of {partition_size}")
    levels = (1 << bits) - 1
    shape = list(x.shape)
    grouped = x.float().reshape(shape[:dim] + [size // partition_size, partition_size] + shape[dim + 1 :])
    minimum = grouped.amin(dim=dim + 1, keepdim=True)
    maximum = grouped.amax(dim=dim + 1, keepdim=True)
    scale = (maximum - minimum) / levels
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = (grouped - minimum) / scale
    rounded = stochastic_round(normalized, generator) if stochastic else torch.round(normalized)
    codes = rounded.clamp_(0, levels).to(torch.uint8).reshape(shape)
    return codes, scale.squeeze(dim + 1), minimum.squeeze(dim + 1)


def dequantize(codes: torch.Tensor, scale: torch.Tensor, minimum: torch.Tensor, dim: int, partition_size: int):
    dim = dim % codes.dim()
    shape = list(codes.shape)
    grouped = codes.float().reshape(shape[:dim] + [shape[dim] // partition_size, partition_size] + shape[dim + 1 :])
    values = grouped * scale.float().unsqueeze(dim + 1) + minimum.float().unsqueeze(dim + 1)
    return values.reshape(shape)


def partition_sums(codes: torch.Tensor, dim: int, partition_size: int, bits: int) -> torch.Tensor:
    """Sum of the codes of every partition along `dim` (the cached term of summation elimination)."""
    dim = dim % codes.dim()
    shape = list(codes.shape)
    grouped = codes.reshape(shape[:dim] + [shape[dim] // partition_size, partition_size] + shape[dim + 1 :])
    return grouped.sum(dim=dim + 1, dtype=torch.int32).to(sum_dtype(partition_size, bits))


def sum_dtype(partition_size: int, bits: int) -> torch.dtype:
    """Smallest integer type that holds the sum of `partition_size` codes of `bits` bits."""
    largest = partition_size * ((1 << bits) - 1)
    if largest <= 255:
        return torch.uint8
    if largest <= 32767:
        return torch.int16
    return torch.int32


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack codes of `bits` bits along the last dimension, 8 // bits codes per byte."""
    if bits == 8:
        return codes
    per_byte = 8 // bits
    if codes.shape[-1] % per_byte != 0:
        raise ValueError("last dimension must be a multiple of the number of codes per byte")
    grouped = codes.reshape(*codes.shape[:-1], codes.shape[-1] // per_byte, per_byte).to(torch.int32)
    shifts = torch.arange(per_byte, device=codes.device, dtype=torch.int32) * bits
    return (grouped << shifts).sum(dim=-1).to(torch.uint8)


def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return packed
    per_byte = 8 // bits
    shifts = torch.arange(per_byte, device=packed.device, dtype=torch.int32) * bits
    codes = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * per_byte).to(torch.uint8)

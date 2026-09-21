"""Homomorphic matrix multiplication on quantized operands (Eq. 4 of the paper), PyTorch reference."""

import torch


def homomorphic_matmul(
    a_codes: torch.Tensor,
    a_scale: torch.Tensor,
    a_min: torch.Tensor,
    b_codes: torch.Tensor,
    b_scale: torch.Tensor,
    b_min: torch.Tensor,
    partition_size: int,
    a_sums: torch.Tensor | None = None,
    b_sums: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate A @ B from the codes of A and B without dequantizing either operand.

    a_codes: [..., M, Z]   a_scale, a_min: [..., M, Z // partition_size]
    b_codes: [..., Z, N]   b_scale, b_min: [..., Z // partition_size, N]
    a_sums / b_sums: optional cached per-partition code sums with the shapes of the scales.
    The inner dimension Z is split into partitions; each partition contributes
        s_a s_b sum(a' b') + m_b s_a sum(a') + m_a s_b sum(b') + partition_size m_a m_b.
    """
    *batch, m, z = a_codes.shape
    n = b_codes.shape[-1]
    parts = z // partition_size
    a = a_codes.reshape(*batch, m, parts, partition_size).transpose(-3, -2).float()
    b = b_codes.reshape(*batch, parts, partition_size, n).float()
    product = torch.matmul(a, b)

    if a_sums is None:
        a_sums = a.sum(dim=-1).transpose(-2, -1)
    if b_sums is None:
        b_sums = b.sum(dim=-2)
    a_scale, a_min, a_sums = (t.float().transpose(-2, -1).unsqueeze(-1) for t in (a_scale, a_min, a_sums))
    b_scale, b_min, b_sums = (t.float().unsqueeze(-2) for t in (b_scale, b_min, b_sums))

    result = a_scale * b_scale * product
    result = result + b_min * a_scale * a_sums
    result = result + a_min * b_scale * b_sums
    result = result + partition_size * a_min * b_min
    return result.sum(dim=-3)

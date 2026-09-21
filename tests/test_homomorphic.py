import pytest
import torch

from hack.homomorphic import homomorphic_matmul
from hack.quant import dequantize, pack, partition_sums, quantize, unpack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_roundtrip(bits):
    codes = torch.randint(0, 1 << bits, (3, 5, 128), dtype=torch.uint8, device=DEVICE)
    assert torch.equal(unpack(pack(codes, bits), bits), codes)


@pytest.mark.parametrize("partition_size", [16, 32, 64, 128])
def test_quantization_error_is_bounded(partition_size):
    x = torch.randn(4, 256, device=DEVICE)
    codes, scale, minimum = quantize(x, 2, -1, partition_size, stochastic=True)
    error = (dequantize(codes, scale, minimum, -1, partition_size) - x).abs()
    bound = scale.repeat_interleave(partition_size, dim=-1)
    assert torch.all(error <= bound + 1e-5)


@pytest.mark.parametrize("partition_size", [32, 64])
@pytest.mark.parametrize("cached_sums", [False, True])
def test_matches_dequantize_then_multiply(partition_size, cached_sums):
    torch.manual_seed(0)
    a = torch.randn(2, 8, 7, 128, device=DEVICE)
    b = torch.randn(2, 8, 128, 50, device=DEVICE)
    a_codes, a_scale, a_min = quantize(a, 8, -1, partition_size)
    b_codes, b_scale, b_min = quantize(b, 2, -2, partition_size)
    b_sums = partition_sums(b_codes, -2, partition_size, 2) if cached_sums else None
    got = homomorphic_matmul(a_codes, a_scale, a_min, b_codes, b_scale, b_min, partition_size, b_sums=b_sums)
    want = dequantize(a_codes, a_scale, a_min, -1, partition_size) @ dequantize(b_codes, b_scale, b_min, -2, partition_size)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-3)

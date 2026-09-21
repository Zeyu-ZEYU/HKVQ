import math

import pytest
import torch

from hack import HackConfig
from hack.attention_ref import hack_attention
from hack.cache import HackLayerCache
from hack.quant import dequantize, unpack

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def full_precision_attention(q, k, v):
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(q.shape[-1])
    q_len, total = q.shape[2], k.shape[2]
    mask = torch.arange(total, device=q.device)[None, :] > torch.arange(total - q_len, total, device=q.device)[:, None]
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1) @ v.float()


def dequantized_kv(cache: HackLayerCache):
    """K and V as floating-point tensors reconstructed from the cache."""
    cfg = cache.config
    keys, values = [], []
    if cache.sink_k is not None:
        keys.append(cache.sink_k.float())
        values.append(cache.sink_v.float())
    if cache.num_quantized_tokens:
        k_codes, v_codes = unpack(cache.k_codes.view(), cfg.kv_bits), unpack(cache.v_codes.view(), cfg.kv_bits)
        keys.append(dequantize(k_codes, cache.k_scale.view(), cache.k_min.view(), 2, cfg.partition_size))
        values.append(dequantize(v_codes, cache.v_scale.view(), cache.v_min.view(), -1, cache.channel_part))
    if cache.k_open is not None:
        keys.append(cache.k_open.float())
        values.append(cache.v_open.float())
    return torch.cat(keys, dim=2), torch.cat(values, dim=2)


def relative_error(got, want):
    return ((got.float() - want).norm() / want.norm()).item()


@pytest.mark.parametrize("length", [64, 200, 257])
@pytest.mark.parametrize("sinks", [0, 4])
def test_prefill_with_8bit_kv_is_close_to_full_precision(length, sinks):
    torch.manual_seed(0)
    q = torch.randn(1, 8, length, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 2, length, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, 2, length, 128, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(HackConfig(partition_size=32, kv_bits=8, sink_tokens=sinks))
    cache.append(k, v)
    assert relative_error(hack_attention(q, cache), full_precision_attention(q, k, v)) < 0.05


@pytest.mark.parametrize("bits", [2, 4])
def test_attention_on_codes_matches_attention_on_dequantized_kv(bits):
    torch.manual_seed(0)
    q = torch.randn(1, 8, 150, 128, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 150, 128, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 150, 128, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(HackConfig(partition_size=32, kv_bits=bits))
    cache.append(k, v)
    k_hat, v_hat = dequantized_kv(cache)
    assert relative_error(hack_attention(q, cache), full_precision_attention(q, k_hat, v_hat)) < 0.03


def test_decode_matches_prefill_state():
    torch.manual_seed(0)
    cfg = HackConfig(partition_size=32, kv_bits=8, sink_tokens=4)
    q = torch.randn(1, 4, 100, 64, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 4, 100, 64, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(1, 4, 100, 64, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(cfg)
    cache.append(k[:, :, :2], v[:, :, :2])
    cache.append(k[:, :, 2:90], v[:, :, 2:90])
    for t in range(90, 100):
        cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
        step = hack_attention(q[:, :, t : t + 1], cache)
        want = full_precision_attention(q[:, :, t : t + 1], k[:, :, : t + 1], v[:, :, : t + 1])
        assert relative_error(step, want) < 0.05
    assert (cache.num_tokens, cache.num_sink_tokens, cache.num_quantized_tokens, cache.num_open_tokens) == (100, 4, 96, 0)


def test_without_requant_elimination_the_open_block_is_lossy():
    torch.manual_seed(0)
    k = torch.randn(1, 2, 40, 64, device=DEVICE, dtype=torch.bfloat16)
    exact = HackLayerCache(HackConfig(partition_size=64, sink_tokens=0))
    lossy = HackLayerCache(HackConfig(partition_size=64, sink_tokens=0, requant_elimination=False))
    exact.append(k, k.clone())
    lossy.append(k, k.clone())
    assert torch.equal(exact.k_open, k)
    assert not torch.equal(lossy.k_open, k)


def test_2bit_cache_is_about_6x_smaller():
    k = torch.randn(1, 8, 1028, 128, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(HackConfig(partition_size=64))
    cache.append(k, k.clone())
    ratio = (2 * k.numel() * 2) / cache.nbytes()
    assert 5.0 < ratio < 8.0


def test_short_sequence_without_sinks_runs_in_16bit():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 10, 64, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 10, 64, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(HackConfig(partition_size=32, sink_tokens=0))
    cache.append(k, k.clone())
    assert relative_error(hack_attention(q, cache), full_precision_attention(q, k, k)) < 1e-2


def test_decode_without_requant_elimination():
    torch.manual_seed(0)
    cfg = HackConfig(partition_size=32, kv_bits=8, requant_elimination=False)
    k = torch.randn(1, 2, 50, 64, device=DEVICE, dtype=torch.bfloat16)
    q = torch.randn(1, 4, 50, 64, device=DEVICE, dtype=torch.bfloat16)
    cache = HackLayerCache(cfg)
    cache.append(k[:, :, :40], k[:, :, :40])
    for t in range(40, 50):
        cache.append(k[:, :, t : t + 1], k[:, :, t : t + 1])
        want = full_precision_attention(q[:, :, t : t + 1], k[:, :, : t + 1], k[:, :, : t + 1])
        assert relative_error(hack_attention(q[:, :, t : t + 1], cache), want) < 0.1

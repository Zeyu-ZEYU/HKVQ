import math

import pytest
import torch
import torch.nn.functional as F

from hack import HackConfig
from hack.attention_ref import hack_attention as reference_attention
from hack.cache import HackLayerCache
from hack.kernels import QuantizedKV, bf16_attention, dequant_attention, hack_attention
from hack.kernels.common import LOAD_ONLY, RESIDENT
from hack.kernels.decode import hack_decode
from hack.kernels.fa2 import bf16_source, fa2_decode_step
from hack.kernels.paged import paged_decode_attention
from hack.kernels import tuning
from hack.kernels.tuning import LaunchConfig
from hack.quant import dequantize

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the Triton kernels need a GPU")
DEVICE = "cuda"

# (query heads, kv heads, head_dim, partition size, sink tokens, cached tokens)
SHAPES = [
    (8, 8, 64, 32, 0, 96),
    (8, 8, 64, 32, 4, 100),
    (8, 8, 64, 64, 4, 200),
    (16, 2, 128, 64, 4, 300),
    (64, 8, 128, 64, 4, 1028),
    (8, 1, 128, 32, 0, 517),
    (8, 1, 128, 128, 4, 700),
    (4, 4, 64, 64, 4, 40),
    (8, 2, 128, 64, 4, 3),
]


def tensors(heads, kv_heads, head_dim, total, q_len, batch=2, dtype=torch.bfloat16):
    q = torch.randn(batch, heads, q_len, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(batch, kv_heads, total, head_dim, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(batch, kv_heads, total, head_dim, device=DEVICE, dtype=torch.bfloat16)
    return q, k, v


def filled_cache(k, v, **settings):
    cache = HackLayerCache(HackConfig(**settings))
    cache.append(k, v)
    return cache


def sdpa(q, k, v, causal=True):
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1).float(), v.repeat_interleave(groups, dim=1).float()
    q_len, total = q.shape[2], k.shape[2]
    mask = torch.ones(q_len, total, dtype=torch.bool, device=q.device).tril(total - q_len) if causal else None
    return F.scaled_dot_product_attention(q.float(), k, v, attn_mask=mask)


def attention_on_dequantized_cache(q, cache):
    """Exact attention over the values that the contents of `cache` stand for."""
    block, part = cache.config.partition_size, cache.channel_part
    k = dequantize(cache.k_unpacked(), cache.k_scale.view(), cache.k_min.view(), 2, block)
    v = dequantize(cache.v_unpacked(), cache.v_scale.view(), cache.v_min.view(), -1, part)
    k_parts = [t.float() for t in (cache.sink_k, k, cache.k_open) if t is not None]
    v_parts = [t.float() for t in (cache.sink_v, v, cache.v_open) if t is not None]
    return sdpa(q, torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2))


def relative_error(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def row_errors(got, want):
    return ((got.float() - want.float()).norm(dim=-1) / want.float().norm(dim=-1)).flatten()


def assert_matches_reference(got, want):
    """Round-to-nearest results agree up to float32 rounding; a rare row differs by one 8-bit code."""
    assert not torch.isnan(got).any()
    assert row_errors(got, want).median().item() < 1e-4
    assert relative_error(got, want) < 5e-3


@pytest.mark.parametrize("heads,kv_heads,head_dim,partition,sinks,total", SHAPES)
def test_decode_matches_reference(heads, kv_heads, head_dim, partition, sinks, total):
    torch.manual_seed(0)
    q, k, v = tensors(heads, kv_heads, head_dim, total, 1, dtype=torch.float32)
    cache = filled_cache(k, v, partition_size=partition, sink_tokens=sinks)
    assert_matches_reference(hack_attention(q, cache), reference_attention(q, cache))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads,kv_heads,head_dim,partition,sinks,total", SHAPES[:4] + SHAPES[5:])
def test_prefill_matches_reference(heads, kv_heads, head_dim, partition, sinks, total, causal):
    torch.manual_seed(0)
    q, k, v = tensors(heads, kv_heads, head_dim, total, total, dtype=torch.float32)
    cache = filled_cache(k, v, partition_size=partition, sink_tokens=sinks)
    assert_matches_reference(hack_attention(q, cache, causal=causal), reference_attention(q, cache, causal=causal))


@pytest.mark.parametrize("q_len", [1, 50])
def test_half_precision_queries_match_reference(q_len):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 333, q_len)
    cache = filled_cache(k, v)
    assert relative_error(hack_attention(q, cache), reference_attention(q, cache)) < 5e-3


def test_prefill_of_the_last_tokens_matches_reference():
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 333, 50, dtype=torch.float32)
    cache = filled_cache(k, v)
    assert_matches_reference(hack_attention(q, cache), reference_attention(q, cache))


def test_decode_after_prefill_matches_reference():
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 150, 150, batch=1, dtype=torch.float32)
    cache = HackLayerCache(HackConfig())
    cache.append(k[:, :, :120], v[:, :, :120])
    assert_matches_reference(hack_attention(q[:, :, :120], cache), reference_attention(q[:, :, :120], cache))
    for t in range(120, 150):
        cache.append(k[:, :, t : t + 1], v[:, :, t : t + 1])
        step = q[:, :, t : t + 1]
        assert_matches_reference(hack_attention(step, cache), reference_attention(step, cache))
    assert cache.num_sink_tokens == 4 and cache.num_quantized_tokens == 128 and cache.num_open_tokens == 18


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("q_len", [1, 77])
def test_wider_kv_codes_match_reference(bits, q_len):
    torch.manual_seed(0)
    q, k, v = tensors(8, 2, 128, 333, q_len, dtype=torch.float32)
    cache = filled_cache(k, v, partition_size=32, kv_bits=bits)
    assert_matches_reference(hack_attention(q, cache), reference_attention(q, cache))


@pytest.mark.parametrize("bits", [2, 8])
@pytest.mark.parametrize("q_len", [1, 60])
def test_recomputed_code_sums_give_the_same_result(bits, q_len):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 300, q_len, dtype=torch.float32)
    with_sums = filled_cache(k, v, kv_bits=bits)
    without_sums = HackLayerCache(HackConfig(kv_bits=bits, summation_elimination=False))
    without_sums.load_state_dict(with_sums.state_dict(), with_sums.num_tokens, 128)
    assert_matches_reference(hack_attention(q, without_sums), hack_attention(q, with_sums))
    assert_matches_reference(hack_attention(q, without_sums), reference_attention(q, without_sums))


def test_number_of_splits_does_not_change_the_result():
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 1000, 1, dtype=torch.float32)
    cache = filled_cache(k, v)
    scaling = 1.0 / math.sqrt(128)
    single = hack_decode(q, cache, scaling, seed=1, config=LaunchConfig(splits=1))
    for splits in (2, 8, 15):
        assert_matches_reference(hack_decode(q, cache, scaling, seed=1, config=LaunchConfig(splits=splits)), single)
    assert_matches_reference(single, reference_attention(q, cache))


def test_without_requantization_elimination_the_open_block_matches_reference():
    torch.manual_seed(0)
    q, k, v = tensors(8, 2, 128, 100, 1, dtype=torch.float32)
    cache = filled_cache(k, v, requant_elimination=False)
    assert_matches_reference(hack_attention(q, cache), reference_attention(q, cache))


@pytest.mark.parametrize("q_len", [1, 48])
def test_stochastic_rounding_agrees_with_reference_statistically(q_len):
    torch.manual_seed(0)
    runs = 24
    q, k, v = tensors(16, 2, 128, 300, q_len, batch=1)
    cache = filled_cache(k, v, stochastic=True)
    got = torch.stack([hack_attention(q, cache).float() for _ in range(runs)])
    want = torch.stack([reference_attention(q, cache).float() for _ in range(runs)])
    noise_got = (got - got.mean(0)).norm() / got.mean(0).norm() / math.sqrt(runs)
    noise_want = (want - want.mean(0)).norm() / want.mean(0).norm() / math.sqrt(runs)
    assert 0.7 < (noise_got / noise_want).item() < 1.4
    expected_gap = noise_want.item() * math.sqrt(2.0 / runs)
    assert relative_error(got.mean(0), want.mean(0)) < 3.0 * expected_gap + 1e-3


@pytest.mark.parametrize("total", [260, 300])
def test_stochastic_rounding_is_unbiased(total):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, total, 1, batch=1)
    cache = filled_cache(k, v, stochastic=True)
    exact = attention_on_dequantized_cache(q, cache)
    runs = torch.stack([hack_attention(q, cache).float() for _ in range(64)])
    single_error = sum(relative_error(run, exact) for run in runs) / len(runs)
    assert relative_error(runs.mean(0), exact) < 0.3 * single_error


def test_sizes_without_a_kernel_use_the_reference():
    torch.manual_seed(0)
    q, k, v = tensors(8, 2, 128, 200, 1, dtype=torch.float32)
    cache = filled_cache(k, v, partition_size=48)
    assert torch.equal(hack_attention(q, cache), reference_attention(q, cache))


@pytest.mark.parametrize("mode", [LOAD_ONLY, RESIDENT])
def test_benchmark_variants_run(mode):
    q, k, v = tensors(64, 8, 128, 1000, 1, batch=1)
    cache = filled_cache(k, v)
    assert hack_decode(q, cache, 0.1, seed=1, mode=mode).shape == q.shape
    assert fa2_decode_step(q, bf16_source(k, v), 0.1, mode=mode).shape == q.shape


def paged_sequences(config, lengths, heads=16, kv_heads=2, head_dim=128):
    """A page pool that holds sequences of the given lengths, the matching decode batch and one query per row."""
    plan = pytest.importorskip("hack.vllm_plugin.plan")
    layout_module = pytest.importorskip("hack.vllm_plugin.layout")
    store_module = pytest.importorskip("hack.vllm_plugin.store")
    block = config.partition_size
    layout = layout_module.PageLayout(config, kv_heads, head_dim)
    width = max(-(-length // block) for length in lengths)
    pool_shape = (len(lengths) * width + 1, 1, block, layout.bytes_per_token_slot)
    pool = torch.zeros(pool_shape, dtype=torch.uint8, device=DEVICE)
    store = store_module.LayerStore(layout, len(lengths), torch.bfloat16, torch.device(DEVICE))
    store.bind(pool)
    pages = torch.randperm(len(lengths) * width, device=DEVICE) + 1
    table = pages.reshape(len(lengths), width).to(torch.int32)
    slots = torch.arange(1, len(lengths) + 1, device=DEVICE)
    for row, length in enumerate(lengths):
        k = torch.randn(1, kv_heads, length, head_dim, device=DEVICE, dtype=torch.bfloat16)
        cache = filled_cache(k, torch.randn_like(k), **vars(config))
        prefill = plan.PrefillRow(0, length, 0, length, table[row].long(), slots[row], slots[row], slots[row])
        store.scatter(cache, prefill)
    closed = [max(length - config.sink_tokens, 0) // block for length in lengths]
    nothing = torch.empty(0, dtype=torch.long, device=DEVICE)
    totals = torch.tensor(lengths, device=DEVICE)
    batch = plan.DecodeBatch(
        len(lengths), max(lengths), max(closed), nothing, totals, totals.cpu().numpy(), table, slots, slots,
        nothing, nothing, nothing, nothing, nothing, nothing,
    )  # fmt: skip
    q = torch.randn(len(lengths), heads, head_dim, device=DEVICE, dtype=torch.float32)
    return store, batch, q


PAGED_SETTINGS = [
    dict(),
    dict(sink_tokens=0),
    dict(partition_size=32, summation_elimination=False),
    dict(partition_size=128, kv_bits=4),
]


@pytest.mark.parametrize("settings", PAGED_SETTINGS)
def test_paged_decode_matches_gather_then_reference(settings):
    torch.manual_seed(0)
    config = HackConfig(**settings)
    lengths = [3, 70, 517, 1028, 260, 131]
    store, batch, q = paged_sequences(config, lengths)
    got = paged_decode_attention(q, store, batch, 1.0 / math.sqrt(q.shape[-1]))
    for row, length in enumerate(lengths):
        cache = store.gather(batch.block_table[row].long(), length, batch.open_slot[row], batch.sink_slot[row])
        assert_matches_reference(got[row : row + 1, :, None], reference_attention(q[row : row + 1, :, None], cache))


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads,kv_heads,head_dim,total,q_len", [
    (8, 8, 64, 96, 96), (16, 2, 128, 300, 300), (64, 8, 128, 1000, 1), (8, 1, 128, 257, 40), (64, 8, 128, 5000, 1),
])  # fmt: skip
def test_bf16_attention_matches_sdpa(heads, kv_heads, head_dim, total, q_len, causal):
    torch.manual_seed(0)
    q, k, v = tensors(heads, kv_heads, head_dim, total, q_len)
    assert relative_error(bf16_attention(q, k, v, causal=causal), sdpa(q, k, v, causal)) < 1e-2


def test_bf16_attention_honours_the_scaling():
    torch.manual_seed(0)
    q, k, v = tensors(8, 2, 64, 100, 100)
    assert relative_error(bf16_attention(q, k, v, scaling=0.05), sdpa(q * (0.05 * 8.0), k, v)) < 1e-2


@pytest.mark.parametrize("bits,group_size", [(2, 32), (2, 64), (4, 64), (4, 128)])
@pytest.mark.parametrize("total,q_len", [(96, 96), (300, 1), (1000, 1), (257, 40)])
def test_dequant_attention_matches_dequantize_then_sdpa(bits, group_size, total, q_len):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, total, q_len)
    qkv = QuantizedKV.from_tensors(k, v, bits=bits, group_size=group_size)
    k_hat, v_hat = qkv.dequantize()
    assert relative_error(dequant_attention(q, qkv), sdpa(q, k_hat, v_hat)) < 1e-2
    assert 2 * k.numel() * k.element_size() / qkv.nbytes() > 8 / (bits + 1)


def levels(bits):
    """A table of 2 ** bits increasing, unevenly spaced levels."""
    even = torch.linspace(-1.0, 1.0, 1 << bits, device=DEVICE)
    return even.sign() * even.abs().pow(1.5)


STORAGE_LAYOUTS = [
    dict(k_axis="token", v_axis="channel"),
    dict(k_axis="token", v_axis="token", group_size=32),
    dict(k_axis="channel", v_axis="token", group_size=128),
    dict(k_lut=levels(2), v_lut=levels(2)),
    dict(bits=4, k_lut=levels(4), v_lut=levels(4), k_axis="token"),
    dict(bits=4, k_axis="token", v_lut=levels(4)),
]


@pytest.mark.parametrize("layout", STORAGE_LAYOUTS)
@pytest.mark.parametrize("total,q_len", [(300, 1), (1000, 1), (257, 40), (64, 64)])
def test_dequant_attention_with_token_groups_and_lookup_tables(layout, total, q_len):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, total, q_len)
    qkv = QuantizedKV.from_tensors(k, v, **layout)
    k_hat, v_hat = qkv.dequantize()
    assert relative_error(dequant_attention(q, qkv), sdpa(q, k_hat, v_hat)) < 1e-2


@pytest.mark.parametrize("layout", STORAGE_LAYOUTS)
def test_storage_only_codes_reconstruct_their_input(layout):
    torch.manual_seed(0)
    _, k, v = tensors(4, 2, 128, 300, 1)
    qkv = QuantizedKV.from_tensors(k, v, **layout)
    k_hat, v_hat = qkv.dequantize(torch.float32)
    bound = 0.6 if qkv.bits == 2 else 0.2
    assert relative_error(k_hat, k) < bound and relative_error(v_hat, v) < bound
    narrowed = qkv.narrow(130).dequantize(torch.float32)
    assert torch.equal(narrowed[0], k_hat[:, :, :130]) and torch.equal(narrowed[1], v_hat[:, :, :130])


@pytest.mark.parametrize("q_len", [1, 40])
def test_skipping_the_dequantization_runs_the_same_kernel(q_len):
    torch.manual_seed(0)
    q, k, v = tensors(16, 2, 128, 300, q_len)
    out = dequant_attention(q, QuantizedKV.from_tensors(k, v), skip_dequant=True)
    assert out.shape == q.shape and torch.isfinite(out).all()


def test_saved_launch_configurations_are_loaded_under_the_same_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "_registry", {})
    key = ("fa2_decode", "device", 0, 64, 128, (0, 2, False, False, False), 11)
    tuning.record(key, LaunchConfig(splits=16, num_warps=2))
    tuning.save(str(tmp_path / "tuning.json"))
    monkeypatch.setattr(tuning, "_registry", {})
    tuning.load(str(tmp_path / "tuning.json"))
    assert tuning.lookup(key, None) == LaunchConfig(splits=16, num_warps=2)

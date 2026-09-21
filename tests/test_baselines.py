"""Tests of the comparison methods (CacheGen-style codec, KVQuant-style quantizer and their caches).

The tests that need a language model use a small one; set HACK_TEST_MODEL to change it.
"""

import json
import math
import os
from dataclasses import replace

import pytest
import torch

from hack.baselines import (
    CacheGenCodec, CacheGenConfig, CacheGenPayload, KVQuantConfig, KVQuantQuantizer, calibration, make_cache, rans,
)  # fmt: skip
from hack.baselines.hf_cache import CacheGenLayer, KVQuantLayer
from hack.baselines.kvquant import fit_levels, outliers_per_side

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("HACK_TEST_MODEL", "Qwen/Qwen3-0.6B")
TARGET_COMPRESSION = 0.86
NEAR_TIE_NATS = 0.5

LIST_PROMPT = (
    "The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday and Sunday. "
    "The months of the year are January, February, March,"
)
PROMPT = (
    "The Amazon River in South America is the largest river in the world by discharge volume of water, and it is "
    "disputed whether it or the Nile is the longest. Its headwaters rise in the Andes of Peru, and the river flows "
    "east across Brazil for thousands of kilometres before it reaches the Atlantic Ocean. The rainforest that "
    "surrounds the river is home to"
)
CALIBRATION_TEXT = (
    "Rivers shape the land they cross. Rain that falls on mountains gathers in streams, the streams join, and the "
    "water carries sand and stones down to the plains, where it slows and drops them. Over thousands of years a "
    "river can cut a canyon through rock or build a delta that reaches far into the sea. People have always "
    "settled near rivers, because they give drinking water, fish, fertile soil and a road for boats. Many of the "
    "oldest cities in the world stand on the banks of a river, and the floods that feed the fields can also "
    "destroy the houses next to them. Engineers build dams and levees to hold the water back, canals to move it "
    "to dry land, and bridges to cross it. A dam stores water for the dry season and can drive turbines that "
    "generate electricity, but it also blocks the fish that swim upstream and holds back the sediment that the "
    "delta needs. "
)


def synthetic_kv(layers=3, batch=1, heads=2, tokens=75, head_dim=32, seed=0):
    """K with per-channel offsets and spreads, V without structure, as in the KV cache of a transformer."""
    generator = torch.Generator().manual_seed(seed)
    kv = []
    for layer in range(layers):
        channels = torch.Generator().manual_seed(1000 + layer)
        offset = 3.0 * torch.randn(1, heads, 1, head_dim, generator=channels)
        spread = torch.rand(1, heads, 1, head_dim, generator=channels) * 2.0 + 0.1
        k = offset + spread * torch.randn(batch, heads, tokens, head_dim, generator=generator)
        v = torch.randn(batch, heads, tokens, head_dim, generator=generator)
        kv.append((k.to(DEVICE, torch.bfloat16), v.to(DEVICE, torch.bfloat16)))
    return kv


def relative_error(got, want):
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def worst_error(dequantized_layers, kv):
    """Largest relative error of the dequantized K and V of all layers."""
    pairs = [pair for layer in zip(dequantized_layers, kv) for pair in zip(*layer)]
    return max(relative_error(got, want) for got, want in pairs)


def assert_same_codes(got, want):
    for name in ("codes", "scale", "anchor_codes", "anchor_scale"):
        assert torch.equal(getattr(got, name), getattr(want, name)), name
    assert (got.levels, got.anchor_levels, got.group_size) == (want.levels, want.anchor_levels, want.group_size)


def bf16_nbytes(kv):
    return sum(2 * (k.numel() + v.numel()) for k, v in kv)


# Entropy coder


def random_streams(alphabets, lengths, period, seed=0):
    """Symbols, stream shapes and tables: one group of `period` tables per alphabet, streams of several lengths."""
    generator = torch.Generator().manual_seed(seed)
    symbols, shapes, groups = [], [], []
    for index, alphabet in enumerate(alphabets):
        probabilities = torch.rand(period, alphabet, generator=generator) ** 4 + 1e-3
        groups.append(rans.normalize_counts(probabilities * 1000))
        for length in lengths:
            steps = torch.arange(max(lengths)) % period
            rows = torch.multinomial(probabilities[steps], 8, replacement=True, generator=generator).T
            symbols.append(rows * (torch.arange(max(lengths)) < length))
            fill = torch.ones(8, dtype=torch.int64)
            shapes.append(rans.Streams(fill * length, fill * index * period, fill * period, fill))
    tables = rans.Tables.from_frequencies([group.to(DEVICE) for group in groups])
    return torch.cat(symbols).to(DEVICE), rans.Streams.cat(shapes).to(DEVICE), tables


STREAM_SHAPES = [((7,), (120,), 1), ((5, 15, 255), (33, 1, 20), 4), ((4,), (1,), 1)]


@pytest.mark.parametrize("alphabets,lengths,period", STREAM_SHAPES)
def test_entropy_coder_round_trip_is_exact(alphabets, lengths, period):
    symbols, streams, tables = random_streams(alphabets, lengths, period)
    words, states = rans.encode(symbols, streams, tables)
    assert words.dtype == torch.int16 and states.dtype == torch.int32
    assert torch.equal(rans.decode(words, states, streams, tables).long(), symbols)


def test_entropy_coder_reaches_the_entropy_of_the_source():
    generator = torch.Generator().manual_seed(0)
    probabilities = torch.tensor([[0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]])
    symbols = torch.multinomial(probabilities[0], 64 * 2000, replacement=True, generator=generator).reshape(64, 2000)
    tables = rans.Tables.from_frequencies([rans.normalize_counts(probabilities)])
    fill = torch.ones(64, dtype=torch.int64)
    words, states = rans.encode(symbols, rans.Streams(fill * 2000, fill * 0, fill, fill * 0), tables)
    entropy = -(probabilities * probabilities.log2()).sum().item()
    coded = (16 * words.numel() + 32 * states.numel()) / symbols.numel()
    assert entropy < coded < 1.03 * entropy


def test_normalized_frequencies_are_positive_and_sum_to_the_coder_precision():
    counts = torch.tensor([[0.0, 0.0, 5.0, 100000.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    freq = rans.normalize_counts(counts)
    assert (freq >= 1).all() and (freq.sum(dim=1) == 1 << rans.PRECISION).all()


# CacheGen-style codec


@pytest.mark.parametrize("tokens,batch", [(1, 1), (9, 1), (10, 2), (75, 1), (230, 2)])
def test_codec_round_trip_is_exact(tokens, batch):
    kv = synthetic_kv(tokens=tokens, batch=batch)
    codec = CacheGenCodec(CacheGenConfig(chunk_size=100), len(kv))
    payload = codec.encode(kv)
    decoded = codec.decode(payload, DEVICE)
    for layer, (k, v) in enumerate(kv):
        for got, want in zip(decoded[layer], codec.quantize(layer, k, v)):
            assert_same_codes(got, want)
            assert got.num_tokens == tokens


def test_codec_round_trip_of_a_cache_that_emits_no_words():
    kv = synthetic_kv(layers=1, heads=1, tokens=1, head_dim=4)
    codec = CacheGenCodec(CacheGenConfig(), len(kv))
    payload = codec.encode(kv)
    assert dict((name, shape) for name, _, shape in payload.meta["sections"])["words"] == [0]
    for got, want in zip(codec.decode(payload, DEVICE)[0], codec.quantize(0, *kv[0])):
        assert_same_codes(got, want)


def test_codec_with_deltas_for_keys_and_values_round_trips():
    kv = synthetic_kv(tokens=57)
    codec = CacheGenCodec(CacheGenConfig(value_delta=True, group_size=5, chunk_size=20), len(kv))
    decoded = codec.decode(codec.encode(kv), DEVICE)
    for layer, (k, v) in enumerate(kv):
        for got, want in zip(decoded[layer], codec.quantize(layer, k, v)):
            assert_same_codes(got, want)
            assert got.anchor_codes.shape[2] == 12


def test_payload_is_a_byte_string_with_a_json_description():
    kv = synthetic_kv(tokens=40)
    codec = CacheGenCodec(CacheGenConfig(), len(kv))
    payload = codec.encode(kv)
    assert isinstance(payload.data, bytes)
    copy = CacheGenPayload(bytes(payload.data), json.loads(json.dumps(payload.meta)))
    assert copy.nbytes() == payload.nbytes() > len(payload.data)
    for got, want in zip(codec.decode(copy, DEVICE), codec.decode(payload, DEVICE)):
        assert_same_codes(got[0], want[0])
        assert_same_codes(got[1], want[1])


def test_decoded_codes_are_integers_with_their_parameters():
    kv = synthetic_kv(tokens=40)
    codec = CacheGenCodec(CacheGenConfig(), len(kv))
    keys, values = codec.decode(codec.encode(kv), DEVICE)[1]
    assert keys.codes.dtype == values.codes.dtype == keys.anchor_codes.dtype == torch.uint8
    assert keys.codes.shape == (1, 2, 36, 16) and keys.scale.shape == (1, 2, 36, 1)
    assert keys.anchor_codes.shape == (1, 2, 4, 32) and keys.anchor_scale.shape == (1, 2, 4, 1)
    assert values.codes.shape == (1, 2, 40, 16) and values.anchor_codes.shape[2] == 0
    k, v = codec.dequantize(codec.decode(codec.encode(kv), DEVICE))[1]
    assert k.dtype == v.dtype == torch.bfloat16 and k.shape == v.shape == kv[1][0].shape
    assert torch.equal(k, keys.dequantize()) and relative_error(v, kv[1][1]) < 0.6


def test_quantization_error_is_bounded_and_shrinks_with_the_number_of_levels():
    kv = synthetic_kv(tokens=200)
    errors = []
    for levels in (7, 31, 255):
        config = CacheGenConfig(key_levels=(levels,), value_levels=(levels,), first_layer_levels=(levels, levels))
        codec = CacheGenCodec(config, len(kv))
        quantized = [codec.quantize(layer, k, v) for layer, (k, v) in enumerate(kv)]
        errors.append(worst_error([(keys.dequantize(), values.dequantize()) for keys, values in quantized], kv))
    assert errors[0] < 0.5 and errors[1] < 0.1 and errors[2] < 0.015
    assert errors[0] > errors[1] > errors[2]


def test_every_element_is_within_half_a_quantization_step():
    k, v = synthetic_kv(layers=1, tokens=64)[0]
    keys, values = CacheGenCodec(CacheGenConfig(), 2).quantize(1, k, v)
    step = values.scale.float()
    assert ((values.dequantize(torch.float32) - v.float()).abs() <= 0.5 * step * 1.01 + 1e-6).all()
    positions = torch.arange(64, device=k.device)
    others = positions % keys.group_size != 0
    bound = 0.5 * keys.scale.float() * 1.01 + 1e-6
    assert ((keys.dequantize(torch.float32) - k.float())[:, :, others].abs() <= bound).all()


def test_earlier_layers_get_more_levels():
    config = CacheGenConfig()
    for is_value, schedule in enumerate((config.key_levels, config.value_levels)):
        levels = [config.levels(layer, 31, bool(is_value)) for layer in range(31)]
        assert levels[0] == config.first_layer_levels[is_value] and levels[1:11] == [schedule[0]] * 10
        assert levels[-1] == schedule[-1] and levels == sorted(levels, reverse=True)


def test_tokens_appended_one_by_one_get_the_codes_of_the_whole_sequence():
    k, v = synthetic_kv(layers=1, tokens=47)[0]
    config = CacheGenConfig()
    layer = CacheGenLayer(config, 1, 4, exact_prefill=False)
    start = 0
    for size in (13, 1, 1, 7, 20, 1, 4):
        layer.update(k[:, :, start : start + size], v[:, :, start : start + size])
        start += size
    for got, want in zip(layer.tensors(), CacheGenCodec(config, 4).quantize(1, k, v)):
        assert_same_codes(got, want)


def test_profiled_channel_statistics_shorten_the_payload():
    config = CacheGenConfig()
    profile = calibration.fit_cachegen(synthetic_kv(tokens=400, batch=4, seed=1), config)
    kv = synthetic_kv(tokens=300, seed=2)
    plain, profiled = CacheGenCodec(config, len(kv)), CacheGenCodec(config, len(kv), profile)
    payload = profiled.encode(kv)
    assert payload.nbytes() < plain.encode(kv).nbytes()
    for got, want in zip(profiled.decode(payload, DEVICE), plain.decode(plain.encode(kv), DEVICE)):
        assert_same_codes(got[0], want[0])


def test_rate_control_picks_the_finest_levels_that_meet_the_target():
    kv = synthetic_kv(tokens=400, batch=2)
    config = CacheGenConfig(first_layer_levels=(15, 15))
    fitted, held_out = [(k[:1], v[:1]) for k, v in kv], [(k[1:], v[1:]) for k, v in kv]
    rates = []
    for key_levels, value_levels in calibration.LEVEL_LADDER:
        candidate = replace(config, key_levels=key_levels, value_levels=value_levels)
        rates.append(calibration.coded_bits(held_out, calibration.profile_cachegen(fitted, candidate)))
    assert rates == sorted(rates, reverse=True)
    chosen = calibration.fit_cachegen(kv, config, target_bits=(rates[4] + rates[5]) / 2).config
    assert (chosen.key_levels, chosen.value_levels) == calibration.LEVEL_LADDER[5]
    unreachable = calibration.fit_cachegen(kv, config, target_bits=0.1).config
    assert (unreachable.key_levels, unreachable.value_levels) == calibration.LEVEL_LADDER[-1]


def test_estimated_rate_matches_the_size_of_the_payload():
    config = CacheGenConfig()
    profile = calibration.fit_cachegen(synthetic_kv(tokens=400, batch=4, seed=1), config)
    kv = synthetic_kv(tokens=300, seed=2)
    payload = CacheGenCodec(config, len(kv), profile).encode(kv)
    estimate = calibration.coded_bits(kv, profile)
    assert estimate < 8 * payload.nbytes() / (bf16_nbytes(kv) / 2) < 1.05 * estimate


# KVQuant-style quantizer


def fitted_quantizer(kv, **settings):
    return KVQuantQuantizer(calibration.fit_kvquant(kv, KVQuantConfig(sink_tokens=0, **settings), DEVICE))


def test_kvquant_error_is_bounded_and_shrinks_with_the_code_width():
    kv = synthetic_kv(tokens=300, head_dim=64)
    errors = []
    for bits in (2, 4, 8):
        quantizer = fitted_quantizer(kv, bits=bits)
        errors.append(
            worst_error([quantizer.quantize(layer, k, v).dequantize() for layer, (k, v) in enumerate(kv)], kv)
        )
    assert errors[0] < 0.45 and errors[1] < 0.12 and errors[2] < 0.012
    assert errors[0] > errors[1] > errors[2]


def test_kvquant_keeps_outliers_in_16_bit():
    k, v = synthetic_kv(layers=1, tokens=100, head_dim=64)[0]
    quantizer = fitted_quantizer([(k, v)])
    k, v = k.clone(), v.clone()
    k[0, 1, 17, 5], v[0, 0, 40, 9], v[0, 1, 41, 63] = 300.0, -250.0, 99.0
    codes = quantizer.quantize(0, k, v)
    k_hat, v_hat = codes.dequantize()
    assert k_hat[0, 1, 17, 5] == k[0, 1, 17, 5]
    assert v_hat[0, 0, 40, 9] == v[0, 0, 40, 9] and v_hat[0, 1, 41, 63] == v[0, 1, 41, 63]
    values = codes.values
    per_token = 2 * outliers_per_side(2 * 64, quantizer.config.outlier_fraction)
    assert per_token > 0 and (values.outliers.counts == per_token).all()
    assert values.outliers.values.dtype == torch.bfloat16


def test_kvquant_keys_are_quantized_per_channel_and_values_per_token():
    kv = synthetic_kv(layers=1, tokens=50, batch=2)
    codes = fitted_quantizer(kv).quantize(0, *kv[0])
    keys, values = codes.keys, codes.values
    assert keys.scale.shape == (1, 2, 1, 32) and values.scale.shape == (2, 1, 50, 1)
    assert keys.codes.shape == values.codes.shape == (2, 2, 50, 8) and keys.levels.numel() == 4
    assert (keys.levels.diff() > 0).all() and keys.levels.abs().max() <= 1


def test_kvquant_levels_follow_the_distribution_of_the_calibration_data():
    samples = torch.randn(1 << 20, generator=torch.Generator().manual_seed(0)).to(DEVICE)
    levels = fit_levels(samples.clamp(-4, 4) / 4, bits=2) * 4
    lloyd_max = torch.tensor([-1.510, -0.4528, 0.4528, 1.510], device=DEVICE)
    assert (levels - lloyd_max).abs().max() < 0.02
    assert torch.equal(levels, fit_levels(samples.clamp(-4, 4) / 4, bits=2) * 4)


def test_kvquant_storage_is_the_codes_the_parameters_and_the_sparse_outliers():
    kv = synthetic_kv(layers=1, tokens=256, head_dim=64)
    codes = fitted_quantizer(kv).quantize(0, *kv[0])
    keys, values = codes.keys, codes.values
    elements = kv[0][0].numel()
    assert keys.nbytes() == elements // 4 + keys.outliers.nbytes()
    assert values.nbytes() == elements // 4 + 2 * 2 * 256 + values.outliers.nbytes()
    assert codes.nbytes() == keys.nbytes() + values.nbytes()
    assert 2.0 < 8 * codes.nbytes() / (2 * elements) < 2.8


def test_kvquant_tokens_appended_in_pieces_get_the_codes_of_the_whole_sequence():
    kv = synthetic_kv(layers=1, tokens=60, batch=2)
    k, v = kv[0]
    fitted = calibration.fit_kvquant(kv, KVQuantConfig(sink_tokens=2), DEVICE)
    layer = KVQuantLayer(fitted.config, fitted.layers[0], exact_prefill=False)
    start = 0
    for size in (1, 1, 17, 1, 40):
        layer.update(k[:, :, start : start + size], v[:, :, start : start + size])
        start += size
    whole = KVQuantQuantizer(fitted).quantize(0, k, v)
    for got, want in zip(layer.codes().dequantize(), whole.dequantize()):
        assert torch.equal(got, want)
    assert layer.codes().nbytes() == whole.nbytes()
    assert layer.codes(10).nbytes() == KVQuantQuantizer(fitted).quantize(0, k[:, :, :10], v[:, :, :10]).nbytes()


def test_kvquant_keeps_the_leading_tokens_in_16_bit():
    kv = synthetic_kv(layers=1, tokens=40)
    k, v = kv[0]
    codes = KVQuantQuantizer(calibration.fit_kvquant(kv, KVQuantConfig(sink_tokens=3), DEVICE)).quantize(0, k, v)
    k_hat, v_hat = codes.dequantize()
    assert codes.sink_k.shape[2] == 3 and codes.keys.codes.shape[2] == 37 and k_hat.shape == k.shape
    assert torch.equal(k_hat[:, :, :3], k[:, :, :3]) and torch.equal(v_hat[:, :, :3], v[:, :, :3])
    assert not torch.equal(k_hat[:, :, 3:], k[:, :, 3:])


@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_calibration_files_round_trip(tmp_path, method):
    kv = synthetic_kv(tokens=120, batch=2)
    if method == "kvquant":
        fitted = calibration.fit_kvquant(kv, KVQuantConfig(), DEVICE)
    else:
        fitted = calibration.fit_cachegen(kv, CacheGenConfig(), target_bits=2.5)
    path = str(tmp_path / "calibration.pt")
    calibration.save(fitted, path)
    loaded = calibration.load(path)
    assert loaded.config == fitted.config
    if method == "kvquant":
        for got, want in zip(loaded.layers, fitted.layers):
            assert all(torch.equal(a, b) for a, b in zip(got.tensors(), want.tensors()))
    else:
        assert all(torch.equal(loaded.frequencies[key], freq) for key, freq in fitted.frequencies.items())
    assert os.path.getsize(path) < 0.5 * bf16_nbytes(kv)


# Caches on a language model


@pytest.fixture(scope="module")
def lm():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    except OSError as error:
        pytest.skip(f"{MODEL} is not available: {error}")
    return tokenizer, model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def calibrations(lm):
    """Calibration of both methods on a text that differs from the prompt."""
    tokenizer, model = lm
    ids = tokenizer(CALIBRATION_TEXT * 4, return_tensors="pt").input_ids[0]
    kv = calibration.collect_kv(model, ids[: ids.numel() // 2 * 2].reshape(2, -1))
    target_bits = 16 * (1 - TARGET_COMPRESSION)
    return {
        "cachegen": {"profile": calibration.fit_cachegen(kv, CacheGenConfig(), target_bits, DEVICE)},
        "kvquant": {"calibration": calibration.fit_kvquant(kv, KVQuantConfig(), DEVICE)},
    }


def generate(lm, cache=None, new_tokens=24, prompt=PROMPT):
    tokenizer, model = lm
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False, past_key_values=cache)
    return out[0, inputs.input_ids.shape[1] :]


def reference_logprobs(lm, prompt, continuation):
    """Log-probabilities of the unmodified model at every position of its own `continuation`."""
    tokenizer, model = lm
    ids = torch.cat([tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE), continuation[None]], dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0, -continuation.numel() - 1 : -1]
    return logits.float().log_softmax(dim=-1)


def agreement(reference, logprobs, tokens):
    """Number of leading tokens that agree, and whether the first disagreement is a near tie of the reference."""
    differing = (reference != tokens).nonzero()
    if differing.numel() == 0:
        return reference.numel(), True
    first = differing[0].item()
    return first, (logprobs[first, reference[first]] - logprobs[first, tokens[first]]).item() <= NEAR_TIE_NATS


def mean_nll(lm, continuation):
    """Negative log-likelihood per token of `continuation` after the prompt under the unmodified model."""
    tokenizer, model = lm
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    ids = torch.cat([prompt, continuation[None]], dim=1)
    with torch.no_grad():
        logits = model(ids).logits[0, prompt.shape[1] - 1 : -1].float()
    return torch.nn.functional.cross_entropy(logits, continuation).item()


def cache_elements(model, tokens):
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * tokens


EIGHT_BIT = {
    "cachegen": {"config": CacheGenConfig(key_levels=(255,), value_levels=(255,))},
    "kvquant": {"config": KVQuantConfig(bits=8)},
}


@pytest.mark.parametrize("exact_prefill", [False, True])
@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_8_bit_caches_generate_the_text_of_bf16(lm, method, exact_prefill):
    for prompt, certain_tokens in ((LIST_PROMPT, 16), (PROMPT, 0)):
        reference = generate(lm, prompt=prompt)
        cache = make_cache(method, lm[1].config, exact_prefill=exact_prefill, **EIGHT_BIT[method])
        logprobs = reference_logprobs(lm, prompt, reference)
        agreeing, near_tie = agreement(reference, logprobs, generate(lm, cache, prompt=prompt))
        assert near_tie and agreeing >= certain_tokens


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_default_settings_generate_sensible_text(lm, calibrations, method, calibrated):
    cache = make_cache(method, lm[1].config, **(calibrations[method] if calibrated else {}))
    text = generate(lm, cache, new_tokens=32)
    assert mean_nll(lm, text) < mean_nll(lm, generate(lm, new_tokens=32)) + 1.0
    assert len(set(text.tolist())) > 8


@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_compression_rate_is_near_the_target(lm, calibrations, method):
    tokenizer, model = lm
    ids = tokenizer(" ".join([PROMPT] * 6 + [LIST_PROMPT]), return_tensors="pt").input_ids.to(DEVICE)
    cache = make_cache(method, model.config, **calibrations[method])
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    bf16 = 2 * cache_elements(model, ids.shape[1])
    assert abs(1 - cache.transfer_nbytes() / bf16 - TARGET_COMPRESSION) < 0.05
    assert cache.transfer_nbytes() <= cache.nbytes() < 0.35 * bf16


def test_transfer_size_is_the_size_of_a_decodable_payload(lm):
    tokenizer, model = lm
    cache = make_cache("cachegen", model.config)
    generate(lm, cache, new_tokens=12)
    payload = cache.encode_prompt()
    assert payload.nbytes() == cache.transfer_nbytes()
    prompt_tokens = tokenizer(PROMPT, return_tensors="pt").input_ids.shape[1]
    assert cache.get_seq_length() == prompt_tokens + 11
    for layer, decoded in zip(cache.layers, cache.codec.decode(payload, DEVICE)):
        for got, stored in zip(decoded, layer.tensors()):
            assert_same_codes(got, stored.prefix(prompt_tokens))


@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_the_first_update_can_return_the_exact_prompt(lm, method):
    tokenizer, model = lm
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        want = model(ids).logits
        exact = model(ids, past_key_values=make_cache(method, model.config, exact_prefill=True), use_cache=True).logits
        lossy = model(ids, past_key_values=make_cache(method, model.config), use_cache=True).logits
    assert torch.equal(exact, want) and not torch.equal(lossy, want)


@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_a_reset_cache_serves_a_new_request(lm, method):
    cache = make_cache(method, lm[1].config)
    first = generate(lm, cache, new_tokens=8)
    cache.reset()
    assert cache.get_seq_length() == 0
    assert torch.equal(generate(lm, cache, new_tokens=8), first)


def test_kvquant_cache_keeps_the_first_token_in_16_bit(lm):
    tokenizer, model = lm
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    reference = model(ids, use_cache=True).past_key_values
    cache = make_cache("kvquant", model.config)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    k, v = cache.layers[3].dequantize()
    assert torch.equal(k[:, :, :1], cache.layers[3].sinks[0]) and k.shape == reference.layers[3].keys.shape
    assert relative_error(v[:, :, 1:], reference.layers[3].values[:, :, 1:]) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused kernel needs a GPU")
@pytest.mark.parametrize("method", ["cachegen", "kvquant"])
def test_caches_convert_to_the_container_of_the_fused_kernel(lm, method):
    kernels = pytest.importorskip("hack.kernels")
    tokenizer, model = lm
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(DEVICE)
    cache = make_cache(method, model.config)
    with torch.no_grad():
        model(ids, past_key_values=cache, use_cache=True)
    layer = cache.layers[5]
    qkv = cache.to_quantized_kv(5)
    k, v = qkv.dequantize()
    assert qkv.num_tokens == ids.shape[1] and qkv.bits == (4 if method == "cachegen" else 2)
    if method == "cachegen":
        assert relative_error(v, layer.dequantize()[1]) < 1e-3
        assert relative_error(k, layer.dequantize()[0]) < 0.1
    else:
        sinks = layer.sinks[0].shape[2]
        exact_k, exact_v = qkv.dequantize(torch.float32)
        assert relative_error(exact_k[:, :, sinks:], layer.codes().keys.dense()) < 1e-5
        assert relative_error(exact_v[:, :, sinks:], layer.codes().values.dense()) < 1e-5
        assert qkv.k_lut.numel() == qkv.v_lut.numel() == 4
    q = torch.randn(1, model.config.num_attention_heads, 1, k.shape[-1], device=DEVICE, dtype=torch.bfloat16)
    groups = q.shape[1] // k.shape[1]
    scores = q.float() @ k.float().repeat_interleave(groups, dim=1).transpose(-2, -1) / math.sqrt(k.shape[-1])
    want = torch.softmax(scores, dim=-1) @ v.float().repeat_interleave(groups, dim=1)
    assert relative_error(kernels.dequant_attention(q, qkv), want) < 2e-2

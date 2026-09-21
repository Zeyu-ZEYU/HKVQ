"""Offline calibration of the comparison methods on public text.

    python -m hack.baselines.calibration --model Qwen/Qwen3-0.6B --method kvquant --output kvquant.pt
    python -m hack.baselines.calibration --model Qwen/Qwen3-0.6B --method cachegen --output cachegen.pt

The KVQuant-style calibration holds the outlier thresholds of the key channels and the levels of
the non-uniform datatype of every layer. The CacheGen-style profile holds the codec settings that
meet a target rate and the symbol statistics of every layer and channel.
"""

import argparse
from dataclasses import asdict, replace

import torch

from hack.baselines import rans
from hack.baselines.cachegen import CacheGenCodec, CacheGenConfig, CacheGenProfile, CacheGenTensor
from hack.baselines.kvquant import KVQuantCalibration, KVQuantConfig, LayerCalibration, fit_layer

KV = list[tuple[torch.Tensor, torch.Tensor]]
BF16_BITS = 16
MAX_CHARACTERS_PER_TOKEN = 16

# Quantization levels of the CacheGen-style codec (keys, values) from fine to coarse; the rate control picks one.
LEVEL_LADDER = (
    ((63, 31, 31), (15, 15, 15)),
    ((31, 31, 31), (15, 7, 7)),
    ((31, 31, 15), (7, 7, 7)),
    ((31, 15, 15), (7, 7, 7)),
    ((31, 15, 15), (7, 5, 5)),
    ((31, 15, 15), (5, 5, 5)),
    ((15, 15, 15), (7, 5, 5)),
    ((15, 15, 15), (5, 5, 5)),
    ((15, 15, 11), (5, 5, 5)),
    ((15, 15, 7), (5, 5, 5)),
    ((15, 11, 7), (5, 5, 5)),
    ((15, 7, 7), (5, 5, 5)),
    ((11, 7, 7), (5, 5, 5)),
    ((9, 7, 7), (5, 5, 5)),
    ((7, 7, 7), (5, 5, 5)),
    ((7, 7, 7), (5, 3, 3)),
    ((7, 5, 5), (3, 3, 3)),
    ((5, 5, 5), (3, 3, 3)),
    ((3, 3, 3), (3, 3, 3)),
)


def calibration_tokens(tokenizer, num_samples: int, sample_length: int) -> torch.Tensor:
    """Token ids [num_samples, sample_length] from the training split of WikiText-2."""
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
    text = text[: MAX_CHARACTERS_PER_TOKEN * num_samples * sample_length]
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < num_samples * sample_length:
        raise ValueError("the calibration text is shorter than the requested number of tokens")
    return ids[: num_samples * sample_length].reshape(num_samples, sample_length)


@torch.no_grad()
def collect_kv(model, input_ids: torch.Tensor) -> KV:
    """K and V of every layer for every sample, [samples, kv_heads, tokens, head_dim], on the CPU."""
    per_sample = []
    for sample in input_ids:
        cache = model(sample[None].to(model.device), use_cache=True).past_key_values
        per_sample.append([(layer.keys.cpu(), layer.values.cpu()) for layer in cache.layers])
    return [tuple(torch.cat(tensors) for tensors in zip(*layers)) for layers in zip(*per_sample)]


def fit_kvquant(kv: KV, config: KVQuantConfig, device: torch.device | str = "cpu") -> KVQuantCalibration:
    skip = config.sink_tokens
    layers = [fit_layer(k[:, :, skip:].to(device), v[:, :, skip:].to(device), config).to("cpu") for k, v in kv]
    return KVQuantCalibration(config, layers)


def _channel_counts(symbols: torch.Tensor, levels: int) -> torch.Tensor:
    """Histogram of `symbols` [batch, kv_heads, tokens, head_dim] for every channel: [channels, levels]."""
    channels = symbols.shape[1] * symbols.shape[3]
    rows = symbols.permute(1, 3, 0, 2).reshape(channels, -1).long()
    ones = torch.ones(rows.shape, device=symbols.device)
    return torch.zeros(channels, levels, device=symbols.device).scatter_add_(1, rows, ones)


def _tensor_counts(tensor: CacheGenTensor) -> dict[bool, torch.Tensor]:
    """Symbol counts of one tensor: per channel for the ordinary tokens, in one table for the anchors."""
    counts = {False: _channel_counts(tensor.symbols(), tensor.levels)}
    if tensor.group_size:
        counts[True] = _channel_counts(tensor.anchor_codes, tensor.anchor_levels).sum(dim=0, keepdim=True)
    return counts


def symbol_counts(kv: KV, config: CacheGenConfig, device: torch.device | str = "cpu") -> tuple[dict, int]:
    """Symbol counts of `kv` for every (layer, is_value, is_anchor), and the bits of the scales."""
    codec = CacheGenCodec(config, len(kv))
    counts, scale_bits = {}, 0
    for layer, (k, v) in enumerate(kv):
        for is_value, tensor in enumerate(codec.quantize(layer, k.to(device), v.to(device))):
            for is_anchor, table in _tensor_counts(tensor).items():
                counts[(layer, bool(is_value), is_anchor)] = table
            scale_bits += 8 * sum(scale.numel() * scale.element_size() for scale in (tensor.scale, tensor.anchor_scale))
    return counts, scale_bits


def profile_cachegen(kv: KV, config: CacheGenConfig, device: torch.device | str = "cpu") -> CacheGenProfile:
    """Symbol statistics of `kv` under `config`."""
    counts, _ = symbol_counts(kv, config, device)
    return CacheGenProfile(config, {key: rans.normalize_counts(table).cpu() for key, table in counts.items()})


def coded_bits(kv: KV, profile: CacheGenProfile, device: torch.device | str = "cpu") -> float:
    """Bits per element of `kv` when it is coded with the statistics of `profile`."""
    counts, bits = symbol_counts(kv, profile.config, device)
    for key, table in counts.items():
        probabilities = profile.frequencies[key].to(device).float() / (1 << rans.PRECISION)
        bits -= (table * torch.log2(probabilities)).sum().item()
    return bits / sum(k.numel() + v.numel() for k, v in kv)


def fit_cachegen(
    kv: KV, config: CacheGenConfig, target_bits: float | None = None, device: torch.device | str = "cpu"
) -> CacheGenProfile:
    """Profile of `kv`. With `target_bits`, the quantization levels are the finest ones of the ladder
    that meet this rate per element, or the coarsest ones if none does. The rate of a candidate is
    measured on every second sample with the statistics of the other samples."""
    if target_bits is None:
        return profile_cachegen(kv, config, device)
    fitted = [(k[0::2], v[0::2]) for k, v in kv]
    held_out = [(k[1::2], v[1::2]) for k, v in kv] if kv[0][0].shape[0] > 1 else fitted
    for key_levels, value_levels in LEVEL_LADDER:
        candidate = replace(config, key_levels=key_levels, value_levels=value_levels)
        if coded_bits(held_out, profile_cachegen(fitted, candidate, device), device) <= target_bits:
            break
    return profile_cachegen(kv, candidate, device)


def save(calibration: CacheGenProfile | KVQuantCalibration, path: str) -> None:
    state = {"config": asdict(calibration.config)}
    if isinstance(calibration, CacheGenProfile):
        state["method"] = "cachegen"
        state["frequencies"] = {
            f"{layer}.{int(is_value)}.{int(is_anchor)}": freq.to(torch.int16)
            for (layer, is_value, is_anchor), freq in calibration.frequencies.items()
        }
    else:
        state["method"] = "kvquant"
        state["layers"] = [asdict(layer) for layer in calibration.layers]
    torch.save(state, path)


def load(path: str) -> CacheGenProfile | KVQuantCalibration:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["method"] == "kvquant":
        layers = [LayerCalibration(**layer) for layer in state["layers"]]
        return KVQuantCalibration(KVQuantConfig(**state["config"]), layers)
    settings = {name: tuple(value) if isinstance(value, list) else value for name, value in state["config"].items()}
    config = CacheGenConfig(**settings)
    frequencies = {}
    for name, freq in state["frequencies"].items():
        layer, is_value, is_anchor = (int(part) for part in name.split("."))
        frequencies[(layer, bool(is_value), bool(is_anchor))] = freq.to(torch.int32)
    return CacheGenProfile(config, frequencies)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True, choices=("cachegen", "kvquant"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-length", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=2, help="code width of the KVQuant-style quantizer")
    parser.add_argument("--compression", type=float, default=0.86, help="target of the CacheGen-style codec vs. BF16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    kv = collect_kv(model, calibration_tokens(tokenizer, args.num_samples, args.sample_length))
    if args.method == "kvquant":
        calibration = fit_kvquant(kv, KVQuantConfig(bits=args.bits), args.device)
    else:
        calibration = fit_cachegen(kv, CacheGenConfig(), BF16_BITS * (1 - args.compression), args.device)
        print(f"quantization levels: keys {calibration.config.key_levels}, values {calibration.config.value_levels}")
    save(calibration, args.output)


if __name__ == "__main__":
    main()

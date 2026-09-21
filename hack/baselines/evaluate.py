"""Perplexity, size and reconstruction error of a comparison method on the test split of WikiText-2.

    python -m hack.baselines.evaluate --model Qwen/Qwen3-0.6B --method bf16
    python -m hack.baselines.evaluate --model Qwen/Qwen3-0.6B --method kvquant --calibration kvquant.pt
    python -m hack.baselines.evaluate --model Qwen/Qwen3-0.6B --method cachegen --calibration cachegen.pt

The text is cut into windows of `--window` tokens. Every window is one forward pass in which the
attention of all tokens reads the dequantized K and V. The sizes are relative to a BF16 KV cache.
"""

import argparse
import json
import math

import torch

from hack.baselines.hf_cache import StorageOnlyCache, make_cache

BF16_BITS = 16


def wikitext_windows(tokenizer, window: int, max_windows: int | None) -> torch.Tensor:
    """Token ids [windows, window] of the test split."""
    from datasets import load_dataset

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    windows = ids[: ids.numel() // window * window].reshape(-1, window)
    return windows if max_windows is None else windows[:max_windows]


def new_cache(model, method: str, calibration: str | None, exact_prefill: bool) -> StorageOnlyCache | None:
    if method == "bf16":
        return None
    options = {} if calibration is None else {"profile" if method == "cachegen" else "calibration": calibration}
    return make_cache(method, model.config, exact_prefill=exact_prefill, **options)


@torch.no_grad()
def perplexity(model, windows: torch.Tensor, method: str, calibration: str | None) -> float:
    total = 0.0
    for ids in windows.to(model.device):
        cache = new_cache(model, method, calibration, exact_prefill=False)
        logits = model(ids[None], past_key_values=cache, use_cache=cache is not None).logits[0, :-1]
        total += torch.nn.functional.cross_entropy(logits.float(), ids[1:], reduction="sum").item()
    return math.exp(total / (windows.shape[0] * (windows.shape[1] - 1)))


@torch.no_grad()
def sizes_and_errors(model, ids: torch.Tensor, method: str, calibration: str | None) -> dict[str, float]:
    """Sizes and relative errors of the cache that holds the unquantized K and V of one window."""
    ids = ids[None].to(model.device)
    reference = model(ids, use_cache=True).past_key_values
    cache = new_cache(model, method, calibration, exact_prefill=True)
    model(ids, past_key_values=cache, use_cache=True)
    errors, elements = torch.zeros(2), 0
    for layer, exact in zip(cache.layers, reference.layers):
        pairs = zip(layer.dequantize(), (exact.keys, exact.values))
        errors += torch.tensor(
            [((got.float() - want.float()).norm() / want.float().norm()).item() for got, want in pairs]
        )
        elements += exact.keys.numel() + exact.values.numel()
    key_error, value_error = (errors / len(cache.layers)).tolist()
    transfer_bits, memory_bits = 8 * cache.transfer_nbytes() / elements, 8 * cache.nbytes() / elements
    return {
        "transfer_bits_per_element": transfer_bits,
        "transfer_compression": 1 - transfer_bits / BF16_BITS,
        "memory_bits_per_element": memory_bits,
        "memory_compression": 1 - memory_bits / BF16_BITS,
        "key_relative_error": key_error,
        "value_relative_error": value_error,
    }


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True, choices=("bf16", "cachegen", "kvquant"))
    parser.add_argument("--calibration", help="file written by hack.baselines.calibration")
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--max-windows", type=int, help="number of windows (default: the whole test split)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()
    windows = wikitext_windows(tokenizer, args.window, args.max_windows)
    result = {"model": args.model, "method": args.method, "windows": windows.shape[0], "window": args.window}
    result["perplexity"] = perplexity(model, windows, args.method, args.calibration)
    if args.method != "bf16":
        result.update(sizes_and_errors(model, windows[0], args.method, args.calibration))
    print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Size of the KV cache of one prompt under each method, measured on real activations.

Example:
    python benchmarks/kv_size/kv_size.py --model Qwen/Qwen3-8B --prompt-tokens 4096 --methods baseline hack
"""

import argparse
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hack import HackConfig
from hack.hf import HackCache, enable_hack


def prompt_ids(tokenizer, tokens: int, device) -> torch.Tensor:
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text[: 8 * tokens], return_tensors="pt").input_ids  # more characters than needed, not the corpus
    if ids.shape[1] < tokens:
        ids = tokenizer(text, return_tensors="pt").input_ids
    return ids[:, :tokens].to(device)


def bf16_nbytes(config, tokens: int) -> int:
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    head_dim = getattr(text, "head_dim", None) or text.hidden_size // text.num_attention_heads
    return 2 * 2 * tokens * text.num_key_value_heads * head_dim * text.num_hidden_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--methods", nargs="+", default=["baseline", "hack"], choices=["baseline", "hack", "cachegen", "kvquant"])
    parser.add_argument("--partition-sizes", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    ids = prompt_ids(tokenizer, args.prompt_tokens, model.device)
    reference = bf16_nbytes(model.config, ids.shape[1])
    default_attention = model.config._attn_implementation

    def report(label: str, cache_bytes: int, transfer_bytes: int):
        print(json.dumps({
            "method": label, "prompt_tokens": ids.shape[1], "cache_bytes": cache_bytes, "transfer_bytes": transfer_bytes,
            "cache_vs_bf16": cache_bytes / reference, "transfer_vs_bf16": transfer_bytes / reference,
        }), flush=True)

    for method in args.methods:
        if method == "baseline":
            model.set_attn_implementation(default_attention)
            cache = DynamicCache(config=model.config)
            with torch.no_grad():
                model(ids, past_key_values=cache, use_cache=True)
            measured = sum(layer.keys.numel() * layer.keys.element_size() * 2 for layer in cache.layers)
            report("baseline", measured, measured)
        elif method == "hack":
            enable_hack(model)
            for size in args.partition_sizes:
                cache = HackCache(model.config, HackConfig(partition_size=size))
                with torch.no_grad():
                    model(ids, past_key_values=cache, use_cache=True)
                report(f"hack-pi{size}", cache.nbytes(), cache.nbytes())
        else:
            from hack.baselines import make_cache

            model.set_attn_implementation(default_attention)
            cache = make_cache(method, model.config)
            with torch.no_grad():
                model(ids, past_key_values=cache, use_cache=True)
            report(method, cache.nbytes(), cache.transfer_nbytes())


if __name__ == "__main__":
    main()

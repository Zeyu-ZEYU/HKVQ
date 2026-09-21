"""Accuracy of a model under different KV-cache methods.

Example:
    python benchmarks/accuracy/run.py --model Qwen/Qwen3-8B --tasks imdb gsm8k \
        --methods baseline hack --partition-sizes 32 64 128 --limit 200 --output results/accuracy
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hack import HackConfig
from hack.hf import HackCache, enable_hack
from tasks import TASKS


def build_inputs(tokenizer, prompt: str, device, max_input_tokens: int):
    if tokenizer.chat_template:
        extra = {"enable_thinking": False} if "enable_thinking" in tokenizer.chat_template else {}
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True, **extra
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    if encoded["input_ids"].shape[1] > max_input_tokens:
        raise ValueError("prompt longer than --max-input-tokens")
    return encoded.to(device)


def method_variants(args):
    for method in args.methods:
        if method == "hack":
            for size in args.partition_sizes:
                yield f"hack-pi{size}", "hack", {"partition_size": size}
        else:
            yield method, method, {}


def make_cache(kind: str, options: dict, model):
    if kind == "baseline":
        return None
    if kind == "hack":
        return HackCache(model.config, HackConfig(**options))
    from hack.baselines import make_cache as make_baseline_cache

    return make_baseline_cache(kind, model.config, exact_prefill=True, **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--methods", nargs="+", default=["baseline", "hack"], choices=["baseline", "hack", "cachegen", "kvquant"])
    parser.add_argument("--partition-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--limit", type=int, default=200, help="examples per task")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/accuracy"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    default_attention = model.config._attn_implementation
    args.output.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        task = TASKS[task_name](args.limit, args.seed)
        for label, kind, options in method_variants(args):
            if kind == "hack":
                enable_hack(model)
            else:
                model.set_attn_implementation(default_attention)
            torch.manual_seed(args.seed)
            scores, started = [], time.time()
            for example in task.examples:
                try:
                    inputs = build_inputs(tokenizer, example.prompt, model.device, args.max_input_tokens)
                except ValueError:
                    continue
                with torch.no_grad():
                    output = model.generate(
                        **inputs, max_new_tokens=task.max_new_tokens, do_sample=False,
                        past_key_values=make_cache(kind, options, model),
                    )
                text = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                scores.append(task.score(text, example.reference))
            result = {
                "model": args.model, "task": task.name, "metric": task.metric, "method": label,
                "examples": len(scores), "score": sum(scores) / max(1, len(scores)), "seconds": time.time() - started,
            }
            print(json.dumps(result), flush=True)
            name = f"{args.model.split('/')[-1]}_{task.name}_{label}.json"
            (args.output / name).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

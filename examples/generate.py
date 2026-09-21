"""Answer a question about a long text with the BF16 KV cache and with HACK, and compare the two KV caches."""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from hack import HackConfig
from hack.hf import HackCache, enable_hack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--context-tokens", type=int, default=2000)
    parser.add_argument("--question", default="Summarize the text above in two sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--partition-size", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    text = text[: 8 * args.context_tokens]  # a few times more characters than needed, not the whole corpus
    context = tokenizer.decode(tokenizer(text).input_ids[: args.context_tokens])
    messages = [{"role": "user", "content": f"{context}\n\n{args.question}"}]
    extra = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **extra
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    def generate(cache):
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, past_key_values=cache)
        return tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)

    bf16_cache = DynamicCache(config=model.config)
    print("BF16:", generate(bf16_cache))
    bf16_bytes = sum(2 * layer.keys.numel() * layer.keys.element_size() for layer in bf16_cache.layers)

    enable_hack(model)
    hack_cache = HackCache(model.config, HackConfig(partition_size=args.partition_size))
    print("HACK:", generate(hack_cache))
    print(f"KV cache: BF16 {bf16_bytes / 2**20:.1f} MiB, HACK {hack_cache.nbytes() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()

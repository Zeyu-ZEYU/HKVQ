"""Workloads for the end-to-end benchmark.

A workload is a JSON-lines file with one request per line:

    {"id": "r0", "prompt": "text ...", "max_tokens": 128}
    {"id": "r1", "prompt_token_ids": [101, 2023, ...], "max_tokens": 64, "ignore_eos": true}

Exactly one of `prompt` and `prompt_token_ids` is given. `ignore_eos` (default false) makes the
output length equal to `max_tokens`.

This module writes synthetic workloads with controlled prompt and output lengths:

    python benchmarks/e2e/workload.py --output synthetic.jsonl --num-requests 64 \
        --prompt-len 2048 4096 --output-len 64 128 --vocab-size 32000
"""

import argparse
import json
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkItem:
    id: str
    max_tokens: int
    prompt: str | None = None
    prompt_token_ids: list[int] | None = None
    ignore_eos: bool = False

    def payload(self, model: str) -> dict:
        body = {
            "model": model,
            "prompt": self.prompt if self.prompt is not None else self.prompt_token_ids,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        if self.ignore_eos:
            body["ignore_eos"] = True
        return body


def read_workload(path: Path) -> list[WorkItem]:
    items = []
    with open(path) as stream:
        for index, line in enumerate(line for line in stream if line.strip()):
            record = json.loads(line)
            if ("prompt" in record) == ("prompt_token_ids" in record):
                raise ValueError(f"{path}: request {index} needs exactly one of prompt and prompt_token_ids")
            items.append(
                WorkItem(
                    id=str(record.get("id", index)),
                    max_tokens=int(record["max_tokens"]),
                    prompt=record.get("prompt"),
                    prompt_token_ids=record.get("prompt_token_ids"),
                    ignore_eos=bool(record.get("ignore_eos", False)),
                )
            )
    return items


def synthetic_items(
    num_requests: int, prompt_len: tuple[int, int], output_len: tuple[int, int], vocab_size: int, seed: int
) -> Iterator[dict]:
    rng = random.Random(seed)
    low, high = 1000, max(1001, vocab_size - 1000)
    for index in range(num_requests):
        tokens = [rng.randrange(low, high) for _ in range(rng.randint(*prompt_len))]
        max_tokens = rng.randint(*output_len)
        yield {"id": f"s{index}", "prompt_token_ids": tokens, "max_tokens": max_tokens, "ignore_eos": True}


def _range(values: list[int]) -> tuple[int, int]:
    if len(values) == 1:
        return values[0], values[0]
    if len(values) == 2 and values[0] <= values[1]:
        return values[0], values[1]
    raise argparse.ArgumentTypeError("give one length or an increasing pair MIN MAX")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, nargs="+", default=[2048], help="LEN or MIN MAX (uniform)")
    parser.add_argument("--output-len", type=int, nargs="+", default=[128], help="LEN or MIN MAX (uniform)")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    items = synthetic_items(
        args.num_requests, _range(args.prompt_len), _range(args.output_len), args.vocab_size, args.seed
    )
    with open(args.output, "w") as stream:
        for item in items:
            stream.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()

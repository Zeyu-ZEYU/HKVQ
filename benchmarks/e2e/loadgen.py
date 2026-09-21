"""Open-loop load generator: Poisson arrivals over the requests of a workload file.

Every request is streamed; the generator records, per request, the scheduled arrival time and
the arrival time of every output chunk, from which TTFT, TPOT and JCT follow.

    python benchmarks/e2e/loadgen.py --url http://127.0.0.1:8000 --model Qwen/Qwen3-8B \
        --workload synthetic.jsonl --rate 0.5 --output runs/demo/requests.jsonl
"""

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workload import WorkItem, read_workload  # noqa: E402


def arrival_offsets(count: int, rate: float, seed: int) -> list[float]:
    """Arrival times of a Poisson process with `rate` requests per second."""
    rng = random.Random(seed)
    now, offsets = 0.0, []
    for _ in range(count):
        offsets.append(now)
        now += rng.expovariate(rate) if rate > 0 else 0.0
    return offsets


async def _run_request(session: aiohttp.ClientSession, url: str, model: str, item: WorkItem, start: float) -> dict:
    request_id = "hk" + uuid.uuid4().hex[:12]
    delay = start - time.time()
    if delay > 0:
        await asyncio.sleep(delay)
    record = {"id": item.id, "request_id": request_id, "arrival": start, "sent": time.time(), "chunk_times": []}
    output_tokens, error = 0, None
    try:
        body = dict(item.payload(model), return_token_ids=True)
        async with session.post(url + "/v1/completions", json=body, headers={"X-Request-Id": request_id}) as response:
            if response.status != 200:
                error = f"HTTP {response.status}: {(await response.text())[:200]}"
            else:
                async for line in response.content:
                    if not line.startswith(b"data:") or b"[DONE]" in line:
                        continue
                    choices = json.loads(line[5:]).get("choices") or []
                    tokens = len(choices[0].get("token_ids") or []) if choices else 0
                    if tokens:
                        output_tokens += tokens
                        record["chunk_times"].append([time.time(), tokens])
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        error = repr(exc)
    record.update(done=time.time(), output_tokens=output_tokens, error=error)
    if item.prompt_token_ids is not None:
        record["prompt_tokens"] = len(item.prompt_token_ids)
    return record


async def run_load(url: str, model: str, items: list[WorkItem], rate: float, seed: int, output: Path) -> list[dict]:
    offsets = arrival_offsets(len(items), rate, seed)
    begin = time.time() + 1.0
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60)
    connector = aiohttp.TCPConnector(limit=0, force_close=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [_run_request(session, url, model, item, begin + offset) for item, offset in zip(items, offsets)]
        records = await asyncio.gather(*tasks)
    with open(output, "w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="proxy (or single instance) base URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True, help="requests per second; 0 sends everything at once")
    parser.add_argument("--num-requests", type=int, default=None, help="use only the first N requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = read_workload(args.workload)[: args.num_requests]
    records = asyncio.run(run_load(args.url, args.model, items, args.rate, args.seed, args.output))
    failed = sum(1 for record in records if record["error"])
    print(f"{len(records)} requests, {failed} failed -> {args.output}")


if __name__ == "__main__":
    main()

"""Turn the raw files of one run into `summary.json` and `per_request.csv`.

Inputs (all in the run directory):
    requests.jsonl           client-side timing of every request (loadgen.py)
    proxy.jsonl              prefill / decode hand-off times (proxy)
    prefill*_events.jsonl    arrive, scheduled, staged, sent, finished (connector, prefill side)
    decode*_events.jsonl     arrive, recv, loaded, scheduled, finished, cache (connector, decode side)
    memory.jsonl             GPU memory and KV-cache usage samples of the decode instances

Per request: TTFT (first output chunk - arrival), TPOT ((last chunk - first chunk) / (tokens - 1)),
JCT (last chunk - arrival) and the decomposition
    queueing     waiting in the prefill queue, waiting for decode blocks, waiting for a decode slot
    prefill      first scheduled on the prefill instance -> prefill finished (includes staging)
    kv_transfer  pull started on the decode instance -> KV installed in its blocks
    decode       first scheduled on the decode instance -> finished
    other        remainder of JCT (HTTP, proxy, tokenization)

    python benchmarks/e2e/metrics.py runs/demo
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REQUEST_ID = re.compile(r"hk[0-9a-f]{12}")
PHASES = ("queueing", "prefill", "kv_transfer", "decode", "other")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def events_by_request(paths: list[Path]) -> dict[str, dict[str, dict]]:
    """{client request id: {event name: first event record}}"""
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        for event in read_jsonl(path):
            match = REQUEST_ID.search(event["request_id"])
            if match:
                grouped[match.group()].setdefault(event["event"], event)
    return grouped


def decompose(jct: float, prefill: dict[str, dict], decode: dict[str, dict]) -> dict[str, float] | None:
    needed_prefill, needed_decode = ("arrive", "scheduled", "finished"), ("arrive", "loaded", "scheduled", "finished")
    if any(k not in prefill for k in needed_prefill) or any(k not in decode for k in needed_decode):
        return None
    load_start = decode["loaded"]["time"] - decode["loaded"]["seconds"]
    phases = {
        "queueing": (prefill["scheduled"]["time"] - prefill["arrive"]["time"])
        + (load_start - decode["arrive"]["time"])
        + (decode["scheduled"]["time"] - decode["loaded"]["time"]),
        "prefill": prefill["finished"]["time"] - prefill["scheduled"]["time"],
        "kv_transfer": decode["loaded"]["seconds"],
        "decode": decode["finished"]["time"] - decode["scheduled"]["time"],
    }
    phases["other"] = jct - sum(phases.values())
    return phases


def request_rows(run_dir: Path) -> list[dict]:
    prefill = events_by_request(sorted(run_dir.glob("prefill*_events.jsonl")))
    decode = events_by_request(sorted(run_dir.glob("decode*_events.jsonl")))
    rows = []
    for record in read_jsonl(run_dir / "requests.jsonl"):
        if record["error"] or not record["chunk_times"]:
            error = record["error"] or "no output"
            rows.append({"id": record["id"], "request_id": record["request_id"], "error": error})
            continue
        first, last = record["chunk_times"][0][0], record["chunk_times"][-1][0]
        tokens = record["output_tokens"]
        row = {
            "id": record["id"],
            "request_id": record["request_id"],
            "error": None,
            "prompt_tokens": record.get("prompt_tokens"),
            "output_tokens": tokens,
            "ttft": first - record["arrival"],
            "tpot": (last - first) / (tokens - 1) if tokens > 1 else None,
            "jct": last - record["arrival"],
        }
        loaded = decode.get(record["request_id"], {}).get("loaded")
        row["transfer_bytes"] = loaded["nbytes"] if loaded else None
        phases = decompose(row["jct"], prefill.get(record["request_id"], {}), decode.get(record["request_id"], {}))
        row.update(phases or dict.fromkeys(PHASES))
        rows.append(row)
    return rows


def memory_summary(run_dir: Path) -> dict:
    samples = read_jsonl(run_dir / "memory.jsonl")
    events = [e for path in sorted(run_dir.glob("decode*_events.jsonl")) for e in read_jsonl(path)]
    caches = [e for e in events if e["event"] == "cache"]
    if not samples:
        return {}
    summary = {
        "decode_gpu_total_mib": samples[0].get("gpu_total_mib"),
        "decode_gpu_peak_used_mib": max(s["gpu_used_mib"] for s in samples),
        "decode_kv_peak_usage": max(s["kv_usage"] for s in samples),
    }
    if caches:
        capacity = caches[0]["num_blocks"] * caches[0]["bytes_per_block"]
        unused = capacity * (1.0 - summary["decode_kv_peak_usage"])
        summary["decode_kv_capacity_mib"] = capacity / 2**20
        summary["decode_side_pool_mib"] = caches[0]["side_pool_bytes"] / 2**20
        summary["decode_peak_memory_in_use_mib"] = summary["decode_gpu_peak_used_mib"] - unused / 2**20
        if summary["decode_gpu_total_mib"]:
            summary["decode_peak_memory_in_use_fraction"] = (
                summary["decode_peak_memory_in_use_mib"] / summary["decode_gpu_total_mib"]
            )
    return summary


def summarize(run_dir: Path) -> dict:
    rows = request_rows(run_dir)
    good = [row for row in rows if not row["error"]]
    summary = {
        "run": run_dir.name,
        "num_requests": len(rows),
        "num_failed": len(rows) - len(good),
        "jct": distribution([row["jct"] for row in good]),
        "ttft": distribution([row["ttft"] for row in good]),
        "tpot": distribution([row["tpot"] for row in good if row["tpot"] is not None]),
        "decomposition_mean": {
            phase: statistics.fmean([row[phase] for row in good if row[phase] is not None])
            for phase in PHASES
            if any(row[phase] is not None for row in good)
        },
        "transfer_bytes": distribution([row["transfer_bytes"] for row in good if row["transfer_bytes"] is not None]),
        "transfer_bytes_total": sum(row["transfer_bytes"] or 0 for row in good),
        "memory": memory_summary(run_dir),
    }
    config = run_dir / "config.json"
    if config.exists():
        summary["config"] = json.loads(config.read_text())
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["id", "request_id", "error", "prompt_tokens", "output_tokens", "ttft", "tpot", "jct", "transfer_bytes"]
    fields += PHASES
    with open(run_dir / "per_request.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, nargs="+")
    for run_dir in parser.parse_args().run_dir:
        print(json.dumps(summarize(run_dir), indent=2))


if __name__ == "__main__":
    main()

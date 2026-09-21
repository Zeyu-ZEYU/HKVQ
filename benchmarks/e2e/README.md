# End-to-end benchmark: disaggregated prefill and decode on vLLM

This directory starts a disaggregated serving cluster (prefill instances, decode instances and a
proxy), replays a workload with Poisson arrivals, and reports job completion time (JCT), time to
first token (TTFT), time per output token (TPOT), the decomposition of JCT into phases, the KV bytes
moved between the instances and the peak memory of the decode GPU.

## Setup

```bash
pip install -e ".[vllm,benchmarks]"     # registers the attention backend as a vLLM plugin
```

The plugin makes the backend selectable with `--attention-backend CUSTOM`. The KV connector is
loaded by module path, see `cluster.py`. Every instance needs one GPU that holds the model.

## One experiment

```bash
python benchmarks/e2e/workload.py --output synthetic.jsonl --num-requests 16 \
    --prompt-len 2000 6000 --output-len 32 64 --vocab-size 100000

python benchmarks/e2e/run_experiment.py --method baseline --model Qwen/Qwen3-8B \
    --prefill-gpus 0 --decode-gpus 1 --bandwidth-gbps 10 \
    --workload synthetic.jsonl --rate 0.2 --run-dir runs/baseline_10gbps

python benchmarks/e2e/run_experiment.py --method hack --model Qwen/Qwen3-8B \
    --prefill-gpus 0 --decode-gpus 1 --bandwidth-gbps 10 \
    --workload synthetic.jsonl --rate 0.2 --run-dir runs/hack_10gbps

python benchmarks/e2e/plot.py --output-dir figures runs/baseline_10gbps runs/hack_10gbps
```

`run_experiment.py` starts the cluster, sends `--warmup-requests` requests that are not measured,
replays the workload, stops the cluster and writes `summary.json` and `per_request.csv` into the run
directory. `plot.py` reads only `summary.json` files.

| Option | Meaning |
|---|---|
| `--method` | `baseline` (default attention backend of vLLM, BF16 KV cache), `baseline_triton` (Triton attention backend of vLLM, BF16 KV cache) or `hack`; all use the same connector and proxy |
| `--prefill-gpus`, `--decode-gpus` | one instance per entry; an entry may list several GPUs (`0,1`) |
| `--bandwidth-gbps` | sender-side rate limit of every prefill instance, emulates the link to the decode instances; on one host the transfer uses the loopback interface, which carries about 12 Gbps with this TCP transport |
| `--rate` | mean arrival rate in requests per second; arrivals are a Poisson process with seed `--seed` |
| `--hack-option KEY=VALUE` | `partition_size`, `kv_bits`, `stochastic`, `summation_elimination`, `requant_elimination`, `sink_tokens`, `attention` (`reference`: PyTorch, `kernels`: `hack.kernels`), `decode` (`batched`, or `loop`: one sequence at a time through a contiguous cache), `side_slots` |
| `--vllm-arg` | extra argument for `vllm serve`, repeatable |

The ablations of the paper map to `--hack-option summation_elimination=0` and
`--hack-option requant_elimination=0`; the sensitivity study to `--hack-option partition_size=32|64|128`.

The same options can be set without the harness, as environment variables (`HACK_PARTITION_SIZE`,
`HACK_KV_BITS`, ...) or with `--additional-config '{"hack": {...}}'`, see `hack/vllm_plugin/settings.py`.

## Request flow

1. The proxy sends the request to the prefill instance with the fewest queued tokens, with
   `max_tokens=1`. At the end of the prefill step the connector copies the KV pages of the request
   to pinned host memory and offers them on a
   TCP port.
2. The response carries the first token and `kv_transfer_params`. The proxy sends prompt + first
   token with these parameters to the decode instance with the fewest queued tokens.
3. The decode instance allocates the blocks, pulls the KV in a background thread, installs it and
   continues decoding. A streamed response starts with the first chunk of the decode instance.

## Workload format

One JSON object per line, with exactly one of `prompt` (text) and `prompt_token_ids`:

```json
{"id": "r0", "prompt": "Summarize the following article ...", "max_tokens": 128}
{"id": "r1", "prompt_token_ids": [101, 2023, 2003], "max_tokens": 64, "ignore_eos": true}
```

## Output files of a run

| File | Content |
|---|---|
| `summary.json` | mean, P50, P95, P99 and maximum of JCT, TTFT, TPOT; mean of every phase; transferred bytes; decode-GPU memory |
| `per_request.csv` | one row per request with the same quantities |
| `requests.jsonl` | client-side arrival time and the arrival time of every output chunk |
| `proxy.jsonl` | chosen instances and hand-off times |
| `prefill*_events.jsonl`, `decode*_events.jsonl` | connector events: `arrive`, `scheduled`, `staged`, `sent`, `recv`, `loaded`, `finished`, `cache` |
| `memory.jsonl` | samples of the used memory of the decode GPU and of the KV-cache usage of the decode instance |

Phases: `queueing` (prefill queue, waiting for decode blocks, waiting for a decode slot), `prefill`
(first scheduled on the prefill instance until the prefill finished, includes staging), `kv_transfer`
(pull started until the KV is installed in the decode blocks), `decode` (first scheduled on the
decode instance until finished), `other` (HTTP, proxy, tokenization).

vLLM reserves its KV cache at start-up, so the memory reported by the driver is constant. The
summary therefore also gives `decode_peak_memory_in_use_mib`: used GPU memory minus the part of the
KV cache that was never occupied (`1 - peak KV usage` times the cache capacity).

## Adding a method

A storage-only quantization baseline plugs in as another attention backend. Register the backend
with vLLM and add one entry to `METHODS` in `cluster.py` with the backend name, the block size it
needs and its environment variables. The connector moves pages as opaque bytes and needs no change
as long as the first dimension of the per-layer cache tensor is the block dimension.

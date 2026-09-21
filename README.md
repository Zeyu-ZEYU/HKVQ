# HACK: Homomorphic Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference

This repository contains HACK, the system of the ATC '26 paper *"HACK: Homomorphic Acceleration via Compression of
the Key-Value Cache for Disaggregated LLM Inference"*, and the scripts that run the experiments of the paper. The
submitted version of the paper calls the system HKVQ (Homomorphic KV Quantization); the section, figure and table
numbers in this file refer to that version.

HACK stores the KV cache as 2-bit codes, sends the codes from the prefill instance to the decode instance, and
computes attention directly on the codes with INT8 matrix multiplications, without dequantizing K and V at
every decode step.

## Artifact evaluation guide

The steps in order. The times were measured on a node with RTX A6000 GPUs (48 GB).

| Step | What to do | Section | Time |
|---|---|---|---|
| 1 | Get a machine with NVIDIA GPUs: your own, or the node that we provide | Environment; Access to GPUs for artifact evaluators | |
| 2 | Clone the repository and install it with pip or with Docker | Installation | 5 to 10 minutes |
| 3 | Check the installation: run the tests and the example | Minimal working example | 10 minutes |
| 4 | Claim C1: size of the KV cache | E2 | 5 minutes |
| 5 | Claim C2: disaggregated serving of long prompts, four runs | E3, long-context workload | 16 minutes |
| 6 | Claim C3: two runs of HACK, with and without requantization elimination | E3, end of the section | 7 minutes |
| 7 | Claim C4: attention-kernel microbenchmark | E4 | 10 minutes |
| 8 | Claim C5: accuracy, quick check | E1 | 30 minutes |

The claims and their expected results are listed under "Main claims". The first run downloads Qwen3-8B (16 GB) and
the datasets from the Hugging Face Hub; no account or token is needed, and the warning about unauthenticated
requests can be ignored. Every experiment reports the measured values as JSON or CSV; the tables of this file show
the values that we measured with the same commands. On a remote machine, run the steps inside `tmux` or `screen`, so
that a lost connection does not end them. Run them one after the other: the serving experiment measures latencies and
needs its two GPUs for itself.

### Access to GPUs for artifact evaluators

The installation check, E1, E2 and E4 need one NVIDIA GPU. E3 needs two GPUs: with 40 GB or more for the
long-context workload, with 20 GB or more for the short-prompt workload. The expected results of E3 and E4 refer to
RTX A6000 GPUs. Evaluators without such a machine get one from us: a cloud node with RTX A6000 GPUs, on which this
repository is installed and the models are downloaded. Please post a comment on the artifact submission site with an
SSH public key and a time window of about four hours, one day ahead if possible. We reply with the login command.
On the node, `cd ~/HACK && source .venv/bin/activate` replaces step 2. The node serves one evaluator at a time, and
we do not log or monitor what evaluators do on it.

## Contents

| Path | Content | Paper |
|---|---|---|
| `hack/quant.py`, `hack/homomorphic.py` | quantization with partitions; matrix multiplication on quantized operands | Sec. 4.2 |
| `hack/cache.py`, `hack/attention_ref.py` | quantized KV cache; reference implementation of attention on codes (PyTorch) | Sec. 4.3 |
| `hack/kernels/` | Triton kernels: `attn_prefill`, `attn_decode`, paged decode, a FlashAttention-2-style BF16 kernel and the same kernel with fused KV dequantization | Sec. 5 |
| `hack/vllm_plugin/` | vLLM integration: attention backend with its own KV-cache layout, KV connector between prefill and decode instances, proxy | Sec. 5 |
| `hack/hf/` | Hugging Face Transformers integration (used by the accuracy experiments) | Sec. 6.3 |
| `hack/baselines/` | the two comparison methods: a CacheGen-style codec and a KVQuant-style quantizer | Sec. 6.1 |
| `benchmarks/accuracy/` | accuracy on IMDb, arXiv, Cocktail, HumanEval and GSM8K | Sec. 6.3, Tables 7, 9-13 |
| `benchmarks/kv_size/` | size of the stored and of the transferred KV | Sec. 6.2 |
| `benchmarks/e2e/` | disaggregated serving: JCT, TTFT, TPOT, decomposition of JCT, KV bytes transferred, peak decode memory; ablation and sensitivity options | Sec. 6.2, 6.4, 6.5; Figs. 9-13, 15; Table 5 |
| `benchmarks/microbench/` | decode attention-kernel microbenchmark | Sec. 6.7, Fig. 18 |
| `tests/` | unit and integration tests | |

vLLM, PyTorch and Transformers are used unmodified, as released on PyPI. The vLLM integration is an out-of-tree
plugin, and the GPU kernels of this repository are written in Triton.

## Environment

The code was developed and tested on:

- Ubuntu 22.04, Linux 5.15, NVIDIA driver 580, four NVIDIA RTX 4000 Ada GPUs (20 GB);
- Ubuntu 22.04, NVIDIA driver 580, four NVIDIA RTX A6000 GPUs (48 GB), with both installation methods below;
- an NVIDIA H200 for the attention-kernel microbenchmark;
- Python 3.12, PyTorch 2.13.0, Triton 3.7.1, Transformers 5.17.0, vLLM 0.29.0.

Requirements: Linux, an NVIDIA GPU with compute capability 8.0 or newer (BF16 and INT8 support), and a driver that
supports the CUDA version of the installed PyTorch wheel. Every experiment states the GPU memory it needs. Disk:
about 26 GB (Python environment 8 GB, Qwen3-8B 16 GB, Qwen3-0.6B 1.5 GB, datasets 0.2 GB); the Docker image takes
32 GB instead of the 8 GB of the Python environment.

## Installation

With pip in a Python 3.12 virtual environment. `uv` provides Python 3.12 where the system has none, for example on
Ubuntu 22.04; with `python3.12` installed, `python3.12 -m venv .venv` replaces the three lines that set up `uv`.

```bash
git clone https://github.com/Zeyu-ZEYU/HACK.git && cd HACK
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
pip install -e ".[vllm,benchmarks,test]"
```

This installs the pinned versions listed in `pyproject.toml` and registers the attention backend as a vLLM
plugin. The first use of every Triton kernel compiles it (a few seconds per kernel, cached afterwards).

With Docker:

```bash
git clone https://github.com/Zeyu-ZEYU/HACK.git && cd HACK
docker build -t hack -f docker/Dockerfile .
docker run --gpus all --ipc=host -it -v $HOME/.cache/huggingface:/root/.cache/huggingface hack
```

The image builds on the vLLM 0.29.0 image (about 32 GB) and needs the NVIDIA Container Toolkit; the container
starts a shell in `/workspace/hack`, where all commands of this file run unchanged.

## Minimal working example (about 10 minutes, one GPU, 4 GB of GPU memory)

```bash
pytest tests -q -k "not vllm"          # quantization, attention on codes, kernels, baselines
python examples/generate.py --model Qwen/Qwen3-0.6B
```

`examples/generate.py` answers a question about a 2,000-token text with the BF16 KV cache and with HACK and prints
both answers and the size of the two KV caches (under a minute). The first test run compiles every Triton kernel once;
later runs are faster. `pytest tests -q` additionally starts vLLM engines and takes about 13 minutes on the first run.

## Configuration

`hack.HackConfig` holds the settings of HACK.

| Field | Default | Meaning |
|---|---|---|
| `partition_size` | 64 | partition size (Pi in the paper); 32, 64 and 128 are used in the sensitivity study |
| `kv_bits` | 2 | bit width of the K and V codes |
| `qp_bits` | 8 | bit width of the transient Q and attention-probability codes |
| `summation_elimination` | `True` | ablation switch (Sec. 6.4) |
| `requant_elimination` | `True` | ablation switch (Sec. 6.4) |
| `stochastic`, `sink_tokens` | `False`, 4 | rounding mode; number of leading tokens kept in 16-bit |
| `quantized_prefill` | `False` | compute the attention of the prefill stage on the codes as well |

With vLLM the same fields are set as `HACK_*` environment variables or with
`--additional-config '{"hack": {...}}'`, see `hack/vllm_plugin/settings.py`.

## Experiments

The paper evaluates models with up to 180B parameters on multi-node clusters. The experiments below exercise the
same mechanisms on one node: Qwen3-8B, one prefill GPU, one decode GPU, and a rate limit on the KV transfer that
emulates the network between the two instances. E1, E3 and E4 write JSON or CSV files with the measured values,
and their plotting and table scripts read only those files; E2 prints JSON lines. Absolute numbers depend on the
GPUs, the network and the models used.

### Main claims

| | Claim | Experiment | Expected result |
|---|---|---|---|
| C1 | The 2-bit KV cache needs about one sixth of the bytes of the BF16 KV cache, in GPU memory and on the network (Sec. 6.2). | E2, 5 minutes | 17.5% to 20.9% of the BF16 bytes for a 4096-token prompt, depending on the partition size. The numbers are deterministic. |
| C2 | In disaggregated serving of long prompts, HACK lowers the average JCT, the tail JCT and the TTFT, and the gain grows with the request rate. It transfers about six times fewer KV bytes, and the decode instance needs several times less KV-cache memory (Sec. 6.2). | E3, long-context workload, 16 minutes | On two RTX A6000 GPUs: average JCT about 10% lower at 0.15 requests/s and about 30% lower at 0.3 requests/s, P99 JCT about 20% and 40% lower, TTFT about 40% and 45% lower, TPOT lower at 0.3 requests/s. Repeated runs differ by a few percentage points. |
| C3 | Requantization elimination is essential for the decode speed (Sec. 6.4). | E3, two runs of HACK with and without `--hack-option requant_elimination=0`, 7 minutes | Without requantization elimination the TPOT is more than five times higher. |
| C4 | Attention on the codes is faster than BF16 attention on long KV caches, and faster than a FlashAttention-style kernel that dequantizes the same codes (Sec. 6.7). | E4, 10 minutes | On an RTX A6000: about 2x faster than BF16 at 4K tokens and about 3x from 32K tokens; 1.2x to 1.3x faster than the kernel with fused dequantization from 16K tokens. |
| C5 | HACK keeps the accuracy of the BF16 KV cache on IMDb, arXiv, HumanEval and GSM8K (Sec. 6.3). | E1, quick check, 30 minutes | Every HACK row is within a few points of the baseline: at most 7 points with 30 examples per task, where one example is 3.3 points, and at most 4 points with 100 examples. It was within 2 points in our runs with 100 examples. |

### E1. Accuracy (Sec. 6.3)

```bash
python benchmarks/accuracy/run.py --model Qwen/Qwen3-8B --tasks imdb humaneval gsm8k \
    --methods baseline cachegen kvquant hack --partition-sizes 32 64 128 --limit 100 --output results/accuracy
python benchmarks/accuracy/run.py --model Qwen/Qwen3-8B --tasks arxiv --limit 20 \
    --methods baseline cachegen kvquant hack --partition-sizes 32 64 128 --output results/accuracy
python benchmarks/accuracy/table.py results/accuracy
```

Quick check of the baseline and HACK (Pi = 64), 30 minutes on one RTX A6000:

```bash
python benchmarks/accuracy/run.py --model Qwen/Qwen3-8B --tasks imdb humaneval gsm8k --limit 30 \
    --methods baseline hack --partition-sizes 64 --output results/accuracy_quick
python benchmarks/accuracy/run.py --model Qwen/Qwen3-8B --tasks arxiv --limit 20 \
    --methods baseline hack --partition-sizes 64 --output results/accuracy_quick
python benchmarks/accuracy/table.py results/accuracy_quick
```

One GPU that holds the model in BF16 plus the KV of the longest prompt: 20 GB for an 8B model with `imdb`,
`humaneval` and `gsm8k`; for the long-context tasks `arxiv` and `cocktail` (prompts of up to 16K tokens) we recommend
a 40 GB GPU. `--limit` sets the number of examples per task; the datasets are downloaded from the Hugging Face Hub.
Metrics follow the paper: exact match (IMDb, Cocktail, GSM8K), ROUGE-1 (arXiv), edit similarity (HumanEval).

Runtime on one RTX A6000 with `--limit 30`: 18 minutes for `imdb`, `humaneval` and `gsm8k` with the baseline and HACK
(Pi = 64), 47 minutes with the two comparison methods. `arxiv` with 20 examples takes 5 minutes per row for the
baseline and HACK, 16 minutes per row for the comparison methods. On one RTX 4000 Ada with `--limit 100`: about
25 minutes per GSM8K row for the baseline and HACK, 1 to 1.5 hours for the two comparison methods, 10 to 25 minutes
per HumanEval row, under 2 minutes per IMDb row.

Measured with Qwen3-8B, 100 examples per task (20 for arXiv):

| Task | baseline | HACK Pi=32 | HACK Pi=64 | HACK Pi=128 | CacheGen-style | KVQuant-style |
|---|---|---|---|---|---|---|
| IMDb (exact match) | 64.0 | 64.0 | 64.0 | 64.0 | 64.0 | 62.0 |
| arXiv (ROUGE-1) | 45.3 | 46.9 | 45.4 | 45.9 | 45.7 | 43.3 |
| HumanEval (edit similarity) | 43.8 | 43.5 | 44.5 | 44.1 | 44.9 | 42.9 |
| GSM8K (exact match) | 90.0 | 88.0 | 89.0 | 89.0 | 89.0 | 92.0 |

With 100 examples the standard error of an accuracy is 3 to 5 points.

### E2. KV size (Sec. 6.2)

```bash
python benchmarks/kv_size/kv_size.py --model Qwen/Qwen3-8B --prompt-tokens 4096 --methods baseline cachegen kvquant hack
```

One GPU (20 GB for an 8B model), about 5 minutes. Prints the bytes of the stored KV cache and of the KV that is
transferred from the prefill instance to the decode instance, absolute and relative to BF16. Measured for
Qwen3-8B and a 4096-token prompt: BF16 576 MiB; HACK 121 / 102 / 101 MiB for Pi = 32 / 64 / 128 (20.9% / 17.7% /
17.5% of BF16); KVQuant-style 84 MiB (14.6%); CacheGen-style 86 MiB transferred (14.9%) and 158 MiB stored (27.3%).

### E3. Disaggregated serving (Sec. 6.2, 6.4, 6.5)

See `benchmarks/e2e/README.md` for all options. One prefill instance and one decode instance on two GPUs;
`--bandwidth-gbps` limits the rate of the KV transfer to emulate the network of the prefill instances in Table 2
of the paper. A run starts its own vLLM servers and stops them when it ends, when it is interrupted, or when its
terminal is closed; it refuses to start while the ports of an earlier run are still in use.

Long-context workload (prompts of 12K to 16K tokens as in the default setting of the paper, 256 to 512 output
tokens; two GPUs with 40 GB or more for an 8B model; 3 to 5 minutes per run):

```bash
python benchmarks/e2e/workload.py --output workload_long.jsonl --num-requests 24 --prompt-len 12000 16000 \
    --output-len 256 512 --vocab-size 100000
for method in baseline hack; do for rate in 0.15 0.3; do
  python benchmarks/e2e/run_experiment.py --method $method --model Qwen/Qwen3-8B --prefill-gpus 0 --decode-gpus 1 \
      --max-model-len 20480 --max-num-seqs 8 --gpu-memory-utilization 0.92 --bandwidth-gbps 10 \
      --workload workload_long.jsonl --rate $rate --warmup-requests 3 --run-dir runs_long/${method}_r${rate}
done; done
python benchmarks/e2e/plot.py --output-dir figures_long runs_long/*
```

Measured on two RTX A6000 GPUs (48 GB), 10 Gbps:

| Requests/s | Method | JCT (s) | P99 JCT (s) | TTFT (s) | TPOT (ms) | KV transfer (s) | KV per request (MiB) | Peak KV-cache usage, decode |
|---|---|---|---|---|---|---|---|---|
| 0.15 | baseline | 17.11 | 24.49 | 5.54 | 31.1 | 2.17 | 1954 | 29% |
| 0.15 | HACK | 15.13 | 19.08 | 3.33 | 31.7 | 0.40 | 326 | 5% |
| 0.30 | baseline | 24.13 | 39.26 | 9.06 | 40.4 | 3.54 | 1954 | 60% |
| 0.30 | HACK | 16.99 | 21.97 | 4.78 | 32.8 | 0.44 | 326 | 7% |

HACK lowers the average JCT by 12% and 30%, the P99 JCT by 22% and 44% and the TTFT by 40% and 47% at the two
request rates; at 0.3 requests/s, where several long requests decode together, it lowers the TPOT by 19%.

Short-prompt workload (2K to 6K prompt tokens, 64 to 128 output tokens; 2 to 5 minutes per run). It shows the effect
of the KV-cache memory of the decode instance and is meant for two GPUs with about 20 GB:

```bash
python benchmarks/e2e/workload.py --output workload.jsonl --num-requests 40 --prompt-len 2000 6000 \
    --output-len 64 128 --vocab-size 100000
for method in baseline hack; do for rate in 0.25 0.5 0.75; do
  python benchmarks/e2e/run_experiment.py --method $method --model Qwen/Qwen3-8B --prefill-gpus 0 --decode-gpus 1 \
      --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --bandwidth-gbps 10 \
      --workload workload.jsonl --rate $rate --warmup-requests 6 --run-dir runs/${method}_r${rate}
done; done
python benchmarks/e2e/plot.py --output-dir figures runs/*
```

Measured on two RTX 4000 Ada GPUs (20 GB), 10 Gbps, where the KV cache of the baseline fills the memory of the
decode instance:

| Requests/s | Method | JCT (s) | P99 JCT (s) | TTFT (s) | TPOT (ms) | KV transfer (s) | KV per request (MiB) | Peak KV-cache usage, decode |
|---|---|---|---|---|---|---|---|---|
| 0.25 | baseline | 6.55 | 9.11 | 1.76 | 49.7 | 0.54 | 542 | 95% |
| 0.25 | HACK | 6.10 | 8.38 | 1.26 | 50.2 | 0.13 | 94 | 23% |
| 0.50 | baseline | 8.69 | 12.92 | 3.73 | 51.5 | 0.71 | 542 | 100% |
| 0.50 | HACK | 6.53 | 8.80 | 1.58 | 51.5 | 0.16 | 94 | 32% |
| 0.75 | baseline | 17.92 | 25.53 | 12.92 | 52.0 | 0.75 | 542 | 100% |
| 0.75 | HACK | 7.09 | 10.16 | 2.01 | 52.8 | 0.18 | 94 | 49% |

HACK lowers the average JCT by 7%, 25% and 60% and the TTFT by 29%, 58% and 84% at the three request rates.

The partition size, the two ablation switches and the bandwidth are set with `--hack-option partition_size=32`,
`--hack-option summation_elimination=0`, `--hack-option requant_elimination=0` and `--bandwidth-gbps`. Claim C3 needs
two runs of HACK on the short-prompt workload, with and without requantization elimination. We measured a TPOT of
31 to 52 ms with it, depending on the GPU, and above 300 ms without it:

```bash
python benchmarks/e2e/workload.py --output workload.jsonl --num-requests 40 --prompt-len 2000 6000 \
    --output-len 64 128 --vocab-size 100000
python benchmarks/e2e/run_experiment.py --method hack --model Qwen/Qwen3-8B --prefill-gpus 0 --decode-gpus 1 \
    --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --bandwidth-gbps 10 \
    --workload workload.jsonl --rate 0.5 --warmup-requests 6 --run-dir runs_c3/hack
python benchmarks/e2e/run_experiment.py --method hack --model Qwen/Qwen3-8B --prefill-gpus 0 --decode-gpus 1 \
    --max-model-len 8192 --max-num-seqs 16 --gpu-memory-utilization 0.92 --bandwidth-gbps 10 \
    --workload workload.jsonl --rate 0.5 --warmup-requests 6 --hack-option requant_elimination=0 \
    --run-dir runs_c3/hack_no_rqe
```

### E4. Attention-kernel microbenchmark (Sec. 6.7)

```bash
python benchmarks/microbench/attn_microbench.py --methods hack bf16 dequant --output attn_microbench.json
python benchmarks/microbench/plot_attn_microbench.py attn_microbench.json
```

See `benchmarks/microbench/README.md`. One GPU with 2 GB of free memory; about 10 minutes including the search for
the launch configurations. `bf16` is a FlashAttention-style kernel on a BF16 KV cache, `dequant` the same kernel with
fused dequantization of 2-bit codes. One step at 128K tokens reads 84 MiB of KV with HACK and 512 MiB with BF16.
Measured time per decode step in microseconds:

| KV length | RTX A6000: HACK | BF16 | dequant | RTX 4000 Ada: HACK | BF16 | dequant | H200: HACK | BF16 | dequant |
|---|---|---|---|---|---|---|---|---|---|
| 1K | 10.0 | 7.5 | 6.6 | 9.0 | 9.0 | 9.9 | 7.6 | 6.0 | 6.3 |
| 4K | 15.5 | 29.4 | 14.8 | 18.2 | 17.7 | 17.5 | 10.2 | 8.6 | 8.9 |
| 16K | 37.5 | 100.4 | 45.9 | 41.4 | 206.8 | 50.9 | 21.5 | 25.5 | 20.2 |
| 32K | 64.9 | 195.1 | 84.8 | 75.0 | 407.3 | 94.6 | 34.8 | 43.2 | 35.5 |
| 64K | 121.7 | 383.2 | 155.4 | 168.5 | 808.3 | 209.8 | 67.8 | 75.8 | 66.2 |
| 128K | 238.4 | 760.5 | 296.2 | 347.7 | 1610.3 | 425.2 | 127.1 | 138.0 | 125.2 |

## Citation

Zeyu Zhang, Haiying Shen, Shay Vargaftik, Ran Ben Basat, Michael Mitzenmacher, and Minlan Yu. HACK: Homomorphic
Acceleration via Compression of the Key-Value Cache for Disaggregated LLM Inference. In 2026 ACM SIGOPS Annual
Technical Conference (ATC '26).

## License

Apache License 2.0, see `LICENSE`.

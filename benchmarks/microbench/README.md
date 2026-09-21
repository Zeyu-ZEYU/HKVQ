# Attention-kernel microbenchmark

Measures the decode attention kernel in isolation: one attention layer with the head
configuration of Llama-3.1 70B (64 query heads, 8 KV heads, head dimension 128), batch 1,
one query token, KV lengths from 1K to 128K tokens.

## Methods

| name | kernel | K and V |
|---|---|---|
| `hack` | `attn_decode` of `hack.kernels` | 2-bit codes; both matrix multiplications run on INT8 codes |
| `bf16` | FlashAttention-2-style Triton kernel | BF16 |
| `dequant` | the same FlashAttention-2-style kernel | 2-bit codes that the kernel dequantizes to BF16 at every step |

All three kernels share the tiling over the sequence, the sequence-level splits, the kernel
that merges the splits, and the search over launch configurations.

## What is measured

A measurement decodes `--steps` consecutive tokens (default 8): step `i` attends to
`L + i` tokens. The steps are captured
in one CUDA graph; after a warm-up the graph is replayed `--replays` times, and the median
of `--rounds` rounds divided by the number of steps is reported. For every method and KV
length the script reports, per decode step:

- `T_us`: latency of the attention kernel,
- `D_us`: time of a load-only kernel that reads exactly the tensors the attention kernel
  reads and folds them into a checksum,
- `C_us`: time of the attention kernel when every read is wrapped onto the first four
  tiles, so that the arithmetic is unchanged and almost nothing is fetched from device memory,
- `kv_bytes_per_step`: size of the K and V tensors that one step reads.

Each of the three kernels is launched with its own best configuration (number of
sequence splits, `num_warps`, `num_stages`, tile length), found by a coordinate search
before the measurement; `--no-tune` uses the defaults instead. The chosen configurations
are stored next to the timings and in `--tuning-file` (default `attn_microbench_tuning.json`);
a later run reads that file and only searches for what is missing. The same file can be
passed to `hack.kernels.tuning.load` so that `hack_attention` launches with these configurations.

## Running

```bash
export PYTHONPATH=$PWD                      # repository root
python benchmarks/microbench/attn_microbench.py --output attn_microbench.json
python benchmarks/microbench/plot_attn_microbench.py attn_microbench.json --output attn_microbench.pdf
```

Useful options of `attn_microbench.py`:

- `--lengths 1024 4096 ...` KV lengths,
- `--methods hack bf16 dequant` methods (default `hack bf16`),
- `--partition-size 64` partition size of HACK and group size of the storage-only codes,
- `--rounding stochastic|nearest` rounding of the INT8 query and attention-weight codes (default: that of `HackConfig`),
- `--no-tune`, `--warmup`, `--replays`, `--rounds`.

The first tuned run compiles every launch configuration once; later runs reuse the Triton cache.
The longest default length needs about 4 GB of device memory.

## Output

`attn_microbench.json` holds the environment (GPU, library versions, settings) and one
record per method and KV length with `T_us`, `D_us`, `C_us`, `kv_bytes_per_step` and the
three launch configurations. A CSV file with the same name holds the timings as a table.

`plot_attn_microbench.py` reads only that JSON file. For every KV length and method it
draws two bottom-aligned bars, data movement `D` (wide) and compute `C` (narrow, in
front), and a marker at the latency `T`. The label above a group is the latency of the
BF16 kernel divided by the latency of HACK. `--log` switches to a logarithmic time axis.

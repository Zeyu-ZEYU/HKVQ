"""Tests of the vLLM integration.

Layer level (one GPU, no engine): a small scheduler drives the paged attention layer with
block tables and slot mappings like vLLM does, and every sequence is compared with a private
contiguous `HackLayerCache` and the reference attention.

Engine level (one GPU, a small model; set HACK_TEST_MODEL to change it):
  (a) with 8-bit codes greedy decoding follows the stock backend,
  (b) the vLLM path follows the Hugging Face path of `hack.hf` for the same settings,
  (c) many concurrent requests of different lengths give the same result as one at a time.
Two greedy generations are taken to agree when they are identical up to the first position
at which the reference distribution has a near-tie between the two chosen tokens.
"""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from hack import HackConfig  # noqa: E402
from hack.attention_ref import hack_attention  # noqa: E402
from hack.cache import HackLayerCache  # noqa: E402

MODEL = os.environ.get("HACK_TEST_MODEL", "Qwen/Qwen3-0.6B")
NEAR_TIE_NATS = 1.0
PROMPTS = [
    "The capital of France is",
    "Write a short poem about the sea.\n",
    "Explain in two sentences why the sky is blue.",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Once upon a time, in a small village by the mountains, there lived an old clockmaker who " * 6,
    "Q: What is 17 times 23? A:",
]


# ---------------------------------------------------------------- layer level


class PagedSequences:
    """Sequences on a paged cache of one layer, scheduled step by step."""

    def __init__(
        self,
        config: HackConfig,
        attention: str = "reference",
        decode: str = "batched",
        heads: int = 8,
        kv_heads: int = 2,
        head_dim: int = 64,
        num_blocks: int = 96,
        slots: int = 8,
    ):
        from hack.vllm_plugin.layer import PagedAttentionLayer
        from hack.vllm_plugin.layout import PageLayout
        from hack.vllm_plugin.plan import StepPlanner
        from hack.vllm_plugin.settings import PluginSettings
        from hack.vllm_plugin.slots import SlotPool
        from hack.vllm_plugin.store import LayerStore

        self.config, self.heads, self.kv_heads, self.head_dim = config, heads, kv_heads, head_dim
        self.device = torch.device("cuda")
        layout = PageLayout(config, kv_heads, head_dim)
        shape = (num_blocks, 1, config.partition_size, layout.bytes_per_token_slot)
        self.kv_cache = torch.zeros(shape, dtype=torch.uint8, device=self.device)
        self.store = LayerStore(layout, slots, torch.bfloat16, self.device)
        self.store.bind(self.kv_cache)
        self.open_slots = SlotPool(slots, self.device, debug_checks=True)
        self.open_slots.bind(num_blocks)
        self.sink_slots = SlotPool(slots, self.device, debug_checks=True) if config.sink_tokens else None
        if self.sink_slots is not None:
            self.sink_slots.bind(num_blocks)
        planner = StepPlanner(config, self.open_slots, self.sink_slots, self.device)
        settings = PluginSettings(config=config, attention=attention, decode=decode, debug_checks=True)
        self.planner = planner
        self.layer = PagedAttentionLayer(self.store, settings, 1.0 / head_dim**0.5)
        self.free = [int(b) for b in np.random.default_rng(0).permutation(np.arange(1, num_blocks))]
        self.sequences: dict[int, tuple[list[int], int]] = {}

    def random(self, tokens: int, heads: int) -> torch.Tensor:
        return torch.randn(tokens, heads, self.head_dim, device=self.device, dtype=torch.bfloat16)

    def step(self, work: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
        block = self.config.partition_size
        starts, totals, rows = [0], [], []
        for sequence, q, _, _ in work:
            blocks, length = self.sequences.setdefault(sequence, ([], 0))
            new_length = length + q.shape[0]
            while len(blocks) * block < new_length:
                blocks.append(self.free.pop())
            self.sequences[sequence] = (blocks, new_length)
            starts.append(starts[-1] + q.shape[0])
            totals.append(new_length)
            rows.append(blocks)
        width = max(len(row) for row in rows)
        table = torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.int32, device=self.device)
        q, k, v = (torch.cat([item[i] for item in work]) for i in (1, 2, 3))
        plan = self.planner.plan(np.array(starts), np.array(totals), table, q.shape[0])
        out = torch.zeros_like(q)
        self.layer.forward(plan, q, k, v, out)
        return [out[a:b] for a, b in zip(starts[:-1], starts[1:])]

    def finish(self, sequence: int) -> None:
        blocks, _ = self.sequences.pop(sequence)
        self.free = blocks + self.free

    def gathered_state(self, sequence: int) -> dict[str, torch.Tensor]:
        blocks, length = self.sequences[sequence]
        table = torch.tensor(blocks, device=self.device)
        after_sinks = max(length - self.config.sink_tokens, 0)
        no_slot = torch.zeros(1, dtype=torch.long, device=self.device)
        open_slot = no_slot
        if after_sinks % self.config.partition_size:
            open_slot = self.open_slots.lookup(table[after_sinks // self.config.partition_size].reshape(1))
        sink = self.sink_slots.lookup(table[:1]) if self.sink_slots is not None else no_slot
        return self.store.gather(table, length, open_slot, sink).state_dict()


def _head_major(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(0, 1).unsqueeze(0)


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return ((got.float() - want.float()).norm() / want.float().norm()).item()


def _run_schedule(paged: PagedSequences, reference_config: HackConfig, steps: int = 60):
    """Prefill four sequences, then decode them while sequences join and leave. Returns the worst
    relative error of any output and the reference caches of the sequences that are still running.
    Without sink tokens every sequence starts with at least one full block."""
    torch.manual_seed(0)
    reference: dict[int, HackLayerCache] = {}
    worst = 0.0

    def submit(work):
        nonlocal worst
        for (sequence, q, k, v), out in zip(work, paged.step(work)):
            cache = reference.setdefault(sequence, HackLayerCache(reference_config))
            cache.append(_head_major(k), _head_major(v))
            want = hack_attention(_head_major(q), cache)[0].transpose(0, 1)
            worst = max(worst, _relative_error(out, want))

    def tokens(sequence: int, count: int):
        q, k, v = (paged.random(count, heads) for heads in (paged.heads, paged.kv_heads, paged.kv_heads))
        return sequence, q, k, v

    extra = 0 if paged.config.sink_tokens else paged.config.partition_size
    live = [0, 1, 2, 3]
    submit([tokens(s, n + extra) for s, n in zip(live, (37, 64, 3, 129))])
    next_sequence = 10
    for step in range(steps):
        work = [tokens(s, 1) for s in live]
        if step % 9 == 4:
            work.append(tokens(next_sequence, 5 + 17 * (step % 4) + extra))
            live.append(next_sequence)
            next_sequence += 1
        if step % 9 == 7 and len(live) > 3:
            gone = live.pop(0)
            paged.finish(gone)
            reference.pop(gone)
            work = [item for item in work if item[0] != gone]
        submit(work)
    return worst, {s: reference[s] for s in live}


LAYER_CONFIGS = [
    HackConfig(quantized_prefill=True, partition_size=32, kv_bits=2),
    HackConfig(quantized_prefill=True, partition_size=32, kv_bits=8),
    HackConfig(quantized_prefill=True, partition_size=16, kv_bits=4, sink_tokens=0),
    HackConfig(quantized_prefill=True, partition_size=32, kv_bits=2, summation_elimination=False),
    HackConfig(quantized_prefill=True, partition_size=32, kv_bits=2, sink_tokens=0),
    HackConfig(quantized_prefill=True, partition_size=32, kv_bits=2, sink_tokens=70),
]


def _config_id(c: HackConfig) -> str:
    return f"pi{c.partition_size}-b{c.kv_bits}-se{int(c.summation_elimination)}-s{c.sink_tokens}"


@pytest.mark.parametrize("decode", ["batched", "loop"])
@pytest.mark.parametrize("config", LAYER_CONFIGS, ids=_config_id)
def test_paged_cache_matches_contiguous_reference(config, decode):
    paged = PagedSequences(config, decode=decode)
    worst, reference = _run_schedule(paged, config)
    assert worst < (1e-6 if decode == "loop" else 2e-3)
    for sequence, cache in reference.items():
        got = paged.gathered_state(sequence)
        for name, want in cache.state_dict().items():
            if name in ("k_sums", "v_sums") and not config.summation_elimination:
                continue
            assert torch.equal(got[name], want), (sequence, name)


def test_requantized_open_block_is_consistent_between_decode_paths():
    config = HackConfig(partition_size=32, kv_bits=8, requant_elimination=False)
    outputs = []
    for decode in ("batched", "loop"):
        torch.manual_seed(1)
        paged = PagedSequences(config, decode=decode)
        results = paged.step([(0, paged.random(40, 8), paged.random(40, 2), paged.random(40, 2))])
        for _ in range(30):
            results += paged.step([(0, paged.random(1, 8), paged.random(1, 2), paged.random(1, 2))])
        outputs.append(torch.cat(results))
    assert _relative_error(outputs[0], outputs[1]) < 2e-3


def test_page_is_about_six_times_smaller_than_bf16():
    from hack.vllm_plugin.layout import PageLayout

    layout = PageLayout(HackConfig(partition_size=64, kv_bits=2), num_kv_heads=8, head_dim=128)
    assert layout.page_bytes == sum(field.nbytes for field in layout.fields.values())
    assert layout.full_precision_page_bytes() / layout.page_bytes > 6.0


def test_buffer_pool_hands_out_the_smallest_free_buffer_that_fits():
    from hack.vllm_plugin.transport import PinnedBufferPool

    pool, unit = PinnedBufferPool(pin=False), PinnedBufferPool.GRANULARITY
    small, large = pool.take(10), pool.take(unit + 1)
    assert (small.numel(), large.numel()) == (unit, 2 * unit)
    pool.give(small)
    pool.give(large)
    assert pool.take(unit // 2) is small and pool.take(unit // 2) is large
    assert pool.take(1).numel() == 2 * unit  # nothing free: as large as the largest buffer so far


def test_rate_limiter_holds_its_rate_when_sending_takes_time():
    import time

    from hack.vllm_plugin.transport import RateLimiter

    limiter = RateLimiter(gbps=0.8)  # 100 MB/s: one chunk of 1 MB every 10 ms
    started = time.monotonic()
    for _ in range(30):
        limiter.acquire(1_000_000)
        time.sleep(0.004)  # the send itself
    elapsed = time.monotonic() - started
    assert 0.29 <= elapsed <= 0.40


def test_slot_pool_recycles_least_recently_used_and_keeps_pinned():
    from hack.vllm_plugin.slots import SlotPool

    device = torch.device("cuda")
    pool = SlotPool(4, device, debug_checks=True)
    pool.bind(32)
    blocks = torch.arange(1, 5, device=device)
    pool.begin_step()
    first = pool.allocate(blocks)
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    pool.pin("waiting", first[:1])
    for _ in range(3):
        pool.begin_step()
        pool.touch(blocks[1:3])
    pool.begin_step()
    pool.touch(blocks[1:3])
    fresh = pool.allocate(torch.tensor([9], device=device))
    assert fresh.tolist() == first[3:].tolist()
    assert pool.lookup(blocks[:1]).tolist() == first[:1].tolist()
    assert pool.block_slot[4].item() == -1


# --------------------------------------------------------------- engine level

_ENGINE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from vllm import LLM, SamplingParams
    spec = json.loads(sys.argv[1])
    kwargs = dict(model=spec["model"], dtype="bfloat16", enforce_eager=True, max_model_len=2048, max_num_seqs=16,
                  gpu_memory_utilization=0.5, enable_prefix_caching=False, enable_chunked_prefill=False,
                  max_num_batched_tokens=4096, seed=0)
    if spec["backend"]:
        kwargs["attention_config"] = {"backend": spec["backend"]}
        kwargs["additional_config"] = {"hack": spec["hack"]}
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=spec["max_tokens"], logprobs=20)
    batches = [spec["prompts"]] if spec["together"] else [[p] for p in spec["prompts"]]
    results = []
    for batch in batches:
        for out in llm.generate(batch, params, use_tqdm=False):
            completion = out.outputs[0]
            top = [{str(t): lp.logprob for t, lp in step.items()} for step in completion.logprobs]
            results.append({"token_ids": list(completion.token_ids), "logprobs": top})
    cache = llm.llm_engine.vllm_config.cache_config
    print("RESULT" + json.dumps({"results": results, "cache_tokens": cache.num_gpu_blocks * cache.block_size}))
    """
)


def _require_plugin() -> None:
    pytest.importorskip("vllm")
    from importlib.metadata import entry_points

    if not any(entry.value.startswith("hack.vllm_plugin") for entry in entry_points(group="vllm.general_plugins")):
        pytest.skip("the vLLM plugin is not installed: run `pip install -e .`")


def _generate(backend: str | None, hack: dict | None = None, together: bool = True, max_tokens: int = 32) -> dict:
    _require_plugin()
    spec = {"model": MODEL, "backend": backend, "hack": hack or {}, "together": together,
            "prompts": PROMPTS, "max_tokens": max_tokens}  # fmt: skip
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    command = [sys.executable, "-c", _ENGINE_SCRIPT, json.dumps(spec)]
    done = subprocess.run(command, env=env, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, done.stderr[-3000:]
    return json.loads(lines[-1][len("RESULT") :])


def _agree_up_to_near_ties(reference: dict, other_tokens: list[int]) -> bool:
    """`reference` holds token ids and the top log-probabilities of every generated position."""
    for position, (want, got) in enumerate(zip(reference["token_ids"], other_tokens)):
        if want == got:
            continue
        top = reference["logprobs"][position]
        return str(got) in top and top[str(want)] - top[str(got)] <= NEAR_TIE_NATS
    return True


@pytest.fixture(scope="module")
def stock() -> dict:
    return _generate(None)


@pytest.fixture(scope="module")
def hack_8bit() -> dict:
    return _generate("CUSTOM", {"kv_bits": 8, "stochastic": False, "debug_checks": True})


def test_engine_8bit_follows_stock_backend(stock, hack_8bit):
    agreeing = [_agree_up_to_near_ties(a, b["token_ids"]) for a, b in zip(stock["results"], hack_8bit["results"])]
    assert all(agreeing), agreeing
    first_tokens = [a["token_ids"][0] == b["token_ids"][0] for a, b in zip(stock["results"], hack_8bit["results"])]
    assert all(first_tokens)


def test_engine_2bit_cache_holds_more_tokens(stock):
    hack = _generate("CUSTOM", {"kv_bits": 2, "debug_checks": True}, max_tokens=4)
    assert hack["cache_tokens"] > 5 * stock["cache_tokens"]


def test_engine_follows_hf_path(hack_8bit):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hack.hf import HackCache, enable_hack

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    enable_hack(model)
    agreeing = []
    with torch.no_grad():
        for prompt, reference in zip(PROMPTS, hack_8bit["results"]):
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            cache = HackCache(model.config, HackConfig(partition_size=64, kv_bits=8, stochastic=False))
            length = len(reference["token_ids"])
            out = model.generate(**inputs, max_new_tokens=length, do_sample=False, past_key_values=cache)
            agreeing.append(_agree_up_to_near_ties(reference, out[0, inputs["input_ids"].shape[1] :].tolist()))
    del model
    torch.cuda.empty_cache()
    assert all(agreeing), agreeing


def test_engine_concurrent_requests_do_not_interfere(hack_8bit):
    alone = _generate("CUSTOM", {"kv_bits": 8, "stochastic": False, "debug_checks": True}, together=False)
    agreeing = [_agree_up_to_near_ties(a, b["token_ids"]) for a, b in zip(hack_8bit["results"], alone["results"])]
    assert all(agreeing), agreeing

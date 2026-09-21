"""Attention of one layer on the paged cache: decode rows in one batch, prefill rows one by one."""

from dataclasses import replace

import torch

from hack.cache import HackLayerCache
from hack.vllm_plugin.plan import PrefillRow, StepPlan
from hack.vllm_plugin.runtime import resolve_attention, resolve_paged_decode
from hack.vllm_plugin.settings import PluginSettings
from hack.vllm_plugin.store import LayerStore, requantized


def _head_major(x: torch.Tensor) -> torch.Tensor:
    """[tokens, heads, head_dim] -> [1, heads, tokens, head_dim]"""
    return x.transpose(0, 1).unsqueeze(0)


class PagedAttentionLayer:
    def __init__(self, store: LayerStore, settings: PluginSettings, scale: float):
        self.store = store
        self.settings = settings
        self.scale = scale
        self.cache_config = replace(settings.config, requant_elimination=True)
        self.attention = resolve_attention(settings.attention)
        self.paged_decode = resolve_paged_decode(settings.attention)

    def forward(
        self, plan: StepPlan, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, output: torch.Tensor
    ) -> None:
        """query/output: [tokens, heads, head_dim]; key/value: [tokens, kv_heads, head_dim]."""
        if plan.decode is not None:
            self._decode(plan, query, key, value, output)
        for row in plan.prefills:
            self._prefill(row, query, key, value, output)

    def _decode(self, plan: StepPlan, query, key, value, output) -> None:
        batch = plan.decode
        index = batch.token_index
        self.store.write_decode(key[index], value[index], batch)
        if self.settings.decode == "batched":
            output[index] = self.paged_decode(query[index], self.store, batch, self.scale)
            return
        for i in range(batch.size):
            cache = self.store.gather(
                batch.block_table[i], int(batch.totals_cpu[i]), batch.open_slot[i : i + 1], batch.sink_slot[i : i + 1]
            )
            q = query[index[i : i + 1]].unsqueeze(2)
            output[index[i : i + 1]] = self.attention(q, cache, scaling=self.scale, causal=True).squeeze(2)

    def _prefill(self, row: PrefillRow, query, key, value, output) -> None:
        tokens = slice(row.start, row.stop)
        if row.context:
            cache = self.store.gather(row.blocks, row.context, row.old_open_slot, row.sink_slot, self.cache_config)
        else:
            cache = HackLayerCache(self.cache_config, self.store.layout.meta_dtype)
        cache.append(_head_major(key[tokens]), _head_major(value[tokens]))
        if not self.settings.config.requant_elimination and cache.k_open is not None:
            self._requantize_open(cache)
        q = _head_major(query[tokens])
        if row.context or self.settings.config.quantized_prefill:
            result = self.attention(q, cache, scaling=self.scale, causal=True)
        else:
            k, v = _head_major(key[tokens]), _head_major(value[tokens])
            result = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale, enable_gqa=k.shape[1] != q.shape[1]
            )
        self.store.scatter(cache, row)
        output[tokens] = result[0].transpose(0, 1)

    def _requantize_open(self, cache: HackLayerCache) -> None:
        cfg, layout = self.settings.config, self.store.layout
        cache.k_open = requantized(cache.k_open, 2, cache.k_open.shape[2], cfg, layout.meta_dtype)
        cache.v_open = requantized(cache.v_open, -1, layout.channel_part, cfg, layout.meta_dtype)

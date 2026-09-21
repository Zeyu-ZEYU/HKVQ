"""Batched decode attention on the paged cache (PyTorch); same result as `hack.attention_ref.hack_attention` per row."""

import torch

from hack.attention_ref import _quantized_scores, _quantized_values
from hack.cache import HackLayerCache
from hack.vllm_plugin.plan import DecodeBatch
from hack.vllm_plugin.store import PER_BLOCK_FIELDS, PER_TOKEN_FIELDS, LayerStore

MAX_GATHERED_ELEMENTS = 1 << 27


def _expand_heads(x: torch.Tensor, groups: int) -> torch.Tensor:
    """[rows, kv_heads, ...] -> [rows, heads, ...]"""
    return x if groups == 1 else x.repeat_interleave(groups, dim=1)


def paged_decode_attention(q: torch.Tensor, store: LayerStore, batch: DecodeBatch, scaling: float) -> torch.Tensor:
    """q: [rows, heads, head_dim] -> [rows, heads, head_dim]"""
    tokens = batch.block_table.shape[1] * store.layout.block_size
    step = max(1, MAX_GATHERED_ELEMENTS // max(1, tokens * q.shape[1] * q.shape[2]))
    pieces = [
        _decode_rows(q, store, batch, slice(start, min(start + step, batch.size)), scaling)
        for start in range(0, batch.size, step)
    ]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _closed_blocks(store: LayerStore, table: torch.Tensor, keep: torch.Tensor) -> HackLayerCache:
    """The closed blocks of several rows as one cache; blocks outside `keep` contribute zero to P V."""
    state = {}
    for name in PER_TOKEN_FIELDS:
        if name in store.fields:
            pages = store.fields[name][table]
            if name in ("v_scale", "v_min"):
                pages = torch.where(keep.view(*keep.shape, 1, 1, 1), pages, 0)
            state[name] = pages.transpose(1, 2).flatten(2, 3)
    for name in PER_BLOCK_FIELDS:
        if name in store.fields:
            state[name] = store.fields[name][table].transpose(1, 2)
    cache = HackLayerCache(store.config, store.layout.meta_dtype)
    cache.load_state_dict(state, table.shape[1] * store.layout.block_size, store.layout.head_dim)
    return cache


def _decode_rows(q: torch.Tensor, store: LayerStore, batch: DecodeBatch, rows: slice, scaling: float) -> torch.Tensor:
    block, sinks = store.layout.block_size, store.config.sink_tokens
    query = q[rows].float().unsqueeze(2)
    groups = query.shape[1] // store.layout.num_kv_heads
    totals = batch.totals[rows]
    after_sinks = (totals - sinks).clamp_min(0)
    closed = after_sinks // block
    width = int(batch.max_closed)
    steps = torch.arange(max(sinks, width * block, block), device=q.device)

    scores, masks = [], []
    if sinks:
        sink_k = _expand_heads(store.sink_k[batch.sink_slot[rows]], groups).float()
        scores.append(query @ sink_k.transpose(-2, -1))
        masks.append(steps[:sinks].view(1, -1) < totals.view(-1, 1))
    if width:
        table = batch.block_table[rows, :width]
        keep = torch.arange(width, device=q.device).view(1, -1) < closed.view(-1, 1)
        cache = _closed_blocks(store, table, keep)
        scores.append(_quantized_scores(query, cache, groups))
        masks.append(steps[: width * block].view(1, -1) < (closed * block).view(-1, 1))
    open_k = _expand_heads(store.open_k[batch.open_slot[rows]], groups).float()
    scores.append(query @ open_k.transpose(-2, -1))
    masks.append(steps[:block].view(1, -1) < (after_sinks - closed * block).view(-1, 1))

    valid = torch.cat(masks, dim=-1)[:, None, None, :]
    probs = torch.softmax((torch.cat(scores, dim=-1) * scaling).masked_fill(~valid, float("-inf")), dim=-1)
    open_v = _expand_heads(store.open_v[batch.open_slot[rows]], groups).float()
    result = probs[..., -block:] @ open_v
    if sinks:
        result = result + probs[..., :sinks] @ _expand_heads(store.sink_v[batch.sink_slot[rows]], groups).float()
    if width:
        result = result + _quantized_values(probs[..., sinks : sinks + width * block], cache, groups)
    return result.squeeze(2).to(q.dtype)

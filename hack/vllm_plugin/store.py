"""Storage of one attention layer: typed views of the paged cache plus the 16-bit side pools."""

import torch

from hack.cache import HackLayerCache, quantize_partitions
from hack.config import HackConfig
from hack.quant import pack, partition_sums
from hack.vllm_plugin.layout import PageLayout
from hack.vllm_plugin.plan import DecodeBatch, PrefillRow

PER_TOKEN_FIELDS = ("k_codes", "k_sums", "v_codes", "v_scale", "v_min")
PER_BLOCK_FIELDS = ("k_scale", "k_min", "v_sums")


def requantized(x: torch.Tensor, dim: int, size: int, config: HackConfig, meta_dtype: torch.dtype) -> torch.Tensor:
    """`x` after one quantization round trip in partitions of `size` along `dim`."""
    dim = dim % x.dim()
    codes, scale, minimum = quantize_partitions(x, config.kv_bits, dim, size, config.stochastic, meta_dtype)
    shape = list(x.shape)
    shape[dim:dim + 1] = [x.shape[dim] // size, size]
    restored = codes.float().reshape(shape) * scale.float().unsqueeze(dim + 1) + minimum.float().unsqueeze(dim + 1)
    return restored.reshape(x.shape).to(x.dtype)


class LayerStore:
    def __init__(self, layout: PageLayout, num_side_slots: int, dtype: torch.dtype, device: torch.device):
        self.layout = layout
        self.config: HackConfig = layout.config
        self.dtype = dtype
        heads, head_dim, sinks = layout.num_kv_heads, layout.head_dim, layout.config.sink_tokens
        open_shape = (num_side_slots + 1, heads, layout.block_size, head_dim)
        sink_shape = (num_side_slots + 1, heads, sinks, head_dim)
        self.open_k = torch.zeros(open_shape, dtype=dtype, device=device)
        self.open_v = torch.zeros(open_shape, dtype=dtype, device=device)
        self.sink_k = torch.zeros(sink_shape, dtype=dtype, device=device) if sinks else None
        self.sink_v = torch.zeros(sink_shape, dtype=dtype, device=device) if sinks else None
        self.pages: torch.Tensor | None = None
        self.fields: dict[str, torch.Tensor] = {}
        self._bound_to: tuple[int, int] | None = None
        self.kernel_layout_checked = False

    def side_pool_bytes(self) -> int:
        pools = [p for p in (self.open_k, self.open_v, self.sink_k, self.sink_v) if p is not None]
        return sum(p.numel() * p.element_size() for p in pools)

    def bind(self, kv_cache: torch.Tensor) -> None:
        key = (kv_cache.data_ptr(), kv_cache.shape[0])
        if key != self._bound_to:
            self.pages = self.layout.as_pages(kv_cache)
            self.fields = self.layout.views(self.pages)
            self._bound_to = key
            self.kernel_layout_checked = False

    # ------------------------------------------------------------------ writes

    def _close_blocks(self, k: torch.Tensor, v: torch.Tensor, pages: torch.Tensor) -> None:
        """Quantize full blocks `k`, `v` [blocks, heads, block_size, head_dim] into `pages`."""
        cfg, layout = self.config, self.layout
        block, part = layout.block_size, layout.channel_part
        codes, scale, minimum = quantize_partitions(k, cfg.kv_bits, 2, block, cfg.stochastic, layout.meta_dtype)
        self.fields["k_codes"][pages] = pack(codes, cfg.kv_bits)
        self.fields["k_scale"][pages] = scale.squeeze(2)
        self.fields["k_min"][pages] = minimum.squeeze(2)
        if cfg.summation_elimination:
            self.fields["k_sums"][pages] = partition_sums(codes, -1, part, cfg.kv_bits)
        codes, scale, minimum = quantize_partitions(v, cfg.kv_bits, -1, part, cfg.stochastic, layout.meta_dtype)
        self.fields["v_codes"][pages] = pack(codes, cfg.kv_bits)
        self.fields["v_scale"][pages] = scale
        self.fields["v_min"][pages] = minimum
        if cfg.summation_elimination:
            self.fields["v_sums"][pages] = partition_sums(codes, 2, block, cfg.kv_bits).squeeze(2)

    def requantize_open(self, slots: torch.Tensor, length: int) -> None:
        """Store the first `length` tokens of open blocks as dequantized codes (requantization elimination off)."""
        cfg, layout = self.config, self.layout
        self.open_k[slots, :, :length] = requantized(self.open_k[slots, :, :length], 2, length, cfg, layout.meta_dtype)
        self.open_v[slots, :, :length] = requantized(
            self.open_v[slots, :, :length], -1, layout.channel_part, cfg, layout.meta_dtype
        )

    def write_decode(self, k: torch.Tensor, v: torch.Tensor, batch: DecodeBatch) -> None:
        """Append one token per decode row; `k` and `v` are [rows, heads, head_dim]."""
        if batch.sink_rows.numel():
            slots = batch.sink_slot[batch.sink_rows]
            self.sink_k[slots, :, batch.sink_pos] = k[batch.sink_rows]
            self.sink_v[slots, :, batch.sink_pos] = v[batch.sink_rows]
        if batch.quant_rows.numel():
            slots = batch.open_slot[batch.quant_rows]
            self.open_k[slots, :, batch.open_offset] = k[batch.quant_rows]
            self.open_v[slots, :, batch.open_offset] = v[batch.quant_rows]
        if batch.seal_rows.numel():
            slots = batch.open_slot[batch.seal_rows]
            self._close_blocks(self.open_k[slots], self.open_v[slots], batch.seal_blocks)
        for length, rows in batch.requant_groups:
            self.requantize_open(batch.open_slot[rows], length)

    def scatter(self, cache: HackLayerCache, row: PrefillRow) -> None:
        """Store what `cache` gained for the tokens [row.context, row.total)."""
        block, sinks = self.layout.block_size, self.config.sink_tokens
        first_sink, last_sink = min(row.context, sinks), min(row.total, sinks)
        if last_sink > first_sink:
            self.sink_k[row.sink_slot, :, first_sink:last_sink] = cache.sink_k[0, :, first_sink:last_sink]
            self.sink_v[row.sink_slot, :, first_sink:last_sink] = cache.sink_v[0, :, first_sink:last_sink]
        first_block = max(row.context - sinks, 0) // block
        last_block = max(row.total - sinks, 0) // block
        if last_block > first_block:
            pages = row.blocks[first_block:last_block]
            for name in PER_TOKEN_FIELDS:
                if name in self.fields:
                    tokens = getattr(cache, name).view()[0, :, first_block * block : last_block * block]
                    self.fields[name][pages] = tokens.unflatten(1, (last_block - first_block, block)).transpose(0, 1)
            for name in PER_BLOCK_FIELDS:
                if name in self.fields:
                    self.fields[name][pages] = getattr(cache, name).view()[0, :, first_block:last_block].transpose(0, 1)
        if cache.k_open is not None:
            length = cache.k_open.shape[2]
            self.open_k[row.open_slot, :, :length] = cache.k_open[0]
            self.open_v[row.open_slot, :, :length] = cache.v_open[0]

    # ------------------------------------------------------------------- reads

    def gather(
        self,
        blocks: torch.Tensor,
        num_tokens: int,
        open_slot: torch.Tensor,
        sink_slot: torch.Tensor,
        config: HackConfig | None = None,
    ) -> HackLayerCache:
        """Rebuild the contiguous cache of one sequence that holds `num_tokens` tokens."""
        block = self.layout.block_size
        sinks = min(self.config.sink_tokens, num_tokens)
        closed = (num_tokens - sinks) // block
        open_tokens = num_tokens - sinks - closed * block
        state: dict[str, torch.Tensor] = {}
        if sinks:
            state["sink_k"] = self.sink_k[sink_slot.reshape(1), :, :sinks]
            state["sink_v"] = self.sink_v[sink_slot.reshape(1), :, :sinks]
        if closed:
            pages = blocks[:closed]
            for name in PER_TOKEN_FIELDS:
                if name in self.fields:
                    state[name] = self.fields[name][pages].transpose(0, 1).flatten(1, 2).unsqueeze(0).contiguous()
            for name in PER_BLOCK_FIELDS:
                if name in self.fields:
                    state[name] = self.fields[name][pages].transpose(0, 1).unsqueeze(0).contiguous()
        if open_tokens:
            state["k_open"] = self.open_k[open_slot.reshape(1), :, :open_tokens].contiguous()
            state["v_open"] = self.open_v[open_slot.reshape(1), :, :open_tokens].contiguous()
        cache = HackLayerCache(config or self.config, self.layout.meta_dtype)
        cache.load_state_dict(state, num_tokens, self.layout.head_dim)
        return cache

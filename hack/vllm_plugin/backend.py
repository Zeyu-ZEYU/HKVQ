"""vLLM attention backend that keeps the KV cache as HACK codes and attends on the codes.

The backend is registered as `AttentionBackendEnum.CUSTOM`; select it with
`--attention-backend CUSTOM`. The vLLM block size equals the partition size, and the
page size reported to vLLM is the true size of a quantized page, so the block budget of
the engine reflects the memory saving.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from hack.vllm_plugin.layer import PagedAttentionLayer
from hack.vllm_plugin.layout import PageLayout
from hack.vllm_plugin.plan import StepPlan
from hack.vllm_plugin.runtime import ensure_runtime, get_runtime
from hack.vllm_plugin.settings import get_settings
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayout

BACKEND_NAME = "CUSTOM"


class HackAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type["HackAttentionImpl"]:
        return HackAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["HackAttentionMetadataBuilder"]:
        return HackAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [get_settings().config.partition_size]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        return block_size is None or block_size == get_settings().config.partition_size

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size % 16 == 0

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC, KVCacheLayout.LBNHC)

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        if spec.state_content_bytes is not None:
            return spec
        layout = PageLayout(get_settings().config, spec.num_kv_heads, spec.head_size)
        return replace(
            spec, dtype=torch.uint8, num_head_slots=1, state_content_bytes=layout.bytes_per_token_slot
        )


@dataclass
class HackAttentionMetadata(AttentionMetadata):
    plan: StepPlan
    num_actual_tokens: int


class HackAttentionMetadataBuilder(AttentionMetadataBuilder[HackAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str], vllm_config: VllmConfig, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        if vllm_config.speculative_config is not None:
            raise ValueError("speculative decoding is not supported by this attention backend")

    def build(
        self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False
    ) -> HackAttentionMetadata:
        meta = common_attn_metadata
        runtime = get_runtime()
        runtime.bind_blocks(self.vllm_config.cache_config.num_gpu_blocks)
        seq_lens = meta.seq_lens_cpu_upper_bound if meta.seq_lens_cpu_upper_bound is not None else meta.seq_lens.cpu()
        plan = runtime.planner.plan(
            meta.query_start_loc_cpu.numpy(),
            seq_lens.numpy(),
            meta.block_table_tensor,
            meta.num_actual_tokens,
        )
        return HackAttentionMetadata(plan=plan, num_actual_tokens=meta.num_actual_tokens)


class HackAttentionImpl(AttentionImpl[HackAttentionMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None or sliding_window is not None or logits_soft_cap is not None:
            raise ValueError("ALiBi, sliding windows and logit soft caps are not supported")
        if kv_sharing_target_layer_name is not None or kwargs.get("sinks") is not None:
            raise ValueError("KV sharing and learned attention sinks are not supported")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

        vllm_config = get_current_vllm_config()
        self.settings = get_settings()
        device = torch.device("cuda", torch.cuda.current_device())
        runtime = ensure_runtime(self.settings, vllm_config.scheduler_config.max_num_seqs, device)
        self.store = runtime.new_store(self.num_kv_heads, head_size, vllm_config.model_config.dtype)
        self.layer = PagedAttentionLayer(self.store, self.settings, self.scale)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HackAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output.fill_(0)
        self.store.bind(kv_cache)
        self.layer.forward(attn_metadata.plan, query, key, value, output)
        return output

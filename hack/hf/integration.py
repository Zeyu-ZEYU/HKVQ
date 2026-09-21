"""Hugging Face Transformers integration.

`enable_hack(model)` switches the attention of a causal LM to HACK. Generation then uses
a `HackCache`, which stores K and V as quantized codes; attention runs on the codes.
"""

import os

import torch
from transformers import AttentionInterface
from transformers.cache_utils import Cache, CacheLayerMixin

from hack.attention_ref import hack_attention as reference_attention
from hack.cache import HackLayerCache
from hack.config import HackConfig

ATTENTION_NAME = "hack"
_active_layers: dict[int, "HackCacheLayer"] = {}


class HackCacheLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False

    def __init__(self, config: HackConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.store = HackLayerCache(config)

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.store.append(key_states, value_states)
        _active_layers[self.layer_idx] = self
        return key_states, value_states

    def get_seq_length(self) -> int:
        return self.store.num_tokens

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.store.num_tokens + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self):
        self.store = HackLayerCache(self.store.config)

    def offload(self):
        pass

    def prefetch(self):
        pass


class HackCache(Cache):
    def __init__(self, model_config, config: HackConfig | None = None):
        config = config or HackConfig()
        text_config = model_config.get_text_config() if hasattr(model_config, "get_text_config") else model_config
        super().__init__(layers=[HackCacheLayer(config, i) for i in range(text_config.num_hidden_layers)])
        self.hack_config = config

    def nbytes(self) -> int:
        return sum(layer.store.nbytes() for layer in self.layers)


def hack_attention_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    layer = _active_layers[module.layer_idx]
    if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
        raise NotImplementedError("padded batches are not supported; use batch size 1 or equally long prompts")
    store = layer.store
    if store.num_tokens == query.shape[2] and query.shape[2] > 1 and not store.config.quantized_prefill:
        output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, scale=scaling, enable_gqa=key.shape[1] != query.shape[1]
        )
    else:
        output = _attention(query)(query, store, scaling=scaling, causal=True)
    return output.transpose(1, 2).contiguous(), None


def _attention(query: torch.Tensor):
    if query.is_cuda and os.environ.get("HACK_ATTENTION", "kernels") == "kernels":
        from hack.kernels import hack_attention

        return hack_attention
    return reference_attention


def enable_hack(model) -> None:
    """Route the attention of `model` through HACK. Pass a `HackCache` as `past_key_values`."""
    AttentionInterface.register(ATTENTION_NAME, hack_attention_forward)
    model.set_attn_implementation(ATTENTION_NAME)

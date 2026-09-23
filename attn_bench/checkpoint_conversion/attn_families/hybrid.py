"""Shared pieces of the hybrid HF architectures (Qwen-style GDN + gated attention, Kimi-style
KDA + MLA): the per-layer mixer pattern and the one cache object covering both mixer types.

layer_types derivation mirrors Megatron's own get_linear_attention_pattern
(experimental_attention_variant_module_specs.py) exactly.
"""

from typing import List, Union

import torch


def compute_layer_types(linear_attention_freq: Union[int, List[int]], num_hidden_layers: int) -> List[str]:
    """freq as int N: full_attention every N-th layer (1-indexed), linear_attention elsewhere.
    freq as an explicit per-layer list (1=linear_attention, 0=full_attention, Megatron's own
    convention): used directly."""
    if isinstance(linear_attention_freq, int):
        pattern = [0 if ((i + 1) % linear_attention_freq == 0) else 1 for i in range(num_hidden_layers)]
    else:
        pattern = list(linear_attention_freq)
        assert len(pattern) == num_hidden_layers, (
            f"linear_attention_freq list length {len(pattern)} != num_hidden_layers {num_hidden_layers}"
        )
    return ["linear_attention" if p else "full_attention" for p in pattern]


class HybridCache:
    """One cache object covering both mixer types, keyed by layer_idx and routed by
    config.layer_types -- same shape as real Qwen3Next's own Qwen3NextDynamicCache
    (transformers/models/qwen3_next/modeling_qwen3_next.py), adapted (not imported: that
    class is tied to Qwen3Next's own module) to the two APIs our reused mixers actually call:

    - linear_attention layers (GDNMixer, KDAMixer) call the GDNCache/KDACache-shaped half:
      has_previous_state(layer_idx) / update(layer_idx, conv_state, recurrent_state) /
      conv_states[layer_idx] / recurrent_states[layer_idx].
    - full_attention layers (GatedLlamaAttention, DeepseekV2Attention) call the
      standard-Cache-shaped half: update(key_states, value_states, layer_idx, cache_kwargs).

    get_seq_length/get_mask_sizes redirect a non-attention layer_idx to the first
    full_attention layer (mirrors Qwen3NextDynamicCache.get_seq_length exactly) -- needed
    because transformers.masking_utils.create_causal_mask defaults to layer_idx=0, which in
    these hybrids is a linear_attention layer with no key_cache entry.

    Static batching only, same as GDNCache -- no reorder_cache/beam-search support.
    """

    def __init__(self, config):
        self.layer_types = config.layer_types
        self.attention_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        num_layers = len(self.layer_types)
        self.conv_states: list = [None] * num_layers
        self.recurrent_states: list = [None] * num_layers
        self.key_cache: list = [None] * num_layers
        self.value_cache: list = [None] * num_layers

    # --- linear_attention side -- GDNCache's/KDACache's exact API ---
    def has_previous_state(self, layer_idx: int) -> bool:
        return self.recurrent_states[layer_idx] is not None

    def update_linear_state(self, layer_idx: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor) -> None:
        self.conv_states[layer_idx] = conv_state
        self.recurrent_states[layer_idx] = recurrent_state

    # The linear mixers call cache_params.update(layer_idx, conv_state, recurrent_state) --
    # positional, 3-arg. Kept as the plain `update` name (not update_linear_state) so the mixers
    # need no changes; the 4-arg attention update below is a different overload, dispatched by
    # arg count since Python has no method overloading.
    def update(self, *args):
        if len(args) == 3:
            layer_idx, conv_state, recurrent_state = args
            self.update_linear_state(layer_idx, conv_state, recurrent_state)
            return
        key_states, value_states, layer_idx = args[0], args[1], args[2]
        cache_kwargs = args[3] if len(args) > 3 else None
        return self._update_attention_state(key_states, value_states, layer_idx, cache_kwargs)

    # --- full_attention side -- Qwen3NextDynamicCache's exact key_cache/value_cache API ---
    def _update_attention_state(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.attention_layers:
            layer_idx = self.attention_layers[0]
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int):
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        return query_length + past_seen_tokens, 0

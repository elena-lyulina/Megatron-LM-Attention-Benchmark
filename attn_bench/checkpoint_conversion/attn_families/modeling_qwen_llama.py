"""HuggingFace config/model classes for the Qwen-style hybrid.

Not built on transformers' own Qwen3NextForCausalLM, despite it being the real "gated-attn +
GDN" architecture this replicates: its softmax layers unconditionally apply q_norm/k_norm, a
Qwen3-era stability feature unrelated to gated attention itself -- the gate's origin paper
never applies it, and GatedLlamaAttention follows the paper, not Qwen3-Next (see its
docstring). Its GDN layers also use a different in_proj/conv1d split, and it uses partial RoPE
where ours is full RoPE (see modeling_gdn_llama.py's docstring and
results/docs/qwen_hybrid_config.md for the full detail). So this file instead composes our own
already-verified GatedLlamaAttention and GDNMixer per layer_types.

QwenLlamaConfig is a straight union of GatedLlamaConfig's attention fields (rope, GQA,
attention_bias/dropout -- inherited from LlamaConfig) and GDNLlamaConfig's linear_* mixer
fields, plus layer_types: which mixer sits at which layer.

layer_types derivation mirrors Megatron's own get_linear_attention_pattern
(experimental_attention_variant_module_specs.py) exactly.
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (LlamaMLP, LlamaRMSNorm,
                                                      LlamaRotaryEmbedding)

from attn_bench.checkpoint_conversion.attn_families.generation_mixin import \
    CacheParamsGenerationMixin
from attn_bench.checkpoint_conversion.attn_families.modeling_gated_llama import \
    GatedLlamaAttention
from attn_bench.checkpoint_conversion.attn_families.modeling_gdn_llama import \
    GDNMixer


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


class QwenLlamaConfig(LlamaConfig):
    model_type = "qwen_llama"

    def __init__(
        self,
        linear_attention_freq: Union[int, List[int]] = 4,
        linear_num_key_heads: int = 8,
        linear_num_value_heads: int = 8,
        linear_key_head_dim: int = 192,
        linear_value_head_dim: int = 384,
        linear_conv_kernel_dim: int = 4,
        linear_conv_bias: bool = False,
        linear_in_proj_bias: bool = False,
        linear_out_proj_bias: bool = False,
        layer_types: Optional[List[str]] = None,
        **kwargs,
    ):
        self.linear_attention_freq = linear_attention_freq
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_conv_bias = linear_conv_bias
        self.linear_in_proj_bias = linear_in_proj_bias
        self.linear_out_proj_bias = linear_out_proj_bias
        super().__init__(**kwargs)
        # After super().__init__ so self.num_hidden_layers is set. Only computed when not
        # given explicitly -- a reload from a saved config.json carries its own layer_types
        # and must not recompute it (same pattern as Qwen3NextConfig).
        self.layer_types = layer_types if layer_types is not None else compute_layer_types(
            self.linear_attention_freq, self.num_hidden_layers
        )


class QwenLlamaCache:
    """One cache object covering both mixer types, keyed by layer_idx and routed by
    config.layer_types -- same shape as real Qwen3Next's own Qwen3NextDynamicCache
    (transformers/models/qwen3_next/modeling_qwen3_next.py), adapted (not imported: that
    class is tied to Qwen3Next's own module) to the two APIs our reused mixers actually call:

    - GDNMixer (linear_attention layers) calls the GDNCache-shaped half:
      has_previous_state(layer_idx) / update(layer_idx, conv_state, recurrent_state) /
      conv_states[layer_idx] / recurrent_states[layer_idx] -- identical to GDNCache, verbatim,
      since GDNMixer's forward (modeling_gdn_llama.py) hardcodes that exact interface.
    - GatedLlamaAttention (full_attention layers) calls the standard-Cache-shaped half:
      update(key_states, value_states, layer_idx, cache_kwargs) -- identical signature to
      GatedLlamaAttention's own past_key_values.update(...) call (modeling_gated_llama.py).

    get_seq_length/get_mask_sizes redirect a non-attention layer_idx to the first
    full_attention layer (mirrors Qwen3NextDynamicCache.get_seq_length exactly) -- needed
    because transformers.masking_utils.create_causal_mask defaults to layer_idx=0, which in
    this hybrid is a linear_attention layer with no key_cache entry.

    Static batching only, same as GDNCache -- no reorder_cache/beam-search support.
    """

    def __init__(self, config: "QwenLlamaConfig"):
        self.layer_types = config.layer_types
        self.attention_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        num_layers = len(self.layer_types)
        self.conv_states: list = [None] * num_layers
        self.recurrent_states: list = [None] * num_layers
        self.key_cache: list = [None] * num_layers
        self.value_cache: list = [None] * num_layers

    # --- linear_attention (GDN) side -- GDNCache's exact API ---
    def has_previous_state(self, layer_idx: int) -> bool:
        return self.recurrent_states[layer_idx] is not None

    def update_linear_state(self, layer_idx: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor) -> None:
        self.conv_states[layer_idx] = conv_state
        self.recurrent_states[layer_idx] = recurrent_state

    # GDNMixer.forward calls cache_params.update(layer_idx, conv_state, recurrent_state) --
    # positional, 3-arg. Kept as the plain `update` name (not update_linear_state) so GDNMixer
    # needs no changes; the 4-arg attention update below is a different overload, dispatched by
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


class QwenLlamaDecoderLayer(nn.Module):
    """Picks GDNMixer or GatedLlamaAttention per config.layer_types[layer_idx] -- both
    imported, not reimplemented (see module docstring). input_layernorm/post_attention_
    layernorm/mlp are identical regardless of mixer type, same as GDNLlamaDecoderLayer's and
    LlamaDecoderLayer's own shape."""

    def __init__(self, config: QwenLlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            self.mixer = GDNMixer(config, layer_idx)
        else:
            self.self_attn = GatedLlamaAttention(config, layer_idx)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
               cache_params=None, cache_position=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        else:
            hidden_states, _ = self.self_attn(
                hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask,
                past_key_values=cache_params, cache_position=cache_position,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class QwenLlamaPreTrainedModel(PreTrainedModel):
    """_is_stateful = True, same flag GDNLlamaPreTrainedModel sets and for the same reason --
    tells GenerationMixin._prepare_cache_for_generation to skip eagerly instantiating a
    DynamicCache under past_key_values, since our own cache_params covers both mixer types."""

    config: QwenLlamaConfig
    config_class = QwenLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["QwenLlamaDecoderLayer"]
    _is_stateful = True


@dataclass
class QwenLlamaOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[QwenLlamaCache] = None


@dataclass
class QwenLlamaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[QwenLlamaCache] = None


class QwenLlamaModel(QwenLlamaPreTrainedModel):
    def __init__(self, config: QwenLlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [QwenLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_params: Optional[QwenLlamaCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> QwenLlamaOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache_params is None:
            cache_params = QwenLlamaCache(self.config)

        if cache_position is None:
            past_seen_tokens = cache_params.get_seq_length() if cache_params is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=cache_params,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings, attention_mask=causal_mask,
                cache_params=cache_params, cache_position=cache_position,
            )
        hidden_states = self.norm(hidden_states)

        return QwenLlamaOutput(last_hidden_state=hidden_states, cache_params=cache_params if use_cache else None)


class QwenLlamaForCausalLM(QwenLlamaPreTrainedModel, CacheParamsGenerationMixin, GenerationMixin):
    """_tied_weights_keys cleared for the same reason GDNLlamaForCausalLM/GatedLlamaForCausalLM
    clear it (confirmed necessary in isolation, job 3118149): without it, from_pretrained's
    meta-device loading leaves lm_head.weight stuck on the meta device for a custom
    architecture class.

    prepare_inputs_for_generation comes from CacheParamsGenerationMixin (shared with GDN and
    KDA) -- CacheParamsGenerationMixin must precede GenerationMixin in the base list so its
    method wins over GenerationMixin's own default via MRO."""

    _tied_weights_keys = None

    def __init__(self, config: QwenLlamaConfig):
        super().__init__(config)
        self.model = QwenLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_params: Optional[QwenLlamaCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> QwenLlamaCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return QwenLlamaCausalLMOutput(logits=logits, cache_params=outputs.cache_params)

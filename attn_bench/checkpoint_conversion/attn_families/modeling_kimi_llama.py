"""HuggingFace config/model classes for the Kimi-style hybrid (--experimental-attention-variant
kimi_delta_attention + --multi-latent-attention + --linear-attention-freq 4): KDA on the
linear_attention layers, MLA on every 4th layer (3/7/11/15).

Not built on Kimi-Linear's own HF model: that one uses NoPE MLA, 32 heads, MoE and a split
q/k/v conv, none of which this checkpoint has. Instead this file composes the two blocks whose
standalone conversions are already verified -- KDAMixer (modeling_kda_llama.py) and transformers'
own DeepseekV2Attention (the mla.py path, DeepSeek-V2-Lite verbatim) -- per layer_types, the same
way modeling_qwen_llama.py composes GDNMixer and GatedLlamaAttention.

KimiLlamaConfig is DeepseekV2Config (MLA dims, plain RoPE, attention_bias/dropout) plus
KDALlamaConfig's linear_* mixer fields plus layer_types (see hybrid.py).
"""

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.deepseek_v2.configuration_deepseek_v2 import \
    DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention, DeepseekV2RotaryEmbedding)
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from attn_bench.checkpoint_conversion.attn_families.generation_mixin import \
    CacheParamsGenerationMixin
from attn_bench.checkpoint_conversion.attn_families.hybrid import (
    HybridCache, compute_layer_types)
from attn_bench.checkpoint_conversion.attn_families.modeling_kda_llama import \
    KDAMixer


class KimiLlamaConfig(DeepseekV2Config):
    model_type = "kimi_llama"

    def __init__(
        self,
        linear_attention_freq: Union[int, List[int]] = 4,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 16,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
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
        # given explicitly -- a reload from a saved config.json carries its own layer_types.
        self.layer_types = layer_types if layer_types is not None else compute_layer_types(
            self.linear_attention_freq, self.num_hidden_layers
        )


class KimiLlamaDecoderLayer(nn.Module):
    """Picks KDAMixer or DeepseekV2Attention per config.layer_types[layer_idx] -- both
    imported, not reimplemented. input_layernorm/post_attention_layernorm/mlp are identical
    regardless of mixer type (dense MLP on every layer, no MoE)."""

    def __init__(self, config: KimiLlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if self.layer_type == "linear_attention":
            self.mixer = KDAMixer(config, layer_idx)
        else:
            self.self_attn = DeepseekV2Attention(config, layer_idx)
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


class KimiLlamaPreTrainedModel(PreTrainedModel):
    """_is_stateful and the sdpa/flash/attention-backend flags for the same reasons as
    QwenLlamaPreTrainedModel (see its docstring): cache_params covers both mixer types, and the
    MLA layers need an efficient attention backend or they fall back to eager and OOM at long
    prefixes."""

    config: KimiLlamaConfig
    config_class = KimiLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["KimiLlamaDecoderLayer"]
    _is_stateful = True
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_attention_backend = True


@dataclass
class KimiLlamaOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[HybridCache] = None


@dataclass
class KimiLlamaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[HybridCache] = None


class KimiLlamaModel(KimiLlamaPreTrainedModel):
    def __init__(self, config: KimiLlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [KimiLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # DeepSeek's complex-valued (interleaved) RoPE on the qk_rope_head_dim slice only --
        # what DeepseekV2Attention expects as position_embeddings.
        self.rotary_emb = DeepseekV2RotaryEmbedding(config=config)
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
        cache_params: Optional[HybridCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> KimiLlamaOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache_params is None:
            cache_params = HybridCache(self.config)

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

        return KimiLlamaOutput(last_hidden_state=hidden_states, cache_params=cache_params if use_cache else None)


class KimiLlamaForCausalLM(KimiLlamaPreTrainedModel, CacheParamsGenerationMixin, GenerationMixin):
    """_tied_weights_keys cleared and CacheParamsGenerationMixin placed before GenerationMixin
    for the same reasons as QwenLlamaForCausalLM (see its docstring)."""

    _tied_weights_keys = None

    def __init__(self, config: KimiLlamaConfig):
        super().__init__(config)
        self.model = KimiLlamaModel(config)
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
        cache_params: Optional[HybridCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> KimiLlamaCausalLMOutput:
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
        return KimiLlamaCausalLMOutput(logits=logits, cache_params=outputs.cache_params)

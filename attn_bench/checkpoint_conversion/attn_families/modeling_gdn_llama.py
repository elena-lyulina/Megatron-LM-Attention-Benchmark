"""HuggingFace config/model classes for pure Gated Delta Net (GDN) checkpoints
(--experimental-attention-variant gated_delta_net).

No native HF architecture to target: every layer is a GDN mixer, not attention, so this follows
HF's pure-SSM pattern (Mamba2ForCausalLM/Mamba2Cache) rather than Qwen3NextForCausalLM, which is
a hybrid GDN+attention MoE model with a different weight layout. GDNMixer keeps Megatron's own
combined in_proj/conv1d layout (not Qwen3-Next's split in_proj_qkvz/in_proj_ba) so gdn.py's
build_state_dict is a straight key rename. Calls `fla` directly for both prefill and decode,
same kernels Megatron uses to train these checkpoints. GDNCache hardcodes static batching (no
reorder_cache/beam search), matching Megatron's own `is_static_batching()` assertion.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules.convolution import causal_conv1d, causal_conv1d_update
from fla.modules.l2norm import l2norm
from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from attn_bench.checkpoint_conversion.attn_families.generation_mixin import \
    CacheParamsGenerationMixin


class GDNLlamaConfig(PretrainedConfig):
    """Llama backbone (embeddings/MLP/final norm) + a pure GDN mixer replacing attention.
    No rope_theta/rope_scaling/attention_bias/num_key_value_heads -- GDN has no rotary
    embeddings and its own head-count fields are independent of the attention ones."""

    model_type = "gdn_llama"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=5824,
        num_hidden_layers=16,
        hidden_act="silu",
        max_position_embeddings=131072,
        initializer_range=0.01,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        tie_word_embeddings=False,
        mlp_bias=False,
        linear_num_key_heads=8,
        linear_num_value_heads=8,
        linear_key_head_dim=192,
        linear_value_head_dim=384,
        linear_conv_kernel_dim=4,
        linear_conv_bias=False,
        linear_in_proj_bias=False,
        linear_out_proj_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.mlp_bias = mlp_bias
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_conv_bias = linear_conv_bias
        self.linear_in_proj_bias = linear_in_proj_bias
        self.linear_out_proj_bias = linear_out_proj_bias
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class GDNCache:
    """Per-layer (conv_state, recurrent_state), lazily populated on prefill -- mirrors
    Megatron's own `inference_context.key_value_memory_dict` (see GatedDeltaNet.forward/_decode)
    rather than pre-allocating fixed-shape state tensors the way HF's `Mamba2Cache` does, since
    fla's internal state-tensor shapes aren't public API to replicate ahead of time.

    Static batching only: batch composition is fixed for the object's lifetime, entries are
    plain per-layer tensors with no reorder_cache/resize support. Threaded through generate()
    as `cache_params` (not `past_key_values`) -- `transformers.generation.utils` special-cases
    that name for exactly this (Mamba-family) cache-object pattern, see
    GDNLlamaForCausalLM.prepare_inputs_for_generation.
    """

    def __init__(self, num_hidden_layers: int):
        self.conv_states: list[Optional[torch.Tensor]] = [None] * num_hidden_layers
        self.recurrent_states: list[Optional[torch.Tensor]] = [None] * num_hidden_layers

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.recurrent_states[layer_idx] is not None

    def update(self, layer_idx: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor) -> None:
        self.conv_states[layer_idx] = conv_state
        self.recurrent_states[layer_idx] = recurrent_state


class GDNMixer(nn.Module):
    """Gated Delta Net mixer. Combined `in_proj`/`conv1d` layout matches Megatron's
    GatedDeltaNet exactly (query/key/value/gate/beta/alpha concatenated in that order, per the
    __init__ below) -- see module docstring for why. Math ported from GatedDeltaNet.forward/
    _decode with tensor/context-parallelism stripped out (tp_size=cp_size=1 at HF-inference
    time, since conversion only supports a single mp_rank_00 checkpoint)."""

    def __init__(self, config: GDNLlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        self.conv_kernel_dim = config.linear_conv_kernel_dim

        self.in_proj_dim = self.qk_dim * 2 + self.v_dim * 2 + self.num_value_heads * 2
        self.in_proj = nn.Linear(self.hidden_size, self.in_proj_dim, bias=config.linear_in_proj_bias)

        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.linear_conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))

        self.out_norm = LlamaRMSNorm(self.value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=config.linear_out_proj_bias)

    def _split_qkvzba(self, qkvzba: torch.Tensor, batch: int, seq_len: int):
        qkv, gate, beta, alpha = torch.split(
            qkvzba, [2 * self.qk_dim + self.v_dim, self.v_dim, self.num_value_heads, self.num_value_heads], dim=-1
        )
        gate = gate.reshape(batch, seq_len, -1, self.value_head_dim)
        beta = beta.reshape(batch, seq_len, -1)
        alpha = alpha.reshape(batch, seq_len, -1)
        return qkv, gate, beta, alpha

    def _prepare_qkv(self, qkv: torch.Tensor, batch: int, seq_len: int):
        query_key, value = torch.split(qkv, [2 * self.qk_dim, self.v_dim], dim=-1)
        query_key = l2norm(query_key.reshape(batch, seq_len, -1, self.key_head_dim).contiguous())
        value = value.reshape(batch, seq_len, -1, self.value_head_dim)
        query, key = torch.split(query_key, [self.num_key_heads, self.num_key_heads], dim=2)
        repeat_factor = self.num_value_heads // self.num_key_heads
        if repeat_factor > 1:
            query = query.repeat_interleave(repeat_factor, dim=2)
            key = key.repeat_interleave(repeat_factor, dim=2)
        return query.contiguous(), key.contiguous(), value.contiguous()

    def _compute_g_and_beta(self, alpha: torch.Tensor, beta: torch.Tensor):
        g = -self.A_log.float().exp() * F.softplus(alpha.float() + self.dt_bias.float())
        return g, torch.sigmoid(beta)

    def _apply_gated_norm(self, core_attn_out: torch.Tensor, gate: torch.Tensor, batch: int, seq_len: int):
        x_dtype = core_attn_out.dtype
        y = self.out_norm(core_attn_out.reshape(-1, self.value_head_dim))
        y = y * F.silu(gate.reshape(-1, self.value_head_dim).float())
        y = y.to(x_dtype)
        return y.reshape(batch, seq_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[GDNCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        decode = (
            cache_params is not None
            and cache_position is not None
            and cache_position[0] > 0
            and cache_params.has_previous_state(self.layer_idx)
        )

        qkvzba = self.in_proj(hidden_states)
        qkv, gate, beta, alpha = self._split_qkvzba(qkvzba, batch, seq_len)

        conv_bias = self.conv1d.bias
        conv_weight = self.conv1d.weight.squeeze(1)
        if decode:
            qkv, conv_state = causal_conv1d_update(
                x=qkv,
                cache=cache_params.conv_states[self.layer_idx],
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
            )
        else:
            qkv, conv_state = causal_conv1d(
                x=qkv,
                weight=conv_weight,
                bias=conv_bias,
                activation="silu",
                initial_state=None,
                output_final_state=cache_params is not None,
                cu_seqlens=None,
            )

        query, key, value = self._prepare_qkv(qkv, batch, seq_len)
        g, beta = self._compute_g_and_beta(alpha, beta)

        if decode:
            core_attn_out, recurrent_state = fused_recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=cache_params.recurrent_states[self.layer_idx],
                output_final_state=True, use_qk_l2norm_in_kernel=False,
            )
        else:
            core_attn_out, recurrent_state = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=None, output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=False, cu_seqlens=None,
            )

        if cache_params is not None:
            cache_params.update(self.layer_idx, conv_state, recurrent_state)

        norm_out = self._apply_gated_norm(core_attn_out, gate, batch, seq_len)
        return self.out_proj(norm_out)


class GDNLlamaDecoderLayer(nn.Module):
    def __init__(self, config: GDNLlamaConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = GDNMixer(config, layer_idx)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, hidden_states, cache_params=None, cache_position=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mixer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GDNLlamaPreTrainedModel(PreTrainedModel):
    """`_is_stateful = True` (same flag `Mamba2PreTrainedModel` sets) is what actually tells
    `GenerationMixin._prepare_cache_for_generation` to skip eagerly instantiating a
    `DynamicCache` under `past_key_values` -- without it, `_supports_default_dynamic_cache()`
    returns True by default and `generate()` would inject an unused `DynamicCache` into
    `model_kwargs` alongside our own lazily-created `cache_params`."""

    config: GDNLlamaConfig
    config_class = GDNLlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["GDNLlamaDecoderLayer"]
    _is_stateful = True


@dataclass
class GDNLlamaOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[GDNCache] = None


@dataclass
class GDNLlamaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[GDNCache] = None


class GDNLlamaModel(GDNLlamaPreTrainedModel):
    def __init__(self, config: GDNLlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [GDNLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_params: Optional[GDNCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> GDNLlamaOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache_params is None:
            cache_params = GDNCache(self.config.num_hidden_layers)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        hidden_states = self.norm(hidden_states)

        return GDNLlamaOutput(last_hidden_state=hidden_states, cache_params=cache_params if use_cache else None)


class GDNLlamaForCausalLM(GDNLlamaPreTrainedModel, CacheParamsGenerationMixin, GenerationMixin):
    """_tied_weights_keys cleared for the same reason modeling_gated_llama.py/
    modeling_sink_llama.py clear it (confirmed necessary in isolation, job 3118149): without it,
    from_pretrained's meta-device loading leaves lm_head.weight stuck on the meta device for a
    custom architecture class.

    prepare_inputs_for_generation comes from CacheParamsGenerationMixin (shared with KDA and
    the Qwen hybrid) -- CacheParamsGenerationMixin must precede GenerationMixin in the base
    list so its method wins over GenerationMixin's own default via MRO."""

    _tied_weights_keys = None

    def __init__(self, config: GDNLlamaConfig):
        super().__init__(config)
        self.model = GDNLlamaModel(config)
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
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_params: Optional[GDNCache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> GDNLlamaCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return GDNLlamaCausalLMOutput(logits=logits, cache_params=outputs.cache_params)

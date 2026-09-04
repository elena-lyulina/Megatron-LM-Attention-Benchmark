"""HuggingFace config/model classes (subclassing Llama) for attention-output-gate checkpoints
(--attention-output-gate).

Not reused from Qwen3-Next's Qwen3NextAttention despite matching gate math: this module
implements the gate as in its origin paper (arXiv:2505.06708), which never applies q_norm/
k_norm -- Qwen3-Next's is a separate stability feature inherited from Qwen3 (Megatron's
qk_layernorm is an orthogonal flag from attention_output_gate; this family only sets the
latter). Its gate is also fused into a widened q_proj, a different weight layout than our
separate gate chunk in linear_qkv (see build_state_dict). Otherwise stock Llama, subclassed
not copied -- same structure as modeling_sink_llama.py.

Unlike sink, the gate is a plain post-hoc multiply on the attention output, not something a
kernel needs to fuse -- GatedLlamaAttention works with any attn_implementation (eager/sdpa/...).
"""

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaConfig,
                                                      LlamaDecoderLayer,
                                                      LlamaForCausalLM,
                                                      LlamaModel,
                                                      LlamaPreTrainedModel,
                                                      LlamaRMSNorm,
                                                      LlamaRotaryEmbedding,
                                                      apply_rotary_pos_emb,
                                                      eager_attention_forward)


class GatedLlamaConfig(LlamaConfig):
    model_type = "gated_llama"


class GatedLlamaAttention(LlamaAttention):
    def __init__(self, config: GatedLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.gate_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )

    def forward(self, hidden_states, position_embeddings, attention_mask,
               past_key_values=None, cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        # [B, S, H, D] -- deliberately NOT transposed like q/k/v: it needs to line up with
        # attn_output's [B, S, H, D] layout below, not query's [B, H, S, D] one.
        gate = self.gate_proj(hidden_states).view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # Matches Megatron's _apply_output_gate exactly (fused_softmax.py path this checkpoint
        # was trained with): upcast to fp32 before sigmoid, multiply, then fall back to the
        # working dtype -- avoids a needless extra logits mismatch in compare_megatron_hf_logits.py.
        gate = gate.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate.float()).to(attn_output.dtype)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class GatedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GatedLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = GatedLlamaAttention(config=config, layer_idx=layer_idx)


class GatedLlamaModel(LlamaModel):
    """Builds via the grandparent's __init__, not LlamaModel's, to avoid constructing a
    throwaway LlamaModel first (see GatedLlamaForCausalLM) -- same pattern as
    modeling_sink_llama.py's SinkLlamaModel."""
    config: GatedLlamaConfig

    def __init__(self, config: GatedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GatedLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


class GatedLlamaForCausalLM(LlamaForCausalLM):
    """Builds via the grandparent's __init__ (skips constructing a throwaway LlamaModel that
    LlamaForCausalLM.__init__ would tie lm_head to before we swap in GatedLlamaModel).
    _tied_weights_keys is cleared too -- same fix modeling_sink_llama.py needed (confirmed
    necessary in isolation, job 3118149): without it, from_pretrained's meta-device loading
    still leaves lm_head.weight stuck on the meta device for this custom class, even with a
    single clean construction."""
    config: GatedLlamaConfig
    _tied_weights_keys = None

    def __init__(self, config: GatedLlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = GatedLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

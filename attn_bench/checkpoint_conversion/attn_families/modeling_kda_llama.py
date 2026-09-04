"""HuggingFace config/model classes for pure Kimi Delta Attention (KDA) checkpoints
(--experimental-attention-variant kimi_delta_attention).

Same shape as modeling_gdn_llama.py (HF's pure-SSM Mamba2ForCausalLM/Mamba2Cache pattern, since
every layer is a linear-attention mixer, not attention). KDAMixer keeps Megatron's own layout --
six separate projections (q/k/v/b_proj, low-rank f_proj_down/up and g_proj_down/up), one fused
conv1d over cat([q,k,v]) -- so kda.py's build_state_dict is a straight key rename. Math ported
from KimiDeltaAttention.forward/_decode with tensor/context-parallelism stripped (tp=cp=1 at
HF-inference time). Calls `fla` directly (>= 0.5.2: the container's 0.4.2 NaNs chunk_kda) for
both prefill and decode -- the same kernels Megatron uses to train these checkpoints.

Differences from GDN: separate projections (no combined in_proj / _split_qkvzba); raw g/beta pass
straight to the kernel with use_gate_in_kernel / use_beta_sigmoid_in_kernel / use_qk_l2norm_in_kernel
set (no _compute_g_and_beta pre-pass); the output gate uses sigmoid, not SiLU; dt_bias is
per-(value-head, key-channel) [num_value_heads * key_head_dim], not per-value-head.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from fla.modules.convolution import causal_conv1d, causal_conv1d_update
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

from attn_bench.checkpoint_conversion.attn_families.generation_mixin import \
    CacheParamsGenerationMixin


class KDALlamaConfig(PretrainedConfig):
    """Llama backbone (embeddings/MLP/final norm) + a pure KDA mixer replacing attention.
    No rope_theta/rope_scaling/attention_bias/num_key_value_heads -- KDA has no rotary
    embeddings (Q/K are L2-normed in-kernel) and its head-count fields are independent of the
    attention ones."""

    model_type = "kda_llama"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=6976,
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
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
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


class KDACache:
    """Per-layer (conv_state, recurrent_state), lazily populated on prefill -- mirrors Megatron's
    own `inference_context.key_value_memory_dict` (see KimiDeltaAttention.forward/_decode).

    Static batching only: batch composition is fixed for the object's lifetime, no
    reorder_cache/resize. Threaded through generate() as `cache_params` (not `past_key_values`),
    same as Mamba2Cache -- see KDALlamaForCausalLM.prepare_inputs_for_generation.
    """

    def __init__(self, num_hidden_layers: int):
        self.conv_states: list[Optional[torch.Tensor]] = [None] * num_hidden_layers
        self.recurrent_states: list[Optional[torch.Tensor]] = [None] * num_hidden_layers

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.recurrent_states[layer_idx] is not None

    def update(self, layer_idx: int, conv_state: torch.Tensor, recurrent_state: torch.Tensor) -> None:
        self.conv_states[layer_idx] = conv_state
        self.recurrent_states[layer_idx] = recurrent_state


class KDAMixer(nn.Module):
    """Kimi Delta Attention mixer. Six separate projections + one fused depthwise conv over
    cat([q,k,v]) -- matches KimiDeltaAttention's own layout (see module docstring). Forward ported
    from KimiDeltaAttention.forward/_decode with tp=cp=1."""

    def __init__(self, config: KDALlamaConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.key_head_dim = config.linear_key_head_dim
        self.value_head_dim = config.linear_value_head_dim
        self.num_key_heads = config.linear_num_key_heads
        self.num_value_heads = config.linear_num_value_heads
        self.conv_kernel_dim = config.linear_conv_kernel_dim

        self.qk_dim = self.key_head_dim * self.num_key_heads
        self.v_dim = self.value_head_dim * self.num_value_heads
        # Per-(value-head, key-channel) fine-grained forget gate (Diag(alpha_t)).
        self.gate_dim = self.key_head_dim * self.num_value_heads
        # Low-rank bottleneck of the f_proj / g_proj gates (FLA uses head_v_dim).
        self.gate_low_rank_dim = self.value_head_dim

        in_bias = config.linear_in_proj_bias
        self.q_proj = nn.Linear(self.hidden_size, self.qk_dim, bias=in_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.qk_dim, bias=in_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.v_dim, bias=in_bias)
        self.b_proj = nn.Linear(self.hidden_size, self.num_value_heads, bias=in_bias)
        self.f_proj_down = nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False)
        self.f_proj_up = nn.Linear(self.gate_low_rank_dim, self.gate_dim, bias=False)
        self.g_proj_down = nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False)
        self.g_proj_up = nn.Linear(self.gate_low_rank_dim, self.v_dim, bias=False)

        self.conv_dim = self.qk_dim * 2 + self.v_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.linear_conv_bias,
            kernel_size=self.conv_kernel_dim,
            groups=self.conv_dim,
            padding=self.conv_kernel_dim - 1,
        )

        # A_log per value-head; dt_bias per gate channel (num_value_heads * key_head_dim).
        self.A_log = nn.Parameter(torch.zeros(self.num_value_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.gate_dim))

        self.out_norm = LlamaRMSNorm(self.value_head_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.v_dim, self.hidden_size, bias=config.linear_out_proj_bias)

    def _apply_gated_norm(self, core_attn_out: torch.Tensor, z: torch.Tensor, batch: int, seq_len: int):
        # RMSNorm(o) * sigmoid(z), per value-head channel (FLA FusedRMSNormGated).
        x_dtype = core_attn_out.dtype
        y = self.out_norm(core_attn_out.reshape(-1, self.value_head_dim))
        y = y * torch.sigmoid(z.reshape(-1, self.value_head_dim).float())
        return y.to(x_dtype).reshape(batch, seq_len, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[KDACache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        decode = (
            cache_params is not None
            and cache_position is not None
            and cache_position[0] > 0
            and cache_params.has_previous_state(self.layer_idx)
        )

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        beta = self.b_proj(hidden_states)
        g_raw = self.f_proj_up(self.f_proj_down(hidden_states))
        z = self.g_proj_up(self.g_proj_down(hidden_states))

        qkv = torch.cat([q, k, v], dim=-1)
        conv_bias = self.conv1d.bias
        conv_weight = self.conv1d.weight.squeeze(1)  # d, 1, w -> d, w
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
        q, k, v = torch.split(qkv, [self.qk_dim, self.qk_dim, self.v_dim], dim=-1)

        q = q.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim).contiguous()
        k = k.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim).contiguous()
        v = v.reshape(batch, seq_len, self.num_value_heads, self.value_head_dim).contiguous()
        g_raw = g_raw.reshape(batch, seq_len, self.num_value_heads, self.key_head_dim).contiguous()
        beta = beta.reshape(batch, seq_len, self.num_value_heads).contiguous()

        # Raw g/beta: the -exp(A_log)*softplus(g_raw + dt_bias) decay, beta sigmoid, and Q/K L2-norm
        # are all fused inside the kernel -- numerically consistent with the trained prefill.
        if decode:
            core_attn_out, recurrent_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g_raw, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=cache_params.recurrent_states[self.layer_idx],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )
        else:
            core_attn_out, recurrent_state = chunk_kda(
                q=q, k=k, v=v, g=g_raw, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                cu_seqlens=None,
            )

        if cache_params is not None:
            cache_params.update(self.layer_idx, conv_state, recurrent_state)

        z = z.reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)
        norm_out = self._apply_gated_norm(core_attn_out, z, batch, seq_len)
        return self.out_proj(norm_out)


class KDALlamaDecoderLayer(nn.Module):
    def __init__(self, config: KDALlamaConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = KDAMixer(config, layer_idx)
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


class KDALlamaPreTrainedModel(PreTrainedModel):
    """`_is_stateful = True` (same flag Mamba2PreTrainedModel sets) tells
    GenerationMixin._prepare_cache_for_generation to skip eagerly instantiating a DynamicCache
    under past_key_values -- otherwise generate() injects an unused one alongside cache_params."""

    config: KDALlamaConfig
    config_class = KDALlamaConfig
    base_model_prefix = "model"
    _no_split_modules = ["KDALlamaDecoderLayer"]
    _is_stateful = True


@dataclass
class KDALlamaOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[KDACache] = None


@dataclass
class KDALlamaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[KDACache] = None


class KDALlamaModel(KDALlamaPreTrainedModel):
    def __init__(self, config: KDALlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [KDALlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
        cache_params: Optional[KDACache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> KDALlamaOutput:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and cache_params is None:
            cache_params = KDACache(self.config.num_hidden_layers)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cache_params=cache_params, cache_position=cache_position)
        hidden_states = self.norm(hidden_states)

        return KDALlamaOutput(last_hidden_state=hidden_states, cache_params=cache_params if use_cache else None)


class KDALlamaForCausalLM(KDALlamaPreTrainedModel, CacheParamsGenerationMixin, GenerationMixin):
    """_tied_weights_keys cleared for the same reason modeling_gdn_llama.py clears it (job 3118149):
    without it, from_pretrained's meta-device loading leaves lm_head.weight on the meta device for
    a custom architecture class.

    prepare_inputs_for_generation comes from CacheParamsGenerationMixin (shared with GDN and
    the Qwen hybrid) -- CacheParamsGenerationMixin must precede GenerationMixin in the base
    list so its method wins over GenerationMixin's own default via MRO."""

    _tied_weights_keys = None

    def __init__(self, config: KDALlamaConfig):
        super().__init__(config)
        self.model = KDALlamaModel(config)
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
        cache_params: Optional[KDACache] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> KDALlamaCausalLMOutput:
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache_params=cache_params,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return KDALlamaCausalLMOutput(logits=logits, cache_params=outputs.cache_params)

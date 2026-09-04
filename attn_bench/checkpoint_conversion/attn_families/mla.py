"""Config + state-dict conversion for Multi-head Latent Attention (MLA) checkpoints
(--multi-latent-attention). Like swa.py this targets a native `transformers` architecture --
`DeepseekV2ForCausalLM` -- so there is no custom `modeling_*.py`: this project's MLA block is
DeepSeek-V2-Lite verbatim (hidden 2048, 16 heads, kv_lora_rank 512, qk_nope 128 / qk_rope 64 /
v 128, `q_lora_rank=None`, plain RoPE), which maps onto `DeepseekV2Config` with no adaptation.

Sources for the MLA-specific mapping:
- megatron/core/transformer/multi_latent_attention.py -- `MLASelfAttention` with `q_lora_rank
  is None`: builds a single `linear_q_proj` (no `linear_q_down/up_proj`, no query-side norm --
  matches HF's `if self.q_lora_rank is None:` branch in modeling_deepseek_v2.py). `kv_layernorm`
  is unconditional but the standard MLA layer spec passes `IdentityOp` for it; with
  `--qk-layernorm` the KV RMSNorm is instead **fused into `linear_kv_up_proj`** as
  `column_parallel_layer_norm_linear` -> `linear_kv_up_proj.layer_norm_weight`.
- megatron/core/models/gpt/gpt_layer_specs.py (`multi_latent_attention` branch) --
  `input_layernorm` standalone (`layer_norm(has_residual=True)`), `pre_mlp_layernorm=IdentityOp`
  (so the pre-MLP norm is fused into `mlp.linear_fc1`, exactly like full.py).
- Megatron-Bridge models/deepseek/{deepseek_v2_bridge,attention,common}.py -- confirms the
  key names and that `linear_kv_up_proj.weight <-> kv_b_proj.weight` is a plain rename with no
  reshape (per-head `[nope|v]` layout matches HF's `.view(...).split([qk_nope, v])`).
- HF: transformers.models.deepseek_v2 (transformers >= 4.48).
"""

from collections import OrderedDict
from typing import Any, Dict

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig


def build_config(args: Any) -> PretrainedConfig:
    """Build the HF config for an MLA checkpoint. Backbone fields follow full.py's build_config
    (see its docstring for the two args-naming fixes this fork needs); MLA block dims + plain
    RoPE from `$MLA_DIMS` (`llama_checkpoints.sh`), re-passed since --use-checkpoint-args does
    not restore them. Dense on every layer (`first_k_dense_replace = num_hidden_layers`), no MoE."""
    num_layers = args.num_layers
    return AutoConfig.for_model(
        "deepseek_v2",
        architectures=["DeepseekV2ForCausalLM"],
        attention_bias=False,
        attention_dropout=args.attention_dropout,
        bos_token_id=128000,
        eos_token_id=128001,
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        max_position_embeddings=131072,
        mlp_bias=False,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=num_layers,
        num_key_value_heads=args.num_attention_heads,  # MLA has no GQA; V2-Lite sets these equal
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see full.py's module docstring.
        rms_norm_eps=args.layernorm_epsilon,
        # MLA block (DeepSeek-V2-Lite). q_lora_rank omitted at train time -> None (Q not compressed).
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=getattr(args, "q_lora_rank", None),
        qk_nope_head_dim=args.qk_head_dim,
        qk_rope_head_dim=args.qk_pos_emb_head_dim,
        v_head_dim=args.v_head_dim,
        # Plain RoPE: --rope-type rope, --rotary-scaling-factor 1.0 (a no-op) -> no yarn scaling.
        rope_theta=args.rotary_base,
        rope_scaling=None,
        # Dense MLP on every layer -> DeepseekV2MLP, never DeepseekV2MoE.
        first_k_dense_replace=num_layers,
        n_routed_experts=None,
        n_shared_experts=None,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        torch_dtype=args.params_dtype,
        use_cache=True,
        vocab_size=args.padded_vocab_size,
    )


def build_model(config: PretrainedConfig) -> AutoModelForCausalLM:
    """Construct an uninitialized model from config -- standard registered architecture,
    no auto_map/trust_remote_code needed (unlike custom families e.g. sink/gdn/kda)."""
    return AutoModelForCausalLM.from_config(config)


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert an MLA Megatron state dict to DeepseekV2ForCausalLM format. Near-literal key
    rename; the non-attention parts (MLP, embeddings, final norm, lm_head) mirror full.py."""
    checkpoint = OrderedDict()

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        p = f'decoder.layers.{layer_idx}.self_attention.'
        a = f'model.layers.{layer_idx}.self_attn.'

        checkpoint[a + 'q_proj.weight'] = model_dict[p + 'linear_q_proj.weight']
        checkpoint[a + 'kv_a_proj_with_mqa.weight'] = model_dict[p + 'linear_kv_down_proj.weight']
        # KV RMSNorm fused into linear_kv_up_proj (column_parallel_layer_norm_linear), not a
        # standalone self_attention.kv_layernorm (that slot is IdentityOp in the MLA spec).
        checkpoint[a + 'kv_a_layernorm.weight'] = model_dict[p + 'linear_kv_up_proj.layer_norm_weight']
        checkpoint[a + 'kv_b_proj.weight'] = model_dict[p + 'linear_kv_up_proj.weight']
        checkpoint[a + 'o_proj.weight'] = model_dict[p + 'linear_proj.weight']

        # Standalone pre-attention RMSNorm (MLA spec: input_layernorm=layer_norm(has_residual=True)).
        checkpoint[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.input_layernorm.weight']

        mlp_weight = model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight']
        ffn_hidden_size = mlp_weight.shape[0] // 2
        checkpoint[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = mlp_weight[:ffn_hidden_size, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = mlp_weight[ffn_hidden_size:, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc2.weight']

        # Pre-MLP RMSNorm fused into linear_fc1 (MLA spec: pre_mlp_layernorm=IdentityOp), same as full.py.
        checkpoint[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight']

    checkpoint['model.norm.weight'] = model_dict['decoder.final_layernorm.weight']

    if not args.untie_embeddings_and_output_weights:
        checkpoint['lm_head.weight'] = checkpoint['model.embed_tokens.weight']
    else:
        checkpoint['lm_head.weight'] = model_dict['output_layer.weight']

    return checkpoint

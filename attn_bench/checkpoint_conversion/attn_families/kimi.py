"""Config + state-dict conversion for the Kimi-style hybrid (12 KDA + 4 MLA layers).

build_state_dict branches per-layer on layer_types: kda.py's convert_kda_mixer_weights for
linear_attention layers, mla.py's convert_mla_attn_weights for full_attention layers. The hybrid
spec keeps each block's own key layout (_get_self_attention_module_spec sets
fuse_input_layernorm=False for MLA, so both mixer types have a standalone input_layernorm).
MLP/embeddings/final-norm/lm_head handling is identical regardless of mixer type.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

from attn_bench.checkpoint_conversion.attn_families.hybrid import \
    compute_layer_types
from attn_bench.checkpoint_conversion.attn_families.kda import \
    convert_kda_mixer_weights
from attn_bench.checkpoint_conversion.attn_families.mla import \
    convert_mla_attn_weights
from attn_bench.checkpoint_conversion.attn_families.modeling_kimi_llama import (
    KimiLlamaConfig, KimiLlamaForCausalLM)


def build_config(args: Any) -> KimiLlamaConfig:
    """Build the HF config for a Kimi-hybrid checkpoint. MLA fields match mla.py's build_config,
    linear-side fields match kda.py's build_config. linear_attention_freq is not restored by
    --use-checkpoint-args (see llama_checkpoints.sh's KIMI_DIMS), so args carries whatever was
    re-passed on the CLI (e.g. 4)."""
    return KimiLlamaConfig(
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
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_attention_heads,  # MLA has no GQA
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see full.py's module docstring.
        rms_norm_eps=args.layernorm_epsilon,
        kv_lora_rank=args.kv_lora_rank,
        q_lora_rank=getattr(args, "q_lora_rank", None),
        qk_nope_head_dim=args.qk_head_dim,
        qk_rope_head_dim=args.qk_pos_emb_head_dim,
        v_head_dim=args.v_head_dim,
        # Plain RoPE: --rope-type rope, --rotary-scaling-factor 1.0 (a no-op) -> no yarn scaling.
        rope_theta=args.rotary_base,
        rope_scaling=None,
        # Dense MLP on every layer, no MoE (the MoE fields are unused by KimiLlamaDecoderLayer).
        first_k_dense_replace=args.num_layers,
        n_routed_experts=None,
        n_shared_experts=None,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        torch_dtype=args.params_dtype,
        use_cache=True,
        vocab_size=args.padded_vocab_size,
        linear_attention_freq=args.linear_attention_freq,
        linear_num_key_heads=args.linear_num_key_heads,
        linear_num_value_heads=args.linear_num_value_heads,
        linear_key_head_dim=args.linear_key_head_dim,
        linear_value_head_dim=args.linear_value_head_dim,
        linear_conv_kernel_dim=args.linear_conv_kernel_dim,
    )


def build_model(config: KimiLlamaConfig) -> KimiLlamaForCausalLM:
    """Construct an uninitialized model and register it for auto_map -- same as qwen.py's/
    kda.py's build_model."""
    model = KimiLlamaForCausalLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    return model


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a Kimi-hybrid Megatron state dict to KimiLlamaForCausalLM format."""
    checkpoint = OrderedDict()
    layer_types = compute_layer_types(args.linear_attention_freq, args.num_layers)

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        if layer_types[layer_idx] == "linear_attention":
            checkpoint.update(convert_kda_mixer_weights(model_dict, layer_idx))
        else:
            checkpoint.update(convert_mla_attn_weights(model_dict, layer_idx))

        mlp_weight = model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight']
        ffn_hidden_size = mlp_weight.shape[0] // 2
        checkpoint[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = mlp_weight[:ffn_hidden_size, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = mlp_weight[ffn_hidden_size:, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc2.weight']

        checkpoint[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight']

    checkpoint['model.norm.weight'] = model_dict['decoder.final_layernorm.weight']

    if not args.untie_embeddings_and_output_weights:
        checkpoint['lm_head.weight'] = checkpoint['model.embed_tokens.weight']
    else:
        checkpoint['lm_head.weight'] = model_dict['output_layer.weight']

    return checkpoint

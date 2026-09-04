"""Config + state-dict conversion for the Qwen-style hybrid.

build_state_dict branches per-layer on layer_types: gdn.py's
convert_gdn_mixer_weights for linear_attention layers, gated.py's convert_gated_attn_weights
for full_attention layers. MLP/embeddings/final-norm/lm_head handling is identical regardless
of mixer type and follows full.py's/gated.py's/gdn.py's shared shape.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

from attn_bench.checkpoint_conversion.attn_families.full import \
    ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS
from attn_bench.checkpoint_conversion.attn_families.gated import \
    convert_gated_attn_weights
from attn_bench.checkpoint_conversion.attn_families.gdn import \
    convert_gdn_mixer_weights
from attn_bench.checkpoint_conversion.attn_families.modeling_qwen_llama import (
    QwenLlamaConfig, QwenLlamaForCausalLM, compute_layer_types)


def build_config(args: Any) -> QwenLlamaConfig:
    """Build the HF config for a Qwen-hybrid checkpoint. Attention-side fields match
    gated.py's build_config (see full.py's docstring for the two args-naming fixes this fork
    needs); linear-side fields match gdn.py's build_config. linear_attention_freq is not
    restored by --use-checkpoint-args (see llama_checkpoints.sh's QWEN_DIMS), so args carries
    whatever was re-passed on the CLI (e.g. 4)."""
    return QwenLlamaConfig(
        attention_bias=False,
        attention_dropout=args.attention_dropout,
        bos_token_id=128000,
        eos_token_id=128001,
        head_dim=int(args.hidden_size / args.num_attention_heads),
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        max_position_embeddings=131072,
        mlp_bias=False,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_query_groups,
        pretraining_tp=1,
        rms_norm_eps=args.layernorm_epsilon,
        rope_scaling={
            "factor": args.rope_scaling_factor,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "rope_type": "llama3"
        },
        rope_theta=args.rotary_base,
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


def build_model(config: QwenLlamaConfig) -> QwenLlamaForCausalLM:
    """Construct an uninitialized model and register it for auto_map -- same as gdn.py's/
    gated.py's build_model."""
    model = QwenLlamaForCausalLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    return model


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a Qwen-hybrid Megatron state dict to QwenLlamaForCausalLM format."""
    checkpoint = OrderedDict()
    # Not model_dict['decoder.layers.0...linear_qkv.weight'].shape[1] (full.py's/gated.py's
    # convention) -- layer 0 is a linear_attention layer in this hybrid and has no linear_qkv
    # key at all. The embedding table's second dim is hidden_size regardless of layer 0's type.
    hidden_size = model_dict['embedding.word_embeddings.weight'].shape[1]
    layer_types = compute_layer_types(args.linear_attention_freq, args.num_layers)

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        if layer_types[layer_idx] == "linear_attention":
            checkpoint.update(convert_gdn_mixer_weights(model_dict, layer_idx))
        else:
            checkpoint.update(convert_gated_attn_weights(
                model_dict, layer_idx, args.num_attention_heads, args.num_query_groups, hidden_size
            ))

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

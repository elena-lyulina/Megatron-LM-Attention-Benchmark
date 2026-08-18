"""Config + state-dict conversion for sink-attention (softmax_type='learnable') checkpoints.

Targets the custom SinkLlamaForCausalLM architecture (modeling_sink_llama.py) rather than a
built-in HF one -- gpt_oss's sink math matches ours, but its MoE/activation function doesn't,
so its architecture can't just be reused (see hf_inference_kernel_support.md). Everything
except the sink parameter itself and attention dispatch is identical to full.py's Llama
conversion, reused directly rather than duplicated.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

from attn_bench.checkpoint_conversion.attn_families.full import (
    ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, convert_qkv_weights)
from attn_bench.checkpoint_conversion.attn_families.modeling_sink_llama import (
    SinkLlamaConfig, SinkLlamaForCausalLM)


def build_config(args: Any) -> SinkLlamaConfig:
    """Build the HF config for a sink-attention checkpoint. Same fields as full.py's
    build_config (see its docstring for the two args-naming fixes this fork needs);
    sink itself is a per-layer weight, not a config field."""
    return SinkLlamaConfig(
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
        vocab_size=args.padded_vocab_size
    )


def build_model(config: SinkLlamaConfig) -> SinkLlamaForCausalLM:
    """Construct an uninitialized model and register it for auto_map -- required for a
    custom (non-built-in) architecture to be loadable later via
    AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True), which is how
    HFBackend loads every HF checkpoint regardless of family."""
    model = SinkLlamaForCausalLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    return model


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a sink-attention Megatron state dict to SinkLlamaForCausalLM format."""
    checkpoint = OrderedDict()
    hidden_size = model_dict['decoder.layers.0.self_attention.linear_qkv.weight'].shape[1]

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        qkv_weights = model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.weight']
        q_weights, k_weights, v_weights = convert_qkv_weights(
            qkv_weights, args.num_attention_heads, args.num_query_groups, hidden_size
        )

        checkpoint[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = q_weights
        checkpoint[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = k_weights
        checkpoint[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = v_weights

        checkpoint[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_proj.weight']

        checkpoint[f'model.layers.{layer_idx}.self_attn.learnable_sink'] = \
            model_dict[f'decoder.layers.{layer_idx}.self_attention.core_attention.softmax_offset']

        mlp_weight = model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight']
        ffn_hidden_size = mlp_weight.shape[0] // 2
        checkpoint[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = mlp_weight[:ffn_hidden_size, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = mlp_weight[ffn_hidden_size:, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc2.weight']

        checkpoint[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight']
        checkpoint[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight']

    checkpoint['model.norm.weight'] = model_dict['decoder.final_layernorm.weight']

    if not args.untie_embeddings_and_output_weights:
        checkpoint['lm_head.weight'] = checkpoint['model.embed_tokens.weight']
    else:
        checkpoint['lm_head.weight'] = model_dict['output_layer.weight']

    return checkpoint

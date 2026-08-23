"""Config + state-dict conversion for pure Gated Delta Net (GDN) checkpoints
(--experimental-attention-variant gated_delta_net). Targets the custom GDNLlamaForCausalLM
architecture (modeling_gdn_llama.py). Everything except the mixer itself (MLP, embeddings,
final norm, lm_head) is identical to full.py's Llama conversion.

build_state_dict is a near-literal key rename, not a reshape-and-split like full.py's/gated.py's
convert_qkv_weights/convert_qkvg_weights, since GDNMixer keeps Megatron's own in_proj/conv1d
layout -- see modeling_gdn_llama.py.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

from attn_bench.checkpoint_conversion.attn_families.modeling_gdn_llama import (
    GDNLlamaConfig, GDNLlamaForCausalLM)


def build_config(args: Any) -> GDNLlamaConfig:
    """Build the HF config for a GDN checkpoint. Backbone fields match full.py's build_config
    (see its docstring for the two args-naming fixes this fork needs); no RoPE fields -- GDN's
    Q/K get L2-normed, not rotated."""
    return GDNLlamaConfig(
        bos_token_id=128000,
        eos_token_id=128001,
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        max_position_embeddings=131072,
        mlp_bias=False,
        num_hidden_layers=args.num_layers,
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see full.py's module docstring.
        rms_norm_eps=args.layernorm_epsilon,
        linear_num_key_heads=args.linear_num_key_heads,
        linear_num_value_heads=args.linear_num_value_heads,
        linear_key_head_dim=args.linear_key_head_dim,
        linear_value_head_dim=args.linear_value_head_dim,
        linear_conv_kernel_dim=args.linear_conv_kernel_dim,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        torch_dtype=args.params_dtype,
        use_cache=True,
        vocab_size=args.padded_vocab_size,
    )


def build_model(config: GDNLlamaConfig) -> GDNLlamaForCausalLM:
    """Construct an uninitialized model and register it for auto_map -- required for a
    custom (non-built-in) architecture to be loadable later via
    AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True), which is how
    HFBackend loads every HF checkpoint regardless of family."""
    model = GDNLlamaForCausalLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    return model


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a GDN Megatron state dict to GDNLlamaForCausalLM format."""
    checkpoint = OrderedDict()

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        prefix = f'decoder.layers.{layer_idx}.self_attention.'
        checkpoint[f'model.layers.{layer_idx}.mixer.in_proj.weight'] = model_dict[prefix + 'in_proj.weight']
        checkpoint[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
            model_dict[prefix + 'in_proj.layer_norm_weight']
        checkpoint[f'model.layers.{layer_idx}.mixer.conv1d.weight'] = model_dict[prefix + 'conv1d.weight']
        if prefix + 'conv1d.bias' in model_dict:
            checkpoint[f'model.layers.{layer_idx}.mixer.conv1d.bias'] = model_dict[prefix + 'conv1d.bias']
        checkpoint[f'model.layers.{layer_idx}.mixer.dt_bias'] = model_dict[prefix + 'dt_bias']
        checkpoint[f'model.layers.{layer_idx}.mixer.A_log'] = model_dict[prefix + 'A_log']
        checkpoint[f'model.layers.{layer_idx}.mixer.out_norm.weight'] = model_dict[prefix + 'out_norm.weight']
        checkpoint[f'model.layers.{layer_idx}.mixer.out_proj.weight'] = model_dict[prefix + 'out_proj.weight']

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

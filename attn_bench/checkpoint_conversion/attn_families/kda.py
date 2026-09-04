"""Config + state-dict conversion for pure Kimi Delta Attention (KDA) checkpoints
(--experimental-attention-variant kimi_delta_attention). Targets the custom KDALlamaForCausalLM
architecture (modeling_kda_llama.py); the non-mixer parts (MLP, embeddings, final norm, lm_head)
follow the standard Llama backbone conversion (as in full.py).

Sources for the KDA-specific mapping:
- megatron/core/ssm/kimi_delta_attention.py -- the module this converts from: the six projection
  names (q_proj / k_proj / v_proj / b_proj, and the low-rank f_proj_down / f_proj_up,
  g_proj_down / g_proj_up), the single fused conv1d over cat([q, k, v]), out_norm / out_proj, and
  the A_log (per value-head) / dt_bias (per value-head * key channel) shapes.
- get_kimi_delta_attention_module_spec (megatron/core/models/gpt/
  experimental_attention_variant_module_specs.py) -- fuse_input_layernorm=False, so the pre-mixer
  RMSNorm is a standalone `decoder.layers.{i}.input_layernorm.weight`, not a projection-fused
  `*.layer_norm_weight` (contrast full/gdn).
- Megatron-Bridge models/kimi/kimi_k3_bridge.py -- independent confirmation that the
  HF <-> Megatron KDA parameter mapping is a straight rename, no reshape (its only non-trivial op
  is an A_log zero-pad strip/restore for inactive heads, which this 16/16 config does not need).
  NB: Kimi-K3's HF layout splits the conv into q_conv1d / k_conv1d / v_conv1d and its model is a
  KDA+MLA+MoE hybrid -- only the projection / gate / norm names transfer, not K3's conv or MoE.
- Algorithm: "Kimi Linear" (arXiv 2510.26692); FLA reference `fla/layers/kda.py`.

build_state_dict is therefore a near-literal key rename.
"""

from collections import OrderedDict
from typing import Any, Dict

import torch

from attn_bench.checkpoint_conversion.attn_families.modeling_kda_llama import (
    KDALlamaConfig, KDALlamaForCausalLM)


def build_config(args: Any) -> KDALlamaConfig:
    """Build the HF config for a KDA checkpoint. Backbone fields follow full.py's build_config
    (see its docstring for the two args-naming fixes this fork needs); no RoPE fields -- KDA has
    no rotary embeddings, its Q/K are L2-normed in-kernel (use_qk_l2norm_in_kernel), see
    megatron/core/ssm/kimi_delta_attention.py."""
    return KDALlamaConfig(
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


def build_model(config: KDALlamaConfig) -> KDALlamaForCausalLM:
    """Construct an uninitialized model and register it for auto_map -- required for a custom
    (non-built-in) architecture to be loadable later via AutoModelForCausalLM.from_pretrained(
    ..., trust_remote_code=True), which is how HFBackend loads every HF checkpoint."""
    model = KDALlamaForCausalLM(config)
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    return model


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a KDA Megatron state dict to KDALlamaForCausalLM format."""
    checkpoint = OrderedDict()

    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    for layer_idx in range(args.num_layers):
        prefix = f'decoder.layers.{layer_idx}.self_attention.'
        m = f'model.layers.{layer_idx}.mixer.'

        for name in ('q_proj', 'k_proj', 'v_proj', 'b_proj',
                     'f_proj_down', 'f_proj_up', 'g_proj_down', 'g_proj_up'):
            checkpoint[m + f'{name}.weight'] = model_dict[prefix + f'{name}.weight']

        checkpoint[m + 'conv1d.weight'] = model_dict[prefix + 'conv1d.weight']
        if prefix + 'conv1d.bias' in model_dict:
            checkpoint[m + 'conv1d.bias'] = model_dict[prefix + 'conv1d.bias']
        checkpoint[m + 'dt_bias'] = model_dict[prefix + 'dt_bias']
        checkpoint[m + 'A_log'] = model_dict[prefix + 'A_log']
        checkpoint[m + 'out_norm.weight'] = model_dict[prefix + 'out_norm.weight']
        checkpoint[m + 'out_proj.weight'] = model_dict[prefix + 'out_proj.weight']

        # Standalone pre-mixer RMSNorm: get_kimi_delta_attention_module_spec sets
        # fuse_input_layernorm=False, so it is its own key, not a projection-fused
        # `*.layer_norm_weight`.
        checkpoint[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.input_layernorm.weight']

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

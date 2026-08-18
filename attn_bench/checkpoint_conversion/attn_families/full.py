"""Config + state-dict conversion for plain full-attention checkpoints -- the only family
with a native HF architecture (LlamaForCausalLM).

Two fixes vs PDM's original convert_megatron_to_hf.py, both needed because this fork
renamed/repurposed things PDM's original assumes are unchanged:

1. rope_scaling.original_max_position_embeddings: ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS
   (below) instead of args.max_position_embeddings -- the latter is this checkpoint's
   buffer-size setting (e.g. 131072), a different value than the true rope-scaling base.

2. rms_norm_eps: args.layernorm_epsilon instead of args.norm_epsilon -- --norm-epsilon is
   still the CLI flag name in this fork, but it sets args.layernorm_epsilon;
   args.norm_epsilon never exists.
"""

from collections import OrderedDict
from typing import Any, Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
# utils.py isn't forked -- it's PDM's own, unmodified. Not a local import: made available
# only via the scoped PYTHONPATH convert_and_validate_hf.slurm sets for Step 2.
from utils import is_rank_0

# Always 8192 in this project, independent of args.max_position_embeddings. NOT the same
# thing as config.original_max_position_embeddings in TransformerConfig (default 4096) --
# that's a YaRN-specific field used only by Multi-Latent Attention models, unrelated to the
# plain RotaryEmbedding._apply_scaling default this constant mirrors. None of this project's
# models use MLA/YaRN.
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192


def build_config(args: Any) -> PretrainedConfig:
    """Build the HF config for a full-attention checkpoint."""
    if is_rank_0():
        used_args = {
            'attention_dropout': args.attention_dropout,
            'hidden_size': args.hidden_size,
            'num_attention_heads': args.num_attention_heads,
            'ffn_hidden_size': args.ffn_hidden_size,
            'num_layers': args.num_layers,
            'num_query_groups': args.num_query_groups,
            'layernorm_epsilon': args.layernorm_epsilon,
            'rope_scaling_factor': args.rope_scaling_factor,
            'max_position_embeddings': args.max_position_embeddings,
            'rotary_base': args.rotary_base,
            'untie_embeddings_and_output_weights': args.untie_embeddings_and_output_weights,
            'params_dtype': args.params_dtype,
            'padded_vocab_size': args.padded_vocab_size
        }
        print("\nModel Arguments:")
        print("=" * 90)
        for key, value in used_args.items():
            dots = "." * (60 - len(key))
            print(f"{key}{dots}{str(value):>30}")

    return AutoConfig.for_model(
        architectures=["LlamaForCausalLM"],
        attention_bias=False,
        attention_dropout=args.attention_dropout,
        bos_token_id=128000,
        eos_token_id=128001,
        head_dim=int(args.hidden_size/args.num_attention_heads),
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        max_position_embeddings=131072,
        mlp_bias=False,
        model_type="llama",
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_query_groups,
        pretraining_tp=1,
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see module docstring.
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


def build_model(config: PretrainedConfig) -> AutoModelForCausalLM:
    """Construct an uninitialized model from config -- standard registered architecture,
    no auto_map/trust_remote_code needed (unlike custom families e.g. sink)."""
    return AutoModelForCausalLM.from_config(config)


def convert_qkv_weights(qkv_weights: torch.Tensor, num_heads: int,
                       num_query_groups: int, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Megatron's merged QKV weight into separate Q, K, V weights for HuggingFace."""
    head_size = hidden_size // num_heads
    heads_per_group = num_heads // num_query_groups
    qkv_total_dim = num_heads + 2 * num_query_groups

    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    q_slice = torch.cat([
        torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
        for i in range(num_query_groups)
    ])
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    return (
        qkv_weights[q_slice].reshape(-1, hidden_size),
        qkv_weights[k_slice].reshape(-1, hidden_size),
        qkv_weights[v_slice].reshape(-1, hidden_size)
    )


def build_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """Convert a full-attention Megatron state dict to HF LlamaForCausalLM format."""
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

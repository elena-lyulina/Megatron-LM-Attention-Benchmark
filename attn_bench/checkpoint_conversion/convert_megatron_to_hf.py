"""
Copied from PDM/src/convert/convert_megatron_to_hf.py. Two fixes in create_llama_config, both
needed because this fork renamed/repurposed things PDM's original assumes are unchanged:

1. rope_scaling.original_max_position_embeddings: ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS (see
   below) instead of args.max_position_embeddings -- the latter is this checkpoint's buffer-size
   setting (e.g. 131072), a different value than the true rope-scaling base.

2. rms_norm_eps: args.layernorm_epsilon instead of args.norm_epsilon -- --norm-epsilon is still
   the CLI flag name in this fork, but it sets args.layernorm_epsilon; args.norm_epsilon never
   exists.

utils.py isn't forked (unmodified) -- convert_and_validate_hf.slurm puts PDM's src/convert/ on
PYTHONPATH for this script only.

Usage (see attn_bench/submissions/convert_and_validate_hf.slurm):
    python attn_bench/checkpoint_conversion/convert_megatron_to_hf.py --experiment-path <dir>
"""
from utils import (
    is_rank_0,
    is_model_converted,
    clear_and_create_directory,
    extract_iteration_number,
)
from typing import Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
import torch
from collections import OrderedDict

# See module docstring -- always 8192 in this project, independent of args.max_position_embeddings.
# NOT the same thing as config.original_max_position_embeddings in TransformerConfig (default
# 4096) -- that's a YaRN-specific field used only by Multi-Latent Attention models, unrelated to
# the plain RotaryEmbedding._apply_scaling default this constant mirrors. Same name, two
# unconnected mechanisms; none of this project's models use MLA/YaRN.
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8192


def create_llama_config(args: Any) -> AutoConfig:
    """
    Create a HuggingFace config object for LLaMA model from Megatron args.

    Args:
        args: Namespace object containing Megatron model arguments

    Returns:
        AutoConfig: HuggingFace config object for LLaMA
    """
    # Print all arguments with aligned formatting
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
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork, not
        # args.norm_epsilon -- see module docstring.
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


def convert_qkv_weights(qkv_weights: torch.Tensor, num_heads: int,
                       num_query_groups: int, hidden_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert merged QKV weights from Megatron format to separate Q, K, V weights for HuggingFace.

    Args:
        qkv_weights: Combined QKV weights tensor
        num_heads: Number of attention heads
        num_query_groups: Number of query groups for grouped query attention
        hidden_size: Hidden size of the model

    Returns:
        Tuple of (query_weights, key_weights, value_weights)
    """
    head_size = hidden_size // num_heads
    heads_per_group = num_heads // num_query_groups
    qkv_total_dim = num_heads + 2 * num_query_groups

    qkv_weights = qkv_weights.reshape([qkv_total_dim, head_size, hidden_size])

    # Calculate indices for Q, K, V separation
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


def convert_megatron_to_hf_state_dict(model_dict: Dict[str, torch.Tensor], args: Any) -> OrderedDict:
    """
    Convert Megatron state dict to HuggingFace format.

    Args:
        model_dict: Megatron model state dictionary
        args: Namespace object containing model arguments

    Returns:
        OrderedDict: HuggingFace format state dictionary
    """
    checkpoint = OrderedDict()
    hidden_size = model_dict['decoder.layers.0.self_attention.linear_qkv.weight'].shape[1]

    # Save embedding
    checkpoint['model.embed_tokens.weight'] = model_dict['embedding.word_embeddings.weight']

    # Process each transformer layer
    for layer_idx in range(args.num_layers):
        # Handle QKV weights
        qkv_weights = model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.weight']
        q_weights, k_weights, v_weights = convert_qkv_weights(
            qkv_weights, args.num_attention_heads, args.num_query_groups, hidden_size
        )

        checkpoint[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = q_weights
        checkpoint[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = k_weights
        checkpoint[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = v_weights

        # Save attention output projection
        checkpoint[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_proj.weight']

        # Handle MLP weights
        mlp_weight = model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.weight']
        ffn_hidden_size = mlp_weight.shape[0] // 2
        checkpoint[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = mlp_weight[:ffn_hidden_size, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = mlp_weight[ffn_hidden_size:, :]
        checkpoint[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc2.weight']

        # Save layer norms
        checkpoint[f'model.layers.{layer_idx}.input_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.self_attention.linear_qkv.layer_norm_weight']
        checkpoint[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = \
            model_dict[f'decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight']

    # Save final layer norm
    checkpoint['model.norm.weight'] = model_dict['decoder.final_layernorm.weight']

    # Handle output layer (weight tying if needed)
    if not args.untie_embeddings_and_output_weights:
        checkpoint['lm_head.weight'] = checkpoint['model.embed_tokens.weight']
    else:
        checkpoint['lm_head.weight'] = model_dict['output_layer.weight']

    return checkpoint


def convert_megatron_checkpoint_to_hf(checkpoint_path: str,
                                    map_location: str = 'cpu') -> Tuple[AutoModelForCausalLM, int]:
    """
    Convert a Megatron-LM checkpoint to a HuggingFace model.

    Implementation adopted from:
    https://github.com/TJ-Solergibert/NeMo/blob/825c246b12e76ee7e9b3cdf01aea9c9dacdc03fe/scripts/checkpoint_converters/convert_llama_nemo_to_hf.py#L106

    ALERT: This implementation only supports loading from a single model parallel rank (mp_rank_00).
    To handle model parallel checkpoints, you would need to merge weights from all mp_ranks first, you can find it here:
    https://github.com/swiss-ai/Megatron-LM/blob/main/README_orig.md#checkpoint-conversion

    Args:
        checkpoint_path: Path to the Megatron checkpoint file
        map_location: Device to load the checkpoint to

    Returns:
        The converted HuggingFace model
        The model's configuration
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
    args = checkpoint['args']

    # Create HF config
    config = create_llama_config(args)

    # Convert state dict
    model_dict = checkpoint['model']
    hf_dict = convert_megatron_to_hf_state_dict(model_dict, args)

    # Create and load model
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(hf_dict)

    # Calculate trainable parameters
    if is_rank_0():
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, config


def convert_and_save_checkpoint(checkpoint_path, output_dir):
    """Convert a checkpoint and save it to the specified output directory."""
    print(f"\nConverting checkpoint from: {checkpoint_path}")
    model, config = convert_megatron_checkpoint_to_hf(str(checkpoint_path))

    # Save model and config
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    print(f"Conversion complete!")
    print(f"Model saved to: {output_dir}")


def handle_single_checkpoint(checkpoint_path, experiment_path):
    """Process a single checkpoint with the original behavior."""
    hf_dir = Path(experiment_path) / "HF"

    if is_model_converted(hf_dir):
        print(f"\nHuggingFace model already exists at: {hf_dir}")
        print("Skipping conversion. Delete the HF directory if you want to reconvert.")
        return

    clear_and_create_directory(hf_dir)
    convert_and_save_checkpoint(checkpoint_path, hf_dir)


def handle_multiple_checkpoints(checkpoint_paths, experiment_path):
    """Process multiple checkpoints, saving each to its own iteration directory."""
    print(f"\nFound {len(checkpoint_paths)} checkpoints. Converting each to its own directory...")

    # Create base HF directory if it doesn't exist
    base_hf_dir = Path(experiment_path) / "HF"
    if not base_hf_dir.exists():
        os.makedirs(base_hf_dir)

    # Process each checkpoint
    for checkpoint_path in sorted(checkpoint_paths):
        iter_num = extract_iteration_number(checkpoint_path)
        if not iter_num:
            print(f"Warning: Could not extract iteration number from {checkpoint_path}, skipping")
            continue

        iter_dir = base_hf_dir / f"iter_{iter_num}"

        # Check if this iteration has already been converted
        if is_model_converted(iter_dir):
            print(f"\nHuggingFace model for iteration {iter_num} already exists at: {iter_dir}")
            print("Skipping conversion. Delete the directory if you want to reconvert.")
            continue

        clear_and_create_directory(iter_dir)
        convert_and_save_checkpoint(checkpoint_path, iter_dir)

    print("\nAll checkpoint conversions completed!")


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    import os

    parser = argparse.ArgumentParser(description='Convert SwissAI Megatron checkpoint to HuggingFace')
    parser.add_argument('--experiment-path', type=str, required=True,
                      help='Path to experiment directory')

    args = parser.parse_args()

    # Check if experiment path exists
    if not os.path.exists(args.experiment_path):
        print(f"Error: Experiment path not found: {args.experiment_path}", file=sys.stderr)
        sys.exit(1)

    # Find all checkpoint paths
    checkpoint_paths = list(Path(args.experiment_path).glob('torch/iter_*/mp_rank_00/model_optim_rng.pt'))

    if not checkpoint_paths:
        print(f"Error: No checkpoints found in {args.experiment_path}/torch/iter_*/mp_rank_00/", file=sys.stderr)
        sys.exit(1)

    # Branch based on number of checkpoints
    if len(checkpoint_paths) == 1:
        handle_single_checkpoint(checkpoint_paths[0], args.experiment_path)
    else:
        handle_multiple_checkpoints(checkpoint_paths, args.experiment_path)

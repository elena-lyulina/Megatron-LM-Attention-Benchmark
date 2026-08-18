"""
Copied from PDM/src/convert/convert_megatron_to_hf.py, then generalized to dispatch across
attention families (attn_families/) instead of assuming plain full-attention Llama -- see
attn_families/registry.py for how a checkpoint's own args determine which attn_family it is, and
attn_families/full.py for the two fixes this fork needed vs PDM's original.

utils.py isn't forked (unmodified) -- convert_and_validate_hf.slurm puts PDM's src/convert/ on
PYTHONPATH for this script only.

Usage (see attn_bench/submissions/convert_and_validate_hf.slurm):
    python attn_bench/checkpoint_conversion/convert_megatron_to_hf.py --experiment-path <dir>
"""
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM
from utils import (clear_and_create_directory, extract_iteration_number,
                   is_model_converted, is_rank_0)

from attn_bench.checkpoint_conversion.attn_families.registry import \
    get_attn_family_module


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
    attn_family = get_attn_family_module(args)

    # Create HF config
    config = attn_family.build_config(args)

    # Convert state dict
    model_dict = checkpoint['model']
    hf_dict = attn_family.build_state_dict(model_dict, args)

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
    import os
    import sys
    from pathlib import Path

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

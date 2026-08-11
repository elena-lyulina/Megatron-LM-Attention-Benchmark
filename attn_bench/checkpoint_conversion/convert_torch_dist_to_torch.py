"""
Copied from PDM/src/convert/convert_torch_dist_to_torch.py, with one change: calls
setup_model_and_optimizer with partial(model_provider, gpt_builder) instead of PDM's bare
pretrain_gpt.model_provider. Required because this fork's model_provider now takes an
explicit model_builder arg -- unrelated to attention variants. Everything else unchanged.

Usage (see attn_bench/submissions/convert_and_validate_hf.slurm for the real invocation).
Architecture flags (num-layers, hidden-size, etc.) come from the checkpoint itself via
--use-checkpoint-args (set below in args_defaults); only rope scaling needs to be passed
explicitly, since the checkpoint can't restore it (NVIDIA/Megatron-LM#6306):
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun attn_bench/checkpoint_conversion/convert_torch_dist_to_torch.py \
    --bf16 \
    --use-precision-aware-optimizer \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --main-grads-dtype bf16 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model TOKENIZER_PATH \
    --use-rope-scaling \
    --rope-scaling-factor 8 \
    --load CHECKPOINT_PATH \
    --ckpt-convert-save INTERMEDIATE_CHECKPOINT_PATH
"""
from functools import partial

from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_and_validate_args
from megatron.training.global_vars import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import setup_model_and_optimizer
from model_provider import \
    model_provider  # this project's own, not PDM's pretrain_gpt.model_provider


def main():

    # Apply ALL model configuration parameters directly
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "ckpt_format": "torch_dist",
        "ckpt_convert_format": "torch",
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        "--untie-embeddings-and-output-weights": True,

        # Fake args for initialization
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }

    # args_defaults now belongs to parse_and_validate_args, not initialize_megatron
    # (arg parsing was moved out of initialize_megatron, see inference_common.py's
    # load_megatron_model for the same pattern).
    parse_and_validate_args(args_defaults=args_defaults)
    initialize_megatron()
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"
    setup_model_and_optimizer(partial(model_provider, gpt_builder), ModelType.encoder_or_decoder)

if __name__ == "__main__":
    main()

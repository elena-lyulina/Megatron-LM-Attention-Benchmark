#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=check-precision-baseline
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# One-off: isolates how much argmax disagreement bf16 rounding alone causes for a given
# Megatron model, with no HF/kernel comparison involved -- runs the SAME Megatron model on the
# SAME random tokens at fp32 and at bf16, and reports self-agreement. This is the precision-only
# baseline to compare against compare_megatron_hf_logits.py's cross-backend argmax agreement
# (e.g. sink-scf1's 94.45% at bf16): if this baseline lands in the same range, the cross-backend
# gap is explained by precision noise rather than a wiring bug.
#
# Usage: sbatch --export=MODEL=sink-scf1 check_megatron_precision_baseline.slurm

set -euo pipefail

CONTAINER_ENV=${CONTAINER_ENV:-nemo_26.04_te2.15}
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL=${MODEL:-sink-scf1}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

SCRATCH_DIR=/users/$USER/scratch/tmp/check_precision_baseline_$SLURM_JOB_ID
mkdir -p "$SCRATCH_DIR"

cat > "$SCRATCH_DIR/check_precision_baseline.py" <<'PYEOF'
import argparse
import torch

from attn_bench.evaluation.inference_backend import MegatronBackend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    backend.load_model()
    device = backend.device

    vocab_size = backend.model.state_dict()['embedding.word_embeddings.weight'].shape[0]

    torch.manual_seed(0)
    tokens = torch.randint(0, vocab_size, (args.batch_size, args.seq_length), device=device)
    position_ids = torch.arange(args.seq_length, device=device).unsqueeze(0).expand(args.batch_size, -1)

    backend.model = backend.model.to(torch.float32)
    with torch.no_grad():
        preds_fp32 = torch.max(backend.forward_logits(tokens, position_ids), dim=-1)[1]
    torch.cuda.empty_cache()

    backend.model = backend.model.to(torch.bfloat16)
    with torch.no_grad():
        preds_bf16 = torch.max(backend.forward_logits(tokens, position_ids), dim=-1)[1]

    agree = torch.sum(preds_fp32 == preds_bf16) / preds_fp32.numel()
    print(f"Megatron self-agreement, fp32 vs bf16, same model/same tokens "
          f"(precision-only baseline): {100 * agree:.2f}% argmax agreement")


if __name__ == "__main__":
    main()
PYEOF

srun --environment=$CONTAINER_ENV bash -c "
export PYTHONPATH='${PYTHONPATH}'
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun --nproc_per_node=1 '$SCRATCH_DIR/check_precision_baseline.py' \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --seq-length 8192 \
    --batch-size 1 \
    --megatron-extra-args $MEGATRON_EXTRA
"

rm -rf "$SCRATCH_DIR"

#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=debug-bf16-inference-backends
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Standalone re-run of convert_and_validate_hf.slurm's Step 3 (compare_megatron_hf_logits.py),
# generalized to any model and a shorter --seq-length -- checks whether a Step-3 failure is
# precision compounding over a long sequence vs a real bug near the start.
#
# SELF_COMPARISON=1 also forwards each backend twice against itself, as a kernel-nondeterminism
# noise floor to compare the cross-backend numbers against.
#
# Does NOT redo Steps 1/2 -- reuses the existing scratch HF/ dir from convert_and_validate_hf.slurm.
#
# Env vars:
#   MODEL            checkpoint tag (default: gdn), see llama_checkpoints.sh
#   SEQ_LENGTH       logits comparison sequence length (default: 128, vs Step 3's default 8192)
#   SELF_COMPARISON  1 to also run the two self-comparisons (default: 0, cross-backend only)
#
# Usage: sbatch debug_bf16_inference_backends.slurm
#        sbatch --export=MODEL=sink-scf1,SELF_COMPARISON=1 debug_bf16_inference_backends.slurm
#        sbatch --export=MODEL=gdn,SELF_COMPARISON=1,SEQ_LENGTH=8192 debug_bf16_inference_backends.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh

MODEL=${MODEL:-gdn}
SEQ_LENGTH=${SEQ_LENGTH:-128}
SELF_COMPARISON=${SELF_COMPARISON:-0}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
SCRATCH_CONVERT_DIR=/iopsstor/scratch/cscs/$USER/hf-checkpoints/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
CONV_DIR=$MEGATRON_LM_DIR/attn_bench/checkpoint_conversion

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ ! -f "$SCRATCH_CONVERT_DIR/HF/config.json" ]; then
    echo "$SCRATCH_CONVERT_DIR/HF not found -- convert_and_validate_hf.slurm's scratch output"
    echo "must have been cleaned up. Re-run convert_and_validate_hf.slurm (Steps 1-2) first."
    exit 1
fi

SELF_COMPARISON_ARG=""
[ "$SELF_COMPARISON" = "1" ] && SELF_COMPARISON_ARG="--self-comparison"

TRITON_ENV=""
if [ "$NEEDS_TRITON" = "1" ]; then
    TRITON_ENV="export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_\$SLURM_PROCID; mkdir -p \$TRITON_CACHE_DIR;"
fi

SINK_ENV=""
if [ "$IS_SINK_FAMILY" = "1" ]; then
    SINK_ENV="export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention; pip install --quiet 'nvidia-cutlass-dsl==4.6.2';"
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME  seq_length=$SEQ_LENGTH  self_comparison=$SELF_COMPARISON"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    $TRITON_ENV
    $SINK_ENV

    echo '### compare_megatron_hf_logits.py (standalone, short seq_length) ###'
    torchrun --nproc_per_node=1 $CONV_DIR/compare_megatron_hf_logits.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        $SELF_COMPARISON_ARG \
        --batch-size 1 \
        --seq-length $SEQ_LENGTH \
        --dtype bfloat16 \
        --megatron-extra-args $MEGATRON_EXTRA
    "

echo "END TIME: $(date)"

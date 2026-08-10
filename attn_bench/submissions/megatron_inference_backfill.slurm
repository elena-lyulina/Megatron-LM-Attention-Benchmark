#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=backfill-mem
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Phase A: backfill one trained model's existing memorization results with sample_idx
# and full per-position ref_nll/gen_nll/p_z, across every (offset, prefix, suffix) point
# already on store for it. One-time migration -- see megatron_inference_backfill.py.
#
# Env vars:
#   MODEL     checkpoint to use, see llama_checkpoints.sh (default: full-scf8)
#
# Usage: sbatch --export=MODEL=sink-scf8 megatron_inference_backfill.slurm
# (normally submitted via megatron_inference_backfill_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh

MODEL=${MODEL:-full-scf8}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_MEM_BASE=/users/$USER/store/mem-results
SCRATCH_MEM_BASE=/iopsstor/scratch/cscs/$USER/mem-results
MEM_DIR=$SCRATCH_MEM_BASE/SparseGutenberg/$EXP_NAME
STORE_EXP_DIR=$STORE_MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_backfill.py

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ ! -d "$STORE_EXP_DIR/inference" ]; then
    echo "No existing inference results in store for $EXP_NAME -- nothing to backfill."
    exit 0
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME"

### Working copy on scratch -- backfill rewrites files in place, never touch store directly ###
mkdir -p "$MEM_DIR"
cp -a "$STORE_EXP_DIR/." "$MEM_DIR/"

COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $MEM_DIR \
    --data-folder $GUTENBERG_JSONL_DIR \
    --batch-size 20"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun --nproc_per_node=4 \
        $PY $COMMON_ARGS $EXTRA_MEGATRON"
BACKFILL_STATUS=$?

copy_scratch_to_store $BACKFILL_STATUS "$MEM_DIR" "$STORE_EXP_DIR"

echo "END TIME: $(date)"

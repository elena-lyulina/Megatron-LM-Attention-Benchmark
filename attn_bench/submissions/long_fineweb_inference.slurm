#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=1:00:00
#SBATCH --job-name=long-fineweb-inf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Per-position loss on long FineWeb-Edu documents (verbatim, no prefix/suffix), one job per
# (model, data file) pair. Results land under RESULTS_BASE/$EXP_NAME/<data file's parent dir>/
# $CONFIG/, keyed by parent dir so different data partitions don't collide on the same basename.
#
# Env vars:
#   MODEL          checkpoint to use, see llama_checkpoints.sh (default: full)
#   CHECKPOINT_BACKEND   megatron (default) or hf -- hf needs the model already converted
#                  via convert_and_validate_hf.slurm
#   DATA_FILE      required; jsonl produced by extract_long_docs.py
#   MAX_LENGTH     cap sequence length (default: uncapped)
#   MAX_SAMPLES    cap docs processed (default: 660)
#   OVERWRITE      recompute even if store already has this bucket
#   LOG_STATE_NORM / STATE_CHUNK   GDN state-norm logging (no-op for attention models)
#   STORE_INDIVIDUAL   also write raw per-sequence records to <key>_individual.jsonl
#   EXP_SUFFIX     appended to EXP_NAME for the results path only, e.g. EXP_SUFFIX=-scf-8;
#                  checkpoint lookup (CKPT_NAME/HF_DIR) is captured before the suffix is applied
#
# Usage: sbatch --export=MODEL=full-scf8,DATA_FILE=/path/to/long_24576_32768.jsonl long_fineweb_inference.slurm
#        sbatch --export=MODEL=full-scf1,CHECKPOINT_BACKEND=hf,DATA_FILE=/path/to/long_24576_32768.jsonl long_fineweb_inference.slurm
# (normally submitted via long_fineweb_inference_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/$USER/store/long-fineweb-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/long-fineweb-results
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
STORE_HF_BASE=/users/$USER/store/hf-checkpoints

DATA_FILE=${DATA_FILE:?ERROR: DATA_FILE must be set to a jsonl produced by extract_long_docs.py}
MAX_LENGTH=${MAX_LENGTH:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
OVERWRITE=${OVERWRITE:-}
LOG_STATE_NORM=${LOG_STATE_NORM:-}
STATE_CHUNK=${STATE_CHUNK:-}
STORE_INDIVIDUAL=${STORE_INDIVIDUAL:-}
EXP_SUFFIX=${EXP_SUFFIX:-}
CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-megatron}

### MODEL -> EXP_NAME + non-restored flags (registry shared with every other eval script) ###
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
MODEL=${MODEL:-full-scf8}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
HF_DIR=$STORE_HF_BASE/$EXP_NAME
EXP_NAME="${EXP_NAME}${EXP_SUFFIX}"
PARTITION_DIR=$(basename "$(dirname "$DATA_FILE")")
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
SCRATCH_RESULTS_DIR=$SCRATCH_RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_fineweb_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# No pre-checkpoint-load "is this already done" check here -- long_fineweb_inference_all.sh
# already decided this job is worth submitting (its own --dry-run check, dual-checking
# scratch+store). Once running, the .py's own pre-checkpoint-load check (same dual-checked
# logic) self-skips if everything requested turns out already complete.

# --experiment-path is the scratch working dir -- results stay there; sync to store manually
# when you want it, scratch is only purged every ~2 weeks.
# --persistent-storage-path lets the done-check see results scratch has lost to a purge.
EXTRA_ENV=""
if [ "$CHECKPOINT_BACKEND" = "hf" ]; then
    if [ ! -f "$HF_DIR/config.json" ]; then
        echo "CHECKPOINT_BACKEND=hf requested but $HF_DIR/config.json not found -- run convert_and_validate_hf.slurm (MODEL=$MODEL) first."
        exit 1
    fi
    BACKEND_ARGS="--checkpoint-backend hf --hf-dir $HF_DIR"
    EXTRA_MEGATRON=""
    if [ "$IS_SINK_FAMILY" = "1" ]; then
        # sink's HF arch needs a patched flash-attn checkout -- see flash-attention-cute-workflow.md
        EXTRA_ENV="export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention; pip install --quiet 'nvidia-cutlass-dsl==4.6.2';"
    fi
else
    BACKEND_ARGS="--checkpoint-backend megatron --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH"
    EXTRA_MEGATRON=""
    [ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"
fi

COMMON_ARGS="--container-env $CONTAINER_ENV \
    $BACKEND_ARGS \
    --experiment-path $SCRATCH_RESULTS_DIR \
    --persistent-storage-path $RESULTS_DIR \
    --data-file $DATA_FILE"
[ -n "$MAX_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS="$COMMON_ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && COMMON_ARGS="$COMMON_ARGS --overwrite"
[ -n "$LOG_STATE_NORM" ] && COMMON_ARGS="$COMMON_ARGS --log-state-norm"
[ -n "$STATE_CHUNK" ] && COMMON_ARGS="$COMMON_ARGS --state-chunk $STATE_CHUNK"
[ -n "$STORE_INDIVIDUAL" ] && COMMON_ARGS="$COMMON_ARGS --store-individual"

echo "RUN: model=$MODEL  exp=$EXP_NAME  backend=$CHECKPOINT_BACKEND  data_file=$DATA_FILE  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-660 (default)}  triton=$NEEDS_TRITON  state_norm=${LOG_STATE_NORM:+on}  individual=${STORE_INDIVIDUAL:+on}"

### Position-wise loss inference ###
# GDN needs a per-rank node-local TRITON_CACHE_DIR: fla kernels autotune on first use and
# a shared/colliding cache crashes with "stale file handle" (job 2570257). torchrun sets
# RANK per worker (SLURM_PROCID is 0 for all four under ntasks-per-node=1), so the cache is
# keyed inside the workers via the --no-python wrapper. Attention variants skip all that.

if [ "$NEEDS_TRITON" = "1" ]; then
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 --no-python \
            bash -c 'export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}_\${RANK}; mkdir -p \"\$TRITON_CACHE_DIR\"; exec python \"\$@\"' _ \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
else
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        $EXTRA_ENV
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
fi

echo "END TIME: $(date)"

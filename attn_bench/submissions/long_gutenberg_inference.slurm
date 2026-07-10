#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=1:30:00
#SBATCH --job-name=long-gutenberg-inf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Per-position loss on long Gutenberg sequences ([BOS, sample, suffix], no prefix), one job
# per model. Network dims come from the checkpoint (--use-checkpoint-args); only flags it
# doesn't restore are re-passed. Work is split across the 4 GPUs by index striding; results
# flush per repetition bucket (<config>/rep_{R}.npz), so a restart skips finished buckets.
#
# Env vars:
#   MODEL          checkpoint to use, see llama_checkpoints.sh (default: full)
#   REPETITIONS    comma-separated rep buckets (default: 0,1,2,4,8,16,32,64,128,256)
#   MAX_LENGTH     cap sequence length (default: uncapped, up to 4L)
#   MAX_SAMPLES    cap docs processed (default: all)
#   OVERWRITE      recompute buckets that already exist
#   LOG_STATE_NORM / STATE_CHUNK   GDN state-norm logging (no-op for attention models)
#
# Usage: sbatch --export=MODEL=full long_gutenberg_inference.slurm
# (normally submitted via long_gutenberg_inference_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/$USER/store/long-gutenberg-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/long-gutenberg-results
GUTENBERG_LONG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl_long
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
MAX_LENGTH=${MAX_LENGTH:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
OVERWRITE=${OVERWRITE:-}
LOG_STATE_NORM=${LOG_STATE_NORM:-}
STATE_CHUNK=${STATE_CHUNK:-}

### MODEL -> EXP_NAME + non-restored flags (registry shared with every other eval script) ###
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh
MODEL=${MODEL:-full}
model_config "$MODEL"
# All models (including sink/off-by-one) run the same way, full length by default -- see
# llama_checkpoints.sh for why.

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME
SCRATCH_RESULTS_DIR=$SCRATCH_RESULTS_BASE/$EXP_NAME
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_gutenberg_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
# CONFIG must match config_name() in long_inference.py.
CONFIG="${MAX_SAMPLES:-all}_samples_${MAX_LENGTH:-full}_tokens"
ALL_DONE=1
IFS=',' read -ra REPS <<< "$REPETITIONS"
for R in "${REPS[@]}"; do
    [ ! -f "$RESULTS_DIR/$CONFIG/rep_${R}.npz" ] && ALL_DONE=0 && break
    if [ -n "$LOG_STATE_NORM" ] && [ ! -f "$RESULTS_DIR/$CONFIG/rep_${R}_state.npz" ]; then
        ALL_DONE=0
        break
    fi
done
if [ -z "$OVERWRITE" ] && [ $ALL_DONE -eq 1 ]; then
    echo "All requested buckets already present in $RESULTS_DIR/$CONFIG -- skipping."
    exit 0
fi

# --experiment-path is the scratch working dir; copy_scratch_to_store handles the store side.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $SCRATCH_RESULTS_DIR \
    --data-folder $GUTENBERG_LONG_JSONL_DIR \
    --repetitions $REPETITIONS"
[ -n "$MAX_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS="$COMMON_ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && COMMON_ARGS="$COMMON_ARGS --overwrite"
[ -n "$LOG_STATE_NORM" ] && COMMON_ARGS="$COMMON_ARGS --log-state-norm"
[ -n "$STATE_CHUNK" ] && COMMON_ARGS="$COMMON_ARGS --state-chunk $STATE_CHUNK"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: model=$MODEL  exp=$EXP_NAME  reps=$REPETITIONS  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-all}  triton=$NEEDS_TRITON  state_norm=${LOG_STATE_NORM:+on}"

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
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
fi
SRUN_STATUS=$?

### Copy scratch results to store ###
copy_scratch_to_store $SRUN_STATUS "$SCRATCH_RESULTS_DIR" "$RESULTS_DIR"

echo "END TIME: $(date)"
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:30:00
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
#   DATA_FILE      required; jsonl produced by extract_long_docs.py
#   TP             tensor-parallel degree (default: 1)
#   MAX_LENGTH     cap sequence length (default: uncapped)
#   MAX_SAMPLES    cap docs processed (default: 660)
#   OVERWRITE      recompute even if store already has this bucket
#   LOG_STATE_NORM / STATE_CHUNK   GDN state-norm logging (no-op for attention models)
#   STORE_INDIVIDUAL   also write raw per-sequence records to <key>_individual.jsonl
#
# Usage: sbatch --export=MODEL=full,DATA_FILE=/path/to/long_24576_32768.jsonl long_fineweb_inference.slurm
# (normally submitted via long_fineweb_inference_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/$USER/store/long-fineweb-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/long-fineweb-results
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

DATA_FILE=${DATA_FILE:?ERROR: DATA_FILE must be set to a jsonl produced by extract_long_docs.py}
MAX_LENGTH=${MAX_LENGTH:-}
MAX_SAMPLES=${MAX_SAMPLES:-}
OVERWRITE=${OVERWRITE:-}
LOG_STATE_NORM=${LOG_STATE_NORM:-}
STATE_CHUNK=${STATE_CHUNK:-}
STORE_INDIVIDUAL=${STORE_INDIVIDUAL:-}

### MODEL -> EXP_NAME + non-restored flags (registry shared with every other eval script) ###
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh
MODEL=${MODEL:-full}
TP=${TP:-1}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
PARTITION_DIR=$(basename "$(dirname "$DATA_FILE")")
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
SCRATCH_RESULTS_DIR=$SCRATCH_RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_fineweb_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
# CONFIG must match config_name() in long_inference.py: unset MAX_SAMPLES falls back to
# long_fineweb_inference.py's own DEFAULT_MAX_SAMPLES (660), not "all"; tp>1 appends "_tp{TP}".
CONFIG="${MAX_SAMPLES:-660}_samples_${MAX_LENGTH:-full}_tokens"
[ "$TP" -gt 1 ] && CONFIG="${CONFIG}_tp${TP}"
KEY=$(basename "$DATA_FILE" .jsonl)
STORE_NPZ=$RESULTS_DIR/$CONFIG/$KEY.npz
if [ -z "$OVERWRITE" ] && [ -f "$STORE_NPZ" ] \
   && { [ -z "$LOG_STATE_NORM" ] || [ -f "$RESULTS_DIR/$CONFIG/${KEY}_state.npz" ]; } \
   && { [ -z "$STORE_INDIVIDUAL" ] || [ -f "$RESULTS_DIR/$CONFIG/${KEY}_individual.jsonl" ]; }; then
    echo "$STORE_NPZ already present in store -- skipping."
    exit 0
fi

# --experiment-path is the scratch working dir; copy_scratch_to_store handles the store side.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $SCRATCH_RESULTS_DIR \
    --data-file $DATA_FILE"
[ "$TP" -gt 1 ] && COMMON_ARGS="$COMMON_ARGS --tensor-parallel $TP"
[ -n "$MAX_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS="$COMMON_ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && COMMON_ARGS="$COMMON_ARGS --overwrite"
[ -n "$LOG_STATE_NORM" ] && COMMON_ARGS="$COMMON_ARGS --log-state-norm"
[ -n "$STATE_CHUNK" ] && COMMON_ARGS="$COMMON_ARGS --state-chunk $STATE_CHUNK"
[ -n "$STORE_INDIVIDUAL" ] && COMMON_ARGS="$COMMON_ARGS --store-individual"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: model=$MODEL  exp=$EXP_NAME  data_file=$DATA_FILE  tp=$TP  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-660 (default)}  triton=$NEEDS_TRITON  state_norm=${LOG_STATE_NORM:+on}  individual=${STORE_INDIVIDUAL:+on}"

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

#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=1:00:00
#SBATCH --job-name=gdn-paired-state
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Paired complete-vs-truncated GDN state comparison, one GDN checkpoint per job, one GPU.
# See evaluation/gdn_paired_state.py. Results: <scratch>/gdn-state-results/<EXP_NAME>/
# rep_R_offset_O_prefix_P.npz (+ run_metadata.json, test.json); a finished file is skipped.
#
# Env vars:
#   MODEL          gdn (default), carry-r0, carry-r1, ... (GDN tags in llama_checkpoints.sh)
#   REPETITIONS    comma-separated rep buckets (default: 32,64,256)
#   POINTS         comma-separated truncated offset:prefix pairs (default: 50:2000); the
#                  full control (0, offset+prefix) is always run alongside
#   SUFFIX_LENGTH  scored suffix tokens (default: 249)
#   MAX_BOOKS      first N books of the seeded per-bucket permutation (default: all)
#   TEST           also compare instrumented vs plain forward on 2 books first
#   DRY_RUN        print resolved books/boundaries/work and exit, no model load
#   OVERWRITE      recompute existing files
#
# Usage: sbatch --export=ALL,MODEL=gdn,MAX_BOOKS=4,TEST=1 gdn_paired_state.slurm
#        sbatch --export=ALL,MODEL=carry-r0 gdn_paired_state.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/$USER/store/gdn-state-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/gdn-state-results
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

REPETITIONS=${REPETITIONS:-32,64,256}
POINTS=${POINTS:-50:2000}
SUFFIX_LENGTH=${SUFFIX_LENGTH:-249}
MAX_BOOKS=${MAX_BOOKS:-}
TEST=${TEST:-}
DRY_RUN=${DRY_RUN:-}
OVERWRITE=${OVERWRITE:-}

### MODEL -> EXP_NAME + non-restored flags ###
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
MODEL=${MODEL:-gdn}
model_config "$MODEL"
if [ "$NEEDS_TRITON" != "1" ]; then
    echo "MODEL=$MODEL is not a GDN checkpoint (no recurrent state to compare)."
    exit 1
fi

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME
SCRATCH_RESULTS_DIR=$SCRATCH_RESULTS_BASE/$EXP_NAME
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/gdn_paired_state.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NEEDS_FLA_052: same side-installed fla as measure_mem.slurm for models whose kernels NaN
# on the container's 0.4.2 (KDA today; harmless for GDN).
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
FLA_ENV=""
if [ "$NEEDS_FLA_052" = "1" ]; then
    FLA_ENV="export PYTHONPATH=$FLA_DIR:\$PYTHONPATH;"
fi

ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $SCRATCH_RESULTS_DIR \
    --persistent-storage-path $RESULTS_DIR \
    --data-folder $GUTENBERG_JSONL_DIR \
    --repetitions $REPETITIONS \
    --suffix-length $SUFFIX_LENGTH \
    --points ${POINTS//,/ }"
[ -n "$MAX_BOOKS" ] && ARGS="$ARGS --max-books $MAX_BOOKS"
[ -n "$TEST" ] && ARGS="$ARGS --test"
[ -n "$DRY_RUN" ] && ARGS="$ARGS --dry-run"
[ -n "$OVERWRITE" ] && ARGS="$ARGS --overwrite"
# --megatron-extra-args swallows the rest of the line, so it goes last
[ -n "$MEGATRON_EXTRA" ] && ARGS="$ARGS --megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: model=$MODEL  exp=$EXP_NAME  reps=$REPETITIONS  points=$POINTS  suffix=$SUFFIX_LENGTH  max_books=${MAX_BOOKS:-all}  test=${TEST:+on}  dry_run=${DRY_RUN:+on}"

### Paired inference ###
# Per-job node-local TRITON_CACHE_DIR: fla kernels autotune on first use and a shared cache
# crashes with "stale file handle" (job 2570257). Single rank, so no per-rank suffix needed.
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}; mkdir -p \$TRITON_CACHE_DIR
    $FLA_ENV
    RANK=0 LOCAL_RANK=0 \
    torchrun --nproc_per_node=1 $PY $ARGS"

echo "END TIME: $(date)"

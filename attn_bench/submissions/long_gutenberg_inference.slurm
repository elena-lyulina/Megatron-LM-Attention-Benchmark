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

# Per-position loss on long Gutenberg sequences ([BOS, sample, suffix], no prefix),
# one job per model. VARIANT picks the checkpoint; the network dims come from the
# checkpoint itself (--use-checkpoint-args), so only the flags that --use-checkpoint-args
# does NOT restore are re-passed here (attention toggles + GDN linear-attention dims).
# Work is split across the 4 GPUs by index striding; results flushed per repetition
# bucket (<config>/rep_{R}.npz), so a restart skips finished buckets.
#
# Usage:
#   sbatch --export=VARIANT=full long_gutenberg_inference.slurm
#   sbatch --export=VARIANT=carry-r1 long_gutenberg_inference.slurm
#   REPETITIONS=0 MAX_SAMPLES=20 sbatch --export=ALL,VARIANT=gdn long_gutenberg_inference.slurm   # calibration
# (normally submitted via long_gutenberg_inference_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/elyulina/store/long-gutenberg-results
GUTENBERG_LONG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl_long
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
MAX_LENGTH=${MAX_LENGTH:-}     # empty = no cap (full suffix, up to 4L)
MAX_SAMPLES=${MAX_SAMPLES:-}   # empty = all samples per bucket
OVERWRITE=${OVERWRITE:-}       # set to anything to recompute buckets that already exist
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to anything to log GDN state norms (no-op on attention models)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128 in the script)

### VARIANT -> EXP_NAME + non-restored flags ###
VARIANT=${VARIANT:-full}
NEEDS_TRITON=0
GDN_DIMS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"
case "$VARIANT" in
    full)       EXP_NAME=llama3-1b-full-attn-fineweb40B-gutenberg3B;             MEGATRON_EXTRA="" ;;
    gated)      EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B;            MEGATRON_EXTRA="--attention-output-gate" ;;
    full-xdoc-leak)  EXP_NAME=llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B; MEGATRON_EXTRA="" ;;
    sink)       EXP_NAME=llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215;       MEGATRON_EXTRA="--softmax-type learnable --attention-backend unfused" ;;
    off-by-one) EXP_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215; MEGATRON_EXTRA="--softmax-type off-by-one --attention-backend unfused";
                # checkpoint lives at the non-te215 path; EXP_NAME (results dir) stays -te215 to match the mem run
                CKPT_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B ;;
    gdn)        EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B;                   MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r0)   EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B;          MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r0.5) EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B;        MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r1)   EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B;          MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    *) echo "Unknown VARIANT=$VARIANT (expected full|gated|full-xdoc-leak|sink|off-by-one|gdn|carry-r0|carry-r0.5|carry-r1)"; exit 1 ;;
esac

# CKPT_NAME defaults to EXP_NAME; only off-by-one overrides it (checkpoint path != results name).
MODEL_DIR=/users/elyulina/store/pretrain-results/${CKPT_NAME:-$EXP_NAME}
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_gutenberg_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Optional caps, and the megatron passthrough only when non-empty.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $RESULTS_DIR \
    --data-folder $GUTENBERG_LONG_JSONL_DIR \
    --repetitions $REPETITIONS"
[ -n "$MAX_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS="$COMMON_ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && COMMON_ARGS="$COMMON_ARGS --overwrite"
[ -n "$LOG_STATE_NORM" ] && COMMON_ARGS="$COMMON_ARGS --log-state-norm"
[ -n "$STATE_CHUNK" ] && COMMON_ARGS="$COMMON_ARGS --state-chunk $STATE_CHUNK"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: variant=$VARIANT  exp=$EXP_NAME  reps=$REPETITIONS  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-all}  triton=$NEEDS_TRITON  state_norm=${LOG_STATE_NORM:+on}"

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

echo "END TIME: $(date)"
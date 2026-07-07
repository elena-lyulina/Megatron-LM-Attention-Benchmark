#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=2:00:00
#SBATCH --job-name=long-gutenberg-inf-tp
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Tensor-parallel long-Gutenberg loss for the unfused-attention models (sink, off-by-one).
# TP shards the attention heads across the 4 GPUs, so the O(S^2) score matrix fits at full
# length -- the reason these OOM under the default data-parallel (TP=1) run. All 4 GPUs form
# one TP group (DP=1): one sequence at a time, no data striding. No state norm.
#
# Equivalence is validated directly on sink: compare this TP=4 full run against the TP=1
# capped run on their overlapping positions. MAX_LENGTH / MAX_SAMPLES are passthrough for smoke tests.
#
# Usage:
#   sbatch --export=ALL,VARIANT=sink       long_gutenberg_inference_tp.slurm
#   sbatch --export=ALL,VARIANT=off-by-one long_gutenberg_inference_tp.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/elyulina/store/long-gutenberg-results
GUTENBERG_LONG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl_long
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
TP=${TP:-4}
MAX_LENGTH=${MAX_LENGTH:-}     # empty = no cap (full length; TP makes it fit)
MAX_SAMPLES=${MAX_SAMPLES:-}
OVERWRITE=${OVERWRITE:-}

### VARIANT -> EXP_NAME + checkpoint + flags ###
# EXP_NAME names the results dir; CKPT_NAME is the checkpoint (off-by-one differs, see main slurm).
VARIANT=${VARIANT:-sink}
case "$VARIANT" in
     sink)       EXP_NAME=llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215;       CKPT_NAME=$EXP_NAME; MEGATRON_EXTRA="--softmax-type learnable --attention-backend unfused" ;;
    off-by-one) EXP_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215; CKPT_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B; MEGATRON_EXTRA="--softmax-type off-by-one --attention-backend unfused" ;;
    *) echo "Unknown VARIANT=$VARIANT (expected sink|off-by-one)"; exit 1 ;;
esac

MODEL_DIR=/users/elyulina/store/pretrain-results/$CKPT_NAME
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_gutenberg_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1   # required for TP>1 on Hopper

ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $RESULTS_DIR \
    --data-folder $GUTENBERG_LONG_JSONL_DIR \
    --repetitions $REPETITIONS \
    --tensor-parallel $TP"
[ -n "$MAX_LENGTH" ] && ARGS="$ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && ARGS="$ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && ARGS="$ARGS --overwrite"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: variant=$VARIANT  exp=$EXP_NAME  tp=$TP  reps=$REPETITIONS  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-all}"

# All 4 GPUs form one TP group (DP=1); no per-rank Triton cache needed (attention, not GDN).
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun --nproc_per_node=4 \
        $PY $ARGS $EXTRA_MEGATRON"

echo "END TIME: $(date)"
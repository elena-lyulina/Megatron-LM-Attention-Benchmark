#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:05:00
#SBATCH --job-name=test-attn-full
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Standard full attention (no sink, no gate) with --use-packed-seq-params.
# Runs all three test suites: xdoc should PASS, sink and gated should FAIL.
#
# Usage:
#   sbatch attn_bench/submissions/test_attn_full.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

### PATHS ###
PROJECT_NAME="test"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/$PROJECT_NAME
EXP_NAME=attn_full
EXP_DIR=$PROJECT_DIR/$EXP_NAME
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $PROJECT_DIR $EXP_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR

### ENV ###
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=localhost
export MASTER_PORT=8888
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

### MEGATRON ARGS ###

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

# Tiny model — seq_len=64 makes the printed attention mask readable
SEQ_LEN=64

NETWORK_SIZE_ARGS=(
    --num-layers 2
    --hidden-size 128
    --ffn-hidden-size 512
    --num-attention-heads 4
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 10000
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-interval 1
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.01
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
)

TRAINING_ARGS=(
    --micro-batch-size 2
    --global-batch-size 2
    --train-iters 2          # just enough for tests to run on the first forward step
    --no-check-for-nan-in-loss-and-grad
    --eval-interval 100000000000
    --eval-iters 0
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

LEARNING_RATE_ARGS=(
    --lr 0.001
    --min-lr 0.0001
    --lr-decay-style cosine
    --lr-warmup-iters 1
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
)

DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /users/elyulina/scratch/tokenizers/llama-3.2-1b
    --make-vocab-size-divisible-by 128
    --reset-position-ids
    --eod-mask-loss
    --no-create-attention-mask-in-dataloader
    --use-packed-seq-params
    --mock-data
)

# No sink or gate flags - standard full attention.
# Expected: xdoc PASS, sink FAIL, gated FAIL.
BENCHMARK_ARGS=(
    --tests xdoc_mask xdoc_loss sink gated
)

TRAINING_CMD="python3 $MEGATRON_LM_DIR/attn_bench/tests/pretrain_gpt_native.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${BENCHMARK_ARGS[@]}"

### W&B ###
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
    WANDB_ENTITY=$(python3 -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null || echo "unknown")
    echo "[$(date)] WandB entity: $WANDB_ENTITY  project: $PROJECT_NAME  exp: $EXP_NAME-$SLURM_JOB_ID"
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] No WANDB API key. WANDB disabled."
fi

ulimit -c 0

### LOG COMPUTE ENV ###
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $DEBUG_DIR/memory_logging.txt
cp $0 $DEBUG_DIR

### LAUNCH ###
# grep -E "PASS|FAIL|### Test|### Summary|IMPLEMENTATION|EXPECTED" attn_bench/logs/<jobid>.out
echo "=== IMPLEMENTATION: full attention (no sink, no gate) ==="
echo "=== EXPECTED: xdoc PASS  |  sink FAIL  |  gated FAIL ==="
echo "[$(date)] Launching..."

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "export PYTHONPATH='${PYTHONPATH}'; ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"

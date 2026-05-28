#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:5:00
#SBATCH --job-name=test-attn-sink-flag-tp2-2nodes
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Sink attention (--softmax-type learnable) on the actual Llama 3.2 1B architecture, TP=2 across 2 nodes.
# Matches the pretraining setup (nemo_26, bf16, TE, same arch/reg/lr args) but with mock data + 2 iters.
# Runs all 5 sink tests: param existence, output sensitivity, init values (explains the 4k params norm),
# norm decomposition, and gradient_flows (key: does TE's flash-attn backprop through softmax_offset?).
#
# Usage:
#   sbatch attn_bench/submissions/test_attn_sink_flag_tp2_2nodes.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

### PATHS ###
PROJECT_NAME="test"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/$PROJECT_NAME
EXP_NAME=attn_sink_flag_tp2_2nodes
EXP_DIR=$PROJECT_DIR/$EXP_NAME
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $PROJECT_DIR $EXP_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR

dump_env() { printenv | grep -v '^WANDB_API_KEY='; }
dump_env > $DEBUG_DIR/env_before.txt

### ENV ###
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

### MEGATRON ARGS ###

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

# Llama 3.2 1B — matches pretrain_llama3_1b_sink_attn_fineweb40B_gutenberg3B.slurm exactly
SEQ_LEN=8192

NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings 131072
    --position-embedding-type rope
    --use-rope-scaling
    --rotary-base 500000
    --rope-scaling-factor 32
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-params-norm
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
    --micro-batch-size 4
    --global-batch-size 4
    --train-iters 2
    --no-check-for-nan-in-loss-and-grad
    --eval-interval 100000000000
    --eval-iters 0
)

INITIALIZATION_ARGS=(
    --seed 28
    --init-method-std 0.02
)

LEARNING_RATE_ARGS=(
    --lr 0.0004
    --min-lr 0.00004
    --lr-decay-style cosine
    --lr-warmup-iters 1
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --sequence-parallel
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

BENCHMARK_ARGS=(
    --softmax-type learnable
    --tests xdoc sink
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

SEP=$(printf '=%.0s' {1..100})
{
    echo "$(date)"
    echo "${SEP}"
    echo "CMD: ${TRAINING_CMD}"
    echo "${SEP}"
    echo "Slurm file: $0"
    cat "$0"
    echo "${SEP}"
    echo "Repo: ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD))"
    echo "${SEP}"
    echo "$(nvidia-smi)"
    echo "${SEP}"
    echo "Environment Variables:"
    dump_env
} > "$DEBUG_DIR/compute_environment.txt"

### LAUNCH ###
# grep -E "PASS|FAIL|### Test|### Summary|IMPLEMENTATION|EXPECTED|norm|grad" attn_bench/logs/<jobid>.out
echo "=== IMPLEMENTATION: sink attention (--softmax-type learnable), Llama 3.2 1B, TP=2, 2 nodes ==="
echo "=== EXPECTED: xdoc PASS  |  sink PASS (gradient FAIL would confirm TE kernel ignores softmax_offset in backward) ==="
echo "[$(date)] Launching..."

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"
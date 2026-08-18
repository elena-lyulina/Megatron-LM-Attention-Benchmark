#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:20:00
#SBATCH --job-name=test-pretrain-llama1B-swa-fineweb40B-gutenberg3B
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Runs the xdoc_mask/xdoc_loss/xdoc_position_ids suites (test_xdoc_attention.py) and the swa suite
# (test_sliding_window_attention.py) against the REAL pretrain_llama3_1b_sliding_window_attn_fineweb40B_gutenberg3B.slurm
# config -- same 8 nodes, same real FineWeb+Gutenberg data, same MBS=3/GBS=288/TP=1 -- instead of the
# tiny synthetic model used by test_attn_swa_flag.slurm. Entry point is pretrain_gpt_native.py, a
# drop-in superset of pretrain_gpt.py that injects --tests on the first forward step, then falls
# through to real training for every step after. train-iters is cut to a handful; no checkpointing
# (no --load/--save) since this run's model state is never meant to be reused.
# --time=00:20:00 (not 00:10:00 like the full-attn version): the swa suite flips one token at a time
# and checks the loss at the query position; with window=4096/seq_len=8192, both the outside- and
# inside-window pools exceed the 200-position exhaustive threshold in test_sliding_window_attention.py,
# so each is randomly sampled down to 100 positions (~202 extra forward passes on the real model).
#
# Usage:
#   sbatch attn_bench/submissions/pretrain_llama3_1b_sliding_window_attn_fineweb40B_gutenberg3B_test.slurm
#   sbatch --export=ALL,WINDOW_SIZE=256 attn_bench/submissions/pretrain_llama3_1b_sliding_window_attn_fineweb40B_gutenberg3B_test.slurm

echo "START TIME: $(date)"

# Container environment (used by srun --environment)
CONTAINER_ENV=nemo_26.04_te2.15

# Megatron source (WARNING: must be on iopsstor)
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

### DATA ###
# no weights — Megatron blends proportionally by sequence count = token count (all seqs are 8192 tokens)
FINEWEB_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25"
GUTENBERG_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/gutenberg_rep_1_256"

DATA_PATH_LIST=($(find -L $FINEWEB_DIR $GUTENBERG_DIR -name '*.bin' | sed 's/\.bin$//' | sort))
DATA_PATH_STR=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
echo "Total dataset prefixes: ${#DATA_PATH_LIST[@]}"

DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR

TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

### TRAINING ###

SEQ_LEN=8192
MBS=3
GBS=288
TRAINING_STEPS=5  # just enough for the injected --tests to run on the first forward step + a couple real steps

# Sliding window size for attention (causal, left-only). See ATTENTION_ARGS below.
WINDOW_SIZE=${WINDOW_SIZE:-4096}

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

# Logging directories & artifacts
PROJECT_NAME="test"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/$PROJECT_NAME
EXP_NAME="swa-w${WINDOW_SIZE}-prod-config"
EXP_DIR=$PROJECT_DIR/$EXP_NAME
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $PROJECT_DIR $EXP_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR

dump_env() { printenv | grep -v '^WANDB_API_KEY='; }
dump_env > $DEBUG_DIR/env_before.txt

# Set up ENV
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

#### Megatron Args ####

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --transformer-impl transformer_engine
)

NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 8192
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

ATTENTION_ARGS=(
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings 8192
    --position-embedding-type rope
    --use-rope-scaling
    --rotary-base 500000
    --rope-scaling-factor 1
    --window-size $WINDOW_SIZE,0  # causal sliding window; no --window-attn-skip-freq -- every layer is SWA
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-params-norm
    --log-interval 1
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --timing-log-level 2
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.1
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.95
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --no-check-for-nan-in-loss-and-grad
    --train-iters $TRAINING_STEPS
    --cross-entropy-loss-fusion
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
    --lr-warmup-iters 1  # must stay < TRAINING_STEPS; LR schedule is irrelevant for this diagnostic run
)

# No checkpointing (no --load/--save) -- this run's model state is never meant to be reused.
MIXED_PRECISION_ARGS=(
    --bf16
)

VOCAB_ARGS=(
    --make-vocab-size-divisible-by 128
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
)

DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --num-workers 2
    --num-dataset-builder-threads 4
    --reset-position-ids
    --eod-mask-loss
    --use-packed-seq-params
    --no-create-attention-mask-in-dataloader
    --data-path $DATA_PATH_STR
    --data-cache-path $DATASET_CACHE_DIR
)

### TRAINING CMD ###

TRAINING_CMD="python3 $MEGATRON_LM_DIR/attn_bench/tests/pretrain_gpt_native.py \
    --tests xdoc_mask=pass xdoc_loss=pass xdoc_position_ids=pass swa=pass \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${ATTENTION_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${VOCAB_ARGS[@]} \
    ${DATA_ARGS[@]}"

# W&B Logging
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
    pip install --quiet wandb --upgrade
    WANDB_ENTITY=$(python3 -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null || echo "unknown")
    echo "[$(date)] WandB logs will be sent to:"
    echo "           Organization/Workspace: $WANDB_ENTITY"
    echo "           Project: $PROJECT_NAME"
    echo "           Experiment: $EXP_NAME-$SLURM_JOB_ID"
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

ulimit -c 0

### LOG COMPUTE ENV ###

srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $DEBUG_DIR/memory_logging.txt
cp $0 $DEBUG_DIR

SEP=$(printf '=%.0s' {1..100})
{
    echo -e "$(date)"
    echo -e "${SEP}"
    echo -e "\nCMD: ${CMD_PREFIX} ${TRAINING_CMD}"
    echo -e "${SEP}"
    echo -e "\nSlurm file: $0\n"
    cat "$0"
    echo -e "${SEP}"
    echo -e "\nTOML file: ${SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment}\n"
    cat "${SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment}"
    echo -e "${SEP}"
    echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
    echo -e "${SEP}"
    echo -e "\nRepo: ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\n$(nvidia-smi)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(dump_env)"
    echo -e "${SEP}"
} > "$DEBUG_DIR/compute_environment.txt"

### LAUNCH ###

echo "Test: swa (window=${WINDOW_SIZE}) on sliding-window-attn prod config  nodes=${SLURM_NNODES}  mbs=${MBS}  gbs=${GBS}  train_iters=${TRAINING_STEPS}  datasets=${#DATA_PATH_LIST[@]}"
echo "[$(date)] Launching..."

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    mkdir -p $TORCH_INDUCTOR_CACHE_DIR
    mkdir -p $TRITON_HOME_DIR
    mkdir -p $TRITON_CACHE_DIR
    mkdir -p $PYTHON_CACHE_DIR
    export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"

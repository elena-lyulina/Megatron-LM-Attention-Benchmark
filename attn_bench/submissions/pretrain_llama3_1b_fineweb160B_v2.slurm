#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=train-llama1B-fineweb160B
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --no-requeue

# Train a standard LLaMA 3.2 1B model on 160B FineWeb Edu Dedup
# Used to obtain a model for perplexity-based dataset curation
#
# Usage:
#   sbatch attn_bench/submissions/pretrain_llama3_1b_fineweb160B_v2.slurm

echo "START TIME: $(date)"


### PATHS & EXPERIMENT ###

CONTAINER_ENV=nemo

# Megatron source (WARNING: must be on iopsstor)
REPO_ROOT=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

PROJECT_NAME="fineweb-160B"
PROJECT_DIR=$REPO_ROOT/attn_bench/results/pretrain/$PROJECT_NAME
EXP_NAME=llama3-1b-fineweb160B
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $PROJECT_DIR $EXP_DIR $CKPT_DIR $TRIGGER_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR

# snapshot env before any sourcing, to compare against compute_environment.txt after the run
printenv > $DEBUG_DIR/env_before.txt

# Clean triggers
rm -f $TRIGGER_DIR/save $TRIGGER_DIR/exit


### DATA ###

SEQ_LEN=8192

BINARY_DATASET_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B"
DATA_PATH=$(python3 $REPO_ROOT/attn_bench/utils/tools/create_data_config.py -p $BINARY_DATASET_DIR)

DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR


### TRAINING ###

MBS=2
GBS=80  # must be divisible by MBS * num_gpus (2 * 40 = 80); Gradient Accumulation Steps (GAS) = 1

# Total tokens: 160,108,458,502 (160.11B)
# 160,108,458,502 / (80 * 8192) = 244306.12 -> 244306
TRAINING_STEPS=244306

CHECKPOINT_STEPS=1000


#### Debugging ####
LOG_NCCL=false  # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false  # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=false  # Set to `true` to use mock data
###################


#### Megatron Args #### Check megatron/training/arguments.py

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

# 1.23B
# taken from https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json, the original values are provided in the comments
NETWORK_SIZE_ARGS=(
    --num-layers 16 # "num_hidden_layers": 16 -- following Llama-3.2-1B config
    --hidden-size 2048 # "hidden_size": 2048  -- following Llama-3.2-1B config
    --ffn-hidden-size 8192  # "intermediate_size": 8192  -- following Llama-3.2-1B config
    --num-attention-heads 32 # "num_attention_heads": 32  -- following Llama-3.2-1B config
    --group-query-attention
    --num-query-groups 8 # "num_key_value_heads": 8  -- following Llama-3.2-1B config
    --max-position-embeddings 131072  # "max_position_embeddings": 131072  -- following Llama-3.2-1B config
    --position-embedding-type rope
    --use-rope-scaling
    --rotary-base 500000 # "rope_theta": 500000.0  -- following Llama-3.2-1B config
    --rope-scaling-factor 32 # "rope_scaling": "factor": 32.0  -- following Llama-3.2-1B config
    # hardcoded in Megatron (megatron/core/models/common/embeddings/rotary_pos_embedding.py:92-98), correspond to Llama-3.2-1B config :
    # high_freq_factor=4
    # low_freq_factor=1
    # original_max_position_embeddings=8192
    # rope_type: llama3
    --normalization RMSNorm
    --norm-epsilon 1e-5 # "rms_norm_eps": 1e-05  -- following Llama-3.2-1B config
    --swiglu # "hidden_act": "silu"  -- following Llama-3.2-1B config
    --disable-bias-linear # "mlp_bias": false, "attention_bias": false  -- following Llama-3.2-1B config
    # Llama-3.2-1B config: "tie_word_embeddings": true -- embeddings tied by default; need to pass --untie-embeddings-and-output-weights to disable
    # Llama-3.2-1B config: "vocab_size": 128256 -- derived from the tokenizer
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
    --attention-dropout 0.0 # "attention_dropout": 0.0  -- following Llama-3.2-1B config
    --hidden-dropout 0.0 # following standard practice / LLama doesn't use it
    --clip-grad 1.0 # 1.0 — from the Llama 3 paper (gradient clipping = 1.0)
    --weight-decay 0.01 # 0.01 -- following Xu et al
    --optimizer adam # AdamW (decoupled weight decay) -- following Llama 3 paper
    --adam-beta1 0.9 # 0.9 — from the Llama 1/2 paper
    --adam-beta2 0.95 # 0.95 — from the Llama 1/2 paper
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --no-check-for-nan-in-loss-and-grad
    --train-iters $TRAINING_STEPS
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 5000
    --eval-interval $CHECKPOINT_STEPS  # must match or be a multiple of checkpoint interval, must be set to avoid TypeError even when EVAL_ITERS=0
    --eval-iters 0                     # no validation split for pretraining (--split 100,0,0)
)

INITIALIZATION_ARGS=(
    --seed 28
    --init-method-std 0.02 # initializer_range: 0.02  -- following Llama-3.2-1B config + default Megatron-LM value (transformer_config.py:327)
)

# NOTE Check all the arguments in megatron/training/arguments.py#L1548
LEARNING_RATE_ARGS=(
    --lr 0.0003 # 3 × 10^−4 -- following Xu et al
    --min-lr 0.00003  # 3 × 10^−5 -- following Xu et al, LLama3 paper
    --lr-decay-style cosine # cosine -- following LLama3 paper
    --lr-warmup-iters 2000 # 2000 -- following LLama3 paper
)

CHECKPOINTING_ARGS=(
    --save $CKPT_DIR
    --load $CKPT_DIR  # delete this to NOT reload from the latest checkpoint
    --save-interval $CHECKPOINT_STEPS
    --save-retain-interval 10000  # keep permanent snapshots every N steps; must be multiple of CHECKPOINT_STEPS
    --ckpt-format torch_dist
    --async-save
)

MIXED_PRECISION_ARGS=(
    --bf16
)

# Pads vocab to a multiple of 128 for even TP splits
VOCAB_ARGS=(
    --make-vocab-size-divisible-by 128
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --wgrad-deferral-limit 50
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model /iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b  # Default path assumes the Llama 3.2 tokenizer is downloaded to the standard location.
)

DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --num-workers 2
    --num-dataset-builder-threads 1
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
)

if [ "$MOCK_DATA" = true ]; then
    DATA_ARGS+=(--mock-data)
else
    DATA_ARGS+=(--data-path $DATA_PATH --data-cache-path $DATASET_CACHE_DIR)
fi


### ENV ###

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will be set inside the srun command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

CMD_PREFIX="numactl --membind=0-3"

# disables core dumps to save disk space
ulimit -c 0

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
    CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi


### TRAINING CMD ###

TRAINING_CMD="python3 $REPO_ROOT/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[*]} \
    ${NETWORK_SIZE_ARGS[*]} \
    ${VOCAB_ARGS[*]} \
    ${REGULARIZATION_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${INITIALIZATION_ARGS[*]} \
    ${LEARNING_RATE_ARGS[*]} \
    ${CHECKPOINTING_ARGS[*]} \
    ${MIXED_PRECISION_ARGS[*]} \
    ${DISTRIBUTED_ARGS[*]} \
    ${TOKENIZER_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    ${DATA_ARGS[*]}"

# W&B Logging
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
    if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
        echo "[$(date)] Syncing WANDB from previous run"
        wandb sync "$LOGGING_DIR/wandb/latest-run"
    fi
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi


### LOG COMPUTE ENV ###

# GPU memory snapshot (before training)
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $DEBUG_DIR/memory_logging.txt

# copy the current slurm script into $DEBUG_DIR
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
    echo -e "\nRepo: ${REPO_ROOT} ($(git -C ${REPO_ROOT} rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\n$(pip list)"
    echo -e "${SEP}"
    echo -e "\nPython sys.path:\n$(python -c 'import sys; print(chr(10).join(sys.path))')"
    echo -e "${SEP}"
    echo -e "\nmegatron location: $(python -c 'import megatron; print(megatron.__file__)' 2>&1)"
    echo -e "${SEP}"
    echo -e "\nmegatron.__path__: $(python -c 'import megatron; print(list(megatron.__path__))' 2>&1)"
    echo -e "${SEP}"
    echo -e "\nmegatron/training exists: $(ls ${REPO_ROOT}/megatron/training 2>&1 | head -5)"
    echo -e "${SEP}"
    echo -e "\n$(nvidia-smi)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(printenv)"
    echo -e "${SEP}"
} > "$DEBUG_DIR/compute_environment.txt"


### LAUNCH ###

echo "Training: fineweb-llama nodes=${SLURM_NNODES} gbs=${GBS} steps=${TRAINING_STEPS}"
echo "[$(date)] Launching training..."

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    --wait 60 \
    -lu \
    bash -c "export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"
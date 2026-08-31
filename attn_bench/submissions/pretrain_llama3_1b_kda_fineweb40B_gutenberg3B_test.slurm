#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:15:00
#SBATCH --job-name=test-pretrain-llama1B-kda-fineweb40B-gutenberg3B
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# SMOKE TEST for pretrain_llama3_1b_kda_fineweb40B_gutenberg3B.slurm -- matches the main config exactly.
# History: 192/384 head dims -> chunk_kda NaN (job 3244703); 8h/128/128 -> NaN cleared, xdoc PASS,
# no OOM at MBS=2/nodes=12 (job 3245161). This run bumps to the MAIN geometry -- 16 heads, ffn 6976
# (param-matched, kda_1B_args_16heads_128_128_ffn6976.txt) -- to confirm 16 heads (2x the 8h mixer)
# still fits at MBS=2 and xdoc still passes. If it OOMs, add --recompute-granularity selective.
# Read the "Summary" block (xdoc) AND whether it OOMs.
# Runs xdoc_mask/xdoc_loss/xdoc_position_ids -- read the "Summary" block in the .out.
# Entry point is pretrain_gpt_native.py, a drop-in superset of pretrain_gpt.py that injects --tests on
# the first forward step (Summary block prints immediately), then falls through to real training.
# TRAINING_STEPS is left at the usual value; the 00:15:00 limit simply times the run out -- that's fine,
# nothing here is meant to be reused. No checkpointing (no --load / --save).
# The --tests harness does NOT fail the job -- read the "Summary" block in the .out.
# NOTE: KDA requires fla.ops.kda.chunk_kda + fla.modules.convolution.causal_conv1d (te2.15 container).
#
# Usage:
#   sbatch attn_bench/submissions/pretrain_llama3_1b_kda_fineweb40B_gutenberg3B_test.slurm

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
MBS=2
GBS=288
# 40,038,865,413 (fineweb 40B) + 2,762,833,920 (gutenberg 3B) = 42,801,699,333 tokens
TARGET_TOKENS=42801699333
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))  # = 18141 (usual value; the 15-min limit cuts it short)

#### Debugging ####
LOG_NCCL=false
NSYS_PROFILER=false
###################

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

# Logging directories & artifacts
PROJECT_NAME="test"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/$PROJECT_NAME
EXP_NAME="xdoc-attn-kda-prod-config"
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
# chunk_kda intermediates fragment the allocator -> OOM at MBS=3 (jobs 3244703/3244890/3244987).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

#### Megatron Args #### (identical to the main script except: no CHECKPOINTING_ARGS, no --exit-signal-handler)

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --transformer-impl transformer_engine
)

NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 6976  # param-matched (kda_1B_args_16heads_128_128_ffn6976.txt) -- matches the main script
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

# 16 heads x 128/128 (symmetric) -- matches the main script. NOT the GDN-mirrored 192/384 (chunk_kda NaN).
ATTENTION_ARGS=(
    --experimental-attention-variant kimi_delta_attention
    --linear-attention-freq [1]*16
    --linear-num-key-heads 16
    --linear-num-value-heads 16
    --linear-key-head-dim 128      # qk_dim = 16*128 = 2048
    --linear-value-head-dim 128    # v_dim = 16*128 = 2048
    --linear-conv-kernel-dim 4
    --num-attention-heads 32       # required to build the model but is not used for KDA
    --position-embedding-type none
    --max-position-embeddings 8192
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
    --manual-gc
    --manual-gc-interval 500
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
    --lr-warmup-iters 500  # < TRAINING_STEPS; LR schedule is irrelevant for this diagnostic run
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
    --tests xdoc_mask=pass xdoc_loss=pass xdoc_position_ids=pass \
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

# disables core dumps to save disk space
ulimit -c 0

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
    CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi


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
    echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
    echo -e "${SEP}"
    echo -e "\nRepo: ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\nmegatron location: $(python -c 'import megatron; print(megatron.__file__)' 2>&1)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(dump_env)"
    echo -e "${SEP}"
} > "$DEBUG_DIR/compute_environment.txt"


### LAUNCH ###

echo "Test: xdoc suites on KDA prod config  nodes=${SLURM_NNODES}  mbs=${MBS}  gbs=${GBS}  train_iters=${TRAINING_STEPS}  datasets=${#DATA_PATH_LIST[@]}"
echo "[$(date)] Launching test..."

# TRITON_CACHE_DIR: per-rank node-local cache. A shared-FS Triton cache crashes fla-kernel runs with
# "OSError: [Errno 116] Stale file handle" when kernels autotune on first backward and ranks collide
# on the same cache files (observed for GDN in job 2570257). Key on SLURM_JOB_ID + SLURM_PROCID.
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    mkdir -p $TORCH_INDUCTOR_CACHE_DIR
    mkdir -p $TRITON_HOME_DIR
    mkdir -p $PYTHON_CACHE_DIR
    export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_\$SLURM_PROCID
    mkdir -p \$TRITON_CACHE_DIR
    export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"

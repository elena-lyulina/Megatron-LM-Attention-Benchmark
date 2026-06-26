#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=01:30:00
#SBATCH --job-name=bench-gdn-carry-throughput
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Throughput A/B for the GDN state-carry / no-pack jitter (r=0 slower + jitterier than r=1).
# Base config is copied VERBATIM from pretrain_llama3_1b_gdn_carry_r0_fineweb40B_gutenberg3B.slurm
# (data, model, MBS/GBS, ENV exports, cache dirs, every constant arg array) so nothing that affects
# throughput differs from the real runs. The ONLY bench deltas, each marked "### BENCH:", are:
#   - TRAIN_ITERS short (throughput only); per-pass exp/ckpt/logging dirs (fresh init, no real load)
#   - four passes in ONE allocation -> same nodes + same container, so node placement is not a confound
# Passes (vary only carry ratio / source tree / allocator):
#   1) r0          current repo, --gdn-state-carry-ratio 0.0      baseline jitter
#   2) r1          current repo, --gdn-state-carry-ratio 1.0      does carry-on smooth it?
#   3) r0_expseg   current repo, r0 + expandable_segments alloc   is it the allocator?
#   4) upstream    upstream clone, GDN no-carry, no pack          our code or inherent?  (LAST: different megatron dir)
# Expected if it is NOT our code: upstream ~ r0 (jittery), r1 smooth. If r0_expseg ~ r1: allocator.
#
# Usage: sbatch attn_bench/submissions/bench_gdn_carry_throughput.slurm

echo "START TIME: $(date)"

# Container environment (used by srun --environment) -- te2.15 ships fla + causal_conv1d for GDN
CONTAINER_ENV=nemo_26.04_te2.15

# Megatron source (WARNING: must be on iopsstor)
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

### BENCH: second source tree for pass 4 (upstream clone, no --gdn-state-carry-ratio) ###
UPSTREAM_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-upstream

### DATA ###
# no weights — Megatron blends proportionally by sequence count = token count (all seqs are 8192 tokens)
# 40,038,865,413 tokens
FINEWEB_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25"
# 2,762,833,920 tokens
GUTENBERG_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/gutenberg_rep_1_256"

DATA_PATH_LIST=($(find -L $FINEWEB_DIR $GUTENBERG_DIR -name '*.bin' | sed 's/\.bin$//' | sort))
DATA_PATH_STR=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
echo "Total dataset prefixes: ${#DATA_PATH_LIST[@]}"

DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR

TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b  # Assumes the Llama 3.2 tokenizer is downloaded to this location

### TRAINING ###

SEQ_LEN=8192

MBS=2
# seq_length * MBS * DP * GAS = 8192 * 2 * 56 * 3 = 2.75 M tokens/step  (TP=1 → DP=56)
# GAS = GBS / (MBS * DP)
GBS=336
# 40,038,865,413 (fineweb 40B) + 2,762,833,920 (gutenberg 3B) = 42,801,699,333 tokens
TARGET_TOKENS=42801699333
#  tokens/step = GBS × SEQ_LEN = 336 × 8192 = 2,752,512
#  raw steps = 42,801,699,333 / 2,752,512 = 15550
CHECKPOINT_STEPS=2000
CHECKPOINT_RETAIN_STEPS=50000
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN))) # = 15550, data exhaustion at 15549 (prod length, shown for reference)

### BENCH: short run, throughput only (prod trains $TRAINING_STEPS). save-interval > TRAIN_ITERS => never saves ###
TRAIN_ITERS=300

#### Debugging ####
LOG_NCCL=false  # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false  # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
###################

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

### BENCH: results root; per-pass exp/ckpt/logging dirs are derived inside run_pass ###
PROJECT_NAME="gdn-carry-throughput-bench"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/pretrain-bench/$PROJECT_NAME
mkdir -p $PROJECT_DIR
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
    echo "[$(date)] No WANDB_API_KEY -> wandb disabled (throughput still in per-pass .out)."
else
    echo "[$(date)] WANDB enabled, project=$PROJECT_NAME (one run per pass: bench-<pass>-$SLURM_JOB_ID)."
fi

# Set up ENV
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

echo "NODES: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

#### Megatron Args (CONSTANT arrays — verbatim from pretrain_..._gdn_carry_r0; per-pass arrays built in run_pass) ####

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --transformer-impl transformer_engine
)

# ~1.239B (GDN mixer + FFN shrunk to match Llama-3.2-1B = 1.236B)
NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 5824
    --num-attention-heads 32
    --max-position-embeddings 8192
    --position-embedding-type none
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

# GDN mixer; carry flag is appended per pass (passes 1-3) / omitted for upstream (pass 4).
ATTENTION_ARGS_BASE=(
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq [1]*16
    --linear-num-key-heads 8
    --linear-num-value-heads 8
    --linear-key-head-dim 192
    --linear-value-head-dim 384
    --linear-conv-kernel-dim 4
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
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --no-check-for-nan-in-loss-and-grad
    --train-iters $TRAIN_ITERS   # BENCH: short (prod uses $TRAINING_STEPS)
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 500
    --exit-signal-handler
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
    --lr-warmup-iters 2000
)

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

# No --use-packed-seq-params (no cu_seqlens -> GDN state leaks across docs in a sequence) and no
# --reset-position-ids (positions unused by GDN). Identical to the carry runs.
DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --num-workers 2
    --num-dataset-builder-threads 4
    --eod-mask-loss
    --no-create-attention-mask-in-dataloader
    --data-path $DATA_PATH_STR
    --data-cache-path $DATASET_CACHE_DIR
)

ulimit -c 0

# run_pass TAG  MEG_DIR  CARRY(0.0/1.0/none)  ALLOC_CONF(or empty)  MASTER_PORT
run_pass() {
    local TAG=$1 MEG_DIR=$2 CARRY=$3 ALLOC=$4 PORT=$5
    echo ""
    echo "######## PASS ${TAG}  dir=${MEG_DIR}  carry=${CARRY}  alloc=${ALLOC:-default}  $(date) ########"

    ### BENCH: per-pass dirs (fresh, empty CKPT_DIR => --load is a no-op => fresh init; never saves at 300 iters) ###
    local EXP_NAME="${TAG}-${SLURM_JOB_ID}"
    local EXP_DIR=$PROJECT_DIR/$EXP_NAME
    local CKPT_DIR=$EXP_DIR/checkpoints
    local LOGGING_DIR=$EXP_DIR/logging
    local TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
    mkdir -p $CKPT_DIR $LOGGING_DIR $TENSORBOARD_DIR

    # ATTENTION: carry flag per pass (matches prod, which hardcodes --gdn-state-carry-ratio here)
    local ATTN=("${ATTENTION_ARGS_BASE[@]}")
    [ "$CARRY" != "none" ] && ATTN+=(--gdn-state-carry-ratio $CARRY)

    # LOGGING: verbatim from prod, tensorboard dir is per pass
    local LOGGING_ARGS=(
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

    # CHECKPOINTING: verbatim from prod, ckpt dir is per pass (empty -> fresh)
    local CHECKPOINTING_ARGS=(
        --load $CKPT_DIR
        --save $CKPT_DIR
        --save-interval $CHECKPOINT_STEPS
        --save-retain-interval $CHECKPOINT_RETAIN_STEPS
        --ckpt-format torch_dist
        --ckpt-fully-parallel-load
        --dist-ckpt-strictness assume_ok_unexpected
    )

    # W&B: one run per pass; empty when no API key
    local WANDB_ARGS=()
    if [ -n "${WANDB_API_KEY:-}" ]; then
        WANDB_ARGS=(--wandb-save-dir $LOGGING_DIR --wandb-project $PROJECT_NAME --wandb-exp-name bench-$EXP_NAME)
    fi

    local CMD="python3 ${MEG_DIR}/pretrain_gpt.py \
        ${TRANSFORMER_ENGINE_ARGS[@]} \
        ${NETWORK_SIZE_ARGS[@]} \
        ${ATTN[@]} \
        ${LOGGING_ARGS[@]} \
        ${REGULARIZATION_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${INITIALIZATION_ARGS[@]} \
        ${LEARNING_RATE_ARGS[@]} \
        ${CHECKPOINTING_ARGS[@]} \
        ${MIXED_PRECISION_ARGS[@]} \
        ${DISTRIBUTED_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${VOCAB_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${WANDB_ARGS[@]}"

    local ALLOC_EXPORT=""
    [ -n "$ALLOC" ] && ALLOC_EXPORT="export PYTORCH_CUDA_ALLOC_CONF=${ALLOC};"   # BENCH: allocator probe (pass 3 only)

    local DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
    mkdir -p $DEBUG_DIR
    cp $0 $DEBUG_DIR   # snapshot this script alongside each pass

    # NCCL Debug (verbatim from prod)
    local CMD_PREFIX=""
    if [ "$LOG_NCCL" = true ]; then
        CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
    fi

    # NSYS profiler (verbatim from prod) -- flip NSYS_PROFILER=true to profile a pass
    if [ "$NSYS_PROFILER" = true ]; then
        local NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
        CMD="$NSYS_LAUNCHER $CMD --profile"
    fi

    # srun verbatim from prod; per-pass node-local Triton cache (avoids Errno 116 stale-handle, job 2570257),
    # per-pass MASTER_PORT, and PYTHONPATH pinned to this pass's tree so upstream imports upstream megatron.
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
        export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_${TAG}_\$SLURM_PROCID
        mkdir -p \$TRITON_CACHE_DIR
        export MASTER_PORT=${PORT}
        export PYTHONPATH=${MEG_DIR}
        ${ALLOC_EXPORT}
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD_PREFIX} ${CMD}" \
        2>&1 | tee $PROJECT_DIR/$EXP_NAME/pass.out

    echo "######## END PASS ${TAG}  $(date) ########"
}

### LOG COMPUTE ENV (once per job; same nodes/container for all passes) ###
dump_env() { printenv | grep -v '^WANDB_API_KEY='; }
JOB_DEBUG_DIR=$PROJECT_DIR/debug/$SLURM_JOB_ID
mkdir -p $JOB_DEBUG_DIR

# GPU memory snapshot (before training)
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $JOB_DEBUG_DIR/memory_logging.txt

SEP=$(printf '=%.0s' {1..100})
{
    echo -e "$(date)"
    echo -e "${SEP}"
    echo -e "\nSlurm file: $0\n"
    cat "$0"
    echo -e "${SEP}"
    echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
    echo -e "${SEP}"
    echo -e "\nRepo (current): ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD 2>&1))"
    echo -e "\nRepo (upstream): ${UPSTREAM_LM_DIR} ($(git -C ${UPSTREAM_LM_DIR} rev-parse --verify HEAD 2>&1))"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(dump_env)"
    echo -e "${SEP}"
} > "$JOB_DEBUG_DIR/compute_environment.txt"

### LAUNCH (sequential; same nodes for all four) ###
echo "[$(date)] Launching 4-pass throughput bench..."
run_pass r0         "$MEGATRON_LM_DIR" 0.0  ""                          8888
run_pass r1         "$MEGATRON_LM_DIR" 1.0  ""                          8889
run_pass r0_expseg  "$MEGATRON_LM_DIR" 0.0  "expandable_segments:True"  8890
run_pass upstream   "$UPSTREAM_LM_DIR" none ""                          8891

echo ""
echo "END TIME: $(date)"
echo "Per-pass logs: $PROJECT_DIR/{r0,r1,r0_expseg,upstream}-$SLURM_JOB_ID/pass.out"
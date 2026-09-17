#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:12:00
#SBATCH --job-name=diag-mla-attn-backend
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# MLA THROUGHPUT DIAGNOSTIC. The prod MLA run (3245261/3248575) did ~2.2 s/iter at 12 nodes,
# ~2x the softmax baselines.
# Round 1 (ATTN_BACKEND, jobs 3393779/3393811/3393816): auto already picks cuDNN FusedAttention;
# forcing fused or FlashAttention-2 gives the same ~2.2 s/iter, so the attention kernel is not the cost.
# The timers pointed elsewhere: forward-compute spread across ranks is ~400 ms (baseline: ~30 ms),
# i.e. something scales with the number of packed documents per micro-batch. That is the RoPE path:
# --no-rope-fusion + THD sends MLA through _apply_rotary_pos_emb_thd (rope_utils.py), a Python loop
# over every packed document with .tolist()/.item() syncs, on q and k in every layer, fwd and bwd.
# The softmax baselines never hit it (rope fusion on -> one TE kernel per tensor).
# Round 2 (ROPE_MODE): the fused MLA-RoPE kernels (fused_apply_mla_rope_for_q/kv, take cu_seqlens)
# only run under rope_type=yarn, but YaRN at --rotary-scaling-factor 1.0 IS plain RoPE:
# inv_freq_inter == inv_freq_extra (ramp mask cancels) and mscale = 1.0, softmax scale unchanged.
#   ROPE_MODE=unfused     : prod config, --rope-type rope --no-rope-fusion (the slow reference)
#   ROPE_MODE=fused_yarn1 : --rope-type yarn --rotary-scaling-factor 1.0, fusion on
# Read: the preflight block, the "[DotProductAttention]" backend lines, `elapsed time per iteration (ms)`
# and the (min, max) forward-compute / backward-compute timer spread.
# 60 iters, no --tests, no checkpointing.
#
# Usage (from attn_bench/):
#   ATTN_BACKEND=auto ROPE_MODE=unfused     sbatch submissions/diag_mla_attn_backend.slurm
#   ATTN_BACKEND=auto ROPE_MODE=fused_yarn1 sbatch submissions/diag_mla_attn_backend.slurm

ATTN_BACKEND=${ATTN_BACKEND:-auto}
ROPE_MODE=${ROPE_MODE:-unfused}

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
# 40,038,865,413 (fineweb 40B) + 2,762,833,920 (gutenberg 3B) = 42,801,699,333 tokens
TARGET_TOKENS=42801699333
TRAINING_STEPS=60  # enough for the per-iter time to settle after the first few warm-up iters

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
EXP_NAME="diag-mla-attn-backend-$ATTN_BACKEND-rope-$ROPE_MODE"
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

# TE prints the attention backend it selects (and why the others were rejected), once per unique config per rank
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2
# Triton prints autotune results whenever it (re)benchmarks a kernel -> shows if the fused RoPE kernels re-autotune
export TRITON_PRINT_AUTOTUNING=1
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
    --ffn-hidden-size 7680  # shrunk from 8192 to absorb the heavier MLA mixer and match Llama-1B param count
    --normalization RMSNorm
    --norm-epsilon 1e-5
    --swiglu
    --disable-bias-linear
)

# FAITHFUL V2-Lite MLA block, identical to the prod run; only --attention-backend and the RoPE wiring vary.
case "$ROPE_MODE" in
    unfused)
        ROPE_ARGS=(
            --rope-type rope            # prod: plain RotaryEmbedding, unfused (fused MLA-RoPE is yarn-only)
            --no-rope-fusion
        )
        ;;
    fused_yarn1)
        ROPE_ARGS=(
            --rope-type yarn            # YaRN at scaling factor 1.0 == plain RoPE, unlocks fused_apply_mla_rope_for_q/kv
        )
        ;;
    *)
        echo "unknown ROPE_MODE=$ROPE_MODE (unfused | fused_yarn1)"; exit 1
        ;;
esac

ATTENTION_ARGS=(
    --multi-latent-attention
    --num-attention-heads 16
    --kv-lora-rank 512
    --qk-head-dim 128
    --qk-pos-emb-head-dim 64
    --v-head-dim 128
    --qk-layernorm                      # DeepSeek kv_a_layernorm: RMSNorm(kv_lora_rank) between down/up proj
    --position-embedding-type rope
    "${ROPE_ARGS[@]}"
    --rotary-base 500000
    --rotary-scaling-factor 1.0         # no-op for rope; for yarn it is what makes it plain RoPE (dataclass default is 40!)
    --max-position-embeddings 8192
    --attention-backend $ATTN_BACKEND
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
    --lr-warmup-iters 10  # must stay < TRAINING_STEPS (init assert); LR schedule is irrelevant here
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

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
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

echo "Diag: MLA attention backend=${ATTN_BACKEND}  rope=${ROPE_MODE}  nodes=${SLURM_NNODES}  mbs=${MBS}  gbs=${GBS}  train_iters=${TRAINING_STEPS}"

### PREFLIGHT: kernel versions inside the container ###
cat > $DEBUG_DIR/preflight.py <<'PY'
import importlib, torch
print("### PREFLIGHT ###")
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("device", torch.cuda.get_device_name(0), "| sm", torch.cuda.get_device_capability(0))
for m in ("transformer_engine", "flash_attn", "flash_attn_3", "flash_attn_interface", "flashattn_hopper", "fla"):
    try:
        mod = importlib.import_module(m)
        print(f"{m:24s} OK  {getattr(mod, '__version__', '?')}  {getattr(mod, '__file__', '')}")
    except Exception as e:
        print(f"{m:24s} --  {type(e).__name__}: {e}")
try:
    from transformer_engine.pytorch.attention.dot_product_attention import utils as u
    for n in ("get_default_fa_version", "_flash_attn_version", "_flash_attn_3_version", "_flash_attn_3_is_installed", "_use_flash_attn_3"):
        if hasattr(u, n):
            v = getattr(u, n)
            print(f"TE utils.{n} = {v() if callable(v) else v}")
except Exception as e:
    print("TE attention utils probe failed:", type(e).__name__, e)
print("### END PREFLIGHT ###")
PY
srun -N1 -n1 --environment=$CONTAINER_ENV python3 $DEBUG_DIR/preflight.py

echo "[$(date)] Launching diagnostic..."

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

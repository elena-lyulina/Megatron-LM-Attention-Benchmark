#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:15:00
#SBATCH --job-name=test-pretrain-llama1B-hybrid-qwen-fineweb40B-gutenberg3B
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# SMOKE TEST for pretrain_llama3_1b_hybrid_qwen_fineweb40B_gutenberg3B.slurm, EXACT prod config
# (8 nodes / TP=1 / MBS=3 / GBS=288). Entry point is pretrain_gpt_native.py -- a drop-in superset of
# pretrain_gpt.py that runs the --tests suites on the first forward step (Summary block prints right
# away), then falls through to real training in the same process. The --tests harness does NOT fail
# the job -- read the "Summary" block in the .out.
#   - Suites: xdoc_mask/xdoc_loss/xdoc_position_ids=pass (doc isolation), gdn=pass (12 GDN layers
#     wired), gated=pass (4 gated-attn layers carry the output gate), sink=fail (no --softmax-type).
#   - Then it trains until the 00:15:00 walltime times it out. CHECKPOINT_STEPS=50 so a torch_dist
#     save of the mixed GDN+attn checkpoint actually fires; per-iter throughput is in the log.
#   - Crucially confirms MBS=3 fits with 12 GDN layers (untested; every prior GDN run used MBS=2).
#     If it OOMs, apply fallback A or B from the main script's header before the full launch.
# Differs from the main script only by: no --exit-signal-handler, CHECKPOINT_STEPS 2000 -> 50,
# test EXP_NAME + fresh CKPT_DIR, and the pretrain_gpt_native.py entry point with --tests.
#
# Train a LLaMA 3.2 1B backbone with a QWEN-STYLE HYBRID token mixer on the FineWeb-Edu 40B +
# Gutenberg 3B blend (~42.8B tokens), param-matched to the softmax-attention scf=1 baselines.
# Replicates the Qwen3.5 / 3.8 hybrid structure (model_type "qwen3_5", full_attention_interval 4):
#   - Pattern: --linear-attention-freq 4 -> [1,1,1,0]*4 = 12 GDN + 4 gated-attn layers,
#              gated-attn at layer indices 3/7/11/15 (int N: SDPA where (i+1)%N==0). Equivalent
#              explicit form "([1]*3+[0]*1)*4" -- but parens break in the bash launch string, so use 4.
#   - GDN layers: identical to the pure-GDN baseline -- 8 K/V heads, key_head_dim 192 /
#                 value_head_dim 384 (paper ratios 0.75/1.5), conv kernel 4.
#   - Gated-attn layers: identical to the gated-attn scf=1 baseline -- Llama GQA 32/8 +
#                        --attention-output-gate, full RoPE (rope-scaling-factor 1).
#   - FFN: --ffn-hidden-size 6208 (shrunk from 8192 to absorb the GDN-heavy mixer; ~1.234B params).
#   - Config: attn_bench/configs/param_count_configs/hybrid_qwen_1B_args_ffn6208.txt
#   - Data / LR / regularization: from the scf=1 baselines (GBS=288, TP=1, wd 0.1, warmup 500,
#     18141 steps). --use-packed-seq-params (DATA_ARGS) resets BOTH the GDN recurrent/conv state
#     and the softmax attention mask at every document boundary via cu_seqlens.
#
# DIST: 8 nodes / TP=1 / MBS=3 / GBS=288 (GAS=3), IDENTICAL to the prod script -- this test exists to
#   prove that config fits. If it OOMs in the first steps, apply fallback A (--recompute-granularity
#   selective, keeps MBS=3 / 8 nodes) or B (nodes=12 + MBS=2) from the prod script's header.
#
# NOTE: the GDN layers need flash-linear-attention (fla) + causal_conv1d -- the te2.15 container
# ships both (fla 0.4.2 is fine for GDN; only KDA needs the 0.5.2 side-install).
#
# Usage:
#   sbatch attn_bench/submissions/pretrain_llama3_1b_hybrid_qwen_fineweb40B_gutenberg3B_test.slurm

echo "START TIME: $(date)"

# Container environment (used by srun --environment) -- te2.15 ships fla + causal_conv1d for GDN
CONTAINER_ENV=nemo_26.04_te2.15

# Megatron source (WARNING: must be on iopsstor)
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}


# Set to `true` to continuously submit jobs to Slurm until training is complete.
# Enable it once you are sure of the cost involved in running this experiment.
AUTO_JOB_REQUEUE=false
if [ "$AUTO_JOB_REQUEUE" = true ]; then
    echo "[$(date)] Submitting follow-up job with singleton dependency and --resume"
    # Export the script path and submit with --resume
    sbatch --dependency=singleton --job-name=$SLURM_JOB_NAME $0
fi

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

MBS=3
# most compute-efficient config from attn_training_perf_bench sweep is tp=1/mbs=3;
# seq_length * MBS * DP * GAS = 8192 * 3 * 32 * 3 = 2.36 M tokens/step
# GBS = MBS * DP * GAS = 3 * 32 * 3 = 288
GBS=288
# 40,038,865,413 (fineweb 40B) + 2,762,833,920 (gutenberg 3B) = 42,801,699,333 tokens
TARGET_TOKENS=42801699333
#  tokens/step = GBS × SEQ_LEN = 288 × 8192 = 2,359,296
#  raw steps = 42,801,699,333 / 2,359,296 = 18,141 (18141.7250455)
CHECKPOINT_STEPS=50   # smoke test: ~50 steps for a stable throughput reading, and a periodic save fires within the 15-min limit; prod uses 2000
CHECKPOINT_RETAIN_STEPS=50000
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))  # = 18141; keep prod value so the LR schedule matches, walltime cuts it short
# round to nearest multiple of CHECKPOINT_STEPS -- drop for memorization studies to ensure no under / over seeing
#TRAINING_STEPS=$((((TRAINING_STEPS + CHECKPOINT_STEPS/2) / CHECKPOINT_STEPS) * CHECKPOINT_STEPS))
# data exhausts at step 18141; data exhaustion fix will checkpoint at the last valid step before crash

#### Debugging ####
LOG_NCCL=false  # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false  # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
###################

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

# Logging directories & artifacts
PROJECT_NAME="fineweb-40B_gutenberg-3B"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/pretrain/$PROJECT_NAME
EXP_NAME="test-llama3-1b-hybrid-qwen-scf1-fineweb40B-gutenberg3B"
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $PROJECT_DIR $EXP_DIR $CKPT_DIR $TRIGGER_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR
# Clean triggers
rm -f $TRIGGER_DIR/save $TRIGGER_DIR/exit
# Clean checkpoints (test script: always start fresh)
rm -rf $CKPT_DIR/*

dump_env() { printenv | grep -v '^WANDB_API_KEY='; }

# snapshot env before training
dump_env > $DEBUG_DIR/env_before.txt


# Set up ENV
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# precautionary: GDN/fla chunked kernels allocate many differently-sized intermediates that fragment
# the allocator; expandable_segments recovers the reserved-but-unallocated pool (needed for KDA at
# MBS=3, kept here as insurance for the GDN-heavy hybrid). Harmless if unused.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

#### Megatron Args ####
# Check megatron/training/arguments.py

TRANSFORMER_ENGINE_ARGS=(
    --main-grads-dtype fp32
    --transformer-impl transformer_engine
)

# ~1.234B, param-matched to the Llama-3.2-1B softmax baseline (1,235,814,400)
# backbone follows Llama-3.2-1B; hybrid mixer + shrunk FFN from hybrid_qwen_1B_args_ffn6208.txt
NETWORK_SIZE_ARGS=(
    --num-layers 16 # "num_hidden_layers": 16 -- following Llama-3.2-1B config
    --hidden-size 2048 # "hidden_size": 2048  -- following Llama-3.2-1B config
    --ffn-hidden-size 6208  # shrunk from 8192 (97x64) to absorb the GDN-heavy mixer and match Llama-1B param count
    --normalization RMSNorm
    --norm-epsilon 1e-5 # "rms_norm_eps": 1e-05  -- following Llama-3.2-1B config
    --swiglu # "hidden_act": "silu"  -- following Llama-3.2-1B config + gating
    --disable-bias-linear # "mlp_bias": false, "attention_bias": false  -- following Llama-3.2-1B config
    # Llama-3.2-1B config: "tie_word_embeddings": true -- embeddings tied by default; need to pass --untie-embeddings-and-output-weights to disable
    # Llama-3.2-1B config: "vocab_size": 128256 -- derived from the tokenizer
)

### ATTENTION ARGS ###
# Hybrid: GDN on 12 layers, gated softmax attention on 4. The GDN dims are part of the model size,
# so they live next to NETWORK_SIZE_ARGS. See hybrid_qwen_1B_args_ffn6208.txt for the full rationale.
ATTENTION_ARGS=(
    # --- layer pattern: 1=GDN, 0=gated softmax attn ---
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq 4   # [1,1,1,0]*4 -> 12 GDN + 4 attn, attn at layers 3/7/11/15 (== Qwen full_attention_interval=4)
    # --- GDN layers (12 of 16): identical to the pure-GDN baseline ---
    --linear-num-key-heads 8       # GDN's K/Q heads
    --linear-num-value-heads 8     # GDN's V heads, must be a multiple of linear_num_key_heads
    --linear-key-head-dim 192      # key_dim = 8*192 = 1536 = 0.75*hidden
    --linear-value-head-dim 384    # value_dim = 8*384 = 3072 = 1.5*hidden = 2*key_dim
    --linear-conv-kernel-dim 4
    # --- gated softmax attn layers (4 of 16): identical to the gated-attn scf=1 baseline ---
    --num-attention-heads 32       # "num_attention_heads": 32  -- Llama-3.2-1B config
    --group-query-attention
    --num-query-groups 8           # "num_key_value_heads": 8  -- Llama-3.2-1B config
    --attention-output-gate        # per-head sigmoid gate on the attention output; no effect on the GDN layers
    --position-embedding-type rope # for the attn layers; GDN ignores positions (conv + decay gate), RoPE has no params
    --rotary-base 500000           # "rope_theta": 500000.0  -- Llama-3.2-1B config
    --rope-scaling-factor 1        # pretraining from scratch | "factor": 32.0 in the Llama-3.2-1B config
    --use-rope-scaling
    --max-position-embeddings 8192 # 8192 since the scaling factor is 1 | 131072 in the Llama-3.2-1B config
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --log-params-norm
    --log-interval 1
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --timing-log-level 2 # level 1 = forward-backward + optimizer; level 2 adds forward-compute + backward-compute
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
    # --config-logger-dir removed: GPTModel._postprocess dumps input_ids/logits via torch.save
    # on every forward call with no guard, which fills scratch storage within one training step
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0 # "attention_dropout": 0.0  -- following Llama-3.2-1B config
    --hidden-dropout 0.0 # following standard practice / LLama doesn't use it
    --clip-grad 1.0 # 1.0 — from the Llama 3 paper (gradient clipping = 1.0)
    --weight-decay 0.1 # 0.1 -- default for pre-training (same as the scf=1 baselines)
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
    --manual-gc-interval 500
    # no --exit-signal-handler: smoke test, the walltime just times it out; periodic saves at CHECKPOINT_STEPS=50 are enough
    --eval-interval 100000000000  # must match or be a multiple of checkpoint interval, must be set to avoid TypeError even when EVAL_ITERS=0
    --eval-iters 0                # no validation split for pretraining (--split 100,0,0)
    # if the smoke test OOMs at MBS=3: uncomment to trade compute for activation memory (keeps 8 nodes / MBS=3)
    #--recompute-granularity selective
)


INITIALIZATION_ARGS=(
    --seed 28
    --init-method-std 0.02 # initializer_range: 0.02  -- following Llama-3.2-1B config + default Megatron-LM value (transformer_config.py:327)
)


# NOTE Check all the arguments in megatron/training/arguments.py#L1548
LEARNING_RATE_ARGS=(
    --lr 0.0004 # 4 × 10^−4 -- following Tiny llama
    --min-lr 0.00004  # x10 reduction, 4 × 10^−5 -- following Tiny llama
    --lr-decay-style cosine # cosine -- following LLama3 paper
    --lr-warmup-iters 500 # ~2.75% of 18141 steps; same as the scf=1 baselines
)


CHECKPOINTING_ARGS=(
    --load $CKPT_DIR  # delete this to NOT reload from the latest checkpoint
    --save $CKPT_DIR
    --save-interval $CHECKPOINT_STEPS
    --save-retain-interval $CHECKPOINT_RETAIN_STEPS
    --ckpt-format torch_dist
    #--async-save  # disabled: sync save ensures checkpoint at the exhaustion step completes before the crash
    --ckpt-fully-parallel-load # all ranks load checkpoint shards in parallel instead of sequentially
    --dist-ckpt-strictness assume_ok_unexpected # suppresses errors when checkpoint has keys not in the current model (in case of minor model changes)
)


MIXED_PRECISION_ARGS=(
    --bf16
)


# Pads vocab to a multiple of 128 for even TP splits (noop sine llama vocab is already divisible)
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
    #    --reset-attention-mask -- noop in megatron either way + we're using packed_seq
    --eod-mask-loss
    --use-packed-seq-params
    --no-create-attention-mask-in-dataloader
    --data-path $DATA_PATH_STR
    --data-cache-path $DATASET_CACHE_DIR
)

# we don't use mock data here
#if [ "$MOCK_DATA" = true ]; then
#    DATA_ARGS+=(--mock-data)
#else
#    DATA_ARGS+=(--data-path $DATA_PATH --data-cache-path $DATASET_CACHE_DIR)
#fi


#CMD_PREFIX="numactl --membind=0-3"


### TRAINING CMD ###
# pretrain_gpt_native.py = pretrain_gpt.py + --tests injected on the first forward step, then it
# falls through to normal training. Suites (attn_bench/tests/util/registry.py):
#   xdoc_mask / xdoc_loss / xdoc_position_ids = pass -- doc-boundary isolation holds (we use
#       --use-packed-seq-params + --reset-position-ids + --eod-mask-loss)
#   gdn = pass    -- the 12 GDN layers are wired into the forward (param_disturbance)
#   gated = pass  -- the 4 gated-attn layers carry the output gate (linear_qkv 4x dim + wired)
#   sink = fail   -- no --softmax-type, so the sink suite has nothing to verify -> expected fail
# The --tests harness does NOT fail the job -- read the "Summary" block in the .out.

TRAINING_CMD="python3 $MEGATRON_LM_DIR/attn_bench/tests/pretrain_gpt_native.py \
    --tests xdoc_mask=pass xdoc_loss=pass xdoc_position_ids=pass gdn=pass gated=pass sink=fail \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${ATTENTION_ARGS[@]} \
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
    ${DATA_ARGS[@]}"

# W&B Logging
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB API key detected. Enabling WANDB logging."

    pip install --quiet wandb --upgrade

    # Get and display WandB entity/workspace
    WANDB_ENTITY=$(python3 -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null || echo "unknown")
    echo "[$(date)] WandB logs will be sent to:"
    echo "           Organization/Workspace: $WANDB_ENTITY"
    echo "           Project: $PROJECT_NAME"
    echo "           Experiment: $EXP_NAME-$SLURM_JOB_ID"

    # Sync any previous run data if present
    if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
        echo "[$(date)] Syncing WANDB from previous run"
        wandb sync "$LOGGING_DIR/wandb/latest-run"
    fi
    # Add wandb-related args to TRAINING_CMD
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
    echo -e "\nRepo: ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\n$(pip list)"
    echo -e "${SEP}"
    echo -e "\nPython sys.path:\n$(python -c 'import sys; print(chr(10).join(sys.path))')"
    echo -e "${SEP}"
    echo -e "\nmegatron location: $(python -c 'import megatron; print(megatron.__file__)' 2>&1)"
    echo -e "${SEP}"
    echo -e "\n$(nvidia-smi)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(dump_env)"
    echo -e "${SEP}"
} > "$DEBUG_DIR/compute_environment.txt"


### LAUNCH ###

# TRITON_CACHE_DIR: per-rank node-local cache. A shared-FS Triton cache crashes GDN/fla runs with
# "OSError: [Errno 116] Stale file handle" when the kernels autotune/compile on first backward and
# ranks collide on the same cache files (observed in job 2570257). Keying on SLURM_JOB_ID +
# SLURM_PROCID gives each rank a clean, private cache per job (cheap: recompiles in ~1-2 min).

echo "Test+train: llama3-1b hybrid-qwen fineweb40B+gutenberg3B  nodes=${SLURM_NNODES}  gbs=${GBS}  mbs=${MBS}  steps=${TRAINING_STEPS}  datasets=${#DATA_PATH_LIST[@]}"
echo "[$(date)] Launching pretrain_gpt_native.py (--tests on first forward step, then trains)..."
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    mkdir -p $TORCH_INDUCTOR_CACHE_DIR
    mkdir -p $TRITON_HOME_DIR   # TRITON_HOME is set at the top to a node-local /tmp path; create it explicitly
    mkdir -p $PYTHON_CACHE_DIR
    export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_\$SLURM_PROCID
    mkdir -p \$TRITON_CACHE_DIR
    export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"

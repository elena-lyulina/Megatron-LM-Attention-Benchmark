#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:30:00
#SBATCH --job-name=attn-training-perf-bench-mbs3-scaling
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Follow-up to attn_training_perf_bench.slurm (job 3056624): that sweep found tp=1/mbs=3 at
# nodes=8 (65.0 node-hrs/model) more COMPUTE-EFFICIENT than tp=1/mbs=2 at nodes=8 (75.5) or
# nodes=10 (75.7) -- but only mbs=2 had been tried at nodes=10, so "8 is more efficient than 10"
# was never actually established, just assumed from an unlike comparison. This checks whether the
# tp=1/mbs=3 efficiency (45,721 tokens/sec/GPU) holds as node count grows, or degrades:
#   nodes=10 tp=1 mbs=3  DP=40  GBS=240  -- same GBS as the existing n10-tp1-mbs2-gbs240 (MBS-only diff)
#   nodes=12 tp=1 mbs=3  DP=48  GBS=288  -- same GBS as the existing n8-tp1-mbs3-gbs288 (DP-only diff)
# mbs=4 is deliberately NOT probed here (expected to OOM at tp=1; not worth the allocation time).
#
# Same PROJECT_NAME as the original sweep so both land in one wandb project for comparison. Model/
# data/optimizer args, run_pass, and the in-container wandb-table upload are copied verbatim from
# attn_training_perf_bench.slurm -- see that file for the full design rationale.
#
# Usage:
#   sbatch attn_bench/submissions/attn_training_perf_bench_mbs3_scaling.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

GPUS_PER_NODE=4

### DATA (verbatim from pretrain_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm) ###
FINEWEB_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25"
GUTENBERG_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/gutenberg_rep_1_256"

DATA_PATH_LIST=($(find -L $FINEWEB_DIR $GUTENBERG_DIR -name '*.bin' | sed 's/\.bin$//' | sort))
DATA_PATH_STR=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
echo "Total dataset prefixes: ${#DATA_PATH_LIST[@]}"

DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR

TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

### TRAINING (recipe constants -- same target as prod; per-pass GBS is pinned in CONFIGS below,
### each within ~15% of the ~256 recipe target; see run_pass for why it isn't auto-derived) ###
SEQ_LEN=8192
TARGET_TOKENS=42801699333   # 40,038,865,413 (fineweb 40B) + 2,762,833,920 (gutenberg 3B)

### BENCH: short passes, throughput only. Per-pass iter count is a CONFIGS column (run_pass's
### 5th arg) so it can be tuned per pass if wall-clock budget gets tight again. ###
WARMUP_ITERS=20   # excluded from the steady-state mean (compile/cudagraph/dataloader-fill noise)
PASS_TIMEOUT_SEC=600   # hard per-pass cap so one hung pass (e.g. OOM mid-collective) can't eat the
                        # rest of the allocation

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

PROJECT_NAME="attn_training_perf_bench"
PROJECT_DIR=$MEGATRON_LM_DIR/attn_bench/results/pretrain-bench/$PROJECT_NAME
mkdir -p $PROJECT_DIR

SUMMARY_CSV=$PROJECT_DIR/summary_$SLURM_JOB_ID.csv
echo "test,nodes,gpu_per_node,seq_len,dp,pp,tp,cp,mbs,gbs,gas,tokens_per_iteration,mean_iter_time_ms,tokens_per_sec,predicted_steps_full,predicted_hours,predicted_days" > $SUMMARY_CSV

if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE=disabled
    echo "[$(date)] No WANDB_API_KEY -> wandb disabled (throughput/predictions still in $SUMMARY_CSV)."
else
    echo "[$(date)] WANDB enabled, project=$PROJECT_NAME (one run per pass + a sweep-summary run)."
fi

### ENV (verbatim from prod) ###
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

echo "NODES: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"

#### Megatron Args (CONSTANT arrays -- verbatim from the real full-attn config; per-pass arrays built in run_pass) ####

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

INITIALIZATION_ARGS=(
    --seed 28
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

VOCAB_ARGS=(
    --make-vocab-size-divisible-by 128
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

ulimit -c 0

# steady-state mean elapsed-time-per-iteration (ms), skipping the first WARMUP_ITERS samples
mean_iter_ms() {
    local LOGFILE=$1
    grep -oP 'elapsed time per iteration \(ms\): \K[0-9.]+' "$LOGFILE" \
        | tail -n +$((WARMUP_ITERS + 1)) \
        | awk '{sum+=$1; n++} END { if (n>0) printf "%.3f", sum/n; else print "" }'
}

# run_pass N TP MBS GBS ITERS PORT -- GBS is pinned explicitly per config (see CONFIGS grid below),
# not auto-rounded, so that designated correctness pairs land on the exact same realized GBS.
# ITERS lets ceiling-probe passes run fewer steps to save wall-clock (still plenty of steady-state
# samples after WARMUP_ITERS is dropped).
run_pass() {
    local N=$1 TP=$2 MBS=$3 GBS=$4 ITERS=$5 PORT=$6
    local TAG="n${N}-tp${TP}-mbs${MBS}-gbs${GBS}"
    echo ""
    echo "######## PASS ${TAG}  nodes=${N}  tp=${TP}  mbs=${MBS}  gbs=${GBS}  iters=${ITERS}  $(date) ########"

    local DP=$(( N * GPUS_PER_NODE / TP ))
    if (( N * GPUS_PER_NODE % TP != 0 )); then
        echo "SKIP ${TAG}: TP=${TP} does not divide world size $((N * GPUS_PER_NODE))"
        return
    fi
    if (( GBS % (MBS * DP) != 0 )); then
        echo "SKIP ${TAG}: GBS=${GBS} not divisible by MBS*DP=$((MBS * DP)) -- fix the CONFIGS entry"
        return
    fi
    local GAS=$(( GBS / (MBS * DP) ))

    local EXP_NAME="${TAG}-${SLURM_JOB_ID}"
    local EXP_DIR=$PROJECT_DIR/$EXP_NAME
    local CKPT_DIR=$EXP_DIR/checkpoints
    local LOGGING_DIR=$EXP_DIR/logging
    local TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
    local DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
    mkdir -p $CKPT_DIR $LOGGING_DIR $TENSORBOARD_DIR $DEBUG_DIR
    cp $0 $DEBUG_DIR

    local DIST_ARGS=(
        --tensor-model-parallel-size $TP
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
    )
    (( TP > 1 )) && DIST_ARGS+=(--sequence-parallel)

    local TRAINING_ARGS=(
        --micro-batch-size $MBS
        --global-batch-size $GBS
        --no-check-for-nan-in-loss-and-grad
        --train-iters $ITERS
        --cross-entropy-loss-fusion
        --manual-gc
        --manual-gc-interval 500
        --exit-signal-handler
        --eval-interval 100000000000
        --eval-iters 0
    )

    # LR schedule doesn't affect throughput; warmup just needs to stay < ITERS.
    local LEARNING_RATE_ARGS=(
        --lr 0.0004
        --min-lr 0.00004
        --lr-decay-style cosine
        --lr-warmup-iters 10
    )

    local LOGGING_ARGS=(
        --log-throughput
        --log-progress
        --log-interval 1
        --tensorboard-dir $TENSORBOARD_DIR
        --log-timers-to-tensorboard
        --timing-log-level 2
        --no-log-loss-scale-to-tensorboard
        --log-memory-to-tensorboard
    )

    # Fresh/empty CKPT_DIR each pass => --load is a no-op => fresh init. save-interval > ITERS
    # => never actually saves (avoids IO skewing the throughput measurement).
    local CHECKPOINTING_ARGS=(
        --load $CKPT_DIR
        --save $CKPT_DIR
        --save-interval $((ITERS + 1))
        --ckpt-format torch_dist
        --ckpt-fully-parallel-load
        --dist-ckpt-strictness assume_ok_unexpected
    )

    local WANDB_ARGS=()
    if [ -n "${WANDB_API_KEY:-}" ]; then
        WANDB_ARGS=(--wandb-save-dir $LOGGING_DIR --wandb-project $PROJECT_NAME --wandb-exp-name bench-$EXP_NAME)
    fi

    local CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
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
        ${DIST_ARGS[@]} \
        ${TOKENIZER_ARGS[@]} \
        ${VOCAB_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${WANDB_ARGS[@]}"

    local NODELIST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n "$N" | paste -sd, -)
    local PASS_OUT=$EXP_DIR/pass.out

    # timeout+kill-on-bad-exit bound a hung/crashed pass so it can't eat the rest of the allocation.
    timeout --kill-after=30s ${PASS_TIMEOUT_SEC}s \
    srun \
        --nodes=$N --ntasks=$((N * GPUS_PER_NODE)) --ntasks-per-node=$GPUS_PER_NODE --nodelist=$NODELIST \
        --kill-on-bad-exit=1 \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --mpi=pmix \
        --environment=$CONTAINER_ENV \
        --network=disable_rdzv_get \
        -lu \
        bash -c "
        mkdir -p $TORCH_INDUCTOR_CACHE_DIR
        mkdir -p $TRITON_HOME_DIR
        mkdir -p $PYTHON_CACHE_DIR
        export TRITON_CACHE_DIR=/tmp/triton_${USER}_${SLURM_JOB_ID}_${TAG}_\$SLURM_PROCID
        mkdir -p \$TRITON_CACHE_DIR
        export MASTER_ADDR=$(echo $NODELIST | cut -d, -f1)
        export MASTER_PORT=${PORT}
        export WORLD_SIZE=$((N * GPUS_PER_NODE))
        export PYTHONPATH='${PYTHONPATH}'
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${CMD}" \
        2>&1 | tee $PASS_OUT
    local SRUN_EXIT=${PIPESTATUS[0]}
    if [ "$SRUN_EXIT" -eq 124 ]; then
        echo "PASS ${TAG}: TIMED OUT after ${PASS_TIMEOUT_SEC}s -- killed, moving to next pass"
    elif [ "$SRUN_EXIT" -ne 0 ]; then
        echo "PASS ${TAG}: exited nonzero ($SRUN_EXIT) -- check $PASS_OUT"
    fi

    local MEAN_MS=$(mean_iter_ms $PASS_OUT)
    if [ -z "$MEAN_MS" ]; then
        echo "PASS ${TAG}: no throughput samples parsed from $PASS_OUT (pass likely failed) -- logging NA row"
        echo "${TAG},${N},${GPUS_PER_NODE},${SEQ_LEN},${DP},1,${TP},1,${MBS},${GBS},${GAS},NA,NA,NA,NA,NA,NA" >> $SUMMARY_CSV
        echo "######## END PASS ${TAG}  $(date) ########"
        return
    fi

    local TOKENS_PER_ITER=$((GBS * SEQ_LEN))
    local TOKENS_PER_SEC=$(awk -v t=$TOKENS_PER_ITER -v ms=$MEAN_MS 'BEGIN { printf "%.1f", t / (ms / 1000) }')
    local STEPS_FULL=$((TARGET_TOKENS / TOKENS_PER_ITER))
    local PRED_HOURS=$(awk -v steps=$STEPS_FULL -v ms=$MEAN_MS 'BEGIN { printf "%.4f", steps * ms / 1000 / 3600 }')
    local PRED_DAYS=$(awk -v h=$PRED_HOURS 'BEGIN { printf "%.4f", h / 24 }')

    echo "PASS ${TAG}: mean_iter_ms=${MEAN_MS}  tokens/sec=${TOKENS_PER_SEC}  predicted_full_run=${PRED_HOURS}h (${PRED_DAYS}d)"
    echo "${TAG},${N},${GPUS_PER_NODE},${SEQ_LEN},${DP},1,${TP},1,${MBS},${GBS},${GAS},${TOKENS_PER_ITER},${MEAN_MS},${TOKENS_PER_SEC},${STEPS_FULL},${PRED_HOURS},${PRED_DAYS}" >> $SUMMARY_CSV

    echo "######## END PASS ${TAG}  $(date) ########"
}

### LOG COMPUTE ENV (once per job; same nodes/container for every pass) ###
dump_env() { printenv | grep -v '^WANDB_API_KEY='; }
JOB_DEBUG_DIR=$PROJECT_DIR/debug/$SLURM_JOB_ID
mkdir -p $JOB_DEBUG_DIR
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $JOB_DEBUG_DIR/memory_logging.txt
{
    echo -e "$(date)"
    echo -e "\nSlurm file: $0\n"
    cat "$0"
    echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
    echo -e "\nRepo: ${MEGATRON_LM_DIR} ($(git -C ${MEGATRON_LM_DIR} rev-parse --verify HEAD 2>&1))"
    echo -e "\nEnvironment Variables:\n\n$(dump_env)"
} > "$JOB_DEBUG_DIR/compute_environment.txt"

### LAUNCH (sequential; nodelist for each pass is the FIRST N hostnames of this allocation) ###
echo "[$(date)] Launching mbs=3 scaling follow-up..."

# N   TP  MBS  GBS  ITERS  PORT
run_pass 10 1 3 240 70 8881   # DP=40  -- same GBS as n10-tp1-mbs2-gbs240 (from job 3056624): MBS-only diff
run_pass 12 1 3 288 70 8882   # DP=48  -- same GBS as n8-tp1-mbs3-gbs288 (from job 3056624): DP-only diff

echo ""
echo "Summary CSV: $SUMMARY_CSV"
cat $SUMMARY_CSV

### Log the summary CSV as a wandb.Table so every config's prediction is visible in one place.
### Runs INSIDE the container (same python/wandb as the training passes) -- the bare allocation
### shell's python3 is an unrelated stale system install and previously crashed this step. ###
if [ -n "${WANDB_API_KEY:-}" ]; then
    cat > $JOB_DEBUG_DIR/log_summary_to_wandb.py <<'PYEOF'
import csv
import sys

import wandb

summary_csv, project_name, run_name = sys.argv[1], sys.argv[2], sys.argv[3]

with open(summary_csv) as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]

run = wandb.init(project=project_name, name=run_name, job_type="scaling-sweep-summary")
run.log({"scaling_sweep": wandb.Table(columns=header, data=data)})
run.finish()
print(f"Logged {len(data)} rows to wandb project {project_name}, run {run_name}")
PYEOF
    srun --nodes=1 --ntasks=1 --environment=$CONTAINER_ENV \
        python3 $JOB_DEBUG_DIR/log_summary_to_wandb.py "$SUMMARY_CSV" "$PROJECT_NAME" "sweep-summary-$SLURM_JOB_ID"
else
    echo "[$(date)] No WANDB_API_KEY -> summary table not uploaded, CSV is at $SUMMARY_CSV"
fi

echo "END TIME: $(date)"
echo "Per-pass logs: $PROJECT_DIR/{n10-tp1-mbs3-gbs240,n12-tp1-mbs3-gbs288}-$SLURM_JOB_ID/pass.out"

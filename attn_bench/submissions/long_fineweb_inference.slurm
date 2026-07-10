#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:30:00
#SBATCH --job-name=long-fineweb-inf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Per-position loss on long FineWeb-Edu documents (verbatim, no prefix/suffix framing),
# one job per (model, data file) pair. VARIANT picks the checkpoint, same as
# long_gutenberg_inference.slurm; DATA_FILE is the full path to one jsonl produced by
# extract_long_docs.py (e.g. the seen or unseen partition at a given length range).
# Results land under RESULTS_BASE/$EXP_NAME/<data file's parent dir name>/$CONFIG/, so
# seen and unseen (different parent dirs) never collide even though they share the same
# long_{lo}_{hi}.jsonl basename. Writes go to Iopsstor scratch during the job and get copied
# to Capstor store at the end (CSCS guidance: never write to capstor from a compute node).
#
# sink/off-by-one use unfused attention (O(S^2) memory) and OOM at full length under the
# default TP=1 -- they auto-select --tensor-parallel 4 (all 4 GPUs form one TP group, DP=1)
# and default to --max-length 20480 (override via MAX_LENGTH; even TP=4 needed this cap on
# Gutenberg, see attn_bench/notebooks/long_gutenberg.ipynb CONFIG_TP4). TP=4 is much slower
# per sequence than the DP=4 default (no data-parallel striding) -- override --time when
# submitting these two variants, the file's default is sized for the fused/DP variants.
#
# Usage:
#   sbatch --export=VARIANT=full,DATA_FILE=/path/to/long_24576_32768.jsonl long_fineweb_inference.slurm
#   sbatch --time=2:00:00 --export=VARIANT=sink,DATA_FILE=/path/to/long_24576_32768.jsonl long_fineweb_inference.slurm
#   REPETITIONS is not used here (no repetition axis); MAX_SAMPLES=20 for calibration.
# (normally submitted via long_fineweb_inference_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RESULTS_BASE=/users/$USER/store/long-fineweb-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/long-fineweb-results
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

DATA_FILE=${DATA_FILE:?ERROR: DATA_FILE must be set to a jsonl produced by extract_long_docs.py}
MAX_LENGTH=${MAX_LENGTH:-}     # empty = no cap (docs are already capped by extract_long_docs.py)
MAX_SAMPLES=${MAX_SAMPLES:-}   # empty = long_fineweb_inference.py's own default (660), not all docs
OVERWRITE=${OVERWRITE:-}       # set to anything to recompute if the bucket already exists
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to anything to log GDN state norms (no-op on attention models)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128 in the script)

### VARIANT -> EXP_NAME + non-restored flags (same model list as long_gutenberg_inference.slurm) ###
VARIANT=${VARIANT:-full}
NEEDS_TRITON=0
TP=1
GDN_DIMS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"
case "$VARIANT" in
    full)       EXP_NAME=llama3-1b-full-attn-fineweb40B-gutenberg3B;             MEGATRON_EXTRA="" ;;
    gated)      EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B;            MEGATRON_EXTRA="--attention-output-gate" ;;
    full-xdoc-leak)  EXP_NAME=llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B; MEGATRON_EXTRA="" ;;
    sink)       EXP_NAME=llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215;       MEGATRON_EXTRA="--softmax-type learnable --attention-backend unfused";
                # unfused attn OOMs at TP=1 for these lengths; TP=4 needs a length cap too (Gutenberg: 20480)
                TP=4; [ -z "$MAX_LENGTH" ] && MAX_LENGTH=20480 ;;
    off-by-one) EXP_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215; MEGATRON_EXTRA="--softmax-type off-by-one --attention-backend unfused";
                # checkpoint lives at the non-te215 path; EXP_NAME (results dir) stays -te215 to match the mem run
                CKPT_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B;
                TP=4; [ -z "$MAX_LENGTH" ] && MAX_LENGTH=20480 ;;
    gdn)        EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B;                   MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r0)   EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B;          MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r0.5) EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B;        MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    carry-r1)   EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B;          MEGATRON_EXTRA="$GDN_DIMS"; NEEDS_TRITON=1 ;;
    *) echo "Unknown VARIANT=$VARIANT (expected full|gated|full-xdoc-leak|sink|off-by-one|gdn|carry-r0|carry-r0.5|carry-r1)"; exit 1 ;;
esac

# CKPT_NAME defaults to EXP_NAME; only off-by-one overrides it (checkpoint path != results name).
MODEL_DIR=/users/$USER/store/pretrain-results/${CKPT_NAME:-$EXP_NAME}
PARTITION_DIR=$(basename "$(dirname "$DATA_FILE")")
RESULTS_DIR=$RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
SCRATCH_RESULTS_DIR=$SCRATCH_RESULTS_BASE/$EXP_NAME/$PARTITION_DIR
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/long_fineweb_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
# CONFIG must match config_name() in long_inference.py: unset MAX_SAMPLES falls back to
# long_fineweb_inference.py's own DEFAULT_MAX_SAMPLES (660), not "all"; tp>1 appends "_tp{TP}".
CONFIG="${MAX_SAMPLES:-660}_samples_${MAX_LENGTH:-full}_tokens"
[ "$TP" -gt 1 ] && CONFIG="${CONFIG}_tp${TP}"
KEY=$(basename "$DATA_FILE" .jsonl)
STORE_NPZ=$RESULTS_DIR/$CONFIG/$KEY.npz
if [ -z "$OVERWRITE" ] && [ -f "$STORE_NPZ" ] \
   && { [ -z "$LOG_STATE_NORM" ] || [ -f "$RESULTS_DIR/$CONFIG/${KEY}_state.npz" ]; }; then
    echo "$STORE_NPZ already present in store -- skipping."
    exit 0
fi

# Optional caps, and the megatron passthrough only when non-empty. --experiment-path is the
# scratch working dir the python script actually reads/writes; store is only touched by the
# copy step at the end (never write to capstor from a compute node).
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $SCRATCH_RESULTS_DIR \
    --data-file $DATA_FILE"
[ "$TP" -gt 1 ] && COMMON_ARGS="$COMMON_ARGS --tensor-parallel $TP"
[ -n "$MAX_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-length $MAX_LENGTH"
[ -n "$MAX_SAMPLES" ] && COMMON_ARGS="$COMMON_ARGS --max-samples $MAX_SAMPLES"
[ -n "$OVERWRITE" ] && COMMON_ARGS="$COMMON_ARGS --overwrite"
[ -n "$LOG_STATE_NORM" ] && COMMON_ARGS="$COMMON_ARGS --log-state-norm"
[ -n "$STATE_CHUNK" ] && COMMON_ARGS="$COMMON_ARGS --state-chunk $STATE_CHUNK"
EXTRA_MEGATRON=""
[ -n "$MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $MEGATRON_EXTRA"

echo "RUN: variant=$VARIANT  exp=$EXP_NAME  data_file=$DATA_FILE  tp=$TP  max_length=${MAX_LENGTH:-none}  max_samples=${MAX_SAMPLES:-660 (default)}  triton=$NEEDS_TRITON  state_norm=${LOG_STATE_NORM:+on}"

### Position-wise loss inference ###
# GDN needs a per-rank node-local TRITON_CACHE_DIR: fla kernels autotune on first use and
# a shared/colliding cache crashes with "stale file handle" (job 2570257). torchrun sets
# RANK per worker (SLURM_PROCID is 0 for all four under ntasks-per-node=1), so the cache is
# keyed inside the workers via the --no-python wrapper. Attention variants skip all that.

if [ "$NEEDS_TRITON" = "1" ]; then
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 --no-python \
            bash -c 'export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}_\${RANK}; mkdir -p \"\$TRITON_CACHE_DIR\"; exec python \"\$@\"' _ \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
else
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
fi
SRUN_STATUS=$?

### Copy scratch results to store ###
if [ $SRUN_STATUS -eq 0 ]; then
    echo "Copying results to store: $RESULTS_DIR"
    mkdir -p "$RESULTS_DIR"
    cp -a "$SCRATCH_RESULTS_DIR/." "$RESULTS_DIR/"
else
    echo "srun exited $SRUN_STATUS -- not copying scratch results to store"
fi

echo "END TIME: $(date)"

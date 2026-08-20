#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:50:00
#SBATCH --job-name=measure-mem
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Measure memorization for one trained model, one (offset, prefix, suffix) point:
#   - Run sparse Gutenberg inference directly from torch_dist checkpoint (no HF conversion),
#     reusing/extending an existing larger-suffix run instead of regenerating if one exists
#   - Compute Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z at every reachable suffix
#     boundary, not just this job's own suffix point
#
# Env vars:
#   MODEL              checkpoint to use, see llama_checkpoints.sh (default: full)
#   CHECKPOINT_BACKEND megatron (default, all attention variants) or hf (already-converted HF
#                      checkpoint, faster generate(); full/sink/gated/swa only, not GDN -- run
#                      convert_and_validate_hf.slurm first)
#   OFFSET         document start offset (default: 0)
#   PREFIX_LENGTH  prompt length in tokens (default: 500)
#   SUFFIX_LENGTH  generated/gold length in tokens (default: 500)
#   REPETITIONS    comma-separated repetition buckets to run (default: 0,1,2,4,8,16,32,64,128,256)
#   SINK_SCALE     sink/off-by-one only: scale the learned sink logit (megatron backend only)
#   OVERWRITE      skip the whole job unless set, even if store already has this suffix's pkl
#   FORCE_METRICS  recompute every reachable boundary's pkl in Step 2, even ones that already
#                  exist (e.g. after backfill added fields an old pkl predates) -- independent
#                  of OVERWRITE, which only gates whether Step 1 runs at all
#
# Usage: sbatch --export=MODEL=sink,OFFSET=0,PREFIX_LENGTH=1000 measure_mem.slurm
# (normally submitted via measure_mem_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-500}
SUFFIX_LENGTH=${SUFFIX_LENGTH:-500}
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-megatron}
SINK_SCALE=${SINK_SCALE:-}
OVERWRITE=${OVERWRITE:-}
FORCE_METRICS=${FORCE_METRICS:-}

### MODEL -> EXP_NAME / MEGATRON_EXTRA / NEEDS_TRITON / IS_SINK_FAMILY / NEEDS_UNFUSED_DECODE ###
MODEL=${MODEL:-full-scf8}
model_config "$MODEL"

# Mirrors InferenceBackend.experiment_path_suffix() in inference_backend.py -- keep in sync.
if [ "$CHECKPOINT_BACKEND" = "hf" ]; then
    PDM_EXP_NAME="${EXP_NAME}_hf"
elif [ -n "$SINK_SCALE" ]; then
    PDM_EXP_NAME="${EXP_NAME}_sscale${SINK_SCALE}"
else
    PDM_EXP_NAME="$EXP_NAME"
fi

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_MEM_BASE=/users/$USER/store/mem-results
SCRATCH_MEM_BASE=/iopsstor/scratch/cscs/$USER/mem-results
MEM_DIR=$SCRATCH_MEM_BASE/SparseGutenberg/$EXP_NAME
SCRATCH_EXP=$SCRATCH_MEM_BASE/SparseGutenberg/$PDM_EXP_NAME
STORE_EXP=$STORE_MEM_BASE/SparseGutenberg/$PDM_EXP_NAME
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/prefix_extraction_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
STORE_PKL=$STORE_MEM_BASE/SparseGutenberg/$PDM_EXP_NAME/metrics/offset_${OFFSET}_prefix_${PREFIX_LENGTH}_suffix_${SUFFIX_LENGTH}_greedy.pkl
if [ -z "$OVERWRITE" ] && [ -f "$STORE_PKL" ]; then
    echo "$STORE_PKL already present in store -- skipping."
    exit 0
fi

# HF checkpoint must already exist (produced separately by convert_and_validate_hf.slurm) --
# this also doubles as the "is this model even HF-convertible" check, since GDN models
# never get one.
if [ "$CHECKPOINT_BACKEND" = "hf" ]; then
    HF_DIR=/users/$USER/store/hf-checkpoints/$EXP_NAME
    if [ ! -f "$HF_DIR/config.json" ]; then
        echo "CHECKPOINT_BACKEND=hf requested but $HF_DIR/config.json not found -- run convert_and_validate_hf.slurm (MODEL=$MODEL) first."
        exit 1
    fi
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME  backend=$CHECKPOINT_BACKEND  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=$SUFFIX_LENGTH${SINK_SCALE:+  sink_scale=$SINK_SCALE}"

# --experiment-path is the scratch working dir; copy_scratch_to_store handles the store side.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --checkpoint-backend $CHECKPOINT_BACKEND \
    --experiment-path $MEM_DIR \
    --data-folder $GUTENBERG_JSONL_DIR \
    --repetitions $REPETITIONS \
    --offset $OFFSET \
    --prefix-length $PREFIX_LENGTH \
    --suffix-length $SUFFIX_LENGTH \
    --batch-size 20"

EXTRA_MEGATRON=""
EXTRA_ENV=""
if [ "$CHECKPOINT_BACKEND" = "hf" ]; then
    COMMON_ARGS="$COMMON_ARGS --hf-dir $HF_DIR"
    if [ "$IS_SINK_FAMILY" = "1" ]; then
        # sink's HF architecture (SinkLlamaForCausalLM) needs a patched flash-attn checkout
        # for its fused kernel -- see flash-attention-cute-workflow.md. Only sink pays this
        # cost: full's HF path never imports flash_attn.cute at all.
        EXTRA_ENV="export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention; pip install --quiet 'nvidia-cutlass-dsl==4.6.2';"
    fi
else
    COMMON_ARGS="$COMMON_ARGS --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH"
    [ "$IS_SINK_FAMILY" = "1" ] && [ -n "$SINK_SCALE" ] && COMMON_ARGS="$COMMON_ARGS --sink-scale $SINK_SCALE"

    # NEEDS_UNFUSED_DECODE: TE 2.15 rejects fused attn in decode mode with a sink token (see
    # llama_checkpoints.sh). Forces the pure-PyTorch UnfusedDotProductAttention path.
    GREEDY_MEGATRON_EXTRA="$MEGATRON_EXTRA"
    if [ "$NEEDS_UNFUSED_DECODE" = "1" ]; then
        GREEDY_MEGATRON_EXTRA="$GREEDY_MEGATRON_EXTRA --attention-backend unfused"
        EXTRA_ENV="export NVTE_FUSED_ATTN=0; export CUDNN_LOGERR_DBG=1; export CUDNN_LOGDEST_DBG=stderr;"
    fi
    [ -n "$GREEDY_MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $GREEDY_MEGATRON_EXTRA"
fi

### STEP 1: sparse Gutenberg inference ###
# Loads torch_dist checkpoint directly (DCP resharding merges TP shards on the fly).
# Saves rank*.jsonl to $MEM_DIR/inference/offset_.../rep_*/ (scratch).
#
# GDN needs a per-rank node-local TRITON_CACHE_DIR: fla kernels autotune on first use and a
# shared/colliding cache crashes with "stale file handle" (job 2570257). torchrun sets RANK
# per worker (SLURM_PROCID is 0 for all four under ntasks-per-node=1), so the cache is keyed
# inside the workers via the --no-python wrapper. Attention variants skip all that.

if [ "$NEEDS_TRITON" = "1" ]; then
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        pip install numba -q
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
        pip install numba -q
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        $EXTRA_ENV
        RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
        torchrun --nproc_per_node=4 \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
fi
INFERENCE_STATUS=$?

# Copy inference results (usually expensive) to store -- Step 1 only ever writes its own
# exact suffix dir, no wildcard needed.
SUFFIX_DIR="offset_${OFFSET}_prefix_${PREFIX_LENGTH}_suffix_${SUFFIX_LENGTH}"
copy_scratch_to_store $INFERENCE_STATUS "$SCRATCH_EXP/inference/$SUFFIX_DIR" "$STORE_EXP/inference/$SUFFIX_DIR"
if [ $INFERENCE_STATUS -eq 0 ] && [ -f "$SCRATCH_EXP/sink_scale_metadata.json" ]; then
    mkdir -p "$STORE_EXP"
    cp -a "$SCRATCH_EXP/sink_scale_metadata.json" "$STORE_EXP/"
fi

### STEP 2: compute memorization metrics ###
# Reads inference outputs from scratch, writes one Rouge-L/lcs_norm/TTR/token_acc/
# divergence_point/p_z summary pkl per suffix boundary to scratch.
# Auto-discovers every (offset, prefix) pair under $MEM_DIR/inference and every boundary
# reachable from whatever's landed on scratch, not just this job's own suffix point.

METRIC_STATUS=1
if [ $INFERENCE_STATUS -eq 0 ]; then
    FORCE_ARG=""
    [ -n "$FORCE_METRICS" ] && FORCE_ARG="--force"
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        pip install numba -q
        export PYTHONPATH='${PYTHONPATH}'
        python $MEGATRON_LM_DIR/attn_bench/evaluation/compute_memorization_metrics.py \
            --exprs $PDM_EXP_NAME \
            --base-path $SCRATCH_MEM_BASE/SparseGutenberg \
            --save-path $SCRATCH_MEM_BASE/SparseGutenberg \
            --offsets $OFFSET \
            --prefix-lengths $PREFIX_LENGTH \
            $FORCE_ARG"
    METRIC_STATUS=$?
else
    echo "Inference (step 1) failed with status $INFERENCE_STATUS -- skipping metric computation"
fi

### Copy metrics to store -- every suffix boundary Step 2 reached for this offset/prefix,
### not just this job's own suffix point (see Step 2 comment above). ###
BOUNDARY_PATTERN="offset_${OFFSET}_prefix_${PREFIX_LENGTH}_suffix_*"
copy_scratch_to_store $METRIC_STATUS "$SCRATCH_EXP/metrics" "$STORE_EXP/metrics" "$BOUNDARY_PATTERN"
copy_scratch_to_store $METRIC_STATUS "$SCRATCH_EXP/metrics_metadata" "$STORE_EXP/metrics_metadata" "$BOUNDARY_PATTERN"

echo "END TIME: $(date)"
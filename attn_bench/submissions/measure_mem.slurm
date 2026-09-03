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

# Measure memorization for one trained model, one or more (offset, prefix) points sharing
# one suffix and one checkpoint load. Step 1: sparse Gutenberg inference from the torch_dist
# checkpoint, reusing/extending an existing larger-suffix run instead of regenerating. Step 2:
# Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z at every reachable suffix boundary.
#
# Env vars:
#   MODEL, CHECKPOINT_BACKEND (megatron/hf), SINK_SCALE  -- see llama_checkpoints.sh / model_config
#   POINTS         comma-separated offset:prefix_length pairs, e.g. "0:500,50:450" (default: "0:500")
#   SUFFIX_LENGTH  shared by every point (default: 500)
#   REPETITIONS    comma-separated rep buckets (default: 0,1,2,4,8,16,32,64,128,256)
#   FORCE_METRICS  recompute every Step-2 boundary even if it already exists
#   MAX_DOC_LENGTH Stage 1 and Stage 2 both skip a point where offset+prefix+suffix exceeds this (default: unset, no filtering)
#   EXPECTED_COUNT records a rep dir needs to count as present, for Stage 2's completeness check (default: 660)
#   CAPTURE_ATTENTION  Step 1 also captures attention maps into per-rep / per-Rouge-L .npz
#                      (megatron backend only; single point per job; prefix+suffix <= 600)
#
# No pre-checkpoint-load "is this already done" check here -- measure_mem_all.sh already
# decided this job is worth submitting (its own dry-run check, dual-checking scratch+store).
# Once running, Step 1 and Step 2 each self-skip internally at the same dual-checked,
# per-point/per-boundary granularity, so nothing here can waste real compute either.
#
# Usage: sbatch --export=MODEL=sink,POINTS=0:1000 measure_mem.slurm
# (normally submitted via measure_mem_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh

POINTS=${POINTS:-0:500}
SUFFIX_LENGTH=${SUFFIX_LENGTH:-500}
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
CHECKPOINT_BACKEND=${CHECKPOINT_BACKEND:-megatron}
EXPECTED_COUNT=${EXPECTED_COUNT:-660}
SINK_SCALE=${SINK_SCALE:-}
FORCE_METRICS=${FORCE_METRICS:-}
MAX_DOC_LENGTH=${MAX_DOC_LENGTH:-}
CAPTURE_ATTENTION=${CAPTURE_ATTENTION:-}

### MODEL -> EXP_NAME / MEGATRON_EXTRA / NEEDS_TRITON / NEEDS_FLA_052 / IS_SINK_FAMILY / NEEDS_UNFUSED_DECODE ###
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
STORE_MEM_DIR=$STORE_MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/prefix_extraction_inference.py

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NEEDS_FLA_052 (KDA): the container ships flash-linear-attention 0.4.2, whose chunk_kda NaNs.
# Side-install 0.5.2 into $FLA_DIR (guarded one-shot, persisted, gitignored) and prepend it to
# PYTHONPATH so it shadows the container copy in every srun below. Mirrors the KDA pretrain slurm.
FLA_VERSION=0.5.2
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
if [ "$NEEDS_FLA_052" = "1" ]; then
    export PYTHONPATH=$FLA_DIR:$PYTHONPATH
    srun --nodes=1 --ntasks=1 --overlap --environment=$CONTAINER_ENV bash -c "
        set -e
        if [ ! -f '$FLA_DIR/fla/__init__.py' ]; then
            pip install --quiet --no-deps --ignore-installed --target='$FLA_DIR' \
                'flash-linear-attention==$FLA_VERSION' 'fla-core==$FLA_VERSION'
        fi
        PYTHONPATH='$FLA_DIR' python3 -c \"import fla; assert fla.__version__.startswith('$FLA_VERSION'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')\"
    "
fi

# Parsed once here, reused by every step below.
IFS=',' read -r -a POINTS_ARR <<< "$POINTS"

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

echo "RUN: model=$MODEL  exp=$EXP_NAME  backend=$CHECKPOINT_BACKEND  points=$POINTS  suffix=$SUFFIX_LENGTH${SINK_SCALE:+  sink_scale=$SINK_SCALE}"

# --experiment-path is the scratch working dir -- results stay there; sync to store manually
# (e.g. copy_mem_results_to_store.sh) when you want it, scratch is only purged every ~2 weeks.
# --persistent-storage-path lets the Python skip-checks see results scratch has lost to a purge.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --checkpoint-backend $CHECKPOINT_BACKEND \
    --experiment-path $MEM_DIR \
    --persistent-storage-path $STORE_MEM_DIR \
    --data-folder $GUTENBERG_JSONL_DIR \
    --repetitions $REPETITIONS \
    --points ${POINTS_ARR[*]} \
    --suffix-length $SUFFIX_LENGTH \
    --batch-size 20"
[ -n "$MAX_DOC_LENGTH" ] && COMMON_ARGS="$COMMON_ARGS --max-doc-length $MAX_DOC_LENGTH"

if [ -n "$CAPTURE_ATTENTION" ]; then
    if [ "$CHECKPOINT_BACKEND" = "hf" ]; then
        echo "CAPTURE_ATTENTION is megatron-only (hooks read Megatron internals) -- drop CHECKPOINT_BACKEND=hf."
        exit 1
    fi
    COMMON_ARGS="$COMMON_ARGS --capture-attention"
fi

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
# Loads torch_dist checkpoint directly; saves rank*.jsonl to $MEM_DIR/inference/offset_.../rep_*/.
#
# GDN needs a per-rank node-local TRITON_CACHE_DIR (fla autotune + shared cache = "stale file
# handle", job 2570257) -- keyed inside the workers via the --no-python wrapper below.

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

### STEP 2: compute memorization metrics ###
# Writes one Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z pkl per reachable suffix
# boundary (not just this job's own suffix point) to scratch.
METRIC_STATUS=1
if [ $INFERENCE_STATUS -eq 0 ]; then
    FORCE_ARG=""
    [ -n "$FORCE_METRICS" ] && FORCE_ARG="--force"
    MAX_DOC_LENGTH_ARG=""
    [ -n "$MAX_DOC_LENGTH" ] && MAX_DOC_LENGTH_ARG="--max-doc-length $MAX_DOC_LENGTH"
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        pip install numba -q
        export PYTHONPATH='${PYTHONPATH}'
        python $MEGATRON_LM_DIR/attn_bench/evaluation/compute_memorization_metrics.py \
            --exp-name $PDM_EXP_NAME \
            --base-path $SCRATCH_MEM_BASE/SparseGutenberg \
            --save-path $SCRATCH_MEM_BASE/SparseGutenberg \
            --persistent-storage-path $STORE_MEM_BASE/SparseGutenberg \
            --points ${POINTS_ARR[*]} \
            --suffix-length $SUFFIX_LENGTH \
            --repetitions $REPETITIONS \
            --expected-count $EXPECTED_COUNT \
            $MAX_DOC_LENGTH_ARG \
            $FORCE_ARG"
    METRIC_STATUS=$?
else
    echo "Inference (step 1) failed with status $INFERENCE_STATUS -- skipping metric computation"
fi

echo "END TIME: $(date)"
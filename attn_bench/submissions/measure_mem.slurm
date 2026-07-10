#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:40:00
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
#   - Run sparse Gutenberg inference directly from torch_dist checkpoint (no HF conversion)
#   - Compute LCS / Rouge-L memorization metrics
#
# Env vars:
#   MODEL          checkpoint to use, see llama_checkpoints.sh (default: full)
#   OFFSET         document start offset (default: 0)
#   PREFIX_LENGTH  prompt length in tokens (default: 500)
#   SUFFIX_LENGTH  generated/gold length in tokens (default: 500)
#   SINK_SCALE     sink/off-by-one only: scale the learned sink logit
#   OVERWRITE      recompute even if store already has this pkl
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
SINK_SCALE=${SINK_SCALE:-}
OVERWRITE=${OVERWRITE:-}

### MODEL -> EXP_NAME / MEGATRON_EXTRA / NEEDS_TRITON / IS_SINK_FAMILY / NEEDS_UNFUSED_DECODE ###
MODEL=${MODEL:-full}
model_config "$MODEL"
[ -n "$SINK_SCALE" ] && PDM_EXP_NAME="${EXP_NAME}_sscale${SINK_SCALE}" || PDM_EXP_NAME="$EXP_NAME"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_MEM_BASE=/users/$USER/store/mem-results
SCRATCH_MEM_BASE=/iopsstor/scratch/cscs/$USER/mem-results
MEM_DIR=$SCRATCH_MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_sparse.py

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
STORE_PKL=$STORE_MEM_BASE/SparseGutenberg/$PDM_EXP_NAME/offset_${OFFSET}_prefix_${PREFIX_LENGTH}_suffix_${SUFFIX_LENGTH}_greedy.pkl
if [ -z "$OVERWRITE" ] && [ -f "$STORE_PKL" ]; then
    echo "$STORE_PKL already present in store -- skipping."
    exit 0
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=$SUFFIX_LENGTH${SINK_SCALE:+  sink_scale=$SINK_SCALE}"

# --experiment-path is the scratch working dir; copy_scratch_to_store handles the store side.
COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --experiment-path $MEM_DIR \
    --data-folder $GUTENBERG_JSONL_DIR \
    --repetitions 0,1,2,4,8,16,32,64,128,256 \
    --offset $OFFSET \
    --prefix-length $PREFIX_LENGTH \
    --suffix-length $SUFFIX_LENGTH \
    --batch-size 20"
[ "$IS_SINK_FAMILY" = "1" ] && [ -n "$SINK_SCALE" ] && COMMON_ARGS="$COMMON_ARGS --sink-scale $SINK_SCALE"

# NEEDS_UNFUSED_DECODE: TE 2.15 rejects fused attn in decode mode with a sink token (see
# llama_checkpoints.sh). Forces the pure-PyTorch UnfusedDotProductAttention path.
GREEDY_MEGATRON_EXTRA="$MEGATRON_EXTRA"
EXTRA_ENV=""
if [ "$NEEDS_UNFUSED_DECODE" = "1" ]; then
    GREEDY_MEGATRON_EXTRA="$GREEDY_MEGATRON_EXTRA --attention-backend unfused"
    EXTRA_ENV="export NVTE_FUSED_ATTN=0; export CUDNN_LOGERR_DBG=1; export CUDNN_LOGDEST_DBG=stderr;"
fi
EXTRA_MEGATRON=""
[ -n "$GREEDY_MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $GREEDY_MEGATRON_EXTRA"

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

### STEP 2: compute memorization metrics ###
# Reads inference outputs from scratch, writes the LCS / Rouge-L summary pkl to scratch.

METRIC_STATUS=1
if [ $INFERENCE_STATUS -eq 0 ]; then
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        pip install numba -q
        export PYTHONPATH='${PYTHONPATH}'
        python $PDM_DIR/src/verbatim_eval/main.py \
            --exprs $PDM_EXP_NAME \
            --mode sparse \
            --base-path $SCRATCH_MEM_BASE \
            --save-path $SCRATCH_MEM_BASE/SparseGutenberg \
            --offsets $OFFSET \
            --prefix-lengths $PREFIX_LENGTH \
            --suffix-lengths $SUFFIX_LENGTH"
    METRIC_STATUS=$?
else
    echo "Inference (step 1) failed with status $INFERENCE_STATUS -- skipping metric aggregation"
fi

### Copy scratch results to store ###
copy_scratch_to_store $METRIC_STATUS "$SCRATCH_MEM_BASE/SparseGutenberg" "$STORE_MEM_BASE/SparseGutenberg"

echo "END TIME: $(date)"
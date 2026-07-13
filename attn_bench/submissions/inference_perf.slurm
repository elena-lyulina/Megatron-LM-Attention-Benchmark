#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:45:00
#SBATCH --job-name=inference-perf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Raw prefill/decode timing profile for one trained model. Single GPU.
#
# Usage: sbatch --export=MODEL=gdn inference_perf.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh

### MODEL -> EXP_NAME / MEGATRON_EXTRA / NEEDS_TRITON / NEEDS_UNFUSED_DECODE ###
MODEL=${MODEL:-full}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_PERF_BASE=/users/$USER/store/inference-perf-results
SCRATCH_PERF_BASE=/iopsstor/scratch/cscs/$USER/inference-perf-results
SCRATCH_DIR=$SCRATCH_PERF_BASE/$EXP_NAME
STORE_DIR=$STORE_PERF_BASE/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
PY=$MEGATRON_LM_DIR/attn_bench/evaluation/inference_perf.py

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: model=$MODEL  exp=$EXP_NAME"

COMMON_ARGS="--container-env $CONTAINER_ENV \
    --ckpt-dir $MODEL_DIR/checkpoints \
    --tokenizer-path $TOKENIZER_PATH \
    --scratch-dir $SCRATCH_DIR \
    --store-dir $STORE_DIR \
    --model-tag $MODEL"

# NEEDS_UNFUSED_DECODE: TE 2.15 rejects fused attn in decode mode with a sink token.
GREEDY_MEGATRON_EXTRA="$MEGATRON_EXTRA"
EXTRA_ENV=""
if [ "$NEEDS_UNFUSED_DECODE" = "1" ]; then
    GREEDY_MEGATRON_EXTRA="$GREEDY_MEGATRON_EXTRA --attention-backend unfused"
    EXTRA_ENV="export NVTE_FUSED_ATTN=0; export CUDNN_LOGERR_DBG=1; export CUDNN_LOGDEST_DBG=stderr;"
fi
EXTRA_MEGATRON=""
[ -n "$GREEDY_MEGATRON_EXTRA" ] && EXTRA_MEGATRON="--megatron-extra-args $GREEDY_MEGATRON_EXTRA"

### LAUNCH ###
# GDN needs a per-job node-local TRITON_CACHE_DIR: fla kernels autotune on first use
# and a shared/colliding cache crashes with "stale file handle" (job 2570257).

if [ "$NEEDS_TRITON" = "1" ]; then
    srun \
        --cpus-per-task $SLURM_CPUS_PER_TASK \
        --environment=$CONTAINER_ENV \
        -lu \
        bash -c "
        pip install numba -q
        export PYTHONPATH='${PYTHONPATH}'
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}
        mkdir -p \"\$TRITON_CACHE_DIR\"
        torchrun --nproc_per_node=1 \
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
        torchrun --nproc_per_node=1 \
            $PY $COMMON_ARGS $EXTRA_MEGATRON"
fi
RUN_STATUS=$?

### Copy scratch results to store ###
copy_scratch_to_store $RUN_STATUS "$SCRATCH_DIR" "$STORE_DIR"

echo "END TIME: $(date)"
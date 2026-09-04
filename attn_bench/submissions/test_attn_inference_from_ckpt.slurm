#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=test-inference-from-ckpt
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Run the inference oracle suite (oracle_selfcheck + decode_matches_oracle) on a REAL checkpoint via
# MegatronBackend -- validates the cached incremental decode against the quadratic oracle on the
# actual trained weights before trusting any memorization number. Runs no memorization, writes
# nothing. Model registry (EXP_NAME + arch flags not restored by --use-checkpoint-args + the
# NEEDS_TRITON / NEEDS_FLA_052 gates) all come from attn_bench/scripts/llama_checkpoints.sh.
#
# Usage:
#   sbatch --export=ALL,MODEL=gdn      attn_bench/submissions/test_attn_inference_from_ckpt.slurm
#   sbatch --export=ALL,MODEL=kda      attn_bench/submissions/test_attn_inference_from_ckpt.slurm
#   sbatch --export=ALL,MODEL=qwen     attn_bench/submissions/test_attn_inference_from_ckpt.slurm
#   sbatch --export=ALL,MODEL=carry-r1 attn_bench/submissions/test_attn_inference_from_ckpt.slurm
#
# Check the log for the "### <label> inference from checkpoint — Summary ###" block (both PASS expected).

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

### MODEL -> EXP_NAME / MEGATRON_EXTRA / NEEDS_TRITON / NEEDS_FLA_052 ###
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
MODEL=${MODEL:-gdn}
model_config "$MODEL"

MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
CKPT_DIR=$MODEL_DIR/checkpoints

# fla 0.5.2 side-install (KDA only, NEEDS_FLA_052=1): the container ships 0.4.2, whose chunk_kda NaNs.
# Persisted, gitignored pip --target dir, prepended to PYTHONPATH so it shadows the container copy.
FLA_VERSION=0.5.2
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
[ "$NEEDS_FLA_052" = "1" ] && FLA_PREFIX="$FLA_DIR:" || FLA_PREFIX=""

export PYTHONPATH=${FLA_PREFIX}$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: model=$MODEL  exp=$EXP_NAME  ckpt=$CKPT_DIR"
echo "     megatron-extra: $MEGATRON_EXTRA"

if [ "$NEEDS_FLA_052" = "1" ]; then
    echo "[$(date)] Ensuring flash-linear-attention $FLA_VERSION in $FLA_DIR"
    srun --nodes=1 --ntasks=1 --overlap --environment=$CONTAINER_ENV bash -c "
        set -e
        if [ ! -f '$FLA_DIR/fla/__init__.py' ]; then
            pip install --quiet --no-deps --ignore-installed --target='$FLA_DIR' \
                'flash-linear-attention==$FLA_VERSION' 'fla-core==$FLA_VERSION'
        fi
        PYTHONPATH='$FLA_DIR' python3 -c \"import fla; assert fla.__version__.startswith('$FLA_VERSION'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')\"
    "
fi

# Single GPU -> single rank, so a per-job node-local TRITON_CACHE_DIR is enough (no cross-rank race).
# The GDN/KDA fla kernels autotune on first use; keep the cache off the shared FS (NEEDS_TRITON).
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}
    mkdir -p \$TRITON_CACHE_DIR
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun --nproc_per_node=1 \
        $MEGATRON_LM_DIR/attn_bench/tests/run_inference_from_ckpt.py \
        --model $MODEL \
        --ckpt-dir $CKPT_DIR \
        --tokenizer-path $TOKENIZER_PATH \
        --megatron-extra-args $MEGATRON_EXTRA"

echo "END TIME: $(date)"

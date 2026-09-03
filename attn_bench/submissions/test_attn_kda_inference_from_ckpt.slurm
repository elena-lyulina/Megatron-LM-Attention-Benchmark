#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=test-kda-inference-from-ckpt
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Run the kda_inference suite (oracle_selfcheck + decode_matches_oracle) on the REAL KDA checkpoint,
# to validate the cached incremental decode against the quadratic oracle on the actual trained
# weights before trusting any memorization number. Loads via the same MegatronBackend the measure
# run uses; runs no memorization, writes nothing.
#
# KDA needs flash-linear-attention >= 0.5.x; the container ships 0.4.2 (its chunk_kda NaNs).
# Side-installed 0.5.2 in $FLA_DIR (guarded one-shot below), same as the KDA pretrain slurm.
#
# Usage:
#   sbatch attn_bench/submissions/test_attn_kda_inference_from_ckpt.slurm
#
# Check the log for the "### KDA inference from checkpoint — Summary ###" block (both PASS expected).

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

EXP_NAME=llama3-1b-kda-scf1-fineweb40B-gutenberg3B
MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

# KDA needs flash-linear-attention >= 0.5.x; the container ships 0.4.2. Side-install 0.5.2 into a pip
# --target dir and put it first on PYTHONPATH so it shadows the container's copy. Persisted, gitignored.
FLA_VERSION=0.5.2
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla

# Mirrors the measure script's env (PDM on PYTHONPATH + numba), with $FLA_DIR prepended.
export PYTHONPATH=$FLA_DIR:$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: exp=$EXP_NAME  ckpt=$MODEL_DIR/checkpoints"

# KDA architecture flags re-passed via --megatron-extra-args (same as the measure script;
# --use-checkpoint-args restores the standard network args). Geometry from
# kda_1B_args_16heads_128_128_ffn6976.txt: pure KDA, 16/16 heads @ 128/128, conv kernel 4.
KDA_EXTRA_ARGS="--experimental-attention-variant kimi_delta_attention \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 16 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-conv-kernel-dim 4"

# One-shot: ensure flash-linear-attention $FLA_VERSION in $FLA_DIR (shadows the container's 0.4.2 via
# the PYTHONPATH prepend above). Guard skips it once installed. Both packages -- they merge into one
# fla/ namespace and flash-linear-attention alone ships no __init__.py. --no-deps keeps torch/triton
# out; --ignore-installed stops pip skipping on the container's 0.4.2.
echo "[$(date)] Ensuring flash-linear-attention $FLA_VERSION in $FLA_DIR"
srun --nodes=1 --ntasks=1 --overlap --environment=$CONTAINER_ENV bash -c "
    set -e
    if [ ! -f '$FLA_DIR/fla/__init__.py' ]; then
        pip install --quiet --no-deps --ignore-installed --target='$FLA_DIR' \
            'flash-linear-attention==$FLA_VERSION' 'fla-core==$FLA_VERSION'
    fi
    PYTHONPATH='$FLA_DIR' python3 -c \"import fla; assert fla.__version__.startswith('$FLA_VERSION'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')\"
"

# Single GPU -> single rank, so a per-job node-local TRITON_CACHE_DIR is enough (no cross-rank race).
# KDA's fla kernels autotune on first use; keep the cache off the shared FS.
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
        $MEGATRON_LM_DIR/attn_bench/tests/run_kda_inference_from_ckpt.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --megatron-extra-args $KDA_EXTRA_ARGS"

echo "END TIME: $(date)"

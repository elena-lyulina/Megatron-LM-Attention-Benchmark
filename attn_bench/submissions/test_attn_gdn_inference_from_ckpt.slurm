#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:10:00
#SBATCH --job-name=test-gdn-inference-from-ckpt
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Run the gdn_inference suite (oracle_selfcheck + decode_matches_oracle) on a REAL GDN checkpoint,
# to validate the cached incremental decode against the quadratic oracle on the actual trained
# weights before trusting any memorization number. Parametrized over the 4 GDN checkpoints via
# GDN_VARIANT (same mapping as the measure script). Loads via the same load_megatron_model the
# measure run uses; runs no memorization, writes nothing.
#
# Usage:
#   sbatch attn_bench/submissions/test_attn_gdn_inference_from_ckpt.slurm                          # base
#   sbatch --export=ALL,GDN_VARIANT=carry-r1 attn_bench/submissions/test_attn_gdn_inference_from_ckpt.slurm
#
# Check the log for the "### GDN inference from checkpoint — Summary ###" block (both PASS expected).

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

### VARIANT -> EXP_NAME ###
GDN_VARIANT=${GDN_VARIANT:-base}
case "$GDN_VARIANT" in
    base)       EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B ;;
    carry-r0)   EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B ;;
    carry-r0.5) EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B ;;
    carry-r1)   EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B ;;
    *) echo "Unknown GDN_VARIANT=$GDN_VARIANT (expected base|carry-r0|carry-r0.5|carry-r1)"; exit 1 ;;
esac

MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

# PDM on PYTHONPATH + numba: importing load_megatron_model pulls in megatron_inference_sparse, whose
# module-level imports include verbatim_eval.LCS (numba). Mirrors the measure script's env.
export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: variant=$GDN_VARIANT  exp=$EXP_NAME  ckpt=$MODEL_DIR/checkpoints"

# GDN architecture flags re-passed via --megatron-extra-args (same as the measure script;
# --use-checkpoint-args restores the standard network args). VERIFY ON FIRST RUN.
GDN_EXTRA_ARGS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"

# Single GPU -> single rank, so a per-job node-local TRITON_CACHE_DIR is enough (no cross-rank race
# like the 4-GPU measure run). GDN's fla kernels autotune on first use; keep the cache off the shared FS.
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
        $MEGATRON_LM_DIR/attn_bench/tests/run_gdn_inference_from_ckpt.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --megatron-extra-args $GDN_EXTRA_ARGS"

echo "END TIME: $(date)"
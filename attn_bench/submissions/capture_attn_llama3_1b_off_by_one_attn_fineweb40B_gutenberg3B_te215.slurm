#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:45:00
#SBATCH --job-name=capture-attn-off-by-one-attn-te215
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Capture decode-time attention patterns for the off-by-one-attention model.
# Writes attn_scores_rouge_l_{NN-MM}_rank{N}.npz and norm_attn_rouge_l_{NN-MM}_rank{N}.npz
# (one per Rouge-L bucket; plus gating_scores_rank{N}.npz for the gated model) at the
# run-level inference/offset_*_prefix_*_suffix_* dir, averaged over all repetition buckets.
# Capturing regenerates greedy generation for every rep (so rank*.jsonl is rewritten too).
#
# Usage:
#   sbatch attn_bench/submissions/capture_attn_llama3_1b_off_by_one_attn_fineweb40B_gutenberg3B_te215.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-50}

EXP_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215
MODEL_DIR=/users/elyulina/store/pretrain-results/llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B
MEM_BASE=/users/elyulina/store/mem-results
MEM_DIR=$MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: exp=$EXP_NAME  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=50"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    # TE 2.15 cuDNN fused attn rejects decode mode (s_q==1) with sink_token (off-by-one).
    # NVTE_FUSED_ATTN=0 disables FusedAttention; FlashAttention is already disabled for
    # non-vanilla softmax_type (transformer_engine/pytorch/attention/dot_product_attention/utils.py:886),
    # so TE falls back to UnfusedDotProductAttention which implements off-by-one in pure
    # PyTorch (appends sink column, softmax, strips it)
    # (transformer_engine/pytorch/attention/dot_product_attention/backends.py:624-651).
    export NVTE_FUSED_ATTN=0
    # cuDNN verbose error logging in case other cuDNN issues surface
    export CUDNN_LOGERR_DBG=1
    export CUDNN_LOGDEST_DBG=stderr
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun \
        --nproc_per_node=4 \
        $MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_sparse.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --offset $OFFSET \
        --prefix-length $PREFIX_LENGTH \
        --suffix-length 50 \
        --batch-size 20 \
        --capture-attention \
        --megatron-extra-args --softmax-type off-by-one --attention-backend unfused"
        # --attention-backend unfused: avoids language_module assertion that rejects NVTE_FUSED_ATTN=0 under AttnBackend.auto

echo "END TIME: $(date)"

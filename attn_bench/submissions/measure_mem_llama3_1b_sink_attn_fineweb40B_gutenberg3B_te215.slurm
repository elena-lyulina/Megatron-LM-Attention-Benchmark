#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:35:00
#SBATCH --job-name=measure-mem-sink-attn-te215
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Measure memorization for the sink-attention model:
#   - Run sparse Gutenberg inference directly from torch_dist checkpoint (no HF conversion)
#   - Compute LCS / Rouge-L memorization metrics
#
# Usage:
#   sbatch attn_bench/submissions/measure_mem_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm
#   sbatch --export=SINK_SCALE=2 attn_bench/submissions/measure_mem_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-500}
SUFFIX_LENGTH=${SUFFIX_LENGTH:-500}
SINK_SCALE=${SINK_SCALE:-}

EXP_NAME=llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215
[ -n "$SINK_SCALE" ] && PDM_EXP_NAME="${EXP_NAME}_sscale${SINK_SCALE}" || PDM_EXP_NAME="$EXP_NAME"
MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
MEM_BASE=/users/elyulina/store/mem-results
MEM_DIR=$MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: exp=$EXP_NAME  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=$SUFFIX_LENGTH${SINK_SCALE:+  sink_scale=$SINK_SCALE}"

### STEP 1: sparse Gutenberg inference ###
# Loads torch_dist checkpoint directly (DCP resharding merges TP=2 shards on the fly).
# Architecture flags (including attention variant) are loaded from the checkpoint itself.
# Saves rank*.jsonl to $MEM_DIR/inference/offset_0_prefix_500_suffix_500/rep_*/

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
        --container-env $CONTAINER_ENV \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --offset $OFFSET \
        --prefix-length $PREFIX_LENGTH \
        --suffix-length $SUFFIX_LENGTH \
        --batch-size 20 \
        ${SINK_SCALE:+--sink-scale $SINK_SCALE} \
        --megatron-extra-args --softmax-type learnable --attention-backend unfused"
        # --attention-backend unfused: avoids language_module assertion that rejects NVTE_FUSED_ATTN=0 under AttnBackend.auto


### STEP 2: compute memorization metrics ###
# Reads inference outputs and computes LCS / Rouge-L summary statistics.
# Saves results to $MEM_BASE/SparseGutenberg/$EXP_NAME/offset_0_prefix_500_suffix_500_greedy.pkl

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
        --base-path $MEM_BASE \
        --save-path $MEM_BASE/SparseGutenberg \
        --offsets $OFFSET \
        --prefix-lengths $PREFIX_LENGTH \
        --suffix-lengths $SUFFIX_LENGTH"

echo "END TIME: $(date)"

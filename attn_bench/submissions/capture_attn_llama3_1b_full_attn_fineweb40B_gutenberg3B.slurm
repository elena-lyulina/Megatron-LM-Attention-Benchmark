#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:35:00
#SBATCH --job-name=capture-attn-full-attn
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Capture decode-time attention patterns for the full-attention baseline.
# Writes attn_scores_rouge_l_{NN-MM}_rank{N}.npz and norm_attn_rouge_l_{NN-MM}_rank{N}.npz
# (one per Rouge-L bucket; plus gating_scores_rank{N}.npz for the gated model) at the
# run-level inference/offset_*_prefix_*_suffix_* dir, averaged over all repetition buckets.
# Capturing regenerates greedy generation for every rep (so rank*.jsonl is rewritten too).
#
# Usage:
#   sbatch attn_bench/submissions/capture_attn_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-50}

EXP_NAME=llama3-1b-full-attn-fineweb40B-gutenberg3B
MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
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
        --capture-attention"

echo "END TIME: $(date)"

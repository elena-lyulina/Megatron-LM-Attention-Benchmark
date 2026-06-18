#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:30:00
#SBATCH --job-name=measure-mem-gated-attn
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Measure memorization for the gated-attention model:
#   - Run sparse Gutenberg inference directly from torch_dist checkpoint (no HF conversion)
#   - Compute LCS / Rouge-L memorization metrics
#
# Usage:
#   sbatch attn_bench/submissions/measure_mem_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-500}

EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B
MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
MEM_BASE=/users/elyulina/store/mem-results
MEM_DIR=$MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/scratch/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: exp=$EXP_NAME  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=500"

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
        --suffix-length 500 \
        --batch-size 20 \
        --megatron-extra-args --attention-output-gate"


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
        --exprs $EXP_NAME \
        --mode sparse \
        --base-path $MEM_BASE \
        --save-path $MEM_BASE/SparseGutenberg \
        --offsets $OFFSET \
        --prefix-lengths $PREFIX_LENGTH \
        --suffix-lengths 500"

echo "END TIME: $(date)"

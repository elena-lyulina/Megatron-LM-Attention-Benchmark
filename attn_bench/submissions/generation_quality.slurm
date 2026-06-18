#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:30:00
#SBATCH --job-name=gen-quality
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Compute distinct-n and perplexity (Qwen2.5-1.5B) for any set of inference experiments.
# Auto-discovers (offset, prefix, suffix, rep, policy) from the inference directory structure.
# Skips experiments where output JSONs already exist.
# Output: $MEM_BASE/SparseGutenberg/<expr>/offset_X_prefix_Y_suffix_Z_policy_{distinct_n,perplexity}.json
#
# Usage:
#   sbatch --export=EXPRS="llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215" attn_bench/submissions/generation_quality.slurm
#   sbatch --export=EXPRS="exp1 exp2 exp3" attn_bench/submissions/generation_quality.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

MEM_BASE=/users/elyulina/store/mem-results
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

OFFSET=0
PREFIX_LENGTH=500
SUFFIX_LENGTH=500

export HF_TOKEN=$(cat $HOME/.hf-token)
export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}

echo "RUN: exprs=${EXPRS}"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export HF_HOME=/iopsstor/scratch/cscs/$USER/hf_cache
    export HF_TOKEN='${HF_TOKEN}'
    python $MEGATRON_LM_DIR/attn_bench/evaluation/compute_generation_quality.py \
        --exprs $EXPRS \
        --base-path $MEM_BASE/SparseGutenberg \
        --tokenizer-path $TOKENIZER_PATH \
        --offsets $OFFSET \
        --prefix-lengths $PREFIX_LENGTH \
        --suffix-lengths $SUFFIX_LENGTH"

echo "END TIME: $(date)"
#!/bin/bash

#SBATCH --account=your-account
#SBATCH --job-name=tokenize-fineweb-edu
#SBATCH --output=/iopsstor/scratch/cscs/elyulina/logs/tokenize-fineweb-edu-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/elyulina/logs/tokenize-fineweb-edu-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml
#SBATCH --partition=normal

echo "START TIME: $(date)"

#### Config ####
TEST=false   # Set to true for a quick throughput test (10M tokens, 1M/shard)
################

export HF_TOKEN=$(cat $HOME/.hf-token)

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b

mkdir -p /iopsstor/scratch/cscs/elyulina/logs

export PYTHONPATH=$MEGATRON_DIR:$PYTHONPATH

if [ "$TEST" = true ]; then
    OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-test
    TOKEN_BUDGET=10000000   # 10M tokens
    mkdir -p $OUTPUT_DIR
    python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu_simple.py \
        --tokenizer-path $TOKENIZER_PATH \
        --output-path $OUTPUT_DIR/fineweb_edu_text_document \
        --token-budget $TOKEN_BUDGET
else
    OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-160B
    TOKEN_BUDGET=160000000000   # 160B tokens
    SHARD_SIZE=10000000000      # 10B tokens/shard
    mkdir -p $OUTPUT_DIR
    python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu.py \
        --tokenizer-path $TOKENIZER_PATH \
        --output-dir $OUTPUT_DIR \
        --token-budget $TOKEN_BUDGET \
        --shard-size $SHARD_SIZE
fi

echo "END TIME: $(date)"
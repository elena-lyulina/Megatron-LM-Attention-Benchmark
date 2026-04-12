#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=tokenize-fineweb-edu
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20  # set to NUM_WORKERS below
#SBATCH --time=00:30:00
#SBATCH --environment=nemo
#SBATCH --partition=normal

echo "START TIME: $(date)"

#### Config ####
# Modes: "simple" (single process, speed test), "parallel" (multi-process), "full" (full 160B run)
MODE=parallel
################

export HF_TOKEN=$(cat $HOME/.hf-token)

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b

mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

export PYTHONPATH=$MEGATRON_DIR:$PYTHONPATH

if [ "$MODE" = "simple" ]; then
    OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-test
    TOKEN_BUDGET=10000000   # 10M tokens
    mkdir -p $OUTPUT_DIR
    python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu_simple.py \
        --tokenizer-path $TOKENIZER_PATH \
        --output-path $OUTPUT_DIR/fineweb_edu_text_document \
        --token-budget $TOKEN_BUDGET

elif [ "$MODE" = "parallel" ]; then
    OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-parallel-test
    TOKEN_BUDGET=100000000   # 100M tokens
    NUM_WORKERS=4
    mkdir -p $OUTPUT_DIR
    python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu_parallel.py \
        --tokenizer-path $TOKENIZER_PATH \
        --output-dir $OUTPUT_DIR \
        --token-budget $TOKEN_BUDGET \
        --num-workers $NUM_WORKERS

else  # full
    OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-160B
    TOKEN_BUDGET=160000000000   # 160B tokens
    NUM_WORKERS=20
    mkdir -p $OUTPUT_DIR
    python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu_parallel.py \
        --tokenizer-path $TOKENIZER_PATH \
        --output-dir $OUTPUT_DIR \
        --token-budget $TOKEN_BUDGET \
        --num-workers $NUM_WORKERS
fi

echo "END TIME: $(date)"
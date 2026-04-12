#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=verify-tokenization
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --environment=nemo
#SBATCH --partition=normal

echo "START TIME: $(date)"

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b
SHARD_PATH=/iopsstor/scratch/cscs/elyulina/dataset/dedup-fineweb-edu-parallel-test/shard_00000

mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

export PYTHONPATH=$MEGATRON_DIR:$PYTHONPATH

python3 $MEGATRON_DIR/attn_bench/data_processing/verify_tokenization.py \
    --shard-path $SHARD_PATH \
    --tokenizer-path $TOKENIZER_PATH \
    --num-sequences 5

echo "END TIME: $(date)"
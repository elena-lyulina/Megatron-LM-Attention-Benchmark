#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=prepare-gutenberg
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue

echo "START TIME: $(date)"

export HF_TOKEN=$(cat $HOME/.hf-token)

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
DATASETS_PATH=/iopsstor/scratch/cscs/elyulina/datasets
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b

mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

python3 $MEGATRON_DIR/attn_bench/data_processing/prepare_gutenberg.py \
    --raw-dir        $DATASETS_PATH/raw/gutenberg-en \
    --output-dir     $DATASETS_PATH/tokenized/gutenberg-en \
    --tokenizer-path $TOKENIZER_PATH

echo "END TIME: $(date)"
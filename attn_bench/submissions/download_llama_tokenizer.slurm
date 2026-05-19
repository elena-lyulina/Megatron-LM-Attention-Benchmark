#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=download-tokenizer
#SBATCH --output=/iopsstor/scratch/cscs/elyulina/logs/download-tokenizer-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/elyulina/logs/download-tokenizer-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue

echo "START TIME: $(date)"

export HF_TOKEN=$(cat $HOME/.hf-token)
OUTPUT_DIR=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b

mkdir -p $OUTPUT_DIR
mkdir -p /iopsstor/scratch/cscs/elyulina/logs

huggingface-cli download meta-llama/Llama-3.2-1B \
    --include "tokenizer*" "special_tokens_map.json" \
    --local-dir $OUTPUT_DIR \
    --token $HF_TOKEN

echo "END TIME: $(date)"
echo "Files downloaded to $OUTPUT_DIR:"
ls -lh $OUTPUT_DIR
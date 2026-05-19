#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=write-megatron-books
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Write repetition_buckets.jsonl to Megatron bin/idx format.
# Produces one rep_N_tokens.bin/.idx pair per repetition level.
# Tokens are written as-is (already BOS + content + EOS, 8192 tokens each).
# Verifies the output afterwards (lengths, BOS/EOS, repetition counts).
#
# Usage:
#   cd attn_bench && sbatch submissions/write_gutenberg_in_megatron_format.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
INPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/repetition_buckets.jsonl
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized/gutenberg_rep_1_256
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
STATS_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/stats/megatron-tokenization

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR

pip install --quiet datatrove


### WRITE MEGATRON FORMAT ###

python3 -u attn_bench/data_processing/books/write_megatron_books.py \
    --input      $INPUT \
    --output-dir $OUTPUT_DIR


### VERIFY ###

python3 -u attn_bench/data_processing/books/verify_gutenberg_tokenization.py \
    --tokenized-dir  $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --stats-dir      $STATS_DIR


echo "END TIME: $(date)"
echo "Done. Files in $OUTPUT_DIR"
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=download-gutenberg-laion
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Download laion/Project-Gutenberg and build two HF arrow datasets:
#   raw-hf-7000/   — first 7,000 books (for pipeline testing)
#   raw-hf-full/   — all books
#
# The tar.gz is downloaded once and reused for both.
# Both dataset builds are idempotent — skipped if already complete.
#
# Usage:
#   sbatch attn_bench/submissions/download_gutenberg_laion.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
BASE_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/gutenberg-laion

mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs
mkdir -p $BASE_DIR

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### DOWNLOAD ###

echo "[$(date)] Building raw-hf-7000..."
python3 -m attn_bench.data_processing.books.download_gutenberg_laion \
    --base-dir $BASE_DIR \
    --n-books 7000

echo "[$(date)] Building raw-hf-full..."
python3 -m attn_bench.data_processing.books.download_gutenberg_laion \
    --base-dir $BASE_DIR


echo "END TIME: $(date)"
echo "Done. Datasets in $BASE_DIR"
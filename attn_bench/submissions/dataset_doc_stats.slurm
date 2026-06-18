#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=dataset-doc-stats
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:10:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Compute per-document token-length distributions for the tokenized datasets.
# Reads only the .idx files (no .bin, no GPU) and saves one raw-lengths .npy per
# dataset for plotting in attn_bench/notebooks/doc_lengths.ipynb.
#
# Usage:
#   cd attn_bench && sbatch submissions/dataset_doc_stats.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
OUT_DIR=/users/$USER/store/datasets/analysis/doc-lengths

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### COMPUTE ###

python3 -u attn_bench/data_processing/dataset_doc_stats.py \
    --out-dir $OUT_DIR

echo "END TIME: $(date)"
echo "Done. Length arrays in $OUT_DIR"
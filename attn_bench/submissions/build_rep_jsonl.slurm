#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:5:00
#SBATCH --job-name=build-rep-jsonl
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Convert repetition_buckets.jsonl and unseen_buckets.jsonl into per-bucket
# JSONL files for PDM sparse memorization inference.
#
# Usage:
#   cd attn_bench && sbatch submissions/build_rep_jsonl.slurm

echo "START TIME: $(date)"

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PREPROCESSING_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
cd $MEGATRON_LM_DIR

python3 attn_bench/data_processing/books/build_rep_jsonl.py \
    --repetition-buckets $PREPROCESSING_DIR/repetition_buckets.jsonl \
    --unseen-buckets     $PREPROCESSING_DIR/unseen_buckets.jsonl \
    --output-dir         /users/elyulina/scratch/datasets/tokenized/gutenberg_rep_jsonl

echo "END TIME: $(date)"
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=filter-and-build-buckets
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:15:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Filter Gutenberg books (ppl p10/p90, max_hit == 0) and partition into
# repetition buckets with equalized perplexity. Writes filtered_buckets.jsonl
# and a 300-book stratified sample for manual inspection.
#
# Usage:
#   cd attn_bench && sbatch submissions/filter_and_build_buckets.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
INPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/sampled_containment.jsonl
OUTPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/repetition_buckets.jsonl
UNSEEN_OUTPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/unseen_buckets.jsonl
STATS_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/stats/repetition-buckets

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### FILTER AND BUILD BUCKETS ###

python3 -u attn_bench/data_processing/books/filter_and_build_buckets.py \
    --input          $INPUT \
    --output         $OUTPUT \
    --unseen-output  $UNSEEN_OUTPUT \
    --stats-dir      $STATS_DIR \
    --ppl-lo-q       10 \
    --ppl-hi-q       90 \
    --seed           42


echo "END TIME: $(date)"
echo "Done. Training output in $OUTPUT"
echo "Done. Unseen output in $UNSEEN_OUTPUT"
echo "Sample in $STATS_DIR/filtered_sample_300.txt"
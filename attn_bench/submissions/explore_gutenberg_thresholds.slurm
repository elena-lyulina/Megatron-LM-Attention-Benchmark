#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=explore-gutenberg-thresholds
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Plot threshold exploration heatmaps for Gutenberg post-contamination filtering.
# Reads sampled_containment.jsonl (must have perplexity, contamination_fraction,
# fineweb_max_ngram_hits). Produces threshold_exploration.png in STATS_DIR.
#
# Usage:
#   cd attn_bench && sbatch submissions/explore_gutenberg_thresholds.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
INPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/sampled_containment.jsonl
STATS_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output/stats/threshold_exploration

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### PLOT THRESHOLD HEATMAPS ###

python3 -u attn_bench/data_processing/books/explore_thresholds.py \
    --input      $INPUT \
    --output-dir $STATS_DIR


echo "END TIME: $(date)"
echo "Done. Plot in $STATS_DIR/threshold_exploration.png"
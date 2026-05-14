#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=check-fineweb-containment
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Check which Gutenberg excerpts (7k pilot) have ≥1 word-level 13-gram match
# in FineWeb-Edu 160B. 200 parquets × 96 workers, ~5 min expected.
#
# Usage:
#   cd attn_bench && sbatch submissions/check_fineweb_containment.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RAW_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/fineweb-edu-dedup
SAMPLED=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-7000-output/sampled.jsonl
OUTPUT=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-7000-output/sampled_containment.jsonl
STATS_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-7000-output/stats/fineweb_containment
PARQUETS_FILE=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/fineweb-edu-dedup/tokenized_parquets.txt

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### INSTALL DEPS ###

pip install --quiet xxhash


### SCAN PARQUETS FOR 13-GRAM MATCHES ###

python3 -u attn_bench/data_processing/books/check_fineweb_containment.py \
    --sampled       $SAMPLED \
    --parquets-file $PARQUETS_FILE \
    --raw-dir       $RAW_DIR \
    --output        $OUTPUT \
    --stats-dir     $STATS_DIR \
    --workers       $SLURM_CPUS_PER_TASK


### PLOT RESULTS ###

python3 -u attn_bench/data_processing/books/plot_fineweb_containment_thresholds.py \
    --input      $OUTPUT \
    --output-dir $STATS_DIR


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT"
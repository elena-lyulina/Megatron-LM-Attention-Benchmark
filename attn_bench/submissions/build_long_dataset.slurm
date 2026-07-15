#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-long-dataset
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Build a token-budget-capped dataset out of the longest documents in a tokenized
# Megatron dataset pool (build_long_dataset.py). Skips entirely if the result is
# already on store. Step 1 reads the source from scratch and writes the built
# dataset on scratch. Step 2 copies the result to store for long-term retention
# (iopsstor scratch purges after 30 days of no access) -- mirrors extract_long_docs.slurm.
#
# Usage:
#   cd attn_bench
#   data_folder=fineweb-edu-dedup-160B-datatrove budget=40000000000 sbatch submissions/build_long_dataset.slurm

set -eo pipefail

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

SCRATCH_TOKENIZED=/iopsstor/scratch/cscs/$USER/datasets/tokenized
STORE_TOKENIZED=/users/$USER/store/datasets/tokenized

### ARGS ###

DATA_FOLDER_NAME=${data_folder:-${1:-}}
if [ -z "$DATA_FOLDER_NAME" ]; then
    echo "ERROR: no data_folder given. Pass data_folder=<dataset dir name under $SCRATCH_TOKENIZED> or as a positional arg." >&2
    exit 1
fi
BUDGET=${budget:-}
if [ -z "$BUDGET" ]; then
    echo "ERROR: no budget given. Pass budget=<target total tokens, e.g. 40000000000>." >&2
    exit 1
fi

SRC_DIR=$SCRATCH_TOKENIZED/$DATA_FOLDER_NAME
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir not found: $SRC_DIR" >&2
    exit 1
fi

OUT_NAME=${DATA_FOLDER_NAME}_long_${BUDGET}
OUT_DIR=$SCRATCH_TOKENIZED/$OUT_NAME
STORE_DIR=$STORE_TOKENIZED/$OUT_NAME

echo "Source:      $SRC_DIR"
echo "Scratch out: $OUT_DIR"
echo "Store out:   $STORE_DIR"
echo "Budget:      $BUDGET tokens"

### SKIP IF ALREADY ON STORE ###

if [ -f "$STORE_DIR/run_metadata.json" ]; then
    echo "Already on store ($STORE_DIR/run_metadata.json exists) -- skipping."
    exit 0
fi

### STEP 1: build on scratch ###

python3 -u attn_bench/data_processing/build_long_dataset.py \
    --data-folder $SRC_DIR \
    --budget $BUDGET \
    --output-folder $OUT_DIR

### STEP 2: copy the result to store ###

echo "Copying to store: $STORE_DIR"
mkdir -p "$STORE_DIR"
cp -a "$OUT_DIR/." "$STORE_DIR/"

echo "END TIME: $(date)"
echo "Done. Long dataset in $STORE_DIR"
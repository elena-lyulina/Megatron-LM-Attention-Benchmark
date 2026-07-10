#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=extract-long-docs
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Extract long documents ([min_length, max_length) tokens) from a tokenized Megatron
# dataset. Step 1 reads the source from scratch and writes the extracted jsonl +
# stats on scratch. Step 2 copies the result to store for long-term retention
# (iopsstor scratch purges after 30 days of no access) -- mirrors Step 5 of
# verify_and_split_fineweb_edu_new_25.slurm.
#
# Usage:
#   cd attn_bench
#   data_folder=fineweb-edu-dedup-160B-datatrove_0.5_unseen sbatch submissions/extract_long_docs.slurm
#   data_folder=... min_length=24576 max_length=32768 sbatch submissions/extract_long_docs.slurm

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
MIN_LENGTH=${min_length:-24576}
MAX_LENGTH=${max_length:-32768}

SRC_DIR=$SCRATCH_TOKENIZED/$DATA_FOLDER_NAME
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir not found: $SRC_DIR" >&2
    exit 1
fi

OUT_DIR=${SRC_DIR}_long
STORE_DIR=$STORE_TOKENIZED/${DATA_FOLDER_NAME}_long

echo "Source:      $SRC_DIR"
echo "Scratch out: $OUT_DIR"
echo "Store out:   $STORE_DIR"
echo "Range:       [$MIN_LENGTH, $MAX_LENGTH)"

### STEP 1: extract on scratch ###

python3 -u attn_bench/data_processing/extract_long_docs.py \
    --data-folder $SRC_DIR \
    --min-length $MIN_LENGTH \
    --max-length $MAX_LENGTH \
    --output-dir $OUT_DIR

### STEP 2: copy the result to store ###

echo "Copying to store: $STORE_DIR"
mkdir -p "$STORE_DIR"
cp -a "$OUT_DIR/." "$STORE_DIR/"

echo "END TIME: $(date)"
echo "Done. Long docs in $STORE_DIR"

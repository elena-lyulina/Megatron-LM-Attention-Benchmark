#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=split-long-dataset
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Build the "short chunk" comparison arm end-to-end:
#   1. Split the long-doc dataset into fixed-length pieces (split_long_dataset.py), on scratch.
#   2. Verify every resulting document is BOS/EOS wrapped (verify_tokenization.py) -- aborts
#      the job (set -eo pipefail) before touching store if any check fails.
#   3. Copy the result to store, parallel single-file rsync (same pattern as
#      verify_and_split_fineweb_edu_new_25.slurm step 5) rather than one big whole-dir copy.
#   4. Submit dataset_doc_stats.slurm (raw doc-length distribution) as a follow-up job --
#      it already does its own scratch-then-store dance, no need to duplicate it here.
#   5. Submit packed_chunk_stats.slurm (actual packed chunk-length distribution, full
#      epoch, accounting for documents fragmented by seq-len training-window boundaries)
#      as a follow-up job, same reason.
#
# Usage:
#   cd attn_bench
#   data_folder=fineweb-edu-dedup-160B-datatrove_long_40B sbatch submissions/split_long_dataset.slurm
#   data_folder=... chunk_size=512 tail_merge_threshold=128 num_workers=32 sbatch submissions/split_long_dataset.slurm

set -eo pipefail

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

SCRATCH_TOKENIZED=/iopsstor/scratch/cscs/$USER/datasets/tokenized
STORE_TOKENIZED=/users/$USER/store/datasets/tokenized

### ARGS ###

DATA_FOLDER_NAME=${data_folder:-${1:-}}
if [ -z "$DATA_FOLDER_NAME" ]; then
    echo "ERROR: no data_folder given. Pass data_folder=<dataset dir name under $SCRATCH_TOKENIZED> or as a positional arg." >&2
    exit 1
fi
CHUNK_SIZE=${chunk_size:-1024}
TAIL_MERGE_THRESHOLD=${tail_merge_threshold:-256}
NUM_WORKERS=${num_workers:-20}

SRC_DIR=$SCRATCH_TOKENIZED/$DATA_FOLDER_NAME
if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: source dir not found: $SRC_DIR" >&2
    exit 1
fi

TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b  # Assumes the Llama 3.2 tokenizer is downloaded to this location

OUT_NAME=${DATA_FOLDER_NAME}_split${CHUNK_SIZE}
OUT_DIR=$SCRATCH_TOKENIZED/$OUT_NAME
STORE_DIR=$STORE_TOKENIZED/$OUT_NAME

echo "Source:      $SRC_DIR"
echo "Scratch out: $OUT_DIR"
echo "Store out:   $STORE_DIR"
echo "chunk_size=$CHUNK_SIZE  tail_merge_threshold=$TAIL_MERGE_THRESHOLD  num_workers=$NUM_WORKERS"

### SKIP IF ALREADY ON STORE ###

if [ -f "$STORE_DIR/run_metadata.json" ]; then
    echo "Already on store ($STORE_DIR/run_metadata.json exists) -- skipping."
    exit 0
fi

### STEP 1: split on scratch ###

python3 -u attn_bench/data_processing/split_long_dataset.py \
    --data-folder $SRC_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --chunk-size $CHUNK_SIZE \
    --tail-merge-threshold $TAIL_MERGE_THRESHOLD \
    --num-workers $NUM_WORKERS \
    --output-folder $OUT_DIR

### STEP 2: verify every resulting document is BOS/EOS wrapped (fails the job if not) ###

python3 -u attn_bench/data_processing/tokenization/verify_tokenization.py \
    --tokenized-dir $OUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --num-workers $NUM_WORKERS

### STEP 3: copy to store, parallel single-file rsync (-P $NUM_WORKERS) ###

echo "Copying to store: $STORE_DIR"
MANIFEST=$(mktemp)
( cd "$OUT_DIR" && find . -type f \( -name "*.idx" -o -name "*.bin" \) ) | while IFS= read -r rel; do
    echo "$OUT_DIR/$rel" "$STORE_DIR/$rel"
done > "$MANIFEST"
echo "Files to copy: $(wc -l < "$MANIFEST")"

xargs -a "$MANIFEST" -n 2 -P $NUM_WORKERS bash -c '
    mkdir -p "$(dirname "$2")"
    rsync -a "$1" "$2"
' _
rm -f "$MANIFEST"

mkdir -p "$STORE_DIR"
cp "$OUT_DIR/run_metadata.json" "$STORE_DIR/run_metadata.json"

### STEP 4 + 5: submit doc-length and packed chunk-length stats as follow-up jobs ###
# Both scripts already handle scratch-then-store themselves -- point them at $OUT_DIR
# (scratch, still present and fast) rather than duplicating that logic here.

DOC_STATS_JOB=$(sbatch --parsable --export=ALL,DATASET_DIRS="$OUT_DIR" submissions/dataset_doc_stats.slurm)
echo "Submitted dataset_doc_stats.slurm: job $DOC_STATS_JOB"

CHUNK_STATS_JOB=$(sbatch --parsable --export=ALL,DATASET_DIRS="$OUT_DIR" submissions/packed_chunk_stats.slurm)
echo "Submitted packed_chunk_stats.slurm: job $CHUNK_STATS_JOB"

echo "END TIME: $(date)"
echo "Done. Split dataset in $STORE_DIR"
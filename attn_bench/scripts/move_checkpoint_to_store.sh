#!/usr/bin/env bash
# Move a finished pretraining experiment folder (checkpoints, wandb/tensorboard logging,
# debug/, triggers/) from scratch to permanent store as one unit, plus the slurm .out/.err
# for the given job IDs, and verify the copy. Run once a pretrain_llama3_1b_*.slurm job has
# completed -- see attn_bench/results/docs/adding_new_model_workflow.md for the full
# lifecycle. Meant to run on a login node (plain rsync between two mounted filesystems).
#
# Does NOT delete the scratch copy -- remove it yourself once confirmed.
#
# Usage: bash attn_bench/scripts/move_checkpoint_to_store.sh <EXP_NAME> [--project <PROJECT_NAME>] [JOB_ID ...]
#   e.g. bash attn_bench/scripts/move_checkpoint_to_store.sh llama3-1b-full-attn-fineweb40B-gutenberg3B 2327225

set -euo pipefail

usage() { echo "Usage: $0 <EXP_NAME> [--project <PROJECT_NAME>] [JOB_ID ...]"; exit 1; }

[[ $# -ge 1 ]] || usage
EXP_NAME="$1"; shift

PROJECT_NAME="fineweb-40B_gutenberg-3B"   # matches PROJECT_NAME in the pretrain_*.slurm scripts
JOB_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --project) PROJECT_NAME="$2"; shift 2 ;;
        *) JOB_IDS+=("$1"); shift ;;
    esac
done

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
SRC_EXP_DIR=$MEGATRON_LM_DIR/attn_bench/results/pretrain/$PROJECT_NAME/$EXP_NAME
DST_EXP_DIR=/users/$USER/store/pretrain-results/$EXP_NAME

[[ -d "$SRC_EXP_DIR" ]] || { echo "ERROR: experiment dir not found: $SRC_EXP_DIR"; exit 1; }

echo "[$(date)] Moving experiment: $SRC_EXP_DIR -> $DST_EXP_DIR"
mkdir -p "$DST_EXP_DIR"
rsync -a --info=progress2 "$SRC_EXP_DIR/" "$DST_EXP_DIR/"

if [[ ${#JOB_IDS[@]} -gt 0 ]]; then
    mkdir -p "$DST_EXP_DIR/slurm-logs"
    for JOB_ID in "${JOB_IDS[@]}"; do
        for EXT in out err; do
            LOG_FILE=$MEGATRON_LM_DIR/attn_bench/logs/$JOB_ID.$EXT
            if [[ -f "$LOG_FILE" ]]; then
                cp "$LOG_FILE" "$DST_EXP_DIR/slurm-logs/"
            else
                echo "WARNING: slurm log not found: $LOG_FILE"
            fi
        done
    done
fi

### VERIFY ###
echo "[$(date)] Verifying copy..."
SRC_SIZE=$(du -sh "$SRC_EXP_DIR" | cut -f1)
DST_SIZE=$(du -sh "$DST_EXP_DIR" | cut -f1)
SRC_FILES=$(find -L "$SRC_EXP_DIR" -type f | wc -l)
DST_FILES=$(find -L "$DST_EXP_DIR" -type f -not -path "*/slurm-logs/*" | wc -l)

echo "  source:      $SRC_EXP_DIR  ($SRC_SIZE, $SRC_FILES files)"
echo "  destination: $DST_EXP_DIR  ($DST_SIZE, $DST_FILES files, excluding slurm-logs)"
if [[ "$SRC_FILES" -eq "$DST_FILES" ]]; then
    echo "  file count OK ($SRC_FILES files)"
else
    echo "  MISMATCH: file counts differ ($SRC_FILES vs $DST_FILES) -- do not delete the scratch copy"
    exit 1
fi

echo "[$(date)] Done. Experiment safely in store: $DST_EXP_DIR"
echo "Scratch copy NOT deleted -- remove $SRC_EXP_DIR manually once confirmed."

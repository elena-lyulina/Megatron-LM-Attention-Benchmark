#!/bin/bash
# Submit Phase A backfill jobs for a chosen subset of models in
# attn_bench/scripts/llama_checkpoints.sh (default: every model in the registry).
# Each job checks its own store results before loading the checkpoint and no-ops if
# already fully backfilled -- safe to resubmit.
#
# Usage: bash attn_bench/submissions/prefix_extraction_inference_backfill_all.sh --models full-scf8 gated-scf8
# Add --dry-run to print the sbatch commands that would run without submitting anything.
# Omit --models to backfill every model in the registry.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

DRY_RUN=0
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1; shift
            ;;
        --models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SELECTED_MODELS+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--models m1 m2 ...]"
            exit 1
            ;;
    esac
done

if [[ ${#SELECTED_MODELS[@]} -eq 0 ]]; then
    SELECTED_MODELS=("${MODELS[@]}")
fi

for MODEL in "${SELECTED_MODELS[@]}"; do
    model_config "$MODEL"  # validates the tag, unused result here
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch --export=ALL,MODEL=$MODEL $SCRIPT_DIR/prefix_extraction_inference_backfill.slurm"
        continue
    fi
    echo "Submitting prefix_extraction_inference_backfill.slurm (model=$MODEL exp=$EXP_NAME)"
    sbatch --export=ALL,MODEL="$MODEL" "$SCRIPT_DIR/prefix_extraction_inference_backfill.slurm"
done

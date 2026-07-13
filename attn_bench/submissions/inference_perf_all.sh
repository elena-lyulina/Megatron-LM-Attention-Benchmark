#!/bin/bash
# Submit inference_perf.slurm for each of the 4 distinct attention architectures
# (models sharing MEGATRON_EXTRA in llama_checkpoints.sh have identical timing,
# so only one checkpoint per architecture needs profiling).
#
# sink gets a longer time limit -- its unfused decode backend is slower per step
# and the base script's 45min default is sized for the fused-attention models.
#
# Usage: bash attn_bench/submissions/inference_perf_all.sh
# Add --force to resubmit even if store already has all files. Add --dry-run to
# print sbatch commands without submitting.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

PERF_MODELS=(full gated sink gdn)
STORE_PERF_BASE=/users/$USER/store/inference-perf-results
SINK_TIME_LIMIT="2:00:00"

# Files inference_perf.py writes per model once done.
UNITS=(prefill_bs20 bs20_prefix50 bs20_prefix5000 bs1_prefix50 bs5_prefix50 bs10_prefix50 bs40_prefix50)

FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force] [--dry-run]"; exit 1 ;;
    esac
done

SKIPPED_COUNT=0
SUBMITTED_COUNT=0

for MODEL in "${PERF_MODELS[@]}"; do
    model_config "$MODEL"
    STORE_DIR=$STORE_PERF_BASE/$EXP_NAME

    ALL_DONE=1
    for UNIT in "${UNITS[@]}"; do
        [[ -f "$STORE_DIR/$UNIT.json" ]] || ALL_DONE=0
    done
    if [[ $FORCE -eq 0 && $ALL_DONE -eq 1 ]]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    EXTRA_SBATCH=()
    [[ "$IS_SINK_FAMILY" == "1" ]] && EXTRA_SBATCH=(--time="$SINK_TIME_LIMIT")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch ${EXTRA_SBATCH[*]} --export=ALL,MODEL=$MODEL $SCRIPT_DIR/inference_perf.slurm"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        continue
    fi

    echo "Submitting inference_perf.slurm (model=$MODEL exp=$EXP_NAME)${EXTRA_SBATCH:+ time=$SINK_TIME_LIMIT}"
    sbatch "${EXTRA_SBATCH[@]}" --export=ALL,MODEL="$MODEL" "$SCRIPT_DIR/inference_perf.slurm"
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT"
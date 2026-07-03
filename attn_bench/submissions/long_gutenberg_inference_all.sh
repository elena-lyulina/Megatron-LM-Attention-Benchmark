#!/bin/bash
# Submit long-Gutenberg position-loss inference for every model. One job per VARIANT,
# each self-parallel across 4 GPUs. A model is skipped when all requested buckets
# (rep_{R}.npz) already exist on store; --force submits regardless.
#
# Env passthrough (optional): REPETITIONS, MAX_LENGTH, MAX_SAMPLES.
#
# Usage (from attn_bench/):
#   bash attn_bench/submissions/long_gutenberg_inference_all.sh                 # full sweep, all 8 models
#   REPETITIONS=0 MAX_SAMPLES=20 bash attn_bench/submissions/long_gutenberg_inference_all.sh --force   # calibration
#   MAX_LENGTH=16384 bash attn_bench/submissions/long_gutenberg_inference_all.sh # cap sequences at 2L

set -e

SCRIPT_DIR=$(dirname "$0")
RESULTS_BASE=/users/elyulina/store/long-gutenberg-results
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}

# "VARIANT|EXP_NAME". EXP_NAME is explicit (it drives the skip-check) and must match the
# case in long_gutenberg_inference.slurm.
JOBS=(
    "full|llama3-1b-full-attn-fineweb40B-gutenberg3B"
    "gated|llama3-1b-gated-attn-fineweb40B-gutenberg3B"
    "sink|llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215"
    "off-by-one|llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215"
    "gdn|llama3-1b-gdn-fineweb40B-gutenberg3B"
    "carry-r0|llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B"
    "carry-r0.5|llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B"
    "carry-r1|llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B"
)

FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force]  (set REPETITIONS/MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

IFS=',' read -ra REPS <<< "$REPETITIONS"

# Config folder (must match config_name() in long_gutenberg_inference.py): unset caps
# become "all"/"full".
CONFIG="${MAX_SAMPLES:-all}_samples_${MAX_LENGTH:-full}_tokens"

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r VARIANT EXP_NAME <<< "$JOB"

    # "Done" = every requested bucket's rep_{R}.npz is present for this config.
    DONE=1
    for R in "${REPS[@]}"; do
        if [[ ! -f "$RESULTS_BASE/$EXP_NAME/$CONFIG/rep_${R}.npz" ]]; then
            DONE=0; break
        fi
    done
    if [[ $FORCE -eq 0 && $DONE -eq 1 ]]; then
        echo "Skipping $EXP_NAME (all buckets present)"
        continue
    fi

    EXPORTS="VARIANT=$VARIANT,REPETITIONS=$REPETITIONS"
    [[ -n "${MAX_LENGTH:-}" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$MAX_LENGTH"
    [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
    # --force also recomputes: without this the resubmitted job would just skip and no-op.
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    echo "Submitting VARIANT=$VARIANT ($EXP_NAME) reps=$REPETITIONS"
    # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
    sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_gutenberg_inference.slurm"
done

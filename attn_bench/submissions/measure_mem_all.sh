#!/bin/bash
# Submit memorization measurement jobs for every model in attn_bench/scripts/llama_checkpoints.sh,
# for every combination of the given offsets and prefix lengths.
#
# Combinations whose final Stage-2 result already exists on store are skipped.
# Even a redundant job still grabs a 4-GPU node and starts the container, so we
# also guard here at submit time. --force submits regardless (and sets OVERWRITE=1 on the
# job itself, so it doesn't just re-check and skip on arrival).
#
# To add a newly trained model to this sweep: add it to attn_bench/scripts/llama_checkpoints.sh,
# not here.
#
# Usage: bash attn_bench/submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 1000 1500 2000 3000 4000 5000
# Add --dry-run to print the sbatch commands that would run without submitting anything.
# Add --models m1,m2 to restrict to a subset (default: every model in the registry).
# Add --force-metrics to recompute every reachable boundary's pkl in Step 2 even where one
# already exists (e.g. a pkl that predates a backfill) -- independent of --force, which
# controls whether Step 1 (generation) reruns at all.
# Add --time HH:MM:SS to override measure_mem.slurm's default time limit for this sweep
# (e.g. a fresh, never-run-before suffix needs more than the default covers).

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

# "Done" marker: the Stage-2 pkl. Its presence means both inference and metric
# aggregation finished.
MEM_BASE=/users/$USER/store/mem-results

OFFSETS=()
PREFIXES=()
SUFFIXES=()
FORCE=0
FORCE_METRICS=0
DRY_RUN=0
JOB_TIME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1; shift
            ;;
        --force-metrics)
            FORCE_METRICS=1; shift
            ;;
        --dry-run)
            DRY_RUN=1; shift
            ;;
        --time)
            JOB_TIME="$2"; shift 2
            ;;
        --models)
            IFS=',' read -r -a MODELS <<< "$2"; shift 2
            ;;
        --offsets)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                OFFSETS+=("$1"); shift
            done
            ;;
        --prefixes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                PREFIXES+=("$1"); shift
            done
            ;;
        --suffixes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SUFFIXES+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--force] [--force-metrics] [--dry-run] [--models m1,m2] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
            exit 1
            ;;
    esac
done

if [[ ${#OFFSETS[@]} -eq 0 || ${#PREFIXES[@]} -eq 0 ]]; then
    echo "Usage: $0 [--force] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
    exit 1
fi

# suffix defaults to 500 when --suffixes is omitted, matching the old behaviour.
if [[ ${#SUFFIXES[@]} -eq 0 ]]; then
    SUFFIXES=(500)
fi

SKIPPED_COUNT=0
SUBMITTED_COUNT=0

for OFFSET in "${OFFSETS[@]}"; do
    for PREFIX in "${PREFIXES[@]}"; do
        for SUFFIX in "${SUFFIXES[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                model_config "$MODEL"

                PKL=$MEM_BASE/SparseGutenberg/$EXP_NAME/metrics/offset_${OFFSET}_prefix_${PREFIX}_suffix_${SUFFIX}_greedy.pkl
                if [[ $FORCE -eq 0 && $FORCE_METRICS -eq 0 && -f "$PKL" ]]; then
                    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                    continue
                fi

                EXPORTS="MODEL=$MODEL,OFFSET=$OFFSET,PREFIX_LENGTH=$PREFIX,SUFFIX_LENGTH=$SUFFIX"
                [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"
                [[ $FORCE_METRICS -eq 1 ]] && EXPORTS="$EXPORTS,FORCE_METRICS=1"
                TIME_ARG=()
                [[ -n "$JOB_TIME" ]] && TIME_ARG=(--time="$JOB_TIME")

                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[dry-run] sbatch ${TIME_ARG[*]} --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/measure_mem.slurm"
                    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
                    continue
                fi

                echo "Submitting measure_mem.slurm (model=$MODEL exp=$EXP_NAME) offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX"
                # ALL = propagate the full submission env (USER, PATH, …) so the scripts'
                # $USER-based paths resolve, then layer our per-job vars on top.
                sbatch "${TIME_ARG[@]}" --export=ALL,"$EXPORTS" "$SCRIPT_DIR/measure_mem.slurm"
                SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
            done
        done
    done
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT"
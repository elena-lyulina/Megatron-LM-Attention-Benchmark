#!/bin/bash
# Submit memorization measurement jobs for every model in attn_bench/scripts/llama_checkpoints.sh,
# for every combination of the given offsets and prefix lengths.
#
# Combinations whose final Stage-2 result already exists on store are skipped.
# Even a redundant job still grabs a 4-GPU node and starts the container, so we
# also guard here at submit time. --force submits regardless.
#
# To add a newly trained model to this sweep: add it to attn_bench/scripts/llama_checkpoints.sh,
# not here.
#
# Usage: bash attn_bench/submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 1000 1500 2000 3000 4000 5000
# Add --dry-run to print the sbatch commands that would run without submitting anything.

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
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1; shift
            ;;
        --dry-run)
            DRY_RUN=1; shift
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
            echo "Usage: $0 [--force] [--dry-run] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
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

for OFFSET in "${OFFSETS[@]}"; do
    for PREFIX in "${PREFIXES[@]}"; do
        for SUFFIX in "${SUFFIXES[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                model_config "$MODEL"

                PKL=$MEM_BASE/SparseGutenberg/$EXP_NAME/offset_${OFFSET}_prefix_${PREFIX}_suffix_${SUFFIX}_greedy.pkl
                if [[ $FORCE -eq 0 && -f "$PKL" ]]; then
                    echo "Skipping $EXP_NAME offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX (exists: $PKL)"
                    continue
                fi

                EXPORTS="MODEL=$MODEL,OFFSET=$OFFSET,PREFIX_LENGTH=$PREFIX,SUFFIX_LENGTH=$SUFFIX"
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[dry-run] sbatch --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/measure_mem.slurm"
                    continue
                fi

                echo "Submitting measure_mem.slurm (model=$MODEL exp=$EXP_NAME) offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX"
                # ALL = propagate the full submission env (USER, PATH, …) so the scripts'
                # $USER-based paths resolve, then layer our per-job vars on top.
                sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/measure_mem.slurm"
            done
        done
    done
done
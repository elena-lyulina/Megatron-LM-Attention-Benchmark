#!/bin/bash
# Submit memorization measurement jobs for all 4 attention variants
# for every combination of the given offsets and prefix lengths.
#
# Usage (from attn_bench/):
#   bash attn_bench/submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 1000 1500 2000 3000 4000 5000
#   bash attn_bench/submissions/measure_mem_all.sh --offsets 0 500 1000 --prefixes 500

set -e

SCRIPT_DIR=$(dirname "$0")

SCRIPTS=(
    measure_mem_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm
    measure_mem_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm
    measure_mem_llama3_1b_off_by_one_attn_fineweb40B_gutenberg3B_te215.slurm
    measure_mem_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm
)

OFFSETS=()
PREFIXES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...]"
            exit 1
            ;;
    esac
done

if [[ ${#OFFSETS[@]} -eq 0 || ${#PREFIXES[@]} -eq 0 ]]; then
    echo "Usage: $0 --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...]"
    exit 1
fi

for OFFSET in "${OFFSETS[@]}"; do
    for PREFIX in "${PREFIXES[@]}"; do
        for SCRIPT in "${SCRIPTS[@]}"; do
            echo "Submitting $SCRIPT with offset=$OFFSET prefix_length=$PREFIX"
            sbatch --export=OFFSET="$OFFSET",PREFIX_LENGTH="$PREFIX" "$SCRIPT_DIR/$SCRIPT"
        done
    done
done
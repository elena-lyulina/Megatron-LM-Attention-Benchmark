#!/bin/bash
# Submit memorization measurement jobs for all variants in JOBS for every
# combination of the given offsets and prefix lengths.
#
# Combinations whose final Stage-2 result already exists on store are skipped.
# Even a redundant job still grabs a 4-GPU node and starts the container before
# the in-job check no-ops (megatron_inference_sparse.py skips the checkpoint
# load when results exist), so we also guard here at submit time. --force submits
# regardless.
#
# Each JOBS entry is "SCRIPT|EXP_NAME|EXTRA_EXPORTS":
#   SCRIPT         the measure_mem_*.slurm file.
#   EXP_NAME       drives the skip-check (the Stage-2 pkl path). Kept explicit
#                  because parametrized scripts (GDN) compute EXP_NAME at runtime.
#   EXTRA_EXPORTS  extra comma-separated KEY=VAL forwarded to sbatch --export, or
#                  empty. The 4 GDN entries reuse one parametrized script via
#                  GDN_VARIANT.
#
# Usage (from attn_bench/):
#   bash attn_bench/submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 1000 1500 2000 3000 4000 5000
#   bash attn_bench/submissions/measure_mem_all.sh --offsets 0 500 1000 --prefixes 500
#   bash attn_bench/submissions/measure_mem_all.sh --force --offsets 0 --prefixes 500

set -e

SCRIPT_DIR=$(dirname "$0")

# "Done" marker: the Stage-2 pkl. Its presence means both inference and metric
# aggregation finished.
MEM_BASE=/users/elyulina/store/mem-results

JOBS=(
    "measure_mem_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm|llama3-1b-full-attn-fineweb40B-gutenberg3B|"
    "measure_mem_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm|llama3-1b-gated-attn-fineweb40B-gutenberg3B|"
    "measure_mem_llama3_1b_off_by_one_attn_fineweb40B_gutenberg3B_te215.slurm|llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215|"
    "measure_mem_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm|llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215|"
    "measure_mem_llama3_1b_full_attn_xdoc_attn_leak_fineweb40B_gutenberg3B.slurm|llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B|"
    "measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm|llama3-1b-gdn-fineweb40B-gutenberg3B|GDN_VARIANT=base"
    "measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm|llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B|GDN_VARIANT=carry-r0"
    "measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm|llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B|GDN_VARIANT=carry-r0.5"
    "measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm|llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B|GDN_VARIANT=carry-r1"
)

OFFSETS=()
PREFIXES=()
SUFFIXES=()
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1; shift
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
            echo "Usage: $0 [--force] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
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
            for JOB in "${JOBS[@]}"; do
                IFS='|' read -r SCRIPT EXP_NAME EXTRA <<< "$JOB"

                PKL=$MEM_BASE/SparseGutenberg/$EXP_NAME/offset_${OFFSET}_prefix_${PREFIX}_suffix_${SUFFIX}_greedy.pkl
                if [[ $FORCE -eq 0 && -f "$PKL" ]]; then
                    echo "Skipping $EXP_NAME offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX (exists: $PKL)"
                    continue
                fi

                EXPORTS="OFFSET=$OFFSET,PREFIX_LENGTH=$PREFIX,SUFFIX_LENGTH=$SUFFIX"
                [[ -n "$EXTRA" ]] && EXPORTS="$EXPORTS,$EXTRA"

                echo "Submitting $SCRIPT ($EXP_NAME) offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX${EXTRA:+  [$EXTRA]}"
                # ALL = propagate the full submission env (USER, PATH, …) so the scripts'
                # $USER-based paths resolve, then layer our per-job vars on top.
                sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/$SCRIPT"
            done
        done
    done
done
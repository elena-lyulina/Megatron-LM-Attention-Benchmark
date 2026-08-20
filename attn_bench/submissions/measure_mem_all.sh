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
# Diagonal mode (disentangles anchor length from cold-start depth): --diagonal-starts <s1> [s2 ...]
# instead of --offsets, still with --prefixes -- offset is derived per pair as start - prefix
# (skipped if negative), so offset+prefix stays fixed at each start while only the anchor
# length varies. Mutually exclusive with --offsets.
# Add --dry-run to print the sbatch commands that would run without submitting anything.
# Add --models m1,m2 to restrict to a subset (default: every model in the registry).
# Add --force-metrics to recompute every reachable boundary's pkl in Step 2 even where one
# already exists (e.g. a pkl that predates a backfill) -- independent of --force, which
# controls whether Step 1 (generation) reruns at all.
# Add --time HH:MM:SS to override measure_mem.slurm's default time limit for this sweep
# (e.g. a fresh, never-run-before suffix needs more than the default covers).
# Add --backend hf to use each model's already-converted HF checkpoint instead of the
# default megatron decode loop (run convert_and_validate_hf.slurm per model first -- this
# script does not check or convert; measure_mem.slurm itself fails fast per-job if a model's
# HF checkpoint is missing).
# Add --repetitions r1,r2,... to restrict which repetition buckets each job generates (default:
# measure_mem.slurm's own default, 0,1,2,4,8,16,32,64,128,256) -- e.g. for a dense offset x
# prefix grid where only the higher reps show any structure, --repetitions 32,64,128,256 cuts
# generation cost roughly proportionally to how many reps are dropped.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

# "Done" marker: the Stage-2 pkl. Its presence means both inference and metric
# aggregation finished.
MEM_BASE=/users/$USER/store/mem-results

OFFSETS=()
PREFIXES=()
SUFFIXES=()
DIAGONAL_STARTS=()
FORCE=0
FORCE_METRICS=0
DRY_RUN=0
JOB_TIME=""
BACKEND="megatron"
REPETITIONS=""

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
        --backend)
            BACKEND="$2"; shift 2
            if [[ "$BACKEND" != "megatron" && "$BACKEND" != "hf" ]]; then
                echo "--backend must be 'megatron' or 'hf', got '$BACKEND'."
                exit 1
            fi
            ;;
        --models)
            IFS=',' read -r -a MODELS <<< "$2"; shift 2
            ;;
        --repetitions)
            REPETITIONS="$2"; shift 2
            ;;
        --offsets)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                OFFSETS+=("$1"); shift
            done
            ;;
        --diagonal-starts)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DIAGONAL_STARTS+=("$1"); shift
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

# Build the (offset, prefix) pairs to sweep -- either the full cartesian product of
# --offsets x --prefixes, or a fixed-suffix_start diagonal (--diagonal-starts x --prefixes,
# offset = start - prefix) so offset+prefix stays fixed while only the anchor length varies.
PAIRS=()
if [[ ${#DIAGONAL_STARTS[@]} -gt 0 ]]; then
    if [[ ${#OFFSETS[@]} -gt 0 ]]; then
        echo "--diagonal-starts and --offsets are mutually exclusive (offset is derived as start - prefix in diagonal mode)."
        exit 1
    fi
    if [[ ${#PREFIXES[@]} -eq 0 ]]; then
        echo "Usage: $0 --diagonal-starts <s1> [s2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
        exit 1
    fi
    for START in "${DIAGONAL_STARTS[@]}"; do
        for PREFIX in "${PREFIXES[@]}"; do
            OFFSET=$((START - PREFIX))
            if [[ $OFFSET -lt 0 ]]; then
                echo "Skipping diagonal point start=$START prefix=$PREFIX: offset would be negative ($OFFSET)."
                continue
            fi
            PAIRS+=("$OFFSET:$PREFIX")
        done
    done
else
    if [[ ${#OFFSETS[@]} -eq 0 || ${#PREFIXES[@]} -eq 0 ]]; then
        echo "Usage: $0 [--force] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffixes <s1> [s2 ...]]"
        exit 1
    fi
    for OFFSET in "${OFFSETS[@]}"; do
        for PREFIX in "${PREFIXES[@]}"; do
            PAIRS+=("$OFFSET:$PREFIX")
        done
    done
fi

# suffix defaults to 500 when --suffixes is omitted, matching the old behaviour.
if [[ ${#SUFFIXES[@]} -eq 0 ]]; then
    SUFFIXES=(500)
fi

SKIPPED_COUNT=0
SUBMITTED_COUNT=0

for PAIR in "${PAIRS[@]}"; do
    OFFSET="${PAIR%%:*}"
    PREFIX="${PAIR##*:}"
    for SUFFIX in "${SUFFIXES[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            model_config "$MODEL"

            # Mirrors measure_mem.slurm's own PDM_EXP_NAME suffixing (its comment: "Mirrors
            # InferenceBackend.experiment_path_suffix() -- keep in sync") -- hf backend results
            # land in a separate _hf-suffixed results dir, so the skip-if-exists check must
            # look there too, not at the megatron-backend path.
            PDM_EXP_NAME="$EXP_NAME"
            [[ "$BACKEND" = "hf" ]] && PDM_EXP_NAME="${EXP_NAME}_hf"

            PKL=$MEM_BASE/SparseGutenberg/$PDM_EXP_NAME/metrics/offset_${OFFSET}_prefix_${PREFIX}_suffix_${SUFFIX}_greedy.pkl
            if [[ $FORCE -eq 0 && $FORCE_METRICS -eq 0 && -f "$PKL" ]]; then
                SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
                continue
            fi

            EXPORTS="MODEL=$MODEL,OFFSET=$OFFSET,PREFIX_LENGTH=$PREFIX,SUFFIX_LENGTH=$SUFFIX,CHECKPOINT_BACKEND=$BACKEND"
            [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"
            [[ $FORCE_METRICS -eq 1 ]] && EXPORTS="$EXPORTS,FORCE_METRICS=1"
            [[ -n "$REPETITIONS" ]] && EXPORTS="$EXPORTS,REPETITIONS=$REPETITIONS"
            TIME_ARG=()
            [[ -n "$JOB_TIME" ]] && TIME_ARG=(--time="$JOB_TIME")

            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[dry-run] sbatch ${TIME_ARG[*]} --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/measure_mem.slurm"
                SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
                continue
            fi

            echo "Submitting measure_mem.slurm (model=$MODEL exp=$EXP_NAME backend=$BACKEND) offset=$OFFSET prefix_length=$PREFIX suffix_length=$SUFFIX"
            # ALL = propagate the full submission env (USER, PATH, …) so the scripts'
            # $USER-based paths resolve, then layer our per-job vars on top.
            sbatch "${TIME_ARG[@]}" --export=ALL,"$EXPORTS" "$SCRIPT_DIR/measure_mem.slurm"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        done
    done
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT"
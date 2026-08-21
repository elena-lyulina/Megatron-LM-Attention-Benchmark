#!/bin/bash
# Submit memorization measurement jobs for every model in attn_bench/scripts/llama_checkpoints.sh.
# One job per model covers every requested (offset, prefix) point at one suffix_length,
# sharing one checkpoint load. Points already done are dropped before submitting; a model
# with nothing left submits no job. To add a model to the sweep: add it to llama_checkpoints.sh.
#
# Usage: bash measure_mem_all.sh --offsets 0 --prefixes 50 100 500 1000 2000
#   --diagonal-starts <s1> [s2 ...]  instead of --offsets: fixed suffix_start diagonal
#                                    (offset = start - prefix), mutually exclusive with --offsets
#   --suffix N                       single value (default: 500) -- Stage 2 always computes
#                                    every boundary <= this from one job, so sweeping over
#                                    multiple suffix values is redundant with that already
#   --dry-run                       print sbatch commands, submit nothing
#   --models m1,m2                  restrict models (default: every model in the registry)
#   --force / --force-metrics       resubmit regardless of store / recompute every Step-2 boundary
#   --time HH:MM:SS                 override measure_mem.slurm's default time limit
#   --backend hf                    use each model's already-converted HF checkpoint
#   --repetitions r1,r2,...         restrict which reps each job generates (default: all 10)

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

# The login node's system python3 is too old (3.6, predates e.g. PDM's use of typing.Literal)
# for the dry-run checks below -- prefer a personal conda env's python3 if one exists
# (per CSCS's own Clariden setup docs), falling back to system python3 otherwise.
PYTHON_BIN=python3
[ -x "$HOME/miniconda3/bin/python3" ] && PYTHON_BIN="$HOME/miniconda3/bin/python3"

# So the dry-run checks below can `import attn_bench` and unpickle existing Results-shaped
# pkls (needs verbatim_eval importable) -- same PYTHONPATH shape measure_mem.slurm exports
# inside the container, minus the container.
PDM_DIR=/users/$USER/scratch/PDM
export PYTHONPATH="$SCRIPT_DIR/../..:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}"

# "Done" marker: the Stage-2 pkl, checked on scratch then store (compute_memorization_metrics.py
# --dry-run) -- its presence means both inference and metric aggregation finished.
STORE_MEM_BASE=/users/$USER/store/mem-results
SCRATCH_MEM_BASE=/iopsstor/scratch/cscs/$USER/mem-results

OFFSETS=()
PREFIXES=()
SUFFIX_LENGTH=500
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
        --suffix)
            SUFFIX_LENGTH="$2"; shift 2
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
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--force] [--force-metrics] [--dry-run] [--models m1,m2] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffix <n>]"
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
        echo "Usage: $0 --diagonal-starts <s1> [s2 ...] --prefixes <p1> [p2 ...] [--suffix <n>]"
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
        echo "Usage: $0 [--force] --offsets <o1> [o2 ...] --prefixes <p1> [p2 ...] [--suffix <n>]"
        exit 1
    fi
    for OFFSET in "${OFFSETS[@]}"; do
        for PREFIX in "${PREFIXES[@]}"; do
            PAIRS+=("$OFFSET:$PREFIX")
        done
    done
fi

SKIPPED_COUNT=0
SUBMITTED_COUNT=0
JOBS_SUBMITTED=0

for MODEL in "${MODELS[@]}"; do
    model_config "$MODEL"

    # Mirrors measure_mem.slurm's own PDM_EXP_NAME suffixing (its comment: "Mirrors
    # InferenceBackend.experiment_path_suffix() -- keep in sync") -- hf backend results
    # land in a separate _hf-suffixed results dir, so the skip-if-exists check must
    # look there too, not at the megatron-backend path.
    PDM_EXP_NAME="$EXP_NAME"
    [[ "$BACKEND" = "hf" ]] && PDM_EXP_NAME="${EXP_NAME}_hf"

    # Points still needing work for this model -- calls Stage 2's own --dry-run (dual-checks
    # scratch then store, rep-aware -- catches a pkl left stale by a REPETITIONS increase,
    # not just "file exists") instead of a separate coarse checker.
    if [[ $FORCE -eq 1 || $FORCE_METRICS -eq 1 ]]; then
        GROUP_POINTS=("${PAIRS[@]}")
    else
        NEEDED=$("$PYTHON_BIN" "$SCRIPT_DIR/../evaluation/compute_memorization_metrics.py" \
            --dry-run \
            --exp-name "$PDM_EXP_NAME" \
            --base-path "$SCRATCH_MEM_BASE/SparseGutenberg" \
            --save-path "$SCRATCH_MEM_BASE/SparseGutenberg" \
            --persistent-storage-path "$STORE_MEM_BASE/SparseGutenberg" \
            --points "${PAIRS[@]}" --suffix-length "$SUFFIX_LENGTH")
        IFS=' ' read -r -a GROUP_POINTS <<< "$NEEDED"
    fi
    SKIPPED_COUNT=$((SKIPPED_COUNT + ${#PAIRS[@]} - ${#GROUP_POINTS[@]}))

    if [[ ${#GROUP_POINTS[@]} -eq 0 ]]; then
        echo "All ${#PAIRS[@]} point(s) already complete for (model=$MODEL suffix=$SUFFIX_LENGTH) -- nothing to submit."
        continue
    fi
    POINTS_CSV=$(IFS=,; echo "${GROUP_POINTS[*]}")

    EXPORTS="MODEL=$MODEL,POINTS=$POINTS_CSV,SUFFIX_LENGTH=$SUFFIX_LENGTH,CHECKPOINT_BACKEND=$BACKEND"
    [[ $FORCE_METRICS -eq 1 ]] && EXPORTS="$EXPORTS,FORCE_METRICS=1"
    [[ -n "$REPETITIONS" ]] && EXPORTS="$EXPORTS,REPETITIONS=$REPETITIONS"
    TIME_ARG=()
    [[ -n "$JOB_TIME" ]] && TIME_ARG=(--time="$JOB_TIME")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch ${TIME_ARG[*]} --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/measure_mem.slurm"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + ${#GROUP_POINTS[@]}))
        JOBS_SUBMITTED=$((JOBS_SUBMITTED + 1))
        continue
    fi

    echo "Submitting measure_mem.slurm (model=$MODEL exp=$EXP_NAME backend=$BACKEND) points=$POINTS_CSV suffix_length=$SUFFIX_LENGTH (${#GROUP_POINTS[@]} points, 1 job)"
    # ALL = propagate the full submission env (USER, PATH, …) so the scripts'
    # $USER-based paths resolve, then layer our per-job vars on top.
    sbatch "${TIME_ARG[@]}" --export=ALL,"$EXPORTS" "$SCRIPT_DIR/measure_mem.slurm"
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + ${#GROUP_POINTS[@]}))
    JOBS_SUBMITTED=$((JOBS_SUBMITTED + 1))
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT points across $JOBS_SUBMITTED job(s)"

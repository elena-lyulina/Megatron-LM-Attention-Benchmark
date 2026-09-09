#!/bin/bash
# Submit lm-eval-harness benchmark jobs for the 13 models of the benchmark table (one job per
# model). Models whose results already exist on scratch or store are skipped.
# The list is not the whole llama_checkpoints.sh registry -- see
# attn_bench/_plans/lm_eval_benchmark_plan.md section 1.
#
# Usage: bash eval_benchmarks_all.sh [--dry-run] [--models m1,m2] [--shots N] [--limit N]
#   --dry-run        print sbatch commands, submit nothing
#   --models m1,m2   restrict to these models (default: all 13)
#   --external-model NAME|all
#                    submit public HF reference models INSTEAD of the 13 (repeatable). These
#                    calibrate the table's absolute numbers; they are not architecture
#                    comparisons. See external_config in llama_checkpoints.sh.
#   --shots N        num_fewshot (default: 0), appears in the output path
#   --limit N        cap documents per task -- smoke tests only, never counts as done
#   --batch-size N   fixed integer, never "auto" (default: 8, must match across all models)
#   --force          resubmit even if results already exist
#   --time HH:MM:SS  override eval_benchmarks.slurm's default time limit

set -e

SCRIPT_DIR=$(dirname "$0")

# No python: the login node's is 3.6 (see measure_mem_all.sh) and the "done" marker is a file.

EVAL_MODELS=(full-scf1 gated-scf1 sink-scf1 swa-w256-scf1 swa-w1024-scf1 swa-w4096-scf1 \
             gdn carry-r0 carry-r0.5 carry-r1 kda mla qwen)

SCRATCH_EVAL_BASE=/iopsstor/scratch/cscs/$USER/eval-results/lm-eval
STORE_EVAL_BASE=/users/$USER/store/eval-results/lm-eval

DRY_RUN=0
FORCE=0
EXTERNAL_SELECTED=()
SHOTS=0
LIMIT=""
BATCH_SIZE=8
JOB_TIME=""
MODELS_CSV=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force) FORCE=1; shift ;;
        --shots) SHOTS="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --time) JOB_TIME="$2"; shift 2 ;;
        --models) MODELS_CSV="$2"; shift 2 ;;
        --external-model) EXTERNAL_SELECTED+=("$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$MODELS_CSV" ]]; then
    IFS=',' read -r -a EVAL_MODELS <<< "$MODELS_CSV"
fi

source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

# --external-model replaces the 13 rather than adding to them: the reference rows are a
# separate, occasional errand, and combining the two would make it easy to resubmit the whole
# sweep by accident while adding one calibration row.
if [[ ${#EXTERNAL_SELECTED[@]} -gt 0 && -n "$MODELS_CSV" ]]; then
    echo "--models and --external-model are mutually exclusive"; exit 1
fi
if [[ " ${EXTERNAL_SELECTED[*]} " == *" all "* ]]; then
    EXTERNAL_SELECTED=("${EXTERNAL_MODELS[@]}")
fi

# Same skip check for both loops: read-only on scratch and store, since results are promoted
# to store by hand and a rerun must not redo a model already copied over.
already_done() {
    local exp_name="$1"
    if [[ $FORCE -eq 1 || -n "$LIMIT" ]]; then return 1; fi
    local found
    found=$(find "$SCRATCH_EVAL_BASE/$exp_name/shots$SHOTS" "$STORE_EVAL_BASE/$exp_name/shots$SHOTS" \
                 -name 'results_*.json' 2>/dev/null | head -1)
    [[ -n "$found" ]] && { echo "$found"; return 0; }
    return 1
}

SUBMITTED=0
SKIPPED=0
MISSING_CKPT=0

for NAME in "${EXTERNAL_SELECTED[@]}"; do
    # Sets EXP_NAME (and the pretrained id / revision / tokenizer / max_length the slurm reads
    # back via external_config). Unknown names exit non-zero from there.
    external_config "$NAME"

    if FOUND=$(already_done "$EXP_NAME"); then
        echo "SKIP $NAME -- results already exist ($FOUND)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    EXPORTS="EXTERNAL_MODEL=$NAME,NUM_FEWSHOT=$SHOTS,BATCH_SIZE=$BATCH_SIZE"
    [[ -n "$LIMIT" ]] && EXPORTS="$EXPORTS,LIMIT=$LIMIT"
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    TIME_ARG=()
    [[ -n "$JOB_TIME" ]] && TIME_ARG=(--time="$JOB_TIME")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch ${TIME_ARG[*]} --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/eval_benchmarks.slurm"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    echo "Submitting eval_benchmarks.slurm (external=$NAME hf=$EXTERNAL_HF_MODEL${EXTERNAL_REVISION:+@$EXTERNAL_REVISION} exp=$EXP_NAME shots=$SHOTS bs=$BATCH_SIZE max_length=$EXTERNAL_MAX_LENGTH)"
    sbatch "${TIME_ARG[@]}" --export=ALL,"$EXPORTS" "$SCRIPT_DIR/eval_benchmarks.slurm"
    SUBMITTED=$((SUBMITTED + 1))
done

[[ ${#EXTERNAL_SELECTED[@]} -gt 0 ]] && EVAL_MODELS=()

for MODEL in "${EVAL_MODELS[@]}"; do
    model_config "$MODEL"

    HF_DIR=/users/$USER/store/hf-checkpoints/$EXP_NAME
    if [[ ! -f "$HF_DIR/config.json" ]]; then
        echo "SKIP $MODEL -- no HF checkpoint at $HF_DIR (run convert_and_validate_hf.slurm first)"
        MISSING_CKPT=$((MISSING_CKPT + 1))
        continue
    fi

    # A --limit run is a smoke test: it neither skips nor counts as done.
    if FOUND=$(already_done "$EXP_NAME"); then
        echo "SKIP $MODEL -- results already exist ($FOUND)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # TASKS stays on the slurm default: --export=ALL,"K=V,..." shreds comma-containing values
    # at the first comma (as measure_mem_all.sh documents for POINTS). To override it, export
    # TASKS as a shell variable before running this and --export=ALL propagates it intact.
    EXPORTS="MODEL=$MODEL,NUM_FEWSHOT=$SHOTS,BATCH_SIZE=$BATCH_SIZE"
    [[ -n "$LIMIT" ]] && EXPORTS="$EXPORTS,LIMIT=$LIMIT"
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    TIME_ARG=()
    [[ -n "$JOB_TIME" ]] && TIME_ARG=(--time="$JOB_TIME")

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch ${TIME_ARG[*]} --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/eval_benchmarks.slurm"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    echo "Submitting eval_benchmarks.slurm (model=$MODEL exp=$EXP_NAME shots=$SHOTS bs=$BATCH_SIZE${LIMIT:+ limit=$LIMIT})"
    sbatch "${TIME_ARG[@]}" --export=ALL,"$EXPORTS" "$SCRIPT_DIR/eval_benchmarks.slurm"
    SUBMITTED=$((SUBMITTED + 1))
done

echo
echo "Submitted: $SUBMITTED job(s)   Skipped (done): $SKIPPED   Missing checkpoint: $MISSING_CKPT"

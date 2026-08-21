#!/bin/bash
# Submit long-Gutenberg position-loss inference for every model in attn_bench/scripts/llama_checkpoints.sh.
# One job per MODEL, each self-parallel across 4 GPUs. A model is skipped when all requested
# buckets are already complete (long_gutenberg_inference.py --dry-run, dual-checks scratch
# then store); --force submits regardless.
#
# Env passthrough (optional): REPETITIONS, MAX_LENGTH, MAX_SAMPLES, LOG_STATE_NORM, STATE_CHUNK,
# STORE_INDIVIDUAL. LOG_STATE_NORM is applied only to GDN variants (attention models have no
# state to log); STORE_INDIVIDUAL applies to every model.
#
# To add a newly trained model to this sweep: add it to attn_bench/scripts/llama_checkpoints.sh, not here.
#
# Usage: bash attn_bench/submissions/long_gutenberg_inference_all.sh   # full sweep, all models
# Add --dry-run to print the sbatch commands that would run without submitting anything.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

RESULTS_BASE=/users/$USER/store/long-gutenberg-results
SCRATCH_RESULTS_BASE=/iopsstor/scratch/cscs/$USER/long-gutenberg-results
GUTENBERG_LONG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl_long
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to log GDN state norms (applied only to GDN variants)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128)
STORE_INDIVIDUAL=${STORE_INDIVIDUAL:-}   # set to also write raw per-sequence records

FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force] [--dry-run]  (set REPETITIONS/MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

SKIPPED_COUNT=0
SUBMITTED_COUNT=0

for MODEL in "${MODELS[@]}"; do
    model_config "$MODEL"

    # All models run at full length now -- MAX_LENGTH (if set) still caps every model.
    VAR_MAXLEN=${MAX_LENGTH:-}

    # State norms are only logged for GDN variants (NEEDS_TRITON is 1 only for the GDN mixer).
    WANT_STATE=0
    [[ -n "$LOG_STATE_NORM" && "$NEEDS_TRITON" == "1" ]] && WANT_STATE=1

    # "Done" = every requested bucket's rep_{R}.npz is present (plus rep_{R}_state.npz/
    # rep_{R}_individual.jsonl when requested) -- checked via the .py's own --dry-run (dual-
    # checks scratch then store), the same logic the real run uses internally, instead of a
    # separate hand-rolled bash loop.
    if [[ $FORCE -eq 1 ]]; then
        NEEDED="force"
    else
        DRY_RUN_ARGS=()
        [[ -n "${MAX_SAMPLES:-}" ]] && DRY_RUN_ARGS+=(--max-samples "$MAX_SAMPLES")
        [[ -n "$VAR_MAXLEN" ]] && DRY_RUN_ARGS+=(--max-length "$VAR_MAXLEN")
        [[ $WANT_STATE -eq 1 ]] && DRY_RUN_ARGS+=(--log-state-norm)
        [[ -n "$STORE_INDIVIDUAL" ]] && DRY_RUN_ARGS+=(--store-individual)
        NEEDED=$(python3 "$SCRIPT_DIR/../evaluation/long_gutenberg_inference.py" --dry-run \
            --ckpt-dir "/users/$USER/store/pretrain-results/$CKPT_NAME/checkpoints" \
            --tokenizer-path "$TOKENIZER_PATH" \
            --experiment-path "$SCRATCH_RESULTS_BASE/$EXP_NAME" \
            --persistent-storage-path "$RESULTS_BASE/$EXP_NAME" \
            --data-folder "$GUTENBERG_LONG_JSONL_DIR" \
            --repetitions "$REPETITIONS" \
            "${DRY_RUN_ARGS[@]}")
    fi
    if [[ -z "$NEEDED" ]]; then
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        echo "All requested buckets already complete for $EXP_NAME -- nothing to submit."
        continue
    fi

    # REPETITIONS is NOT passed here: sbatch --export=NAME=VALUE splits VALUE on commas, which would
    # truncate the list to the first bucket. The slurm hardcodes the default list; an override set in
    # the environment (e.g. REPETITIONS=0 bash ...) rides to the job via --export=ALL instead.
    EXPORTS="MODEL=$MODEL"
    [[ -n "$VAR_MAXLEN" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$VAR_MAXLEN"
    [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
    [[ $WANT_STATE -eq 1 ]] && EXPORTS="$EXPORTS,LOG_STATE_NORM=1"
    [[ $WANT_STATE -eq 1 && -n "${STATE_CHUNK:-}" ]] && EXPORTS="$EXPORTS,STATE_CHUNK=$STATE_CHUNK"
    [[ -n "$STORE_INDIVIDUAL" ]] && EXPORTS="$EXPORTS,STORE_INDIVIDUAL=1"
    # --force also recomputes: without this the resubmitted job would just skip and no-op.
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/long_gutenberg_inference.slurm"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        continue
    fi

    echo "Submitting MODEL=$MODEL ($EXP_NAME) reps=$REPETITIONS state_norm=$WANT_STATE individual=${STORE_INDIVIDUAL:-0}"
    # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
    sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_gutenberg_inference.slurm"
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT"

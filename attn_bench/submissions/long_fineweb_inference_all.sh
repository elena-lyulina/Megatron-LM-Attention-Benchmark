#!/bin/bash
# Submit long-FineWeb-Edu position-loss inference for every model in attn_bench/scripts/llama_checkpoints.sh,
# on both the seen and unseen partitions. One job per (MODEL, DATA_FILE) pair, each self-parallel
# across 4 GPUs. A job is skipped when its bucket (<data file stem>.npz) already exists on store;
# --force submits regardless.
#
# Env passthrough (optional): MAX_LENGTH, MAX_SAMPLES, LOG_STATE_NORM, STATE_CHUNK,
# STORE_INDIVIDUAL. LOG_STATE_NORM is applied only to GDN variants (attention models have no
# state to log); STORE_INDIVIDUAL applies to every model.
#
# To add a newly trained model to this sweep: add it to attn_bench/scripts/llama_checkpoints.sh, not here.
#
# Usage: bash attn_bench/submissions/long_fineweb_inference_all.sh   # full sweep, all models x 2 partitions
# Add --dry-run to print the sbatch commands that would run without submitting anything.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

RESULTS_BASE=/users/$USER/store/long-fineweb-results
STORE_TOKENIZED=/users/$USER/store/datasets/tokenized
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to log GDN state norms (applied only to GDN variants)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128)
STORE_INDIVIDUAL=${STORE_INDIVIDUAL:-}   # set to also write raw per-sequence records

# Length range must match the extract_long_docs.py run that produced these files.
MIN_LENGTH=${MIN_LENGTH:-24576}
MAX_LENGTH_RANGE=${MAX_LENGTH_RANGE:-32768}

# "TAG|DATA_FOLDER_NAME" for the two partitions to sweep. DATA_FOLDER_NAME must match
# what was passed as data_folder= to extract_long_docs.slurm (its output then lives at
# $STORE_TOKENIZED/${DATA_FOLDER_NAME}_long/long_${MIN_LENGTH}_${MAX_LENGTH_RANGE}.jsonl).
DATA_FOLDERS=(
    "seen|fineweb-edu-dedup-160B-datatrove_0.25"
    "unseen|fineweb-edu-dedup-160B-datatrove_0.75_unseen"
)

FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force] [--dry-run]  (set MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

SKIPPED_COUNT=0
SUBMITTED_COUNT=0

for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    IFS='|' read -r TAG DATA_FOLDER_NAME <<< "$DATA_FOLDER"
    DATA_FILE=$STORE_TOKENIZED/${DATA_FOLDER_NAME}_long/long_${MIN_LENGTH}_${MAX_LENGTH_RANGE}.jsonl
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "Skipping partition=$TAG: $DATA_FILE not found (run extract_long_docs.slurm first)"
        continue
    fi
    KEY=$(basename "$DATA_FILE" .jsonl)

    for MODEL in "${MODELS[@]}"; do
        model_config "$MODEL"

        # All models run the same way now: TP=1, no length cap, default walltime.
        VAR_MAXLEN=${MAX_LENGTH:-}
        # Config folder (must match config_name() in long_inference.py): unset MAX_SAMPLES falls
        # back to long_fineweb_inference.py's own DEFAULT_MAX_SAMPLES (660), not "all" -- the
        # python script itself defaults --max-samples to 660, unlike the Gutenberg script.
        CONFIG="${MAX_SAMPLES:-660}_samples_${VAR_MAXLEN:-full}_tokens"

        # State norms are only logged for GDN variants (NEEDS_TRITON is 1 only for the GDN mixer).
        WANT_STATE=0
        [[ -n "$LOG_STATE_NORM" && "$NEEDS_TRITON" == "1" ]] && WANT_STATE=1

        RESULTS_DIR=$RESULTS_BASE/$EXP_NAME/${DATA_FOLDER_NAME}_long/$CONFIG
        DONE=1
        [[ ! -f "$RESULTS_DIR/$KEY.npz" ]] && DONE=0
        [[ $WANT_STATE -eq 1 && ! -f "$RESULTS_DIR/${KEY}_state.npz" ]] && DONE=0
        [[ -n "$STORE_INDIVIDUAL" && ! -f "$RESULTS_DIR/${KEY}_individual.jsonl" ]] && DONE=0
        if [[ $FORCE -eq 0 && $DONE -eq 1 ]]; then
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
            continue
        fi

        EXPORTS="MODEL=$MODEL,DATA_FILE=$DATA_FILE"
        [[ -n "$VAR_MAXLEN" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$VAR_MAXLEN"
        [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
        [[ $WANT_STATE -eq 1 ]] && EXPORTS="$EXPORTS,LOG_STATE_NORM=1"
        [[ $WANT_STATE -eq 1 && -n "${STATE_CHUNK:-}" ]] && EXPORTS="$EXPORTS,STATE_CHUNK=$STATE_CHUNK"
        [[ -n "$STORE_INDIVIDUAL" ]] && EXPORTS="$EXPORTS,STORE_INDIVIDUAL=1"
        # --force also recomputes: without this the resubmitted job would just skip and no-op.
        [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[dry-run] sbatch --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/long_fineweb_inference.slurm"
            SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
            continue
        fi

        echo "Submitting MODEL=$MODEL ($EXP_NAME) partition=$TAG state_norm=$WANT_STATE individual=${STORE_INDIVIDUAL:-0}"
        # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
        sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_fineweb_inference.slurm"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
    done
done

echo "Skipped: $SKIPPED_COUNT"
echo "Submitted: $SUBMITTED_COUNT"
#!/bin/bash
# Submit long-FineWeb-Edu position-loss inference for every model, on both the seen
# and unseen partitions. One job per (VARIANT, DATA_FILE) pair, each self-parallel
# across 4 GPUs. A job is skipped when its bucket (<data file stem>.npz) already
# exists on store; --force submits regardless.
#
# Env passthrough (optional): MAX_LENGTH, MAX_SAMPLES, LOG_STATE_NORM, STATE_CHUNK.
# LOG_STATE_NORM is applied only to GDN variants (attention models have no state to log).
#
# Usage (from attn_bench/):
#   bash attn_bench/submissions/long_fineweb_inference_all.sh                 # full sweep, all 9 models x 2 partitions
#   MAX_SAMPLES=20 bash attn_bench/submissions/long_fineweb_inference_all.sh --force   # calibration
#   MAX_LENGTH=16384 bash attn_bench/submissions/long_fineweb_inference_all.sh # cap sequences

set -e

SCRIPT_DIR=$(dirname "$0")
RESULTS_BASE=/users/$USER/store/long-fineweb-results
STORE_TOKENIZED=/users/$USER/store/datasets/tokenized
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to log GDN state norms (applied only to GDN variants)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128)

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

# "VARIANT|EXP_NAME". EXP_NAME is explicit (it drives the skip-check) and must match the
# case in long_fineweb_inference.slurm.
JOBS=(
    "full|llama3-1b-full-attn-fineweb40B-gutenberg3B"
    "gated|llama3-1b-gated-attn-fineweb40B-gutenberg3B"
    "full-xdoc-leak|llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B"
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
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force]  (set MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

# sink / off-by-one use unfused attention (O(S^2) memory) and OOM at full length, so cap them.
# A global MAX_LENGTH (if set) overrides this for every model.
UNFUSED_MAXLEN=12288

for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
    IFS='|' read -r TAG DATA_FOLDER_NAME <<< "$DATA_FOLDER"
    DATA_FILE=$STORE_TOKENIZED/${DATA_FOLDER_NAME}_long/long_${MIN_LENGTH}_${MAX_LENGTH_RANGE}.jsonl
    if [[ ! -f "$DATA_FILE" ]]; then
        echo "Skipping partition=$TAG: $DATA_FILE not found (run extract_long_docs.slurm first)"
        continue
    fi
    KEY=$(basename "$DATA_FILE" .jsonl)

    for JOB in "${JOBS[@]}"; do
        IFS='|' read -r VARIANT EXP_NAME <<< "$JOB"

        # Per-variant length cap: the unfused-attention models can't run at full length.
        VAR_MAXLEN=${MAX_LENGTH:-}
        if [[ -z "$VAR_MAXLEN" && ( "$VARIANT" == sink || "$VARIANT" == off-by-one ) ]]; then
            VAR_MAXLEN=$UNFUSED_MAXLEN
        fi
        # Config folder (must match config_name() in long_inference.py): unset MAX_SAMPLES falls
        # back to long_fineweb_inference.py's own DEFAULT_MAX_SAMPLES (660), not "all" -- the
        # python script itself defaults --max-samples to 660, unlike the Gutenberg script.
        CONFIG="${MAX_SAMPLES:-660}_samples_${VAR_MAXLEN:-full}_tokens"

        # State norms are only logged for GDN variants (their exp name contains "gdn").
        WANT_STATE=0
        [[ -n "$LOG_STATE_NORM" && "$EXP_NAME" == *gdn* ]] && WANT_STATE=1

        RESULTS_DIR=$RESULTS_BASE/$EXP_NAME/${DATA_FOLDER_NAME}_long/$CONFIG
        DONE=1
        [[ ! -f "$RESULTS_DIR/$KEY.npz" ]] && DONE=0
        [[ $WANT_STATE -eq 1 && ! -f "$RESULTS_DIR/${KEY}_state.npz" ]] && DONE=0
        if [[ $FORCE -eq 0 && $DONE -eq 1 ]]; then
            echo "Skipping $EXP_NAME / $TAG (bucket present)"
            continue
        fi

        EXPORTS="VARIANT=$VARIANT,DATA_FILE=$DATA_FILE"
        [[ -n "$VAR_MAXLEN" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$VAR_MAXLEN"
        [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
        [[ $WANT_STATE -eq 1 ]] && EXPORTS="$EXPORTS,LOG_STATE_NORM=1"
        [[ $WANT_STATE -eq 1 && -n "${STATE_CHUNK:-}" ]] && EXPORTS="$EXPORTS,STATE_CHUNK=$STATE_CHUNK"
        # --force also recomputes: without this the resubmitted job would just skip and no-op.
        [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

        echo "Submitting VARIANT=$VARIANT ($EXP_NAME) partition=$TAG state_norm=$WANT_STATE"
        # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
        sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_fineweb_inference.slurm"
    done
done
#!/bin/bash
# Submit long-Gutenberg position-loss inference for every model. One job per VARIANT,
# each self-parallel across 4 GPUs. A model is skipped when all requested buckets
# (rep_{R}.npz) already exist on store; --force submits regardless.
#
# Env passthrough (optional): REPETITIONS, MAX_LENGTH, MAX_SAMPLES, LOG_STATE_NORM, STATE_CHUNK.
# LOG_STATE_NORM is applied only to GDN variants (attention models have no state to log).
#
# Usage (from attn_bench/):
#   bash attn_bench/submissions/long_gutenberg_inference_all.sh                 # full sweep, all 9 models
#   REPETITIONS=0 MAX_SAMPLES=20 bash attn_bench/submissions/long_gutenberg_inference_all.sh --force   # calibration
#   MAX_LENGTH=16384 bash attn_bench/submissions/long_gutenberg_inference_all.sh # cap sequences at 2L

set -e

SCRIPT_DIR=$(dirname "$0")
RESULTS_BASE=/users/elyulina/store/long-gutenberg-results
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to log GDN state norms (applied only to GDN variants)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128)

# "VARIANT|EXP_NAME". EXP_NAME is explicit (it drives the skip-check) and must match the
# case in long_gutenberg_inference.slurm.
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
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force]  (set REPETITIONS/MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

IFS=',' read -ra REPS <<< "$REPETITIONS"

# sink / off-by-one use unfused attention (O(S^2) memory) and OOM at full length, so cap them.
# A global MAX_LENGTH (if set) overrides this for every model.
UNFUSED_MAXLEN=12288

for JOB in "${JOBS[@]}"; do
    IFS='|' read -r VARIANT EXP_NAME <<< "$JOB"

    # Per-variant length cap: the unfused-attention models can't run at full length.
    VAR_MAXLEN=${MAX_LENGTH:-}
    if [[ -z "$VAR_MAXLEN" && ( "$VARIANT" == sink || "$VARIANT" == off-by-one ) ]]; then
        VAR_MAXLEN=$UNFUSED_MAXLEN
    fi
    # Config folder (must match config_name() in long_gutenberg_inference.py): unset caps -> "all"/"full".
    CONFIG="${MAX_SAMPLES:-all}_samples_${VAR_MAXLEN:-full}_tokens"

    # State norms are only logged for GDN variants (their exp name contains "gdn").
    WANT_STATE=0
    [[ -n "$LOG_STATE_NORM" && "$EXP_NAME" == *gdn* ]] && WANT_STATE=1

    # "Done" = every requested bucket's rep_{R}.npz is present (plus rep_{R}_state.npz when
    # state norms are requested) for this config.
    DONE=1
    for R in "${REPS[@]}"; do
        if [[ ! -f "$RESULTS_BASE/$EXP_NAME/$CONFIG/rep_${R}.npz" ]]; then
            DONE=0; break
        fi
        if [[ $WANT_STATE -eq 1 && ! -f "$RESULTS_BASE/$EXP_NAME/$CONFIG/rep_${R}_state.npz" ]]; then
            DONE=0; break
        fi
    done
    if [[ $FORCE -eq 0 && $DONE -eq 1 ]]; then
        echo "Skipping $EXP_NAME (all buckets present)"
        continue
    fi

    # REPETITIONS is NOT passed here: sbatch --export=NAME=VALUE splits VALUE on commas, which would
    # truncate the list to the first bucket. The slurm hardcodes the default list; an override set in
    # the environment (e.g. REPETITIONS=0 bash ...) rides to the job via --export=ALL instead.
    EXPORTS="VARIANT=$VARIANT"
    [[ -n "$VAR_MAXLEN" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$VAR_MAXLEN"
    [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
    [[ $WANT_STATE -eq 1 ]] && EXPORTS="$EXPORTS,LOG_STATE_NORM=1"
    [[ $WANT_STATE -eq 1 && -n "${STATE_CHUNK:-}" ]] && EXPORTS="$EXPORTS,STATE_CHUNK=$STATE_CHUNK"
    # --force also recomputes: without this the resubmitted job would just skip and no-op.
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    echo "Submitting VARIANT=$VARIANT ($EXP_NAME) reps=$REPETITIONS state_norm=$WANT_STATE"
    # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
    sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_gutenberg_inference.slurm"
done

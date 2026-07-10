#!/bin/bash
# Submit long-Gutenberg position-loss inference for every model in attn_bench/scripts/llama_checkpoints.sh.
# One job per MODEL, each self-parallel across 4 GPUs. A model is skipped when all requested
# buckets (rep_{R}.npz) already exist on store; --force submits regardless.
#
# Env passthrough (optional): REPETITIONS, MAX_LENGTH, MAX_SAMPLES, LOG_STATE_NORM, STATE_CHUNK.
# LOG_STATE_NORM is applied only to GDN variants (attention models have no state to log).
#
# To add a newly trained model to this sweep: add it to attn_bench/scripts/llama_checkpoints.sh, not here.
#
# Usage: bash attn_bench/submissions/long_gutenberg_inference_all.sh   # full sweep, all models
# Add --dry-run to print the sbatch commands that would run without submitting anything.

set -e

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/../scripts/llama_checkpoints.sh"

RESULTS_BASE=/users/$USER/store/long-gutenberg-results
REPETITIONS=${REPETITIONS:-0,1,2,4,8,16,32,64,128,256}
LOG_STATE_NORM=${LOG_STATE_NORM:-}   # set to log GDN state norms (applied only to GDN variants)
STATE_CHUNK=${STATE_CHUNK:-}         # override the state readout stride (default 128)

FORCE=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--force] [--dry-run]  (set REPETITIONS/MAX_LENGTH/MAX_SAMPLES via env)"; exit 1 ;;
    esac
done

IFS=',' read -ra REPS <<< "$REPETITIONS"

for MODEL in "${MODELS[@]}"; do
    model_config "$MODEL"

    # All models run at full length now -- MAX_LENGTH (if set) still caps every model.
    VAR_MAXLEN=${MAX_LENGTH:-}
    # Config folder (must match config_name() in long_gutenberg_inference.py): unset caps -> "all"/"full".
    CONFIG="${MAX_SAMPLES:-all}_samples_${VAR_MAXLEN:-full}_tokens"

    # State norms are only logged for GDN variants (NEEDS_TRITON is 1 only for the GDN mixer).
    WANT_STATE=0
    [[ -n "$LOG_STATE_NORM" && "$NEEDS_TRITON" == "1" ]] && WANT_STATE=1

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
    EXPORTS="MODEL=$MODEL"
    [[ -n "$VAR_MAXLEN" ]] && EXPORTS="$EXPORTS,MAX_LENGTH=$VAR_MAXLEN"
    [[ -n "${MAX_SAMPLES:-}" ]] && EXPORTS="$EXPORTS,MAX_SAMPLES=$MAX_SAMPLES"
    [[ $WANT_STATE -eq 1 ]] && EXPORTS="$EXPORTS,LOG_STATE_NORM=1"
    [[ $WANT_STATE -eq 1 && -n "${STATE_CHUNK:-}" ]] && EXPORTS="$EXPORTS,STATE_CHUNK=$STATE_CHUNK"
    # --force also recomputes: without this the resubmitted job would just skip and no-op.
    [[ $FORCE -eq 1 ]] && EXPORTS="$EXPORTS,OVERWRITE=1"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] sbatch --export=ALL,\"$EXPORTS\" $SCRIPT_DIR/long_gutenberg_inference.slurm"
        continue
    fi

    echo "Submitting MODEL=$MODEL ($EXP_NAME) reps=$REPETITIONS state_norm=$WANT_STATE"
    # ALL propagates the submission env (USER, PATH, ...) so $USER-based paths resolve.
    sbatch --export=ALL,"$EXPORTS" "$SCRIPT_DIR/long_gutenberg_inference.slurm"
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/llama_checkpoints.sh"   # provides MODELS + model_config -> EXP_NAME

### ARGS ###
# --individual / -i        pull *_individual.jsonl files (skipped by default, see below)
# --reps R [R ...]         only pull these repetitions' individual files (implies --individual)
# --models TAG [TAG ...]   only pull these llama_checkpoints.sh model tags (default: all MODELS)
PULL_INDIVIDUAL=false
REPS_FILTER=()
MODELS_FILTER=()
while [ $# -gt 0 ]; do
    case "$1" in
        --individual|-i)
            PULL_INDIVIDUAL=true
            shift
            ;;
        --reps)
            PULL_INDIVIDUAL=true
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                REPS_FILTER+=("$1")
                shift
            done
            ;;
        --models)
            shift
            while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
                MODELS_FILTER+=("$1")
                shift
            done
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done
if [ ${#MODELS_FILTER[@]} -eq 0 ]; then
    MODELS_TO_PULL=("${MODELS[@]}")
else
    MODELS_TO_PULL=("${MODELS_FILTER[@]}")
fi

### CONFIG ###
REMOTE_HOST="elyulina@clariden"
GUTENBERG_REMOTE_SRC="$REMOTE_HOST:/users/elyulina/store/long-gutenberg-results/"
GUTENBERG_LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/long-gutenberg-results/"
FINEWEB_REMOTE_SRC="$REMOTE_HOST:/users/elyulina/store/long-fineweb-results/"
FINEWEB_LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/long-fineweb-results/"

### BUILD INCLUDE FILTERS ###
# $EXP_NAME/*** pulls the dir and every config subdir under it.
GUTENBERG_INC=()
FINEWEB_INC=()
for MODEL in "${MODELS_TO_PULL[@]}"; do
    model_config "$MODEL"
    GUTENBERG_INC+=(--include="$EXP_NAME/***")
    FINEWEB_INC+=(--include="$EXP_NAME/***")
done

### INDIVIDUAL SAMPLE FILES: skipped by default, --reps narrows to specific repetitions ###
# Order matters: rsync filter rules are first-match-wins, so any rep-specific includes
# must precede the catch-all exclude, which must itself precede the EXP_NAME/*** includes
# above (that broad include would otherwise match individual files first).
INDIVIDUAL_FILTER=()
if [ "$PULL_INDIVIDUAL" = true ]; then
    if [ ${#REPS_FILTER[@]} -gt 0 ]; then
        for REP in "${REPS_FILTER[@]}"; do
            INDIVIDUAL_FILTER+=(--include="rep_${REP}_individual.jsonl")
        done
        INDIVIDUAL_FILTER+=(--exclude='*_individual.jsonl')
    fi
else
    INDIVIDUAL_FILTER=(--exclude='*_individual.jsonl')
fi

### PULL ###
mkdir -p "$GUTENBERG_LOCAL_DST" "$FINEWEB_LOCAL_DST"
rsync -avm "${INDIVIDUAL_FILTER[@]}" "${GUTENBERG_INC[@]}" --exclude='*' "$GUTENBERG_REMOTE_SRC" "$GUTENBERG_LOCAL_DST"
rsync -avm "${INDIVIDUAL_FILTER[@]}" "${FINEWEB_INC[@]}" --exclude='*' "$FINEWEB_REMOTE_SRC" "$FINEWEB_LOCAL_DST"

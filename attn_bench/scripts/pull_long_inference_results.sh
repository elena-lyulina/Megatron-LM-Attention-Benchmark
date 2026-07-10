#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/llama_checkpoints.sh"   # provides MODELS + model_config -> EXP_NAME

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
for MODEL in "${MODELS[@]}"; do
    model_config "$MODEL"
    GUTENBERG_INC+=(--include="$EXP_NAME/***")
    FINEWEB_INC+=(--include="$EXP_NAME/***")
done

### PULL ###
mkdir -p "$GUTENBERG_LOCAL_DST" "$FINEWEB_LOCAL_DST"
rsync -avm "${GUTENBERG_INC[@]}" --exclude='*' "$GUTENBERG_REMOTE_SRC" "$GUTENBERG_LOCAL_DST"
rsync -avm "${FINEWEB_INC[@]}" --exclude='*' "$FINEWEB_REMOTE_SRC" "$FINEWEB_LOCAL_DST"

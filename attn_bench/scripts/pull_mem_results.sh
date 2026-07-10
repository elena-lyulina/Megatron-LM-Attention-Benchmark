#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/llama_checkpoints.sh"   # provides MODELS + model_config -> EXP_NAME

### CONFIG ###
REMOTE_SRC="elyulina@clariden:/users/elyulina/store/mem-results/SparseGutenberg/"
LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/mem-results/SparseGutenberg2/"

### BUILD INCLUDE FILTERS ###
# Pull each experiment dir's *_greedy.pkl summaries only, not the raw per-sample jsonls.
INC=()
for MODEL in "${MODELS[@]}"; do
    model_config "$MODEL"
    INC+=(--include="$EXP_NAME/" --include="$EXP_NAME/*_greedy.pkl")
done

### PULL ###
mkdir -p "$LOCAL_DST"
rsync -avm "${INC[@]}" --exclude='*' "$REMOTE_SRC" "$LOCAL_DST"

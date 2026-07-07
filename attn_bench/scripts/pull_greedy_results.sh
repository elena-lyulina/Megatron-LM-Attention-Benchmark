#!/usr/bin/env bash
set -euo pipefail

### CONFIG ###
REMOTE_SRC="elyulina@clariden:/users/elyulina/store/mem-results/SparseGutenberg/"
LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/mem-results/SparseGutenberg2/"

### EXPERIMENTS ###
EXPS=(
    llama3-1b-full-attn-fineweb40B-gutenberg3B
    llama3-1b-gated-attn-fineweb40B-gutenberg3B
    llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215
    llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215
    llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B
    llama3-1b-gdn-fineweb40B-gutenberg3B
    llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B
    llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B
    llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B
)

### BUILD INCLUDE FILTERS ###
# only pull each experiment dir and its *_greedy.pkl files
INC=()
for e in "${EXPS[@]}"; do
    INC+=(--include="$e/" --include="$e/*_greedy.pkl")
done

### PULL ###
mkdir -p "$LOCAL_DST"
rsync -avm "${INC[@]}" --exclude='*' "$REMOTE_SRC" "$LOCAL_DST"
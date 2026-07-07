#!/usr/bin/env bash
set -euo pipefail

### CONFIG ###
REMOTE_SRC="elyulina@clariden:/users/elyulina/store/long-gutenberg-results/"
LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/long-gutenberg-results/"

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

### PER-MODEL CONFIG SELECTION ###
# By default pull each experiment dir and everything under it: all config subdirs
# (all_samples_full_tokens, all_samples_12288_tokens, all_samples_..._tp4, ...) with
# their rep_*.npz, rep_*_state.npz and run_metadata.json.
#
# sink and off-by-one are the exception: they have extra oomed/partial configs on the
# cluster (all_samples_16384_tokens_tp4, all_samples_full_tokens_tp4), so pull only the
# two that actually completed -- the 12k full-token DP run and the 20480 TP4 run.
SINK_OBO="llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215 llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215"
SINK_OBO_CONFIGS=(all_samples_12288_tokens all_samples_20480_tokens_tp4)

### BUILD INCLUDE FILTERS ###
# $e/*** matches the dir and all its contents recursively; missing paths are skipped.
# For the per-config (sink/obo) case we must also include the experiment dir itself
# (--include="$e/") -- otherwise the trailing --exclude='*' drops the top-level $e dir
# before rsync ever descends into it, and nothing under it transfers. The bare "$e/"
# only lets rsync traverse in; the unwanted config subdirs still get excluded.
INC=()
for e in "${EXPS[@]}"; do
    if [[ " $SINK_OBO " == *" $e "* ]]; then
        INC+=(--include="$e/")
        for c in "${SINK_OBO_CONFIGS[@]}"; do
            INC+=(--include="$e/$c/***")
        done
    else
        INC+=(--include="$e/***")
    fi
done

### PULL ###
mkdir -p "$LOCAL_DST"
rsync -avm "${INC[@]}" --exclude='*' "$REMOTE_SRC" "$LOCAL_DST"
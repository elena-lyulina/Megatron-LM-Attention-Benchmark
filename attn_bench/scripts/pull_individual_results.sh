#!/usr/bin/env bash
# Pull specific rep_{R}_individual.jsonl files for one model from store to local -- unlike
# pull_long_inference_results.sh (which mirrors whole experiment dirs), these files are large
# (~500MB+ each), so this only fetches the reps you actually ask for.
#
# Usage: MODEL=full REPS=0,256 bash attn_bench/scripts/pull_individual_results.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/llama_checkpoints.sh"

### CONFIG ###
REMOTE_HOST="elyulina@clariden"
REMOTE_STORE="/users/elyulina/store"
LOCAL_ROOT="$SCRIPT_DIR/../results"

MODEL=${MODEL:-full}
REPS=${REPS:-0,256}
CONFIG=${CONFIG:-all_samples_full_tokens}   # must match config_name() in long_inference.py

model_config "$MODEL"

REMOTE_DIR="$REMOTE_STORE/long-gutenberg-results/$EXP_NAME/$CONFIG"
LOCAL_DIR="$LOCAL_ROOT/long-gutenberg-results/$EXP_NAME/$CONFIG"
mkdir -p "$LOCAL_DIR"

IFS=',' read -ra REP_LIST <<< "$REPS"
INCLUDES=()
for R in "${REP_LIST[@]}"; do
    INCLUDES+=(--include="rep_${R}_individual.jsonl")
done

echo "Pulling reps=$REPS for $MODEL ($EXP_NAME) -> $LOCAL_DIR"
rsync -avm --progress "${INCLUDES[@]}" --exclude='*' "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

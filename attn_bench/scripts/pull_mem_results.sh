#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/llama_checkpoints.sh"   # provides MODELS + model_config -> EXP_NAME

### CONFIG ###
REMOTE_SRC="elyulina@clariden:/users/elyulina/store/mem-results/SparseGutenberg/"
LOCAL_DST="/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/mem-results/SparseGutenberg2/"

### MODELS TO PULL ###
# Positional args restrict which models to pull (default: all). --backend hf pulls the
# _hf-suffixed results dir instead -- run twice to pull a mixed backend set.
BACKEND="megatron"
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKEND="$2"; shift 2
            if [[ "$BACKEND" != "megatron" && "$BACKEND" != "hf" ]]; then
                echo "--backend must be 'megatron' or 'hf', got '$BACKEND'."
                exit 1
            fi
            ;;
        *)
            TARGETS+=("$1"); shift
            ;;
    esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${MODELS[@]}")

### BUILD INCLUDE FILTERS ###
# Pull each experiment dir's metrics/*_greedy.pkl summaries and metrics_metadata/*.json
# (which job/reps/timestamp wrote each pkl) only, not the raw per-sample jsonls.
INC=()
for MODEL in "${TARGETS[@]}"; do
    model_config "$MODEL"
    # Mirrors measure_mem.slurm's own PDM_EXP_NAME suffixing -- keep in sync.
    PDM_EXP_NAME="$EXP_NAME"
    [[ "$BACKEND" = "hf" ]] && PDM_EXP_NAME="${EXP_NAME}_hf"
    INC+=(--include="$PDM_EXP_NAME/" --include="$PDM_EXP_NAME/metrics/" --include="$PDM_EXP_NAME/metrics/*_greedy.pkl"
          --include="$PDM_EXP_NAME/metrics_metadata/" --include="$PDM_EXP_NAME/metrics_metadata/*.json")
done

### PULL ###
mkdir -p "$LOCAL_DST"
rsync -avm "${INC[@]}" --exclude='*' "$REMOTE_SRC" "$LOCAL_DST"

#!/bin/bash
set -euo pipefail

# Promote long-inference results (Gutenberg + FineWeb) and GDN paired-state results from
# scratch to store, scoped per model.
# The slurms write scratch only -- their skip-checks read both, see --persistent-storage-path --
# so run this whenever you want results kept past scratch's ~biweekly purge. Sibling of
# copy_mem_results_to_store.sh; run from a login node, never from inside a compute job.
#
# Takes llama_checkpoints.sh model tags (gdn, carry-r0, kda, ...), not experiment names, and
# copies whatever exists for each: both corpora, both backends. The _hf suffix sits on the model
# dir for Gutenberg and on the partition dir for FineWeb, so Gutenberg needs two source paths
# and FineWeb's whole model tree covers it.
#
# Usage: bash copy_long_results_to_store.sh gdn carry-r0 kda mla

SCRIPT_DIR=$(dirname "$0")
source "$SCRIPT_DIR/llama_checkpoints.sh"

SCRATCH=/iopsstor/scratch/cscs/$USER
STORE=/users/$USER/store

if [ $# -eq 0 ]; then
    echo "Usage: $0 model [model ...]   (tags from llama_checkpoints.sh: ${MODELS[*]})"
    exit 1
fi

copy_dir() {
    local src="$1" dst="$2"
    [ -d "$src" ] || return 0
    mkdir -p "$dst"
    rsync -a "$src/" "$dst/"
    echo "  $src -> $dst"
}

for MODEL in "$@"; do
    model_config "$MODEL"
    echo "$MODEL ($EXP_NAME)"
    for NAME in "$EXP_NAME" "${EXP_NAME}_hf"; do
        copy_dir "$SCRATCH/long-gutenberg-results/$NAME" "$STORE/long-gutenberg-results/$NAME"
    done
    copy_dir "$SCRATCH/long-fineweb-results/$EXP_NAME" "$STORE/long-fineweb-results/$EXP_NAME"
    copy_dir "$SCRATCH/gdn-state-results/$EXP_NAME" "$STORE/gdn-state-results/$EXP_NAME"
done

#!/bin/bash
# Shared final step for every MODEL-driven eval slurm: copy scratch results to store, only
# if the job succeeded. Never write to capstor from a compute node.
#
# Usage: source this file, then call copy_scratch_to_store <exit_status> <scratch_dir> <store_dir>
copy_scratch_to_store() {
    local status=$1
    local scratch_dir=$2
    local store_dir=$3
    if [ "$status" -eq 0 ]; then
        echo "Copying results to store: $store_dir"
        mkdir -p "$store_dir"
        cp -a "$scratch_dir/." "$store_dir/"
    else
        echo "Job failed (exit $status) -- not copying scratch results to store"
    fi
}

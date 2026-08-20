#!/bin/bash
# Shared final step for every MODEL-driven eval slurm: copy scratch results to store, only
# if the job succeeded. Never write to capstor from a compute node.
#
# Usage: source this file, then call copy_scratch_to_store <exit_status> <scratch_dir> <store_dir> [<glob_pattern>]
# <glob_pattern>, if given, copies only entries directly under scratch_dir matching it
# (e.g. "offset_0_prefix_500_suffix_*") instead of the whole scratch_dir.
copy_scratch_to_store() {
    local status=$1
    local scratch_dir=$2
    local store_dir=$3
    local pattern=$4
    if [ "$status" -ne 0 ]; then
        echo "Job failed (exit $status) -- not copying scratch results to store"
        return
    fi
    echo "Copying results to store: $store_dir"
    mkdir -p "$store_dir"
    if [ -z "$pattern" ]; then
        if [ -d "$scratch_dir" ]; then
            cp -a "$scratch_dir/." "$store_dir/"
        fi
    else
        shopt -s nullglob
        local matches=("$scratch_dir"/$pattern)
        shopt -u nullglob
        if [ ${#matches[@]} -gt 0 ]; then
            cp -a "${matches[@]}" "$store_dir/"
        fi
    fi
}

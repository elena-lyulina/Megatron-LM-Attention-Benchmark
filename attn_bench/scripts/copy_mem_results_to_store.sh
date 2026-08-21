#!/bin/bash
set -euo pipefail

# Manually copy mem-results from scratch to store, scoped per experiment. measure_mem.slurm
# and friends write scratch only (skip-checks look at both, see --persistent-storage-path);
# run this yourself whenever you want an experiment's results promoted to store, e.g. before
# scratch's ~biweekly purge. Run from a login node, never from inside a compute job.
#
# Usage: bash copy_mem_results_to_store.sh exp1 [exp2 ...]

SCRATCH_BASE=/iopsstor/scratch/cscs/$USER/mem-results/SparseGutenberg
STORE_BASE=/users/$USER/store/mem-results/SparseGutenberg

if [ $# -eq 0 ]; then
    echo "Usage: $0 exp1 [exp2 ...]"
    exit 1
fi

for EXP in "$@"; do
    echo "Copying $EXP"
    mkdir -p "$STORE_BASE/$EXP"
    cp -a "$SCRATCH_BASE/$EXP/." "$STORE_BASE/$EXP/"
done

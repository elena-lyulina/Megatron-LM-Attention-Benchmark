#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=dataset-doc-stats
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:10:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Compute per-document token-length distributions for the tokenized datasets.
# Reads only the .idx files (no .bin, no GPU) and saves one raw-lengths .npy per
# dataset for plotting in attn_bench/notebooks/doc_lengths.ipynb.
#
# Env vars:
#   DATASET_DIRS   optional; space-separated dataset dirs. Unset falls back to the .py
#                  script's own DEFAULT_DATASETS (always run, no store check, in that case).
#   OVERWRITE      recompute even if store already has a dataset's <name>.npy
#
# Usage:
#   cd attn_bench && sbatch submissions/dataset_doc_stats.slurm
#   cd attn_bench && sbatch --export=DATASET_DIRS="/path/to/new_dataset" submissions/dataset_doc_stats.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
STORE_DIR=/users/$USER/store/datasets/analysis/doc-lengths
SCRATCH_DIR=/iopsstor/scratch/cscs/$USER/datasets/analysis/doc-lengths

DATASET_DIRS=${DATASET_DIRS:-}
OVERWRITE=${OVERWRITE:-}

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR
source attn_bench/scripts/scratch_to_store.sh
mkdir -p $SCRATCH_DIR

### CHECK STORE (only possible when DATASET_DIRS is explicit -- the .py's own
### DEFAULT_DATASETS aren't known here, so that fallback path always runs) ###

if [ -n "$DATASET_DIRS" ] && [ -z "$OVERWRITE" ]; then
    TODO=()
    for d in $DATASET_DIRS; do
        name=$(basename "${d%/}")
        if [ -f "$STORE_DIR/$name.npy" ]; then
            echo "$name.npy already present in store -- skipping."
        else
            TODO+=("$d")
        fi
    done
    if [ ${#TODO[@]} -eq 0 ]; then
        echo "All requested datasets already present in store -- nothing to do."
        exit 0
    fi
    DATASET_DIRS="${TODO[*]}"
fi

### COMPUTE (on scratch) ###

python3 -u attn_bench/data_processing/dataset_doc_stats.py \
    ${DATASET_DIRS:+--datasets $DATASET_DIRS} \
    --out-dir $SCRATCH_DIR
STATUS=$?

### Copy scratch results to store ###

copy_scratch_to_store $STATUS "$SCRATCH_DIR" "$STORE_DIR"

echo "END TIME: $(date)"
echo "Done. Length arrays in $STORE_DIR"
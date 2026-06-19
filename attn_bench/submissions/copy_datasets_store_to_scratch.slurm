#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=01:00:00
#SBATCH --job-name=copy-datasets-store-to-scratch
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Copy the tokenized datasets used by the pretraining slurms from permanent store
# (/users/$USER/store, slower IO) to scratch (/iopsstor, faster IO) where training reads them.
# rsync -a preserves the nested folder structure (e.g. fineweb .../dump_0, .../dump_1) and
# copies both .bin and .idx files. Resumable: re-run to finish an interrupted transfer.
#
# Usage:
#   sbatch attn_bench/submissions/copy_datasets_store_to_scratch.slurm

echo "START TIME: $(date)"

### PATHS ###
STORE_DIR=/users/$USER/store/datasets/tokenized
SCRATCH_DIR=/users/$USER/scratch/datasets/tokenized/

DATASETS=(
    fineweb-edu-dedup-160B-datatrove_0.25
    gutenberg_rep_1_256
)

mkdir -p $SCRATCH_DIR

### COPY ###
for ds in "${DATASETS[@]}"; do
    echo "[$(date)] rsync $STORE_DIR/$ds -> $SCRATCH_DIR/"
    rsync -a --info=progress2 "$STORE_DIR/$ds" "$SCRATCH_DIR/"
done

### VERIFY ###
# compare .bin/.idx counts between source and destination for each dataset
for ds in "${DATASETS[@]}"; do
    src=$(find -L "$STORE_DIR/$ds" -type f \( -name '*.bin' -o -name '*.idx' \) | wc -l)
    dst=$(find -L "$SCRATCH_DIR/$ds" -type f \( -name '*.bin' -o -name '*.idx' \) | wc -l)
    echo "[$(date)] $ds: store=$src scratch=$dst .bin/.idx files $([ "$src" = "$dst" ] && echo OK || echo MISMATCH)"
done

echo "END TIME: $(date)"
echo "Done. Datasets in $SCRATCH_DIR"
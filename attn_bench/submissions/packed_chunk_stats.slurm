#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=packed-chunk-stats
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Packed-segment ("chunk") length histogram for one or more tokenized datasets, one full
# epoch each (documents split at both document and sample-window boundaries) -- an
# intrinsic property of each dataset + seq_len, independent of any specific training run's
# blend, so results are cached per-dataset and reused across whatever blend it appears in
# next. CPU-only, no GPU needed. Checks store per-dataset, writes to scratch, copies to
# store on success.
#
# Env vars:
#   DATASET_DIRS   required; space-separated dataset dirs, one histogram per dir
#   SEQ_LEN        default: 8192
#   TOKENIZER_PATH default: /iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
#   CACHE_PATH     default: /iopsstor/scratch/cscs/$USER/datasets/cache
#   SEED           default: 28
#   OVERWRITE      recompute even if a dataset's store dir already has results
#
# Usage:
#   cd attn_bench && sbatch --export=DATASET_DIRS="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25 /iopsstor/scratch/cscs/$USER/datasets/tokenized/gutenberg_rep_1_256" \
#     submissions/packed_chunk_stats.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
STORE_DIR=/users/$USER/store/datasets/packed_chunk_lengths
SCRATCH_DIR=/iopsstor/scratch/cscs/$USER/datasets/packed_chunk_lengths

DATASET_DIRS=${DATASET_DIRS:?ERROR: DATASET_DIRS must be set (space-separated dataset dirs)}
SEQ_LEN=${SEQ_LEN:-8192}
TOKENIZER_PATH=${TOKENIZER_PATH:-/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b}
CACHE_PATH=${CACHE_PATH:-/iopsstor/scratch/cscs/$USER/datasets/cache}
SEED=${SEED:-28}
OVERWRITE=${OVERWRITE:-}

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR
source attn_bench/scripts/scratch_to_store.sh
mkdir -p $SCRATCH_DIR

### COMPUTE (one dataset at a time, on scratch) ###

for DATASET_DIR in $DATASET_DIRS; do
    NAME=$(basename "${DATASET_DIR%/}")

    if [ -z "$OVERWRITE" ] && [ -f "$STORE_DIR/${NAME}_run_metadata.json" ]; then
        echo "$NAME already present in store ($STORE_DIR) -- skipping."
        continue
    fi

    python3 -u attn_bench/data_processing/packed_chunk_stats.py \
        --dataset-dir $DATASET_DIR \
        --tokenizer-path $TOKENIZER_PATH \
        --cache-path $CACHE_PATH \
        --seq-len $SEQ_LEN \
        --seed $SEED \
        --out-dir $SCRATCH_DIR
    STATUS=$?

    copy_scratch_to_store $STATUS "$SCRATCH_DIR" "$STORE_DIR"
done

echo "END TIME: $(date)"
echo "Done. Results in $STORE_DIR/"

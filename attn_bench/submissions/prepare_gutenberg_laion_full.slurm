#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=prepare-gutenberg-laion-full
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=2:00:00
#SBATCH --environment=nemo
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Run the full 20-step Gutenberg LAION processing pipeline on the full dataset,
# including perplexity + Min-K%++ scoring (step 20).
#
# Usage:
#   sbatch attn_bench/submissions/prepare_gutenberg_laion_full.slurm
#
#   TO_STEP=13 sbatch attn_bench/submissions/prepare_gutenberg_laion_full.slurm
#     stop after the given step, e.g. 13 to regenerate stats and match files up to
#     content deduplication without running excerpt sampling or scoring
#
#   DATASET_DIR=/users/$USER/store/datasets/raw/gutenberg-laion/raw-hf-full sbatch ...
#     read the raw dataset from somewhere other than scratch, e.g. store after a
#     scratch purge, avoiding a copy back

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
DATASET_DIR=${DATASET_DIR:-/iopsstor/scratch/cscs/$USER/datasets/raw/gutenberg-laion/raw-hf-full}
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output
MEGATRON_CKPT_DIR=/users/$USER/store/pretrain-results/llama3-1b-fineweb160B/checkpoints

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MASTER_ADDR=localhost
export MASTER_PORT=8888
cd $MEGATRON_DIR


### INSTALL DEPS ###

# transformers>=5 drops the "tf" backend that sentence-transformers still references,
# breaking the step-5 title-embedding import. The attn_bench[data] extra is unpinned, so
# a fresh resolve picks up whatever is newest on PyPI; pin to what the container expects.
pip install --quiet -e "$MEGATRON_DIR/attn_bench[data]" "transformers<5" "numpy<2"


### PIPELINE ###

python3 -u -m attn_bench.data_processing.books.main_gutenberg_laion \
    --dataset-dir    $DATASET_DIR \
    --output-dir     $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --checkpoint \
    --megatron-ckpt-dir $MEGATRON_CKPT_DIR \
    ${TO_STEP:+--to-step $TO_STEP} \
    --num-workers    $SLURM_CPUS_PER_TASK


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
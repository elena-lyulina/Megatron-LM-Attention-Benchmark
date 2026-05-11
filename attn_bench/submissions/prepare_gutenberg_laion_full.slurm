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

# Run the 19-step Gutenberg LAION processing pipeline on the full dataset.
#
# Usage:
#   sbatch attn_bench/submissions/prepare_gutenberg_laion_full.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
DATASET_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/gutenberg-laion/raw-hf-full
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### PIPELINE ###

python3 -m attn_bench.data_processing.books.main_gutenberg_laion \
    --dataset-dir    $DATASET_DIR \
    --output-dir     $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --checkpoint \
    --num-workers    $SLURM_CPUS_PER_TASK


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
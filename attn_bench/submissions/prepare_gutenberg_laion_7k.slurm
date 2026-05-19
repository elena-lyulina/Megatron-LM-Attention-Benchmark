#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=prepare-gutenberg-laion-7k
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=1:30:00
#SBATCH --environment=nemo
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Run the full 20-step Gutenberg LAION processing pipeline on the 7k sample,
# including perplexity + Min-K%++ scoring (step 20).
#
# Usage:
#   sbatch attn_bench/submissions/prepare_gutenberg_laion_7k.slurm

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
DATASET_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/gutenberg-laion/raw-hf-7000
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-7000-output
MEGATRON_CKPT_DIR=/users/$USER/store/pretrain-results/llama3-1b-fineweb160B/checkpoints

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MASTER_ADDR=localhost
export MASTER_PORT=8888
cd $MEGATRON_DIR


### INSTALL DEPS ###

pip install --quiet -e "$MEGATRON_DIR/attn_bench[data]"


### PIPELINE ###

python3 -u -m attn_bench.data_processing.books.main_gutenberg_laion \
    --dataset-dir    $DATASET_DIR \
    --output-dir     $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --checkpoint \
    --megatron-ckpt-dir $MEGATRON_CKPT_DIR \
    --num-workers    $SLURM_CPUS_PER_TASK


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
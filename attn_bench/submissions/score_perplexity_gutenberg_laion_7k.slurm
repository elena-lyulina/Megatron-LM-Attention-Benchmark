#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=score-gutenberg-laion-7k
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:15:00
#SBATCH --environment=nemo
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Score perplexity and Min-K%++ for the 7k Gutenberg LAION sample (step 20).
# Resumes from the step-19 checkpoint; both metrics computed in one forward pass.
#
# Usage:
#   sbatch attn_bench/submissions/score_perplexity_gutenberg_laion_7k.slurm

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


### STEP 20: PERPLEXITY + MIN-K%++ SCORING ###

python3 -u -m attn_bench.data_processing.books.main_gutenberg_laion \
    --dataset-dir    $DATASET_DIR \
    --output-dir     $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --checkpoint \
    --megatron-ckpt-dir $MEGATRON_CKPT_DIR \
    --num-workers    $SLURM_CPUS_PER_TASK


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
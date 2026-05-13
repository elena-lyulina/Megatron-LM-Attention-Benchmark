#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=score-perplexity-gutenberg-full
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Score perplexity for the full Gutenberg LAION dataset (step 20).
# Resumes from the step-19 checkpoint and runs only step 20.
# TP=1, DP=8: each GPU scores an independent slice of books, results gathered to rank 0.
#
# Usage:
#   sbatch attn_bench/submissions/score_perplexity_gutenberg_laion_full.slurm

CONTAINER_ENV=nemo

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
DATASET_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/gutenberg-laion/raw-hf-full
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/gutenberg-laion/hf-full-output
MEGATRON_CKPT_DIR=/users/$USER/store/pretrain-results/llama3-1b-fineweb160B/checkpoints

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export WORLD_SIZE=$SLURM_NTASKS


### INSTALL DEPS ###

srun --ntasks=1 --environment=$CONTAINER_ENV \
    pip install --quiet -e "$MEGATRON_DIR/attn_bench[data]"


### STEP 20: PERPLEXITY SCORING ###

srun \
    --environment=$CONTAINER_ENV \
    bash -c "
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
python3 -m attn_bench.data_processing.books.main_gutenberg_laion \
    --dataset-dir    $DATASET_DIR \
    --output-dir     $OUTPUT_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --checkpoint \
    --megatron-ckpt-dir $MEGATRON_CKPT_DIR \
    --num-workers    $SLURM_CPUS_PER_TASK"


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
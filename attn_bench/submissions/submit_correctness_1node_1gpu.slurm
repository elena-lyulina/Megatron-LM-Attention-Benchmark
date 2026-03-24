#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=attn-correctness-1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=200GB
#SBATCH --time=00:10:00
#SBATCH --environment=nemo # in my ~/.edf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/correctness/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/correctness/%x_%j.err
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure)


# 1-node correctness run
# Parallelism env vars are set here, attn and impl are passed at submission time.
# All the other params (model configuration, training params, etc) are set in scripts/run_correctness.sh

# Usage:
#   sbatch submissions/submit_correctness_1node_1gpu.slurm full flash

export TP_SIZE=1
export PP_SIZE=1

ATTN="${1:?usage: sbatch submit_correctness_1node.slurm <attn> <impl>}"
IMPL="${2:?usage: sbatch submit_correctness_1node.slurm <attn> <impl>}"

# W&B
export LOG_WANDB=true  # Set to false to disable W&B logging
export WANDB_PROJECT="Megatron-LM-Attention-Benchmark"
export WANDB_EXP_NAME="correctness_${ATTN}_${IMPL}_1node_1gpu_tp${TP_SIZE}_pp${PP_SIZE}_${SLURM_JOB_ID}"

# create log directory if it doesn't exist
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/logs/correctness/

# passing attention params to run_correctness.sh
bash /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/scripts/run_correctness.sh "$@"
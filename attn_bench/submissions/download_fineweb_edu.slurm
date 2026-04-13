#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=download-fineweb-edu
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue	# Prevent Slurm to requeue the job if the execution crashes (e.g. node failure)

echo "START TIME: $(date)"

export HF_TOKEN=$(cat $HOME/.hf-token)

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
RAW_DIR=/iopsstor/scratch/cscs/elyulina/datasets/dedup-fineweb-edu/raw

mkdir -p $RAW_DIR
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

python3 $MEGATRON_DIR/attn_bench/data_processing/download_fineweb_edu.py \
    --output-dir $RAW_DIR

echo "END TIME: $(date)"
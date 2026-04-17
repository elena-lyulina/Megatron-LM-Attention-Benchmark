#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=prepare-dumps-fineweb-edu
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
RAW_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/fineweb-edu-dedup
PATH_TO_PREPROCESSING_METADATA=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/fineweb-edu-dedup
NUMBER_OF_DUMPS=2

python3 $MEGATRON_DIR/attn_bench/data_processing/tokenization/prepare_dumps.py \
    --dataset-folder $RAW_DIR \
    --preprocessing-metadata-folder $PATH_TO_PREPROCESSING_METADATA \
    --n-dumps $NUMBER_OF_DUMPS
#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=tokenize-fineweb-edu-datatrove
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:00:00
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --no-requeue

echo "START TIME: $(date)"

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b
DATASETS_PATH=/iopsstor/scratch/cscs/elyulina/datasets
RAW_DIR=$DATASETS_PATH/raw/fineweb-edu-dedup
OUTPUT_DIR=$DATASETS_PATH/tokenized/fineweb-edu-dedup-160B-datatrove

mkdir -p $OUTPUT_DIR
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

export PYTHONPATH=$MEGATRON_DIR:$PYTHONPATH

TOKEN_BUDGET=160000000000   # 160B tokens
NUM_WORKERS=$SLURM_CPUS_PER_TASK   # 128 workers = 128 shards

pip install datatrove --quiet

python3 $MEGATRON_DIR/attn_bench/data_processing/tokenize_fineweb_edu_datatrove.py \
    --tokenizer-path $TOKENIZER_PATH \
    --raw-dir $RAW_DIR \
    --output-dir $OUTPUT_DIR \
    --token-budget $TOKEN_BUDGET \
    --num-workers $NUM_WORKERS

echo "END TIME: $(date)"
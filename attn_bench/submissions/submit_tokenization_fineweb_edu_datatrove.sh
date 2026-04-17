#!/bin/bash

NUMBER_OF_DATATROVE_TASKS=28
TOKEN_BUDGET=160000000000          # 160B tokens total

TOKENIZER_NAME=llama-3.2-1b
DATASET_NAME=fineweb-edu-dedup

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/$TOKENIZER_NAME
RAW_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/$DATASET_NAME
PATH_TO_PREPROCESSING_METADATA=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/$DATASET_NAME
PATH_TO_DATATROVE_LOGGING_DIR=/iopsstor/scratch/cscs/$USER/datasets/logs/datatrove/$DATASET_NAME
PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_DIR/attn_bench/logs
PATH_TO_OUTPUT_FOLDER=/iopsstor/scratch/cscs/$USER/datasets/tokenized/$DATASET_NAME-160B-datatrove

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$DATASET_NAME.csv

mkdir -p $PATH_TO_OUTPUT_FOLDER
mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed_dumps

ln -sfn $PATH_TO_OUTPUT_FOLDER $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" > $CSV_RESULTS_FILE
# Calculate node budget from number of dumps
NUMBER_OF_DUMPS=$(ls "$PATH_TO_PREPROCESSING_METADATA/dumps"/*.txt | wc -l)
NODE_BUDGET=$((TOKEN_BUDGET / NUMBER_OF_DUMPS))

# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*.txt; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<< $paths_file)
  output_folder=$PATH_TO_OUTPUT_FOLDER/dump_$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/dump_$dump
  sbatch --job-name=tokenize-$DATASET_NAME --output=$PATH_TO_SLURM_LOGGING_DIR/%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/%j.err $MEGATRON_DIR/attn_bench/submissions/tokenize_fineweb_edu_datatrove.slurm $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER_PATH $logging_dir $CSV_RESULTS_FILE $paths_file $NUMBER_OF_DATATROVE_TASKS $MEGATRON_DIR $NODE_BUDGET
done
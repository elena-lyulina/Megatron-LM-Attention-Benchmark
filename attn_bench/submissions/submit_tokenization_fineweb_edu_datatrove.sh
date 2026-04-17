#!/bin/bash

NUMBER_OF_DATATROVE_TASKS=28
BATCH_SIZE=10000
TOKEN_BUDGET=160000000000          # 160B tokens total
#TOKEN_BUDGET=1000000000          # 1B tokens total

TOKENIZER_NAME=llama-3.2-1B
DATASET_NAME=fineweb-edu-dedup
TOKENIZED_DATASET_NAME=$DATASET_NAME-160B

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=meta-llama/Llama-3.2-1B
RAW_DIR=/iopsstor/scratch/cscs/$USER/datasets/raw/$DATASET_NAME
PATH_TO_PREPROCESSING_METADATA=/iopsstor/scratch/cscs/$USER/datasets/preprocessing/$DATASET_NAME
PATH_TO_DATATROVE_LOGGING_DIR=/iopsstor/scratch/cscs/$USER/datasets/logs/datatrove/$TOKENIZED_DATASET_NAME
PATH_TO_SLURM_LOGGING_DIR=$MEGATRON_DIR/attn_bench/logs
PATH_TO_OUTPUT_FOLDER=/iopsstor/scratch/cscs/$USER/datasets/tokenized/$TOKENIZED_DATASET_NAME-datatrove

CSV_RESULTS_FILE=$PATH_TO_PREPROCESSING_METADATA/tokenize-$TOKENIZER_NAME-$TOKENIZED_DATASET_NAME.csv

mkdir -p $PATH_TO_OUTPUT_FOLDER
mkdir -p $PATH_TO_SLURM_LOGGING_DIR
mkdir -p $PATH_TO_PREPROCESSING_METADATA/completed_dumps

ln -sfn $PATH_TO_OUTPUT_FOLDER $PATH_TO_PREPROCESSING_METADATA/tokenized-dir-link

# Write header only if file doesn't exist yet
if [ ! -f $CSV_RESULTS_FILE ]; then
  echo "slurm_job_id,node,start,end,paths_file,output_folder,dataset_total_size,processed_total_size,number_of_workers_per_node,batch_size,token_budget,time,bw,total_tokens_processed,throughput (Million Tokens/Second/Node)" > $CSV_RESULTS_FILE
fi
# Move dumps back from completed_dumps if dumps folder is empty
if [ -z "$(ls "$PATH_TO_PREPROCESSING_METADATA/dumps"/*.txt 2>/dev/null)" ]; then
  echo "No dumps found, moving back from completed_dumps..."
  mv "$PATH_TO_PREPROCESSING_METADATA/completed_dumps"/*.txt "$PATH_TO_PREPROCESSING_METADATA/dumps"/
fi

# Calculate node budget from number of dumps
NUMBER_OF_DUMPS=$(ls "$PATH_TO_PREPROCESSING_METADATA/dumps"/*.txt | wc -l)
NODE_BUDGET=$((TOKEN_BUDGET / NUMBER_OF_DUMPS))

# Iterate through all dumps paths files
for paths_file in "$PATH_TO_PREPROCESSING_METADATA/dumps"/*.txt; do
  dump=$(grep -oP '(?<=paths_file_)\d+(?=\.txt)' <<< $paths_file)
  output_folder=$PATH_TO_OUTPUT_FOLDER/dump_$dump
  logging_dir=$PATH_TO_DATATROVE_LOGGING_DIR/dump_$dump
  sbatch --job-name=tokenize-$DATASET_NAME --output=$PATH_TO_SLURM_LOGGING_DIR/%j.out --error=$PATH_TO_SLURM_LOGGING_DIR/%j.err $MEGATRON_DIR/attn_bench/submissions/tokenize_fineweb_edu_datatrove.slurm $PATH_TO_PREPROCESSING_METADATA/raw-dataset-link $output_folder $TOKENIZER_PATH $logging_dir $CSV_RESULTS_FILE $paths_file $NUMBER_OF_DATATROVE_TASKS $MEGATRON_DIR $NODE_BUDGET $BATCH_SIZE
done
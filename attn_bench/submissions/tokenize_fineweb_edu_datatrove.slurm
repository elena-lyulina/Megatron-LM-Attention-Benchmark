#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --environment=nemo
#SBATCH --partition=normal
#SBATCH --mem=800G  # fineweb edu parquet files are big (2.4G) so need a lot of memory for all workers
#SBATCH --no-requeue

input_folder=$1
output_folder=$2
tokenizer=$3
logging_dir=$4
CSV_RESULTS_FILE=$5
paths_file=$6
number_of_tasks=$7
MEGATRON_DIR=$8
node_budget=$9
batch_size=${10}
continue_run=${11:-false}

# Setup ENV
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_TOKEN=$(cat $HOME/.hf-token)
export PYTHONPATH=$MEGATRON_DIR:$PYTHONPATH
# Setup directories
rm -rf $output_folder
if [ "$continue_run" != "true" ]; then
  rm -rf $logging_dir
fi
mkdir -p $output_folder

pip install datatrove --quiet

echo "START TIME: $(date) | Preprocessing $paths_file with $number_of_tasks tasks per node with the $tokenizer tokenizer. Node budget: $node_budget tokens. Storing tokenized dataset in $output_folder"
start_s=`date`
start=`date +%s`
numactl --membind=0-3 python3 $MEGATRON_DIR/attn_bench/data_processing/tokenization/preprocess_megatron_budgeted.py --tokenizer-name-or-path $tokenizer --output-folder $output_folder --logging-dir $logging_dir --n-tasks $number_of_tasks --dataset $input_folder --paths-file $paths_file --node-budget $node_budget --batch-size $batch_size
end=`date +%s`
end_s=`date`
echo "FINISH TIME: $(date) | Preprocessed $paths_file ! Stored in $output_folder"

# Stats
wc=$((end-start))
dataset_total_size=$(python3 $MEGATRON_DIR/attn_bench/data_processing/tokenization/compute_dump_size.py $paths_file)
processed_total_size=$(du -shLb $output_folder | cut -f1)
bw=$(python3 -c "print($dataset_total_size/$wc)")
total_tokens_processed=$(($(du -shLcb $output_folder/*.bin | tail -n1 | sed -r 's/([^0-9]*([0-9]*)){1}.*/\2/')/4))
throughput=$(python3 -c "print($total_tokens_processed/$wc)")
echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$number_of_tasks,$batch_size,$node_budget,$wc,$bw,$total_tokens_processed,$throughput"
echo "$SLURM_JOB_ID,$(hostname),$start_s,$end_s,$paths_file,$output_folder,$dataset_total_size,$processed_total_size,$number_of_tasks,$batch_size,$node_budget,$wc,$bw,$total_tokens_processed,$throughput" >> $CSV_RESULTS_FILE

sleep 10
ls -lS $output_folder
mv $paths_file $(dirname $(dirname $paths_file))/completed_dumps
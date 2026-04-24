#!/bin/bash

#SBATCH --account=infra01
#SBATCH --job-name=verify-tokenization
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --environment=nemo
#SBATCH --partition=normal

echo "START TIME: $(date)"

MEGATRON_DIR=/iopsstor/scratch/cscs/elyulina/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/elyulina/tokenizers/llama-3.2-1b
TOKENIZED_DIR=/iopsstor/scratch/cscs/elyulina/datasets/tokenized/fineweb-edu-dedup-160B-datatrove

mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs

echo "PYTHONPATH before: ${PYTHONPATH:-unset}"
export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
echo "PYTHONPATH after:  $PYTHONPATH"
python3 -c "
import megatron
print('megatron __path__:', list(megatron.__path__))
import megatron.core
print('megatron.core __path__:', list(megatron.core.__path__))
import megatron.core.datasets
print('megatron.core.datasets __path__:', list(megatron.core.datasets.__path__))
" 2>&1 || echo "import chain broken above"

python3 $MEGATRON_DIR/attn_bench/data_processing/tokenization/verify_tokenization.py \
    --tokenized-dir $TOKENIZED_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --num-sequences 5 \
    --num-workers 20

echo "END TIME: $(date)"
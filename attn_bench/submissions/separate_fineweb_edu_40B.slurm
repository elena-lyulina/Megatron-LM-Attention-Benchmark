#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=04:00:00
#SBATCH --job-name=separate-fineweb-40B
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Split FineWeb-Edu-dedup-160B into a 40B partition (ratio 0.25) and 120B remainder (ratio 0.75).
# Documents are randomly assigned per shard using Megatron's greedy blending sampler.
#
# Output (written to datasets/tokenized/ alongside the original):
#   fineweb-edu-dedup-160B-datatrove_0.25/  ← 40B partition, use this for training
#   fineweb-edu-dedup-160B-datatrove_0.75/  ← 120B remainder, can be deleted
#
# Usage:
#   sbatch attn_bench/submissions/separate_fineweb_edu_40B.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

INPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized

mkdir -p $MEGATRON_LM_DIR/attn_bench/logs

echo "Splitting: $INPUT_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Ratios:    0.25 (40B) + 0.75 (120B)"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    python3 $MEGATRON_LM_DIR/attn_bench/utils/tools/separate_binary.py \
        --input-dir $INPUT_DIR \
        --output-dir $OUTPUT_DIR \
        --ratios 0.25 0.75 \
        --seed 42 \
        --workers $SLURM_CPUS_PER_TASK"

echo "END TIME: $(date)"
echo "40B partition: ${OUTPUT_DIR}/fineweb-edu-dedup-160B-datatrove_0.25"
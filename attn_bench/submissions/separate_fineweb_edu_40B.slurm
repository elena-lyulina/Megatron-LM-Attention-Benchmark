#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:15:00
#SBATCH --job-name=separate-fineweb-40B-re
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Regenerate the FineWeb-Edu-dedup-160B 0.25/0.75 split (same input, ratios, and
# seed as the original run) so we can recover the 0.75 remainder, which was
# cleaned up after being unused. The live 0.25 partition (fineweb-edu-dedup-160B-datatrove_0.25
# under datasets/tokenized/) is currently used by in-progress training runs, so
# this writes to a separate tokenized_re/ directory instead of overwriting it.
# separate_binary.py is deterministic given the same input dir, ratios, and seed
# (see attn_bench/utils/tools/separate_binary.py docstring), so the regenerated
# 0.25 here should be byte-identical to the live one — verified by
# verify_and_split_fineweb_edu_new_25.slurm before it touches anything.
#
# Output (written to datasets/tokenized_re/, NOT alongside the live split):
#   fineweb-edu-dedup-160B-datatrove_0.25/  ← throwaway, only used to verify determinism
#   fineweb-edu-dedup-160B-datatrove_0.75/  ← the 120B remainder we need next
#
# Usage:
#   sbatch attn_bench/submissions/separate_fineweb_edu_40B.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

INPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized_re

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

if [ $? -ne 0 ]; then
    echo "FAIL: separate_binary.py crashed (see .err log for the traceback) — output in $OUTPUT_DIR is incomplete/missing."
    exit 1
fi

echo "END TIME: $(date)"
echo "Regenerated 0.25 (verify only): ${OUTPUT_DIR}/fineweb-edu-dedup-160B-datatrove_0.25"
echo "Regenerated 0.75 (next input):  ${OUTPUT_DIR}/fineweb-edu-dedup-160B-datatrove_0.75"
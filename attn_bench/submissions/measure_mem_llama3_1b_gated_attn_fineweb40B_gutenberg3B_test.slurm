#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:05:00
#SBATCH --job-name=measure-mem-gated-attn-test
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# End-to-end pipeline test: 50 samples from rep=0 of gated-attn model.
# Tests model loading, inference, and PDM metric computation.
#
# Usage:
#   sbatch attn_bench/submissions/measure_mem_llama3_1b_gated_attn_fineweb40B_gutenberg3B_test.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B-test
MODEL_DIR=/users/elyulina/store/pretrain-results/llama3-1b-gated-attn-fineweb40B-gutenberg3B
MEM_BASE=/users/elyulina/store/mem-results
MEM_DIR=$MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1


### STEP 1: sparse Gutenberg inference (50 samples, rep=0 only) ###

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun \
        --nproc_per_node=4 \
        $MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_sparse.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0 \
        --offset 0 \
        --prefix-length 500 \
        --suffix-length 500 \
        --batch-size 20 \
        --max-samples 50 \
        --megatron-extra-args --attention-output-gate"


### STEP 2: compute memorization metrics ###

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    python $PDM_DIR/src/verbatim_eval/main.py \
        --exprs $EXP_NAME \
        --mode sparse \
        --base-path $MEM_BASE \
        --save-path $MEM_BASE/SparseGutenberg \
        --offsets 0 \
        --prefix-lengths 500 \
        --suffix-lengths 500"

echo "END TIME: $(date)"

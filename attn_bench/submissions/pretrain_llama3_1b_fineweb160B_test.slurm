#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=train-llama1B-fineweb-test
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=01:00:00
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --no-requeue

# Smoke test for pretrain_llama3_1b_fineweb160B.slurm
# Checks:
#   - checkpointing saves and resumes correctly
#   - env vars are not silently overridden by system defaults (compare env_before.txt vs compute_environment.txt)
#   - training speed (tokens/s logged by Megatron)
#
# Usage:
#   sbatch attn_bench/submissions/pretrain_llama3_1b_fineweb160B_test.slurm

echo "START TIME: $(date)"


####### PATHS & EXPERIMENT #######
REPO_ROOT=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}

PROJECT_NAME="fineweb-160B"
PROJECT_DIR=$REPO_ROOT/attn_bench/results/pretrain/$PROJECT_NAME
EXP_NAME=llama3-1b-fineweb160B-test
EXP_DIR=$PROJECT_DIR/$EXP_NAME
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID

mkdir -p $PROJECT_DIR $EXP_DIR $DEBUG_DIR

printenv > $DEBUG_DIR/env_before.txt
###################################


####### MODEL #####################
source $REPO_ROOT/attn_bench/configs/models/llama3_1b.sh
###################################


####### DATA ######################
SEQ_LEN=8192

BINARY_DATASET_DIR="/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B"
DATA_PATH=$(python3 $REPO_ROOT/attn_bench/utils/tools/create_data_config.py -p $BINARY_DATASET_DIR)

DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache
mkdir -p $DATASET_CACHE_DIR

DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --num-workers 2
    --num-dataset-builder-threads 1
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
    --data-path $DATA_PATH
    --data-cache-path $DATASET_CACHE_DIR
)
###################################


####### TRAINING ##################
MBS=2
GBS=80  # must be divisible by MBS * num_gpus (2 * 40 = 80); Gradient Accumulation Steps (GAS) = 1

TRAINING_STEPS=500

# <--override any variables before sourcing-->
# overrides CHECKPOINT_STEPS from configs/training.sh
CHECKPOINT_STEPS=100

source $REPO_ROOT/attn_bench/configs/training.sh

TRAINING_ARGS+=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
)
###################################


####### DISTRIBUTED ###############
source $REPO_ROOT/attn_bench/configs/distributed.sh
###################################


####### LOGGING ###################
source $REPO_ROOT/attn_bench/configs/logging.sh
###################################


####### Launching Training ########
TRAINING_CMD="torchrun ${TORCHRUN_ARGS[*]} $REPO_ROOT/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[*]} \
    ${NETWORK_SIZE_ARGS[*]} \
    ${VOCAB_ARGS[*]} \
    ${REGULARIZATION_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${INITIALIZATION_ARGS[*]} \
    ${LEARNING_RATE_ARGS[*]} \
    ${CHECKPOINTING_ARGS[*]} \
    ${MIXED_PRECISION_ARGS[*]} \
    ${DISTRIBUTED_ARGS[*]} \
    ${TOKENIZER_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    ${DATA_ARGS[*]}"

setup_wandb
setup_profiler

cp $0 $DEBUG_DIR
log_compute_env

echo "Training: fineweb-llama-test nodes=${SLURM_NNODES} gpus_per_node=${SLURM_GPUS_PER_NODE} gbs=${GBS} steps=${TRAINING_STEPS}"
echo "[$(date)] Launching training..."

srun -lu --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "export PYTHONPATH='${PYTHONPATH}'; ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"
###################################
#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=train-llama1B-fineweb
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --time=24:00:00
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --no-requeue

# Train a standard LLaMA 3.2 1B model on FineWeb Edu (no custom attention).
# Used to obtain a model for perplexity-based dataset curation.
#
# Usage:
#   sbatch attn_bench/submissions/submit_pretrain_llama3_fineweb.slurm

echo "START TIME: $(date)"

################ Configs ################

FINEWEB_PREFIX="/capstor/store/cscs/swissai/infra01/users/xyixuan/finewebedu-sample-100BT/megatrontokenized/finewebedu-llama3tok_text_document"

MBS=2
GBS=80
SEQ_LEN=8192
# ~100B tokens: 100e9 / (80 * 8192) ≈ 152588
TRAINING_STEPS=152588
CHECKPOINT_STEPS=5000

#### Debugging ####
LOG_NCCL=false
NSYS_PROFILER=false
MOCK_DATA=false
###################

# Megatron source (WARNING: must be on iopsstor)
REPO_ROOT=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

# Logging
PROJECT_NAME=Llama-Fineweb
EXP_NAME=llama3-1b-fineweb-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz
PROJECT_DIR=$REPO_ROOT/attn_bench/results/train/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $CKPT_DIR $PROJECT_DIR $TRIGGER_DIR $DEBUG_DIR $LOGGING_DIR $TENSORBOARD_DIR
mkdir -p /iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark/attn_bench/logs/
mkdir -p $DATASET_CACHE_DIR

rm -f $TRIGGER_DIR/save $TRIGGER_DIR/exit

# Environment
cd $REPO_ROOT
export PYTHONPATH=$REPO_ROOT:${PYTHONPATH:-}
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=22
MASTER_ADDR=$(hostname)
MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))

srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > $GPU_MEM_LOGGING
ulimit -c 0

#### Megatron Args (Llama 3.2 1B) ####

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --rotary-base 500000
    --use-rope-scaling
    --rope-scaling-factor 32
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
    --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --no-check-for-nan-in-loss-and-grad
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --cross-entropy-loss-fusion
    --optimizer adam
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 5000
)

INITIALIZATION_ARGS=(
    --seed 28
    --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 0.0003
    --min-lr 0.00003
    --lr-decay-style cosine
    --lr-warmup-iters 2000
)

CHECKPOINTING_ARGS=(
    --save $CKPT_DIR
    --save-interval $CHECKPOINT_STEPS
    --load $CKPT_DIR
    --ckpt-format torch_dist
    --async-save
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --wgrad-deferral-limit 50
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model nvidia/OpenMath2-Llama3.1-8B
)

DATA_ARGS=(
    --split 100,0,0
    --seq-length $SEQ_LEN
    --num-workers 2
    --num-dataset-builder-threads 1
    --reset-position-ids
    --reset-attention-mask
    --eod-mask-loss
)

if [ "$MOCK_DATA" = true ]; then
    DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
    DATA_ARGS="${DATA_ARGS[@]} --data-path $FINEWEB_PREFIX --data-cache-path $DATASET_CACHE_DIR"
fi

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
    --log-dir $DEBUG_DIR
)

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[*]} $REPO_ROOT/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[*]} \
    ${NETWORK_SIZE_ARGS[*]} \
    ${REGULARIZATION_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${INITIALIZATION_ARGS[*]} \
    ${LEARNING_RATE_ARGS[*]} \
    ${CHECKPOINTING_ARGS[*]} \
    ${MIXED_PRECISION_ARGS[*]} \
    ${DISTRIBUTED_ARGS[*]} \
    ${TOKENIZER_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    $DATA_ARGS"

# W&B logging
LOG_WANDB="${LOG_WANDB:-false}"
if [ "${LOG_WANDB}" = "true" ]; then
    echo "[$(date)] W&B logging enabled."
    if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
        wandb sync "$LOGGING_DIR/wandb/latest-run"
    fi
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOGGING_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] W&B logging disabled."
fi

# NCCL debug
if [ "$LOG_NCCL" = true ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_FILE="$DEBUG_DIR/nccl_debug_${SLURM_PROCID}.log"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

cp $0 $DEBUG_DIR

# Log compute environment
SEP=$(printf '=%.0s' {1..100})
{
    echo -e "$(date)"
    echo -e "${SEP}"
    echo -e "\nCMD: ${CMD_PREFIX} ${TRAINING_CMD}"
    echo -e "${SEP}"
    echo -e "\nNODES: $(scontrol show hostnames ${SLURM_JOB_NODELIST})"
    echo -e "${SEP}"
    echo -e "\nRepo: ${REPO_ROOT} ($(git -C ${REPO_ROOT} rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\n$(pip list)"
    echo -e "${SEP}"
    echo -e "\n$(nvidia-smi)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(printenv)"
    echo -e "${SEP}"
} > "${COMPUTE_ENVIRONMENT_DIR}"

echo "Training: fineweb-llama nodes=${SLURM_NNODES} gpus_per_node=${SLURM_GPUS_PER_NODE} gbs=${GBS} steps=${TRAINING_STEPS}"

echo "[$(date)] Launching training..."
srun -lu --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "export PYTHONPATH='${PYTHONPATH}'; ${CMD_PREFIX} ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"
#!/bin/bash

# Runs the correctness benchmark with parallelism args set as env vars (see submissions/submit_correctness_1node_1gpu.slurm)
# All the default args about the model and the training are set here.

# Args:
#   $1  attention mechanism, e.g. full
#   $2  kernel implementation, e.g. flash, torch

# Usage:
#   bash attn_bench/scripts/run_correctness.sh full flash

set -euox pipefail # -e exit immediately on error, -u treat unset vars as errors, -o pipefail fail if any command in a pipe fails, -x print each command before executing it (for debugging)

echo "START TIME: $(date)"

ATTN="${1:?usage: run_correctness.sh <attn> <impl>}"
IMPL="${2:?usage: run_correctness.sh <attn> <impl>}"

##### Paths #####
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" # points at Megatron-LM-Attention-Benchmark. BASH_SOURCE[0] has the path of the current file, we just cd a few dirs up and get the repo abs path dynamically (works from everywhere)
ATTN_BENCH_ROOT="${REPO_ROOT}/attn_bench"
RESULTS_DIR="${ATTN_BENCH_ROOT}/results/correctness"
EXP_NAME="${WANDB_EXP_NAME:?WANDB_EXP_NAME must be set}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
TENSORBOARD_DIR="${EXP_DIR}/tensorboard"

# Set up directories
echo "[$(date)] Setting up directories..."
mkdir -p $EXP_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $TENSORBOARD_DIR
# w&b will create its folder automatically

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

##### Logging & Debugging #####
WANDB_PROJECT="${WANDB_PROJECT:?WANDB_PROJECT must be set}"

LOG_NCCL="${LOG_NCCL:-false}"  # Set to true to enable NCCL_DEBUG=INFO logging (one file per process)
if [ "${LOG_NCCL}" = "true" ]; then
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_FILE="${EXP_DIR}/nccl_debug_${SLURM_PROCID}.log"
fi


##### Environment #####
echo "[$(date)] Setting up environment..."
cd $REPO_ROOT
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -c 0  # Disable core dumps to avoid filling up scratch space

# Log initial GPU memory usage on each node before training starts
echo "[$(date)] Logging GPU memory usage..."
srun -l bash -c 'echo $(hostname) $(nvidia-smi | grep -o "|\s*[0-9]*MiB")' > "${GPU_MEM_LOGGING}"
echo "[$(date)] GPU memory logged to ${GPU_MEM_LOGGING}"


##### Distributed #####
export TORCH_NCCL_AVOID_RECORD_STREAMS=1  # reduce GPU memory fragmentation during NCCL communication
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # surface NCCL errors immediately instead of hanging
export CUDA_DEVICE_MAX_CONNECTIONS=1       # limit CUDA kernel launch queues to improve gradient communication overlap
export NCCL_IB_TIMEOUT=22                 # increasing the default timeout (2^22 ms ~4s), avoids false failures under network load
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))  # CPU threads per GPU for OpenMP operations

TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
MASTER_ADDR=$(hostname)
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-30000 -n 1)}" # random port to avoid problems with multiple jobs on the same node

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --context-parallel-size 1
	  --wgrad-deferral-limit 50
	  --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

TORCHRUN_ARGS=(
    --nproc-per-node "${SLURM_GPUS_PER_NODE:-1}"
    --nnodes "${SLURM_NNODES:-1}"  # number of nodes (set by SLURM)
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}"   # rendezvous endpoint for node coordination
    --rdzv_backend c10d   # more robust than env var method for multi-node
    --max_restarts 0  # don't auto-restart on failure so we don't lose logs
    --tee 3  # redirect stdout+stderr from all workers to files
    --log-dir "${DEBUG_DIR}"  # write per-worker logs here instead of cluttering repo root
)


##### Megatron Args (Llama 3.2 1B) #####

TRANSFORMER_ENGINE_ARGS=(
	--transformer-impl transformer_engine
	--use-precision-aware-optimizer
	--main-grads-dtype bf16
)

# Network size (Llama 3.2 1B)

SEQ_LEN="${SEQ_LEN:-8192}"

NETWORK_SIZE_ARGS=(
    --num-layers 16
    --hidden-size 2048
    --ffn-hidden-size 8192
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 8
    --max-position-embeddings "${SEQ_LEN}"
    --position-embedding-type rope
    --rotary-base 500000
    --use-rope-scaling
    --rope-scaling-factor 32
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
)

##### Regularization #####
REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
	  --adam-beta2 0.95
)

##### Training #####
MBS=2
GBS=80
TRAIN_ITERS="${TRAIN_ITERS:-15}"

TRAINING_ARGS=(
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
#    --no-check-for-nan-in-loss-and-grad   keep nan for attention correctness check
    --train-iters "${TRAIN_ITERS}"
    --cross-entropy-loss-fusion
    --optimizer adam
    --dataloader-type single
)

INITIALIZATION_ARGS=(
	--seed 28
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
    --lr 0.0003
    --min-lr 0.00003
    --lr-decay-style cosine
    --lr-warmup-iters 2
)

CHECKPOINTING_ARGS=(
    --save "${EXP_DIR}/checkpoints" # in case megatron needs this set?
    --save-interval 10000  # effectively disables checkpointing for a short run
    --no-save-optim
    --no-save-rng
)

MIXED_PRECISION_ARGS=(
    --bf16
)

##### Data #####
DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer # we're using mock data
    --vocab-size 32768 # need to set if using mock data, using real vocab size of Llama 3.2 1B
    --seq-length "${SEQ_LEN}"
    --make-vocab-size-divisible-by 128
)

##### Logging #####
LOGGING_ARGS=(
    --log-interval 1
    --log-throughput
    --log-progress
    --log-timers-to-tensorboard
    --no-log-loss-scale-to-tensorboard
    --log-memory-to-tensorboard
    --tensorboard-dir "${TENSORBOARD_DIR}"
)

EVAL_ARGS=(
    --eval-interval 100
    --eval-iters 0  # no evaluation needed for a correctness benchmark
)


##### Benchmark-specific #####
BENCHMARK_ARGS=(
    --attn "${ATTN}"
    --impl "${IMPL}"
)

##### Run #####

# CMD_PREFIX="numactl --membind=0-3"  # bind memory to NUMA nodes for better locality (disabled until node NUMA topology is verified)
CMD_PREFIX=""

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[*]} ${ATTN_BENCH_ROOT}/benchmarks/correctness.py \
    ${BENCHMARK_ARGS[*]} \
    ${TRANSFORMER_ENGINE_ARGS[*]} \
    ${DISTRIBUTED_ARGS[*]} \
    ${NETWORK_SIZE_ARGS[*]} \
    ${INITIALIZATION_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${MIXED_PRECISION_ARGS[*]} \
    ${REGULARIZATION_ARGS[*]} \
    ${LEARNING_RATE_ARGS[*]} \
    ${DATA_ARGS[*]} \
    ${LOGGING_ARGS[*]} \
    ${EVAL_ARGS[*]} \
    ${CHECKPOINTING_ARGS[*]}"


# WANDB Logging 
LOG_WANDB="${LOG_WANDB:-false}"
if [ "${LOG_WANDB}" = "true" ]; then
    echo "[$(date)] W&B logging enabled."
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $EXP_DIR \
    --wandb-project $WANDB_PROJECT \
    --wandb-exp-name $EXP_NAME"
else
    export WANDB_MODE=disabled
    echo "[$(date)] W&B logging disabled."
fi


# NSYS profiler
NSYS_PROFILER="${NSYS_PROFILER:-false}"
if [ "${NSYS_PROFILER}" = "true" ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=${DEBUG_DIR}/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="${NSYS_LAUNCHER} ${TRAINING_CMD} --profile"
fi

# Copy this script to debug dir for reproducibility
cp "$0" "${DEBUG_DIR}"

# Log compute environment
SEP=$(printf '=%.0s' {1..100})
{
    echo -e "$(date)"
    echo -e "${SEP}"
    echo -e "\nCMD: ${CMD_PREFIX} ${TRAINING_CMD}"
    echo -e "${SEP}"
    echo -e "\nNODES: $(scontrol show hostnames "${SLURM_JOB_NODELIST}")"
    echo -e "${SEP}"
    echo -e "\nRepo: ${REPO_ROOT} ($(git -C "${REPO_ROOT}" rev-parse --verify HEAD))"
    echo -e "${SEP}"
    echo -e "\n$(pip list)"
    echo -e "${SEP}"
    echo -e "\nPython sys.path:\n$(python -c 'import sys; print(chr(10).join(sys.path))')" 
    echo -e "${SEP}"
    echo -e "\nmegatron location: $(python -c 'import megatron; print(megatron.__file__)' 2>&1)"
    echo -e "${SEP}"
    echo -e "\n$(nvidia-smi)"
    echo -e "${SEP}"
    echo -e "\nEnvironment Variables:\n\n$(printenv)"
    echo -e "${SEP}"
} > "${COMPUTE_ENVIRONMENT_DIR}"

echo "Correctness benchmark: attn=${ATTN} impl=${IMPL} tp=${TP_SIZE} pp=${PP_SIZE} nodes=${SLURM_NNODES:-1} gpus_per_node=${SLURM_GPUS_PER_NODE:-1}"

echo "[$(date)] Launching training..."
srun -lu --cpus-per-task "${SLURM_CPUS_PER_TASK}" --wait 60 bash -c "${CMD_PREFIX} ${TRAINING_CMD}"
echo "[$(date)] Training finished."

echo "END TIME: $(date)"
echo "Done. Results in ${EXP_DIR}"

#!/usr/bin/env bash
# Distributed training configuration for ALPS cluster.
# Requires: EXP_DIR, DEBUG_DIR, REPO_ROOT set before sourcing.

### PARALLELISM ###
# TP/PP/CP default to 1. Override for larger models
: ${TP:=1}
: ${PP:=1}
: ${CP:=1}

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --wgrad-deferral-limit 50
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

### LAUNCH ###
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

: ${MASTER_ADDR:=$(hostname)}
: ${MASTER_PORT:=$(shuf -i 20000-30000 -n 1)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$((SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE))}

: ${CMD_PREFIX:="numactl --membind=0-3"}

# disables core dumps to save disk space
ulimit -c 0

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
    --log-dir $DEBUG_DIR
)

### DEBUG ###
: ${LOG_NCCL:=false}

if [ "$LOG_NCCL" = true ]; then
    CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

### PROFILER ###
: ${NSYS_PROFILER:=false}

# prepends nsys launcher to TRAINING_CMD; call after TRAINING_CMD is built
setup_profiler() {
    if [ "$NSYS_PROFILER" = true ]; then
        NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
        TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
    fi
}
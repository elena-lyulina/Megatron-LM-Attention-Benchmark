#!/usr/bin/env bash
# Check that none of our config variables are already set in the environment.
# Source this at the very start of a slurm script, before any overrides or sourcing.
# If anything is pre-set, the job exits immediately with a clear error.

### VARIABLES OWNED BY OUR CONFIGS ###
# Any of these found in the environment before sourcing = silent override risk.

_PROTECTED_VARS=(
    # distributed.sh
    TP PP CP
    TORCH_NCCL_AVOID_RECORD_STREAMS
    TORCH_NCCL_ASYNC_ERROR_HANDLING
    CUDA_DEVICE_MAX_CONNECTIONS
    OMP_NUM_THREADS
    LOG_NCCL
    NSYS_PROFILER

    # training.sh
    CHECKPOINT_STEPS
    SAVE_RETAIN_INTERVAL
    EVAL_INTERVAL
    EVAL_ITERS

    # logging.sh
    LOG_WANDB

    # models/llama3_1b.sh
    SEQ_LEN
    NUM_LAYERS
    HIDDEN_SIZE
    FFN_HIDDEN_SIZE
    NUM_ATTENTION_HEADS
    NUM_QUERY_GROUPS
    ROTARY_BASE
    ROPE_SCALING_FACTOR
    NORM_EPSILON
    MAX_POSITION_EMBEDDINGS
    LR MIN_LR LR_DECAY LR_WARMUP
    ATTENTION_DROPOUT
    HIDDEN_DROPOUT
    CLIP_GRAD
    WEIGHT_DECAY
    ADAM_BETA1
    ADAM_BETA2
    SEED
    INIT_STD
    TOKENIZER_PATH
)

_FOUND=()
for _var in "${_PROTECTED_VARS[@]}"; do
    if [[ -v "$_var" ]]; then
        _FOUND+=("  $_var=${!_var}")
    fi
done
unset _var _PROTECTED_VARS

if [[ ${#_FOUND[@]} -gt 0 ]]; then
    echo "ERROR [check_clean_env]: the following variables are already set in the environment."
    echo "      They would silently override our config defaults via the := / :- syntax."
    echo "      Unset them before running, or explicitly override them in the slurm script."
    printf '%s\n' "${_FOUND[@]}"
    unset _FOUND
    exit 1
fi
unset _FOUND
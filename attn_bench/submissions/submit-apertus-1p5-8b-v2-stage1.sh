#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=12:00:00
#SBATCH --job-name=apertus-1p5-8b
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclude=nid006931
#SBATCH --no-requeue

echo "START TIME: $(date)"

# Per batch ratio is determined by the physical data distribution under STAGE1_DATA_DIR:
# ~40% text / ~46% vision / ~14% audio (Megatron blends proportionally by token count)

# Container environment (used by srun --environment)
CONTAINER_ENV=/capstor/store/cscs/swissai/infra01/containers/ngc_25-12-alps2.toml

# Vetnode tests disabled until scale issues are resolved
# srun  -N ${SLURM_JOB_NUM_NODES} --tasks-per-node=4 --mpi=pmix --environment=$CONTAINER_ENV --network=disable_rdzv_get --container-writable bash -c '
#
#     if [ "${SLURM_LOCALID}" = "0" ]; then
#         mkdir -p /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}
#         cd /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}
#         wget -q -O config.yaml https://raw.githubusercontent.com/theely/vetnode/refs/heads/main/examples/alps-ml-vetting/config.yaml
#         python -m venv --system-site-packages .venv
#         source .venv/bin/activate
#         pip install -q --no-cache-dir --index-url "https://jfrog.svc.cscs.ch/artifactory/api/pypi/pypi-remote/simple" vetnode
#         touch /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}/.setup_done
#     else
#         while [ ! -f /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}/.setup_done ]; do
#             sleep 2
#         done
#         cd /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}
#         source .venv/bin/activate
#     fi
#
#     vetnode diagnose config.yaml
#
#     if [ "${SLURM_LOCALID}" = "0" ]; then
#         sleep 5
#         rm -rf /tmp/vetnode-$(hostname)-${SLURM_JOB_ID}
#     fi
# '

# Stage 1 data root (symlinks to tokenized datasets)
STAGE1_DATA_DIR=/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-stage1

echo "------------------------ Debug info ------------------------"

# Megatron source and dataset cache
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/apertus/Megatron-LM
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

AUTO_JOB_REQUEUE=true # Set to `true` to continuously submit jobs to Slurm until training is complete. Enable it once you are sure of the cost involved in running this experiment.
if [ "$AUTO_JOB_REQUEUE" = true ]; then
    echo "[$(date)] Submitting follow-up job with singleton dependency and --resume"
    # Export the script path and submit with --resume
    sbatch --dependency=singleton --job-name=$SLURM_JOB_NAME $0 --resume
fi

# Enumerate all data prefixes (no weights — Megatron blends proportionally by token count)
# The physical data is already curated to ~40% text / 46% vision / 14% audio
DATA_PATH_LIST=($(find -L /iopsstor/scratch/cscs/ahuang/apertus_dataset/megatron_lm_data_split/split_A_20pct $STAGE1_DATA_DIR/vision $STAGE1_DATA_DIR/audio \
    -name '*.bin' | sed 's/\.bin$//' | sort))

DATA_PATH_STR=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
RESUME_TRAINING=false
MBS=1 # Micro batch size
GBS=1024 # Global batch size
SEQ_LEN=8192 # Sequence length
TARGET_TOKENS=805600000000 # ~806B: 1 epoch of all data — 371.2B vision (46.1%), 110.7B audio (13.7%), 323.7B text (40.2%)
CHECKPOINT_STEPS=2000
TRAINING_STEPS=$((TARGET_TOKENS / (GBS * SEQ_LEN)))
TRAINING_STEPS=$((((TRAINING_STEPS + CHECKPOINT_STEPS/2) / CHECKPOINT_STEPS) * CHECKPOINT_STEPS)) # Round to nearest multiple of CHECKPOINT_STEPS

# Base model checkpoint (Stage 5 Apertus 8B model) - specify to laod mddel to extend
BASE_MODEL_DIR=/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/base_model_megatron_ngc/
# The pretrained extended model will be loaded from here if --extend-model-vocab is not specified
EXTENDED_MODEL_DIR=/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/extended_model_vocab_266440

# Uncomment to extend the vocabulary for multimodal training
MULTIMODAL_ARGS=(
        #--extend-model-vocab
)

################ Parse Command Line Arguments ################

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
                        echo "---> Resume Training <---"
            RESUME_TRAINING=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--resume]"
            exit 1
            ;;
    esac
done


#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info. Every process will dump the logging into separate files, check `NCCL_DEBUG_FILE`
NSYS_PROFILER=false # Turn on the NSYS profiler. Check the `--profile-*` args available in megatron/training/arguments.py
MOCK_DATA=false # Set to `true` to use mock data
###################

TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
TRITON_HOME_DIR=/tmp/.triton
PYTHON_CACHE_DIR=/tmp/.python_cache

# Logging directories & artifacts
PROJECT_NAME=main-runs-v2-apertus-1p5
EXP_NAME=apertus-1p5-8b-256
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

#########################################

EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

# Set up ENV
# export your WANDB_API_KEY here.
# export WANDB_API_KEY=$(grep -A2 "api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
export WANDB__FILE_STREAM_RETRY_MAX=10
export HF_HUB_OFFLINE=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# We are preparing for torch.distributed programs so it wants:
# - MASTER_ADDR, MASTER_PORT, WORLD_SIZE - already known before `srun`
# - RANK, LOCAL_RANK - will set at `srun` command
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

#### Megatron Args #### Check megatron/training/arguments.py
# Based on the Llama 3.2 8B model.
# --ddp-bucket-size 10000000000
# --fp8-recipe tensorwise
# --fp8-format hybrid
# --log-intermediate-metrics mean rms kurtosis
# --log-intermediate-metrics-interval 500
# --fix-old-xielu
TRANSFORMER_ENGINE_ARGS=(
        --main-grads-dtype fp32
        --log-params-norm
)

NETWORK_SIZE_ARGS=(
        --num-layers 32
        --hidden-size 4096
        --ffn-hidden-size 21504  # xielu
        --num-attention-heads 32
        --group-query-attention
        --num-query-groups 8
        --max-position-embeddings $SEQ_LEN
        --position-embedding-type rope
        --rotary-base 500000
        --use-rope-scaling
        --rope-scaling-factor 8
        --make-vocab-size-divisible-by 128
        --normalization RMSNorm
        --xielu  # xielu
        --qk-layernorm  # op-block
        --qknorm-impl apex  # op-block
        --untie-embeddings-and-output-weights
)

LOGGING_ARGS=(
        --log-throughput
        --tensorboard-dir $TENSORBOARD_DIR
        --no-log-loss-scale-to-tensorboard
        --log-memory-to-tensorboard
        # --log-pre-final-ln-norm
)

REGULARIZATION_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --weight-decay 0.1
        --weight-decay-on-xielu-alphas
        --clip-grad 0.1  # ademamix
        --adam-beta1 0.9
        --adam-beta2 0.999  # ademamix
        --ademamix-alpha 8 # ademamix
        --ademamix-beta3 0.9999  # ademamix
        --ademamix-beta3-warmup 100000  # ademamix
        --ademamix-alpha-warmup 100000  # ademamix
)

TRAINING_ARGS=(
        --micro-batch-size $MBS
        --global-batch-size $GBS
        --no-check-for-nan-in-loss-and-grad
        --train-iters $TRAINING_STEPS # --train-samples 3662109375
        --log-interval 1
        --cross-entropy-loss-fusion
        --disable-bias-linear
        --optimizer ademamix  # ademamix
        --dataloader-type single
        --manual-gc
        --manual-gc-interval 500
        --exit-signal-handler
        --eval-interval 100000000000
        --eval-iters 0
)

INITIALIZATION_ARGS=(
        --seed 41
        --init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
        --lr 0.00011
        --min-lr 0.000011 # x10 reduction
        --lr-decay-style WSD  # WSD schedule
        --lr-warmup-iters 500
        --lr-wsd-decay-style minus_sqrt  # WSD schedule
        --lr-wsd-decay-iters 0  # WSD decay will be a different run
)

# Check if --extend-model-vocab is in MULTIMODAL_ARGS
# This determines whether we're extending the vocab (first step) or doing normal training (second step)
EXTEND_MODEL_VOCAB=false
for arg in "${MULTIMODAL_ARGS[@]}"; do
    if [[ "$arg" == "--extend-model-vocab" ]]; then
        EXTEND_MODEL_VOCAB=true
        break
    fi
done

# Determine checkpoint loading strategy based on flags
# Priority order: vocab extension > resume training > normal multimodal training
if [ "$EXTEND_MODEL_VOCAB" = true ]; then
    echo "=========================================="
    echo "[$(date)] VOCAB EXTENSION MODE: Extending vocab from base model"
    echo "[$(date)] Loading from: $BASE_MODEL_DIR"
    echo "[$(date)] Saving to: $EXTENDED_MODEL_DIR"
    echo "=========================================="
    LOAD_DIR=$BASE_MODEL_DIR
    SAVE_DIR=$EXTENDED_MODEL_DIR
    FINETUNE_FLAG="--finetune"
    # Disable logging during vocab extension
    LOGGING_ARGS=()
    unset WANDB_API_KEY
    mkdir -p $SAVE_DIR
elif [ "$RESUME_TRAINING" = true ]; then
    echo "=========================================="
    echo "[$(date)] RESUME MODE: Loading from previous checkpoint"
    echo "[$(date)] Loading from: $CKPT_DIR"
    echo "=========================================="
    LOAD_DIR=$CKPT_DIR
    SAVE_DIR=$CKPT_DIR
    FINETUNE_FLAG=""  # No finetune flag when resuming - preserves optimizer states
else
    echo "=========================================="
    echo "[$(date)] MULTIMODAL TRAINING MODE: Starting from extended model"
    echo "[$(date)] Loading from: $EXTENDED_MODEL_DIR"
    echo "[$(date)] Saving to: $CKPT_DIR"
    echo "=========================================="
    LOAD_DIR=$EXTENDED_MODEL_DIR
    SAVE_DIR=$CKPT_DIR
    FINETUNE_FLAG="--finetune"
fi

CHECKPOINTING_ARGS=(
        --load $LOAD_DIR
        --save $SAVE_DIR
        --save-interval $CHECKPOINT_STEPS
        --ckpt-format torch_dist
        --async-save
        --ckpt-fully-parallel-load
        --dist-ckpt-strictness assume_ok_unexpected
        --override-opt_param-scheduler
        $FINETUNE_FLAG
)

MIXED_PRECISION_ARGS=(
        --bf16
)

DISTRIBUTED_ARGS=(
        --tensor-model-parallel-size 2
        --pipeline-model-parallel-size 1
        --use-distributed-optimizer
        --overlap-grad-reduce
        --overlap-param-gather
)

TOKENIZER_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok
)

DATA_ARGS=(
        --split 100,0,0
        --seq-length $SEQ_LEN
        --reset-position-ids  # crossDocAttn: need to reset position ids to properly separate samples! (not multiple pos ids in one sample)
        #--reset-attention-mask  # crossDocAttn / ignore as attn mask from dataloader is ignore
        --use-packed-seq-params # Use packed seq params to enable proper xdoc masking (ASSUMES THIS FIX IS AVAILABLE!)
        --no-create-attention-mask-in-dataloader # Dont create attnmask in datalaoder as we use packed seq params
        --eod-mask-loss  # crossDocAttn
        --num-workers 64        # workers per rank
        --num-dataset-builder-threads 4 # only needed for initialization
        --goldfish-loss  # goldfish
        --goldfish-k 50  # goldfish
        --goldfish-h 50  # goldfish
)

# Set up directories
mkdir -p $CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR
export PYTHONPATH=$MEGATRON_LM_DIR

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $DATA_PATH_STR --data-cache-path $DATASET_CACHE_DIR"
fi

#CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    $DATA_ARGS"

# WANDB Logging Setup - Only if API Key in Env
if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."

  # Get and display WandB entity/workspace
  WANDB_ENTITY=$(python3 -c "import wandb; api = wandb.Api(); print(api.default_entity)" 2>/dev/null || echo "unknown")
  echo "[$(date)] WandB logs will be sent to:"
  echo "           Organization/Workspace: $WANDB_ENTITY"
  echo "           Project: $PROJECT_NAME"
  echo "           Experiment: $EXP_NAME-$SLURM_JOB_ID"

  # Sync any previous run data if present
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  # Add wandb-related args to TRAINING_CMD
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    --network=disable_rdzv_get \
    -lu \
    bash -c "
    mkdir -p $TORCH_INDUCTOR_CACHE_DIR
    mkdir -p $TRITON_HOME_DIR
    mkdir -p $TRITON_CACHE_DIR
    mkdir -p $PYTHON_CACHE_DIR
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD
    "

echo "END TIME: $(date)"
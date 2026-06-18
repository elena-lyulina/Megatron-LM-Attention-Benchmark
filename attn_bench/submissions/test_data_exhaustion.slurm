#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:20:00
#SBATCH --job-name=test-data-exhaustion
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Smoke test for the data-exhaustion fix (Option D).
# Trains a tiny model on a tiny 2-file blend with --train-iters set far higher than
# the data supports. Expect: no StopIteration crash, a "data supports N iterations" log,
# rolling periodic checkpoints, and a forced final checkpoint at the true last step.
#
# Usage:
#   sbatch attn_bench/submissions/test_data_exhaustion.slurm

echo "START TIME: $(date)"

### ENV ###
CONTAINER_ENV=nemo_26
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

### TINY DATA ###
# ~20k tokens per file -> blend caps at a few dozen samples (one epoch, no repetition)
TINY_DATA_DIR=/iopsstor/scratch/cscs/$USER/datasets/tiny_exhaustion_test
DATASET_CACHE_DIR=$TINY_DATA_DIR/cache
mkdir -p $TINY_DATA_DIR $DATASET_CACHE_DIR

# Generate the tiny dataset once (inside the container so megatron imports work)
if [ ! -f "$TINY_DATA_DIR/tiny_0.bin" ]; then
    echo "[$(date)] Generating tiny tokenized data..."
    srun --ntasks=1 --environment=$CONTAINER_ENV bash -c \
        "export PYTHONPATH='${PYTHONPATH}'; python3 $MEGATRON_LM_DIR/attn_bench/tests/make_tiny_tokenized_data.py --out-dir $TINY_DATA_DIR"
fi

DATA_PATH_LIST=($(find -L $TINY_DATA_DIR -name '*.bin' | sed 's/\.bin$//' | sort))
DATA_PATH_STR=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")
echo "Tiny dataset prefixes: ${#DATA_PATH_LIST[@]}"

### TRAINING ###
SEQ_LEN=512
MBS=1
# GBS = mbs * DP = 1 * 4 = 4  (1 node, 4 GPUs, TP=1, PP=1)
GBS=4
# huge on purpose: data will run out long before this
TRAINING_STEPS=200

### Logging dirs ###
EXP_DIR=$MEGATRON_LM_DIR/attn_bench/results/pretrain/tests/data_exhaustion
CKPT_DIR=$EXP_DIR/checkpoints/$SLURM_JOB_ID
mkdir -p $EXP_DIR $CKPT_DIR

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8888
export WORLD_SIZE=$SLURM_NPROCS

### Megatron args (tiny model) ###
NETWORK_SIZE_ARGS=(
    --num-layers 2
    --hidden-size 256
    --ffn-hidden-size 512
    --num-attention-heads 4
    --max-position-embeddings 4096
    --position-embedding-type rope
    --rotary-base 10000
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
)

CHECKPOINTING_ARGS=(
    --load $CKPT_DIR
    --save $CKPT_DIR
    --save-interval 5            # periodic checkpoint every 5 steps
    --save-retain-interval 20    # keep only the rolling latest (must be a multiple of save-interval)
    --ckpt-format torch_dist
    --dist-ckpt-strictness assume_ok_unexpected
)

DATA_ARGS=(
    --split "100,0,0"
    --dataloader-type single
    --seq-length $SEQ_LEN
    --num-workers 1
    --reset-position-ids
    --eod-mask-loss
    --use-packed-seq-params
    --no-create-attention-mask-in-dataloader
    --data-path $DATA_PATH_STR
    --data-cache-path $DATASET_CACHE_DIR
)

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    --transformer-impl transformer_engine \
    ${NETWORK_SIZE_ARGS[@]} \
    --micro-batch-size $MBS \
    --global-batch-size $GBS \
    --train-iters $TRAINING_STEPS \
    --lr 0.0004 \
    --min-lr 0.00004 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --bf16 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --use-distributed-optimizer \
    --eval-interval 1000000 \
    --eval-iters 0 \
    --log-interval 1 \
    --seed 28 \
    --make-vocab-size-divisible-by 128 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model $TOKENIZER_PATH \
    ${CHECKPOINTING_ARGS[@]} \
    ${DATA_ARGS[@]}"

### LAUNCH ###
echo "[$(date)] Launching tiny training (train-iters=$TRAINING_STEPS, gbs=$GBS, seq=$SEQ_LEN)..."
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --mpi=pmix \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "export PYTHONPATH='${PYTHONPATH}'; RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${TRAINING_CMD}"

echo "END TIME: $(date)"
echo "Checkpoints in $CKPT_DIR"
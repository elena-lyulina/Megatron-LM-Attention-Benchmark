#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:40:00
#SBATCH --job-name=measure-mem-gdn
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Measure memorization for the Gated Delta Net (GDN) models. Parametrized over the 4 GDN
# checkpoints via GDN_VARIANT:
#   base       -> llama3-1b-gdn-fineweb40B-gutenberg3B            (packed, no carry)
#   carry-r0   -> llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B   (no carry, ratio 0)
#   carry-r0.5 -> llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B
#   carry-r1   -> llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B
#
#   - Run sparse Gutenberg inference directly from torch_dist checkpoint (no HF conversion)
#   - Compute LCS / Rouge-L memorization metrics
#
# GDN inference uses the cached incremental decode path (see _plans/gdn_inference_plan.md). The
# recurrent + conv state is zero-initialised here (the saved carry states of r0.5/r1 are NOT loaded;
# that would be a separate experiment later). Container nemo_26.04_te2.15 ships flash-linear-attention +
# causal_conv1d, required by the GDN layer.
#
# Usage:
#   sbatch attn_bench/submissions/measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm                       # base
#   sbatch --export=GDN_VARIANT=carry-r1 attn_bench/submissions/measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm
#   sbatch --export=GDN_VARIANT=carry-r0.5,OFFSET=0,PREFIX_LENGTH=500 attn_bench/submissions/measure_mem_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

OFFSET=${OFFSET:-0}
PREFIX_LENGTH=${PREFIX_LENGTH:-500}

### VARIANT -> EXP_NAME ###
GDN_VARIANT=${GDN_VARIANT:-base}
case "$GDN_VARIANT" in
    base)       EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B ;;
    carry-r0)   EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B ;;
    carry-r0.5) EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B ;;
    carry-r1)   EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B ;;
    *) echo "Unknown GDN_VARIANT=$GDN_VARIANT (expected base|carry-r0|carry-r0.5|carry-r1)"; exit 1 ;;
esac

MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
MEM_BASE=/users/elyulina/store/mem-results
MEM_DIR=$MEM_BASE/SparseGutenberg/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: variant=$GDN_VARIANT  exp=$EXP_NAME  offset=$OFFSET  prefix=$PREFIX_LENGTH  suffix=500"

# GDN architecture flags. --use-checkpoint-args restores the standard network args (ffn-hidden-size,
# position-embedding-type none, etc.), but the experimental GDN dims are re-passed here, same reason
# the gated/sink/obo scripts re-pass their attention flags. VERIFY ON FIRST RUN: if the build fails on
# ffn / heads / positions, add those here too.
GDN_EXTRA_ARGS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"

### STEP 1: sparse Gutenberg inference ###
# Loads torch_dist checkpoint directly (DCP resharding merges shards on the fly).
# Saves rank*.jsonl to $MEM_DIR/inference/offset_0_prefix_500_suffix_500/rep_*/
#
# torchrun --no-python wrapper: each worker sets its own node-local TRITON_CACHE_DIR keyed on RANK
# (torchrun sets RANK per worker; SLURM_PROCID is 0 for all four under ntasks-per-node=1). GDN's fla
# kernels autotune on first use and a shared-FS / colliding cache crashes with "stale file handle"
# (job 2570257). Per-rank + per-job under /tmp isolates them.

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun --nproc_per_node=4 --no-python \
        bash -c 'export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}_\${RANK}; mkdir -p \"\$TRITON_CACHE_DIR\"; exec python \"\$@\"' _ \
        $MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_sparse.py \
        --container-env $CONTAINER_ENV \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --offset $OFFSET \
        --prefix-length $PREFIX_LENGTH \
        --suffix-length 500 \
        --batch-size 20 \
        --megatron-extra-args $GDN_EXTRA_ARGS"


### STEP 2: compute memorization metrics ###
# Reads inference outputs and computes LCS / Rouge-L summary statistics.
# Saves results to $MEM_BASE/SparseGutenberg/$EXP_NAME/offset_0_prefix_500_suffix_500_greedy.pkl

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
        --offsets $OFFSET \
        --prefix-lengths $PREFIX_LENGTH \
        --suffix-lengths 500"

echo "END TIME: $(date)"
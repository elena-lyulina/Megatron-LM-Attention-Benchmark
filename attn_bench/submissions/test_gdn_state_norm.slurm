#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:10:00
#SBATCH --job-name=test-gdn-state-norm
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Check the state-norm segment-and-carry against a single chunk-kernel call on random inputs
# (no checkpoint needed -- it is a property of the kernel + our slicing). Look for the
# "### GDN state-norm — Summary ###" block; the job exits nonzero on failure.
#
# Usage:
#   sbatch attn_bench/submissions/test_gdn_state_norm.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Single GPU -> single rank, so a per-job node-local TRITON_CACHE_DIR is enough (fla kernels
# autotune on first use; keep the cache off the shared FS to avoid a stale-handle crash).
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TRITON_CACHE_DIR=/tmp/triton_\${USER}_\${SLURM_JOB_ID}
    mkdir -p \$TRITON_CACHE_DIR
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun --nproc_per_node=1 \
        $MEGATRON_LM_DIR/attn_bench/tests/test_gdn_state_norm.py"

echo "END TIME: $(date)"

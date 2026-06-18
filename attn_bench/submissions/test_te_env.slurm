#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=test-te-env
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00
#SBATCH --environment=nemo_26
#SBATCH --no-requeue

# Usage:
#   sbatch attn_bench/submissions/test_te_env.slurm

srun bash -c "
nvcc --version 2>/dev/null || echo 'nvcc not found'
echo '---'
cat /usr/local/cuda/version.json 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null || echo 'no cuda version file'
echo '---'
python3 -u << 'EOF'
import transformer_engine
print('TE version:', transformer_engine.__version__)

import transformer_engine.pytorch as te
print('TE pytorch ok')

import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA runtime (torch):', torch.version.cuda)
print('cuDNN version:', torch.backends.cudnn.version())

import transformer_engine_torch as tex
print('cuDNN version (TE):', tex.get_cudnn_version())
EOF
"
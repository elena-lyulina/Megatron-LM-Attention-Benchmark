#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=test-nemo-env
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:05:00
#SBATCH --environment=nemo
#SBATCH --no-requeue

srun bash -c "
pip install --quiet --no-deps datasketch inscriptis
echo 'pip done'
python3 -u << 'EOF'
print('Python started')
print('importing datasets...', end=' ', flush=True); from datasets import load_dataset; print('ok')
print('importing sentence_transformers...', end=' ', flush=True); from sentence_transformers import SentenceTransformer; print('ok')
print('importing datasketch...', end=' ', flush=True); from datasketch import MinHash; print('ok')
print('all imports ok')
EOF
"
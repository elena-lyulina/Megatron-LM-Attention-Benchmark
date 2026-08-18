#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:05:00
#SBATCH --job-name=check-container-env
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Writes a snapshot of key package versions inside a container environment to a file next to
# where the sqsh lives, so you can check what's installed without digging through job logs.
# Writes to scratch first, then copies to OUTPUT_DIR (store) only on success -- never write to
# store directly from a compute node.
#
# Usage: sbatch --export=CONTAINER_ENV=nemo_26.04_te2.15,OUTPUT_DIR=/path/to/containers check_container_env.slurm

set -euo pipefail

CONTAINER_ENV=${CONTAINER_ENV:?set CONTAINER_ENV, e.g. nemo_26.04_te2.15}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR, e.g. /users/$USER/store/containers}
SCRATCH_DIR=/users/$USER/scratch/tmp/check_container_env_$SLURM_JOB_ID
OUT_FILE="$SCRATCH_DIR/${CONTAINER_ENV}_env_info.txt"

mkdir -p "$SCRATCH_DIR"

set +e
srun --environment=$CONTAINER_ENV python -c "
import subprocess, sys, socket, os, datetime, flash_attn, inspect, torch, transformer_engine

header = [
    f'container: $CONTAINER_ENV',
    f'checked at: {datetime.datetime.now().isoformat()}',
    f'host: {socket.gethostname()}',
    f'slurm job id: {os.environ.get(\"SLURM_JOB_ID\", \"n/a\")}',
    f'python version: {sys.version.split()[0]}',
    f'transformer_engine version: {transformer_engine.__version__}',
    f'supports s_aux (sink attention): ' + str('s_aux' in inspect.signature(flash_attn.flash_attn_varlen_func).parameters),
    f'cuda version: {torch.version.cuda}',
]
if torch.cuda.is_available():
    header.append(f'gpu: {torch.cuda.get_device_name(0)}')
    header.append(f'nccl version: {\".\".join(str(v) for v in torch.cuda.nccl.version())}')
else:
    header.append('gpu: none allocated to this job')
header += ['', 'pip freeze:']

freeze = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True).stdout
content = '\n'.join(header) + '\n' + freeze
print(content)
with open('$OUT_FILE', 'w') as f:
    f.write(content)
"
STATUS=$?
set -e

if [ "$STATUS" -eq 0 ]; then
    mkdir -p "$OUTPUT_DIR"
    cp -a "$SCRATCH_DIR/." "$OUTPUT_DIR/"
else
    echo "Job failed (exit $STATUS) -- not copying scratch results to store"
fi
rm -rf "$SCRATCH_DIR"

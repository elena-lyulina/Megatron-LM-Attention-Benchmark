#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:10:00
#SBATCH --job-name=count-model-param
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Count the parameters of one Megatron model built from a flag file, writing the
# result next to the config as <config_stem>_param_counts. Needs a GPU + TE +
# flash-linear-attention + causal_conv1d (GDN spec asserts TE), so it runs in the
# container even though it only counts param shapes.
#
# Pass just the config file name via the `config` env var (exported by sbatch's
# default --export=ALL); a positional arg also works. The file is looked up in
# attn_bench/data/param_count_configs/.
#
# Usage:
#   config=args.txt sbatch attn_bench/submissions/count_model_param.slurm
#   sbatch attn_bench/submissions/count_model_param.slurm args.txt

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

### ENV ###
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

### ARGS ###
CONFIG_DIR=$MEGATRON_LM_DIR/attn_bench/data/param_count_configs
CONFIG=${config:-${1:-}}
if [ -z "$CONFIG" ]; then
    echo "ERROR: no config given. Pass config=<file> or as a positional arg." >&2
    exit 1
fi
# Fail now (cheap) instead of after the container + torch startup just to hit a FileNotFoundError.
CONFIG=$CONFIG_DIR/$CONFIG
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

SCRIPT=$MEGATRON_LM_DIR/attn_bench/evaluation/count_model_param.py
# Launched under torchrun so the script gets the distributed rank env Megatron needs.
CMD="torchrun --nproc_per_node=1 $SCRIPT $CONFIG"

ulimit -c 0

### LAUNCH ###
echo "=== Counting params for: $CONFIG ==="
echo "[$(date)] Launching..."

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "export PYTHONPATH='${PYTHONPATH}'; ${CMD}"

echo "END TIME: $(date)"
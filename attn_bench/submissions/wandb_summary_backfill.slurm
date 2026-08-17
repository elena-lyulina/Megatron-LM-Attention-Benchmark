#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:10:00
#SBATCH --job-name=wandb-summary-backfill
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# One-off backfill: push an attn_training_perf_bench sweep's summary CSV to wandb as a Table, for
# sweeps submitted before attn_training_perf_bench.slurm's inline upload step was fixed to run
# inside the container (it used to crash on a stale bare-shell python3 missing `dataclasses`).
#
# Usage:
#   sbatch attn_bench/submissions/wandb_summary_backfill.slurm <SWEEP_JOB_ID>
#   e.g. sbatch attn_bench/submissions/wandb_summary_backfill.slurm 3056624

SWEEP_JOB_ID=${1:?Usage: sbatch wandb_summary_backfill.slurm <SWEEP_JOB_ID>}

CONTAINER_ENV=nemo_26.04_te2.15
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PROJECT_NAME=attn_training_perf_bench
SUMMARY_CSV=$MEGATRON_LM_DIR/attn_bench/results/pretrain-bench/$PROJECT_NAME/summary_${SWEEP_JOB_ID}.csv
RUN_NAME=sweep-summary-$SWEEP_JOB_ID

[ -f "$SUMMARY_CSV" ] || { echo "ERROR: not found: $SUMMARY_CSV"; exit 1; }

echo "START TIME: $(date)"
echo "Backfilling $SUMMARY_CSV -> wandb project=$PROJECT_NAME run=$RUN_NAME"

srun --nodes=1 --ntasks=1 --environment=$CONTAINER_ENV python3 - "$SUMMARY_CSV" "$PROJECT_NAME" "$RUN_NAME" <<'PYEOF'
import csv
import sys

import wandb

summary_csv, project_name, run_name = sys.argv[1], sys.argv[2], sys.argv[3]

with open(summary_csv) as f:
    rows = list(csv.reader(f))
header, data = rows[0], rows[1:]

run = wandb.init(project=project_name, name=run_name, job_type="scaling-sweep-summary")
run.log({"scaling_sweep": wandb.Table(columns=header, data=data)})
run.finish()
print(f"Logged {len(data)} rows to wandb project {project_name}, run {run_name}")
PYEOF

echo "END TIME: $(date)"

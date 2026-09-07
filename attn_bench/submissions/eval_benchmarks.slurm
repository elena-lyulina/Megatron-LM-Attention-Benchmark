#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=4:00:00
#SBATCH --job-name=eval-bench
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Zero-shot downstream benchmarks (lm-eval-harness) for one trained model, all tasks in a
# single lm_eval call so the checkpoint loads once. Feeds the benchmark table in results 4.1.
# See attn_bench/_plans/lm_eval_benchmark_plan.md for the full design and its rationale.
#
# Loads the HF checkpoint through the stock --model hf path: every attention family already
# has a conversion registered for auto_class (checkpoint_conversion/attn_families/), so the
# custom modeling_*.py ships inside the checkpoint dir and trust_remote_code picks it up.
# No custom LM subclass needed.
#
# Env vars:
#   MODEL        see llama_checkpoints.sh (default: full-scf1)
#   NUM_FEWSHOT  shots, appears in the output path (default: 0)
#   BATCH_SIZE   fixed integer, never "auto" -- see the pinning note below (default: 8)
#   TASKS        comma-separated lm-eval task names (default: the 8-task thesis set)
#   LIMIT        cap documents per task, for smoke tests only (default: unset = full)
#   OVERWRITE    re-run even if results already exist in scratch or store
#
# Usage: sbatch --export=MODEL=full-scf1 eval_benchmarks.slurm
#        sbatch --export=MODEL=full-scf1,LIMIT=20 eval_benchmarks.slurm   # smoke test
# (normally submitted via eval_benchmarks_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

# Pinned so all models share one lm-eval version, and therefore one set of task versions --
# per-question results are only comparable across models when the task version matches.
# The job asserts the installed version equals this before doing any work.
# 0.4.13 was the current stable on PyPI as of 07/09/2026. Set LM_EVAL_VERSION=latest to
# install unpinned and skip the assert (the resolved version still lands in run_metadata.json).
LM_EVAL_VERSION=${LM_EVAL_VERSION:-0.4.13}

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh

MODEL=${MODEL:-full-scf1}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
TASKS=${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,sciq,wikitext}
LIMIT=${LIMIT:-}
OVERWRITE=${OVERWRITE:-}

### MODEL -> EXP_NAME / NEEDS_TRITON / NEEDS_FLA_052 / IS_SINK_FAMILY ###
model_config "$MODEL"

HF_DIR=/users/$USER/store/hf-checkpoints/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
OUT_DIR=/iopsstor/scratch/cscs/$USER/eval-results/lm-eval/$EXP_NAME/shots$NUM_FEWSHOT
STORE_OUT_DIR=/users/$USER/store/eval-results/lm-eval/$EXP_NAME/shots$NUM_FEWSHOT

# Dataset + dynamic-module cache on scratch, shared by every model so hellaswag downloads once.
# Never store: compute nodes must not write to capstor.
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf-home
export HF_DATASETS_TRUST_REMOTE_CODE=1

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

# The trained context length. The HF configs carry max_position_embeddings=131072, so without
# this wikitext's rolling window would be scored 16x past anything these models ever saw --
# especially wrong for the SWA and linear-attention families. Runtime only, the checkpoints
# are untouched.
MAX_LENGTH=8192

SEED=1234

if [ ! -f "$HF_DIR/config.json" ]; then
    echo "No HF checkpoint at $HF_DIR -- run convert_and_validate_hf.slurm (MODEL=$MODEL) first."
    exit 1
fi

### SKIP CHECK ###
# Read-only on both sides: the job writes to scratch and never to store, but you promote
# results to store by hand, so a rerun must not redo a model already copied over.
if [ -z "$OVERWRITE" ] && [ -z "$LIMIT" ]; then
    FOUND=$(find "$OUT_DIR" "$STORE_OUT_DIR" -name 'results_*.json' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        echo "Results already exist ($FOUND) -- set OVERWRITE=1 to redo. Exiting."
        exit 0
    fi
fi

mkdir -p "$OUT_DIR"

echo "RUN: model=$MODEL  exp=$EXP_NAME  shots=$NUM_FEWSHOT  bs=$BATCH_SIZE  tasks=$TASKS${LIMIT:+  limit=$LIMIT}"

### PER-FAMILY ENV ###
# NEEDS_FLA_052 (KDA): the container ships flash-linear-attention 0.4.2, whose chunk_kda NaNs.
# Side-installed 0.5.2 in $FLA_DIR (populated by the pretrain / measure_mem slurms), prepended
# to PYTHONPATH so it shadows the container copy.
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
FLA_ENV=""
if [ "$NEEDS_FLA_052" = "1" ]; then
    FLA_ENV="export PYTHONPATH=$FLA_DIR:\$PYTHONPATH;"
fi

# NEEDS_TRITON (GDN / KDA / qwen / carry-*): fla autotune on a shared Triton cache dies with
# "stale file handle" (job 2570257). One process here, so a per-job dir is enough.
TRITON_ENV=""
if [ "$NEEDS_TRITON" = "1" ]; then
    TRITON_ENV="export TRITON_CACHE_DIR=/tmp/triton_${USER}_${SLURM_JOB_ID}; mkdir -p \$TRITON_CACHE_DIR;"
fi

# IS_SINK_FAMILY: SinkLlamaForCausalLM's fused kernel needs the patched flash-attn checkout.
SINK_ENV=""
if [ "$IS_SINK_FAMILY" = "1" ]; then
    SINK_ENV="export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention; pip install --quiet 'nvidia-cutlass-dsl==4.6.2';"
fi

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

### RUN ###
# --log_samples writes one record per document (doc_id, target, filtered_resps, per-doc metric
# values). That is the only way to recover per-question outcomes afterwards, so it is always on
# and the analysis (CIs, paired model comparisons, bootstrapped wikitext ppl) can be decided later.
#
# batch_size is a fixed integer on purpose. Under bf16 the batch composition perturbs logits in
# the last decimals, so "auto" -- which picks a different size per model and per run -- would make
# per-document scores irreproducible and flip near-ties.
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    set -e
    if [ '$LM_EVAL_VERSION' = 'latest' ]; then
        pip install --quiet lm-eval
        python -c \"import lm_eval; print('lm_eval', lm_eval.__version__, '(unpinned)')\"
    else
        pip install --quiet 'lm-eval==$LM_EVAL_VERSION'
        python -c \"import lm_eval; assert lm_eval.__version__ == '$LM_EVAL_VERSION', 'got lm_eval '+lm_eval.__version__; print('lm_eval', lm_eval.__version__, 'OK')\"
    fi
    $FLA_ENV
    $TRITON_ENV
    $SINK_ENV
    python -m lm_eval \
        --model hf \
        --model_args pretrained=$HF_DIR,tokenizer=$TOKENIZER_PATH,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_path $OUT_DIR \
        --log_samples \
        $LIMIT_ARG
    "
EVAL_STATUS=$?

### RUN METADATA ###
# Written on success only, next to the results. The pins live in the script, this is the audit
# trail that lets the analysis assert all models really did share a version / max_length / batch.
if [ $EVAL_STATUS -eq 0 ]; then
    RESOLVED_VERSION=$(srun --overlap --environment=$CONTAINER_ENV bash -c "python -c 'import lm_eval; print(lm_eval.__version__)'" 2>/dev/null | tail -1)
    cat > "$OUT_DIR/run_metadata.json" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL",
  "exp_name": "$EXP_NAME",
  "hf_dir": "$HF_DIR",
  "tokenizer": "$TOKENIZER_PATH",
  "tasks": "$TASKS",
  "num_fewshot": $NUM_FEWSHOT,
  "batch_size": $BATCH_SIZE,
  "max_length": $MAX_LENGTH,
  "seed": $SEED,
  "limit": "${LIMIT:-null}",
  "lm_eval_version": "${RESOLVED_VERSION:-$LM_EVAL_VERSION}",
  "container": "$CONTAINER_ENV",
  "git_sha": "$(git -C $MEGATRON_LM_DIR rev-parse --short HEAD 2>/dev/null)"
}
EOF
    echo "Wrote $OUT_DIR/run_metadata.json"
else
    echo "lm_eval failed with status $EVAL_STATUS"
fi

echo "END TIME: $(date)"
exit $EVAL_STATUS

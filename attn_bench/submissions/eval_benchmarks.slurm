#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=3:00:00
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

# Downstream benchmarks (lm-eval-harness) for one model, all tasks in a single lm_eval call
# so the checkpoint loads once. Loads the HF checkpoint via the stock --model hf path.
#
# Env vars:
#   MODEL          see llama_checkpoints.sh (default: full-scf1)
#   EXTERNAL_MODEL public HF hub reference model (see external_config in llama_checkpoints.sh).
#                  Takes over from MODEL: brings its own pretrained id, revision, tokenizer and
#                  trained context length, and has no local checkpoint to look for.
#   NUM_FEWSHOT  shots, appears in the output path (default: 0)
#   BATCH_SIZE   fixed integer, never "auto" -- see the pinning note below (default: 8)
#   TASKS        comma-separated lm-eval task names (default: the 8-task thesis set)
#   LIMIT        cap documents per task, for smoke tests only (default: unset = full)
#   OVERWRITE    re-run even if results already exist in scratch or store
#
# Usage: sbatch --export=MODEL=full-scf1 eval_benchmarks.slurm
#        sbatch --export=MODEL=full-scf1,LIMIT=20 eval_benchmarks.slurm   # smoke test
#        sbatch --export=EXTERNAL_MODEL=pythia-1.4b-43b eval_benchmarks.slurm
# (normally submitted via eval_benchmarks_all.sh)

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

# All models must share one version, so task versions match. Side-installed rather than taken
# from the container: /opt/venv's copy mixes files from two releases and dies importing the hf
# backend (job 3314815). Must be recent enough for the image's transformers -- 0.4.8 fails on
# transformers.AutoModelForVision2Seq, removed from newer transformers (job 3315048).
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
EXTERNAL_MODEL=${EXTERNAL_MODEL:-}

### MODEL -> EXP_NAME / NEEDS_TRITON / NEEDS_FLA_052 / IS_SINK_FAMILY ###
# PRETRAINED / REVISION / TOKENIZER_PATH / MAX_LENGTH are what the two paths have to agree on;
# everything downstream (output dirs, cache, skip check, metadata) keys off EXP_NAME alone.
if [ -n "$EXTERNAL_MODEL" ]; then
    # A hub reference model: no local checkpoint, no model_config entry, and none of the
    # per-family kernel quirks -- those are properties of our Megatron conversions only.
    external_config "$EXTERNAL_MODEL"
    NEEDS_TRITON=0
    NEEDS_FLA_052=0
    IS_SINK_FAMILY=0
    PRETRAINED=$EXTERNAL_HF_MODEL
    REVISION=$EXTERNAL_REVISION
    TOKENIZER_PATH=$EXTERNAL_TOKENIZER
    # The reference model's own trained context, NOT our 8192 -- see external_config.
    MAX_LENGTH=$EXTERNAL_MAX_LENGTH
else
    model_config "$MODEL"
    PRETRAINED=/users/$USER/store/hf-checkpoints/$EXP_NAME
    REVISION=""
    TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
    # Trained context length. The HF configs say 131072, so without this wikitext's rolling
    # window runs 16x past anything these models saw. Runtime only, checkpoints untouched.
    MAX_LENGTH=8192
fi

# Version-scoped so changing the pin installs cleanly instead of reusing the old tree.
LM_EVAL_DIR=$MEGATRON_LM_DIR/attn_bench/packages-lm-eval/$LM_EVAL_VERSION
OUT_DIR=/iopsstor/scratch/cscs/$USER/eval-results/lm-eval/$EXP_NAME/shots$NUM_FEWSHOT
STORE_OUT_DIR=/users/$USER/store/eval-results/lm-eval/$EXP_NAME/shots$NUM_FEWSHOT

# A --limit run writes a results_*.json like any other, so it must never land in the real
# output dir -- the sweep's skip check would find it and leave that model on 20-doc numbers.
[ -n "$LIMIT" ] && OUT_DIR="$OUT_DIR/limit$LIMIT"

# Request cache, written as it goes -- results are only written after the whole task list,
# so this is what makes a killed job cheap to resubmit. Keyed by request and NOT by model:
# two models sharing a path would silently get each other's loglikelihoods.
CACHE_DIR=/iopsstor/scratch/cscs/$USER/eval-results/lm-eval-cache/$EXP_NAME/shots$NUM_FEWSHOT

# Dataset + dynamic-module cache on scratch, shared by every model so hellaswag downloads once.
# Never store: compute nodes must not write to capstor.
export HF_HOME=/iopsstor/scratch/cscs/$USER/hf-home
export HF_DATASETS_TRUST_REMOTE_CODE=1

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

SEED=1234

# Local checkpoints only: a hub model is resolved (and downloaded, or read from HF_HOME) by
# transformers at load time, so there is nothing to stat here.
if [ -z "$EXTERNAL_MODEL" ] && [ ! -f "$PRETRAINED/config.json" ]; then
    echo "No HF checkpoint at $PRETRAINED -- run convert_and_validate_hf.slurm (MODEL=$MODEL) first."
    exit 1
fi

### SKIP CHECK ###
# Read-only on both sides -- results are promoted to store by hand, so a rerun must not
# redo a model already copied over.
if [ -z "$OVERWRITE" ] && [ -z "$LIMIT" ]; then
    FOUND=$(find "$OUT_DIR" "$STORE_OUT_DIR" -name 'results_*.json' 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        echo "Results already exist ($FOUND) -- set OVERWRITE=1 to redo. Exiting."
        exit 0
    fi
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR"

echo "RUN: model=${EXTERNAL_MODEL:-$MODEL}  exp=$EXP_NAME  shots=$NUM_FEWSHOT  bs=$BATCH_SIZE  max_length=$MAX_LENGTH  tasks=$TASKS${REVISION:+  revision=$REVISION}${LIMIT:+  limit=$LIMIT}"

### PER-FAMILY ENV ###
# NEEDS_FLA_052 (KDA): container's fla 0.4.2 NaNs chunk_kda; shadow it with the side-installed
# 0.5.2 (populated by the pretrain / measure_mem slurms).
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
FLA_ENV=""
if [ "$NEEDS_FLA_052" = "1" ]; then
    FLA_ENV="export PYTHONPATH=$FLA_DIR:\$PYTHONPATH;"
fi

# NEEDS_TRITON: fla autotune on a shared Triton cache dies with "stale file handle"
# (job 2570257). One process here, so a per-job dir is enough.
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

MODEL_ARGS="pretrained=$PRETRAINED,tokenizer=$TOKENIZER_PATH,trust_remote_code=True,dtype=bfloat16,max_length=$MAX_LENGTH"
[ -n "$REVISION" ] && MODEL_ARGS="$MODEL_ARGS,revision=$REVISION"

### RUN ###
# --log_samples is always on: per-document records are the only way to recover per-question
# outcomes later. batch_size is a fixed integer, never "auto" -- under bf16 the batch
# composition perturbs logits, so a varying size makes per-document scores irreproducible.
srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    set -e
    # Guarded one-shot side-install, same shape as packages-fla. PYTHONPATH precedes
    # /opt/venv, so this shadows the container's broken copy.
    if [ ! -f '$LM_EVAL_DIR/lm_eval/__init__.py' ] || [ ! -f '$LM_EVAL_DIR/sqlitedict.py' ]; then
        # sqlitedict is the only lm-eval dep the container lacks (everything else, incl. torch
        # and transformers, comes from the image -- deliberately not shadowed here).
        pip install --quiet --no-deps --ignore-installed --target='$LM_EVAL_DIR' 'lm-eval==$LM_EVAL_VERSION' sqlitedict
    fi
    # Expanded from the batch script, not the container's own: the custom modeling_*.py inside
    # each HF checkpoint imports attn_bench, and transformers' check_imports refuses to load a
    # dynamic module whose imports are missing (job 3315463).
    export PYTHONPATH='$LM_EVAL_DIR:$PYTHONPATH'
    # Importing the hf backend is the check that matters -- a version match alone would not
    # have caught the container's mixed-release install (job 3314815).
    python -c \"import lm_eval, lm_eval.models.huggingface as h; assert lm_eval.__version__ == '$LM_EVAL_VERSION', 'got lm_eval '+lm_eval.__version__; assert lm_eval.__file__.startswith('$LM_EVAL_DIR'), 'shadowed by '+lm_eval.__file__; print('lm_eval', lm_eval.__version__, 'hf backend OK')\"
    $FLA_ENV
    $TRITON_ENV
    $SINK_ENV
    python -m lm_eval \
        --model hf \
        --model_args $MODEL_ARGS \
        --tasks $TASKS \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output_path $OUT_DIR \
        --log_samples \
        --use_cache $CACHE_DIR/cache \
        $LIMIT_ARG
    "
EVAL_STATUS=$?

### RUN METADATA ###
# Audit trail: lets the analysis assert all models shared a version / max_length / batch.
if [ $EVAL_STATUS -eq 0 ]; then
    RESOLVED_VERSION=$(srun --overlap --environment=$CONTAINER_ENV bash -c "PYTHONPATH=$LM_EVAL_DIR:\$PYTHONPATH python -c 'import lm_eval; print(lm_eval.__version__)'" 2>/dev/null | tail -1)
    cat > "$OUT_DIR/run_metadata.json" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "timestamp": "$(date -Iseconds)",
  "model": "${EXTERNAL_MODEL:-$MODEL}",
  "external_model": "${EXTERNAL_MODEL:-null}",
  "exp_name": "$EXP_NAME",
  "pretrained": "$PRETRAINED",
  "revision": "${REVISION:-null}",
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

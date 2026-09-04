#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:40:00
#SBATCH --job-name=convert-hf
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Convert a Megatron torch_dist checkpoint to a real HF model (steps 1 and 2 are both
# locally-edited copies of PDM's converters, see checkpoint_conversion/ for what and why),
# then validate the conversion two ways: a single-forward-pass logits diff on random tokens
# (step 3), and a real multi-step greedy-decode comparison on real excerpts (step 4) -- the
# latter is the only one that exercises this project's own KV-cache decode loop.
# Step 2 auto-detects the checkpoint's attention family from its own restored args and picks
# the matching attn_families/ module (see checkpoint_conversion/attn_families/registry.py) --
# families without one yet fail there with a clear NotImplementedError.
#
# Env vars:
#   MODEL checkpoint to use, see llama_checkpoints.sh (default: full-scf1)
#   OVERWRITE reconvert even if the HF checkpoint already exists in store
#   EXPECTED_ROPE_FACTOR expected rope-scaling-factor for the sanity check (default: 1)
#
# Usage: sbatch --export=MODEL=full-scf1 convert_and_validate_hf.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh

MODEL=${MODEL:-full-scf1}
OVERWRITE=${OVERWRITE:-}
EXPECTED_ROPE_FACTOR=${EXPECTED_ROPE_FACTOR:-1}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_HF_BASE=/users/$USER/store/hf-checkpoints
SCRATCH_CONVERT_DIR=/iopsstor/scratch/cscs/$USER/hf-checkpoints/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
CONV_DIR=$MEGATRON_LM_DIR/attn_bench/checkpoint_conversion

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

# NEEDS_FLA_052 (KDA): steps 1-4 all run Megatron chunk_kda; the container ships
# flash-linear-attention 0.4.2, which NaNs it. Side-install 0.5.2 into $FLA_DIR (guarded one-shot,
# persisted, gitignored) and prepend it to PYTHONPATH so it shadows the container copy in the srun
# below. Mirrors the KDA pretrain / measure_mem slurms.
FLA_VERSION=0.5.2
FLA_DIR=$MEGATRON_LM_DIR/attn_bench/packages-fla
if [ "$NEEDS_FLA_052" = "1" ]; then
    export PYTHONPATH=$FLA_DIR:$PYTHONPATH
fi

# Steps 2-4: sink's HF architecture (SinkLlamaForCausalLM) needs a patched flash-attn checkout
# for its fused kernel -- see flash-attention-cute-workflow.md
SINK_ENV=""
if [ "$IS_SINK_FAMILY" = "1" ]; then
    SINK_ENV="export FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention; pip install --quiet 'nvidia-cutlass-dsl==4.6.2';"
fi

# Step 3: fused kernels (sink's flash_attn.cute, GDN's fla) only support fp16/bf16, not the
# fp32 the logits comparison defaults to for precision. Step 3 casts both backends' models to
# this dtype -- GDN's fla kernels run on the Megatron side too, so this isn't HF-specific.
DTYPE_ARG=""
{ [ "$IS_SINK_FAMILY" = "1" ] || [ "$NEEDS_TRITON" = "1" ]; } && DTYPE_ARG="--dtype bfloat16"

# Step 4: TE 2.15 rejects fused attn in decode mode with a sink token (see llama_checkpoints.sh).
# Only the decode loop needs this -- step 3 is a single forward pass, no decode.
STEP4_MEGATRON_EXTRA="$MEGATRON_EXTRA"
STEP4_ENV=""
if [ "$NEEDS_UNFUSED_DECODE" = "1" ]; then
    STEP4_MEGATRON_EXTRA="$STEP4_MEGATRON_EXTRA --attention-backend unfused"
    STEP4_ENV="export NVTE_FUSED_ATTN=0; export CUDNN_LOGERR_DBG=1; export CUDNN_LOGDEST_DBG=stderr;"
fi

### CHECK STORE (before doing anything expensive) ###
STORE_HF_DIR=$STORE_HF_BASE/$EXP_NAME
if [ -z "$OVERWRITE" ] && [ -f "$STORE_HF_DIR/config.json" ]; then
    echo "$STORE_HF_DIR already present in store -- skipping."
    exit 0
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME"

# One-shot: ensure flash-linear-attention $FLA_VERSION in $FLA_DIR (KDA only). Guard skips it once
# installed. Both packages -- they merge into one fla/ namespace and flash-linear-attention alone
# ships no __init__.py. --no-deps keeps torch/triton out; --ignore-installed stops pip skipping on
# the container's 0.4.2.
if [ "$NEEDS_FLA_052" = "1" ]; then
    echo "[$(date)] Ensuring flash-linear-attention $FLA_VERSION in $FLA_DIR"
    srun --nodes=1 --ntasks=1 --overlap --environment=$CONTAINER_ENV bash -c "
        set -e
        if [ ! -f '$FLA_DIR/fla/__init__.py' ]; then
            pip install --quiet --no-deps --ignore-installed --target='$FLA_DIR' \
                'flash-linear-attention==$FLA_VERSION' 'fla-core==$FLA_VERSION'
        fi
        PYTHONPATH='$FLA_DIR' python3 -c \"import fla; assert fla.__version__.startswith('$FLA_VERSION'), 'got fla '+fla.__version__; print('fla', fla.__version__, 'OK')\"
    "
fi

# TRITON_CACHE_DIR: per-rank node-local cache for GDN's fla kernels -- a shared-FS cache
# crashes with "OSError: [Errno 116] Stale file handle" when fla autotunes/compiles on first
# use (see pretrain_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm). Single-rank here, but the
# default ~/.triton is still a shared filesystem path, so the fix still applies.
TRITON_ENV=""
if [ "$NEEDS_TRITON" = "1" ]; then
    TRITON_ENV="export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_\$SLURM_PROCID; mkdir -p \$TRITON_CACHE_DIR;"
fi

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    $TRITON_ENV

    echo '### STEP 1: torch_dist -> torch ###'
    # Rope scaling isn't restored from checkpoint, so \$MEGATRON_EXTRA supplies it.
    torchrun --nproc_per_node=1 $CONV_DIR/convert_torch_dist_to_torch.py \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model $TOKENIZER_PATH \
        --bf16 \
        --use-precision-aware-optimizer \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --use-distributed-optimizer \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --main-grads-dtype bf16 \
        $MEGATRON_EXTRA \
        --load $MODEL_DIR/checkpoints \
        --ckpt-convert-save $SCRATCH_CONVERT_DIR
    STEP1_STATUS=\$?
    if [ \$STEP1_STATUS -ne 0 ]; then echo 'Step 1 failed'; exit \$STEP1_STATUS; fi

    echo '### STEP 2: torch -> HF ###'
    $SINK_ENV
    # Forces every rerun to regenerate rather than skip (convert_megatron_to_hf.py no-ops if
    # $SCRATCH_CONVERT_DIR/HF already exists) -- also clears stale custom modeling_*.py
    # (sink, gated) from HF's trust_remote_code cache; that second rm is a no-op for full/swa.
    rm -rf '$SCRATCH_CONVERT_DIR/HF'
    rm -rf ~/.cache/huggingface/modules/transformers_modules/HF
    # Local copy (see convert_megatron_to_hf.py). utils.py is unforked -- pull PDM's via PYTHONPATH.
    PYTHONPATH=\"$PDM_DIR/src/convert:\$PYTHONPATH\" python $CONV_DIR/convert_megatron_to_hf.py --experiment-path $SCRATCH_CONVERT_DIR
    STEP2_STATUS=\$?
    if [ \$STEP2_STATUS -ne 0 ]; then echo 'Step 2 failed'; exit \$STEP2_STATUS; fi

    if [ "$HAS_ROPE" = "1" ]; then
        echo '### Sanity check: rope factor before the GPU-heavy steps 3/4 ###'
        python -c \"
import json
cfg = json.load(open('$SCRATCH_CONVERT_DIR/HF/config.json'))
# This transformers version stores rope scaling under 'rope_parameters', not 'rope_scaling'.
rope_cfg = cfg.get('rope_parameters') or cfg.get('rope_scaling', {})
factor = rope_cfg.get('factor')
assert factor == $EXPECTED_ROPE_FACTOR, f'expected rope factor == $EXPECTED_ROPE_FACTOR, got {factor!r} -- conversion likely dropped it silently'
print(f'rope factor = {factor} (expected $EXPECTED_ROPE_FACTOR) -- OK')
\"
        CFG_CHECK_STATUS=\$?
        if [ \$CFG_CHECK_STATUS -ne 0 ]; then echo 'Config sanity check failed'; exit \$CFG_CHECK_STATUS; fi
    else
        echo '### Sanity check: skipped -- HAS_ROPE=0 for this model (GDN has no rotary embeddings) ###'
    fi

    echo '### STEP 3: compare logits (random tokens, single forward pass) ###'
    # batch-size 1 (default 4 OOMs: fp32 SDPA has no flash kernel, so it materializes the full
    # batch x heads x seq x seq attention matrix -- 4x32x8192x8192x4 bytes = 32 GiB).
    # --megatron-extra-args must stay last -- it's argparse.REMAINDER, swallows everything after it.
    torchrun --nproc_per_node=1 $CONV_DIR/compare_megatron_hf_logits.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        --batch-size 1 \
        $DTYPE_ARG \
        --megatron-extra-args $MEGATRON_EXTRA
    STEP3_STATUS=\$?
    # Don't exit on a Step 3 failure -- run Step 4 regardless so both results land in the
    # same run instead of needing a separate resubmission to see the generation comparison.
    if [ \$STEP3_STATUS -ne 0 ]; then echo 'Step 3 failed (continuing to Step 4 anyway)'; fi

    echo '### STEP 4: compare generation (real excerpts, multi-step decode loop) ###'
    $STEP4_ENV
    torchrun --nproc_per_node=1 $CONV_DIR/compare_megatron_hf_generation.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        --data-folder $GUTENBERG_JSONL_DIR \
        --rep 0 --offset 0 --prefix-length 500 --suffix-length 500 --num-samples 5 \
        --megatron-extra-args $STEP4_MEGATRON_EXTRA
    STEP4_STATUS=\$?
    if [ \$STEP4_STATUS -ne 0 ]; then echo 'Step 4 failed'; fi

    if [ \$STEP3_STATUS -ne 0 ] || [ \$STEP4_STATUS -ne 0 ]; then exit 1; fi
    "
CONVERT_STATUS=$?

### Copy only the HF output to store (not the deletable intermediate torch/ format) ###
# Synchronous, unlike the other scripts' xfer-queued copy: downstream jobs look up the HF
# checkpoint in store only (no scratch fallback), so it needs to be there the moment this
# job exits. Low-frequency (one conversion per model), so inline store I/O here is fine.
if [ $CONVERT_STATUS -eq 0 ] && [ -d "$SCRATCH_CONVERT_DIR/HF" ]; then
    mkdir -p "$STORE_HF_DIR"
    cp -a "$SCRATCH_CONVERT_DIR/HF/." "$STORE_HF_DIR/"
fi

echo "END TIME: $(date)"

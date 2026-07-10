#!/bin/bash
# Canonical registry of trained models -- single source of truth for every eval driver,
# MODEL-driven inference slurm, and puller in attn_bench/{scripts,submissions}.
#
# To add a model everywhere at once: add its tag to MODELS and a case entry to
# model_config() below. Nothing else needs to change.
#
# Usage: source this file, call `model_config <tag>`. Sets (resets each call): EXP_NAME,
# CKPT_NAME, MEGATRON_EXTRA, NEEDS_TRITON, IS_SINK_FAMILY, NEEDS_UNFUSED_DECODE.
#   EXP_NAME             results/experiment dir name
#   CKPT_NAME            checkpoint dir name, if it differs from EXP_NAME
#   MEGATRON_EXTRA       flags not restored by --use-checkpoint-args
#   NEEDS_TRITON         1 if the model needs a per-rank node-local TRITON_CACHE_DIR (GDN)
#   IS_SINK_FAMILY       1 for sink/off-by-one (model identity). Used by the pull script's
#                        config-subset selection and --sink-scale in measure_mem.slurm.
#   NEEDS_UNFUSED_DECODE 1 if decode needs --attention-backend unfused + NVTE_FUSED_ATTN=0
#                        (TE 2.15 rejects fused attn at s_q==1 with a sink token). Only
#                        measure_mem.slurm decodes -- the long_* scripts do a plain forward
#                        pass, where TE's FusedAttention already supports softmax_type
#                        natively, so they never need this.

MODELS=(full gated full-xdoc-leak sink off-by-one gdn carry-r0 carry-r0.5 carry-r1)

# GDN linear-attention dims -- not restored by --use-checkpoint-args, must be re-passed.
GDN_DIMS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"

model_config() {
    local model="$1"
    CKPT_NAME=""
    NEEDS_TRITON=0
    IS_SINK_FAMILY=0
    NEEDS_UNFUSED_DECODE=0
    case "$model" in
        full)
            EXP_NAME=llama3-1b-full-attn-fineweb40B-gutenberg3B
            MEGATRON_EXTRA=""
            ;;
        gated)
            EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="--attention-output-gate"
            ;;
        full-xdoc-leak)
            EXP_NAME=llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B
            MEGATRON_EXTRA=""
            ;;
        sink)
            EXP_NAME=llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215
            MEGATRON_EXTRA="--softmax-type learnable"
            IS_SINK_FAMILY=1
            NEEDS_UNFUSED_DECODE=1
            ;;
        off-by-one)
            EXP_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215
            MEGATRON_EXTRA="--softmax-type off-by-one"
            IS_SINK_FAMILY=1
            NEEDS_UNFUSED_DECODE=1
            # checkpoint lives at the non-te215 path; EXP_NAME (results dir) stays -te215 to match the mem run
            CKPT_NAME=llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B
            ;;
        gdn)
            EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            ;;
        carry-r0)
            EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            ;;
        carry-r0.5)
            EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            ;;
        carry-r1)
            EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            ;;
        *)
            echo "Unknown MODEL=$model (expected one of: ${MODELS[*]})"
            exit 1
            ;;
    esac
    CKPT_NAME="${CKPT_NAME:-$EXP_NAME}"
}
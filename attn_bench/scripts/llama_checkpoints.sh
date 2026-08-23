#!/bin/bash
# Canonical registry of trained models -- single source of truth for every eval driver,
# MODEL-driven inference slurm, and puller in attn_bench/{scripts,submissions}.
#
# To add a model everywhere at once: add its tag to MODELS and a case entry to
# model_config() below. Nothing else needs to change.
#
# Usage: source this file, call `model_config <tag>`. Sets (resets each call): EXP_NAME,
# CKPT_NAME, MEGATRON_EXTRA, NEEDS_TRITON, IS_SINK_FAMILY, NEEDS_UNFUSED_DECODE, HAS_ROPE.
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
#   HAS_ROPE             0 for GDN (no rotary embeddings at all) -- skips
#                        convert_and_validate_hf.slurm's rope-scaling-factor sanity check,
#                        which otherwise fails on a family that never had rope to drop.

MODELS=(full-scf8 gated-scf8 full-xdoc-leak-scf8 sink-scf8 off-by-one-scf8 gdn carry-r0 carry-r0.5 carry-r1 full-goldfish-scf8 gdn-goldfish full-fineweb80B-scf8 full-long-scf8 full-long-split-1024-scf8 full-scf1 gated-scf1 sink-scf1 swa-w256-scf1 swa-w1024-scf1 swa-w4096-scf1)

# GDN linear-attention dims -- not restored by --use-checkpoint-args, must be re-passed.
GDN_DIMS="--experimental-attention-variant gated_delta_net \
    --linear-attention-freq '[1]*16' \
    --linear-num-key-heads 8 \
    --linear-num-value-heads 8 \
    --linear-key-head-dim 192 \
    --linear-value-head-dim 384 \
    --linear-conv-kernel-dim 4"

# Actual RoPE scaling factor these checkpoints trained with (see gpt_builders.py) -- not restored by --use-checkpoint-args, must be re-passed.
ROPE_SCF8="--use-rope-scaling --rope-scaling-factor 8"
ROPE_SCF1="--use-rope-scaling --rope-scaling-factor 1"

model_config() {
    local model="$1"
    CKPT_NAME=""
    NEEDS_TRITON=0
    IS_SINK_FAMILY=0
    NEEDS_UNFUSED_DECODE=0
    HAS_ROPE=1
    case "$model" in
        full-scf8)
            EXP_NAME=llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        gated-scf8)
            EXP_NAME=llama3-1b-gated-attn-scf8-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8 --attention-output-gate"
            ;;
        full-xdoc-leak-scf8)
            EXP_NAME=llama3-1b-full-attn-xdoc-attn-leak-scf8-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        sink-scf8)
            EXP_NAME=llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-te215
            MEGATRON_EXTRA="$ROPE_SCF8 --softmax-type learnable"
            IS_SINK_FAMILY=1
            NEEDS_UNFUSED_DECODE=1
            ;;
        off-by-one-scf8)
            EXP_NAME=llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-te215
            MEGATRON_EXTRA="$ROPE_SCF8 --softmax-type off-by-one"
            IS_SINK_FAMILY=1
            NEEDS_UNFUSED_DECODE=1
            # checkpoint lives at the non-te215 path; EXP_NAME (results dir) stays -te215 to match the mem run
            CKPT_NAME=llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B
            ;;
        gdn)
            EXP_NAME=llama3-1b-gdn-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            HAS_ROPE=0
            ;;
        carry-r0)
            EXP_NAME=llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            HAS_ROPE=0
            ;;
        carry-r0.5)
            EXP_NAME=llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            HAS_ROPE=0
            ;;
        carry-r1)
            EXP_NAME=llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            HAS_ROPE=0
            ;;
        full-goldfish-scf8)
            EXP_NAME=llama3-1b-full-attn-goldfish-scf8-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        gdn-goldfish)
            EXP_NAME=llama3-1b-gdn-goldfish-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$GDN_DIMS"
            NEEDS_TRITON=1
            HAS_ROPE=0
            ;;
        full-fineweb80B-scf8)
            EXP_NAME=llama3-1b-full-attn-scf8-fineweb80B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        full-long-scf8)
            EXP_NAME=llama3-1b-full-attn-scf8-fineweb40B-long-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        full-long-split-1024-scf8)
            EXP_NAME=llama3-1b-full-attn-scf8-fineweb40B-long-split-1024-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF8"
            ;;
        full-scf1)
            EXP_NAME=llama3-1b-full-attn-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1"
            ;;
        gated-scf1)
            EXP_NAME=llama3-1b-gated-attn-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1 --attention-output-gate"
            ;;
        sink-scf1)
            EXP_NAME=llama3-1b-sink-attn-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1 --softmax-type learnable"
            IS_SINK_FAMILY=1
            NEEDS_UNFUSED_DECODE=1
            ;;
        swa-w256-scf1)
            EXP_NAME=llama3-1b-swa-w256-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1 --window-size 256,0"
            ;;
        swa-w1024-scf1)
            EXP_NAME=llama3-1b-swa-w1024-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1 --window-size 1024,0"
            ;;
        swa-w4096-scf1)
            EXP_NAME=llama3-1b-swa-w4096-scf1-fineweb40B-gutenberg3B
            MEGATRON_EXTRA="$ROPE_SCF1 --window-size 4096,0"
            ;;
        *)
            echo "Unknown MODEL=$model (expected one of: ${MODELS[*]})"
            exit 1
            ;;
    esac
    CKPT_NAME="${CKPT_NAME:-$EXP_NAME}"
}
#!/usr/bin/env bash
# Llama 3.2 1B model config, defines NETWORK_SIZE_ARGS, LEARNING_RATE_ARGS, REGULARIZATION_ARGS, INITIALIZATION_ARGS, TOKENIZER_ARGS.
# Override a value by setting it before sourcing, e.g.:
#   LR=0.0001
#   source attn_bench/configs/models/llama3_1b.sh

# Add one-off args after sourcing, e.g.:
#   TRAINING_ARGS+=(--goldfish-loss ...)

# Numeric params are overridable
# Flags without a variable (RoPE, GQA, RMSNorm, SwiGLU, no bias) define the Llama 3.x architecture; if they need to be changed, create a new config


### ARCHITECTURE ###
# taken from https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json, the original values are provided in the comments

: ${SEQ_LEN:=8192}

: ${NUM_LAYERS:=16}
: ${HIDDEN_SIZE:=2048}
: ${FFN_HIDDEN_SIZE:=8192}
: ${NUM_ATTENTION_HEADS:=32}
: ${NUM_QUERY_GROUPS:=8}
: ${ROTARY_BASE:=500000}
: ${ROPE_SCALING_FACTOR:=32}
: ${NORM_EPSILON:=1e-5}
: ${MAX_POSITION_EMBEDDINGS:=131072}

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS # "num_hidden_layers": 16 -- following Llama-3.2-1B config
    --hidden-size $HIDDEN_SIZE # "hidden_size": 2048  -- following Llama-3.2-1B config
    --ffn-hidden-size $FFN_HIDDEN_SIZE  # "intermediate_size": 8192  -- following Llama-3.2-1B config
    --num-attention-heads $NUM_ATTENTION_HEADS # "num_attention_heads": 32  -- following Llama-3.2-1B config
    --group-query-attention
    --num-query-groups $NUM_QUERY_GROUPS # "num_key_value_heads": 8  -- following Llama-3.2-1B config
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS  # "max_position_embeddings": 131072  -- following Llama-3.2-1B config
    --position-embedding-type rope
    --use-rope-scaling
    --rotary-base $ROTARY_BASE # "rope_theta": 500000.0  -- following Llama-3.2-1B config
    --rope-scaling-factor $ROPE_SCALING_FACTOR # "rope_scaling": "factor": 32.0  -- following Llama-3.2-1B config
    # hardcoded in Megatron (megatron/core/models/common/embeddings/rotary_pos_embedding.py:92-98), correspond to Llama-3.2-1B config :
    # high_freq_factor=4
    # low_freq_factor=1
    # original_max_position_embeddings=8192
    # rope_type: llama3
    --normalization RMSNorm
    --norm-epsilon $NORM_EPSILON # "rms_norm_eps": 1e-05  -- following Llama-3.2-1B config
    --swiglu # "hidden_act": "silu"  -- following Llama-3.2-1B config
    --disable-bias-linear # "mlp_bias": false, "attention_bias": false  -- following Llama-3.2-1B config
    # Llama-3.2-1B config: "tie_word_embeddings": true -- embeddings tied by default; need to pass --untie-embeddings-and-output-weights to disable
    # Llama-3.2-1B config: "vocab_size": 128256 -- derived from the tokenizer
)

### LEARNING RATE ###
: ${LR:=0.0003}
: ${MIN_LR:=0.00003}
: ${LR_DECAY:=cosine}
: ${LR_WARMUP:=2000}

LEARNING_RATE_ARGS=(
    --lr $LR # 3 × 10^−4 -- following Xu et al
    --min-lr $MIN_LR # 3 × 10^−5 -- following Xu et al, LLama3 paper
    --lr-decay-style $LR_DECAY # cosine -- following LLama3 paper
    --lr-warmup-iters $LR_WARMUP # 2000 -- following LLama3 paper
)

### REGULARIZATION ###
: ${ATTENTION_DROPOUT:=0.0}
: ${HIDDEN_DROPOUT:=0.0}
: ${CLIP_GRAD:=1.0}
: ${WEIGHT_DECAY:=0.01}
: ${ADAM_BETA1:=0.9}
: ${ADAM_BETA2:=0.95}

REGULARIZATION_ARGS=(
    --attention-dropout $ATTENTION_DROPOUT # "attention_dropout": 0.0  -- following Llama-3.2-1B config
    --hidden-dropout $HIDDEN_DROPOUT # following standard practice / LLama doesn't use it
    --clip-grad $CLIP_GRAD # 1.0 — from the Llama 3 paper (gradient clipping = 1.0)
    --weight-decay $WEIGHT_DECAY # 0.01 -- following Xu et al
    --optimizer adam # AdamW (decoupled weight decay) -- following Llama 3 paper
    --adam-beta1 $ADAM_BETA1 # 0.9 — from the Llama 1/2 paper
    --adam-beta2 $ADAM_BETA2 # 0.95 — from the Llama 1/2 paper
)

### INITIALIZATION ###
: ${SEED:=28}
: ${INIT_STD:=0.02}
#: ${INIT_STD:=0.008944}

INITIALIZATION_ARGS=(
    --seed $SEED
    --init-method-std $INIT_STD # initializer_range: 0.02  -- following Llama-3.2-1B config + default Megatron-LM value (transformer_config.py:327)
)

### TOKENIZER ###
# Default path assumes the Llama 3.2 tokenizer is downloaded to the standard location.
# Override: TOKENIZER_PATH=/path/to/tokenizer
: ${TOKENIZER_PATH:=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b}

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
)
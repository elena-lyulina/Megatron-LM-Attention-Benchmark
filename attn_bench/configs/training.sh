#!/usr/bin/env bash
# Training defaults and checkpointing.
# Requires: EXP_DIR set before sourcing.

### TRANSFORMER ENGINE ###
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

### MIXED PRECISION ###
MIXED_PRECISION_ARGS=(
    --bf16
)

### VOCAB ###
# Pads vocab to a multiple of 128 for even TP splits
VOCAB_ARGS=(
    --make-vocab-size-divisible-by 128
)

### TRAINING ###
TRAINING_ARGS=(
    --no-check-for-nan-in-loss-and-grad
    --cross-entropy-loss-fusion
    --manual-gc
    --manual-gc-interval 5000
)

### CHECKPOINTING ###
: ${CHECKPOINT_STEPS:=1000}
: ${SAVE_RETAIN_INTERVAL:=10000}  # keep permanent snapshots every N steps; must be multiple of CHECKPOINT_STEPS
: ${CKPT_DIR:=$EXP_DIR/checkpoints}

mkdir -p $CKPT_DIR

CHECKPOINTING_ARGS=(
    --save $CKPT_DIR
    --load $CKPT_DIR # delete this to NOT reload from the latest checkpoint
    --save-interval $CHECKPOINT_STEPS
    --save-retain-interval $SAVE_RETAIN_INTERVAL
    --ckpt-format torch_dist
    --async-save
)

### TRIGGERS ###
: ${TRIGGER_DIR:=$EXP_DIR/triggers}
mkdir -p $TRIGGER_DIR
rm -f $TRIGGER_DIR/save $TRIGGER_DIR/exit
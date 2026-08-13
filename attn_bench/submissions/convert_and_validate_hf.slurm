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

# Convert a Megatron torch_dist checkpoint to a real HF LlamaForCausalLM (steps 1 and 2 are
# both locally-edited copies of PDM's converters, see checkpoint_conversion/ for what and why),
# then validate the conversion two ways: a single-forward-pass logits diff on random tokens
# (step 3), and a real multi-step greedy-decode comparison on real excerpts (step 4) -- the
# latter is the only one that exercises this project's own KV-cache decode loop.
# Full-attention only -- no path here for the custom attention variants (sink/off-by-one/
# gated/GDN aren't representable in plain HF Llama).
#
# Env vars:
#   MODEL checkpoint to use, see llama_checkpoints.sh (default: full-scf8)
#   OVERWRITE reconvert even if the HF checkpoint already exists in store
#   EXPECTED_ROPE_FACTOR expected rope-scaling-factor for the sanity check (default: 8)
#
# Usage: sbatch --export=MODEL=full-scf8 convert_and_validate_hf.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/$USER/scratch/PDM
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh
source $MEGATRON_LM_DIR/attn_bench/scripts/scratch_to_store.sh

MODEL=${MODEL:-full-scf8}
OVERWRITE=${OVERWRITE:-}
EXPECTED_ROPE_FACTOR=${EXPECTED_ROPE_FACTOR:-8}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
STORE_HF_BASE=/users/$USER/store/hf-checkpoints
SCRATCH_CONVERT_DIR=/iopsstor/scratch/cscs/$USER/hf-checkpoints/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
CONV_DIR=$MEGATRON_LM_DIR/attn_bench/checkpoint_conversion

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

### CHECK STORE (before doing anything expensive) ###
STORE_HF_DIR=$STORE_HF_BASE/$EXP_NAME
if [ -z "$OVERWRITE" ] && [ -f "$STORE_HF_DIR/config.json" ]; then
    echo "$STORE_HF_DIR already present in store -- skipping."
    exit 0
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1

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
    # Local copy (see convert_megatron_to_hf.py). utils.py is unforked -- pull PDM's via PYTHONPATH.
    PYTHONPATH=\"$PDM_DIR/src/convert:\$PYTHONPATH\" python $CONV_DIR/convert_megatron_to_hf.py --experiment-path $SCRATCH_CONVERT_DIR
    STEP2_STATUS=\$?
    if [ \$STEP2_STATUS -ne 0 ]; then echo 'Step 2 failed'; exit \$STEP2_STATUS; fi

    echo '### Sanity check: rope factor before the GPU-heavy steps 3/4 ###'
    python -c \"
import json
cfg = json.load(open('$SCRATCH_CONVERT_DIR/HF/config.json'))
# transformers>=5 uses 'rope_parameters', not 'rope_scaling'.
rope_cfg = cfg.get('rope_parameters') or cfg.get('rope_scaling', {})
factor = rope_cfg.get('factor')
assert factor == $EXPECTED_ROPE_FACTOR, f'expected rope factor == $EXPECTED_ROPE_FACTOR, got {factor!r} -- conversion likely dropped it silently'
print(f'rope factor = {factor} (expected $EXPECTED_ROPE_FACTOR) -- OK')
tie = cfg.get('tie_word_embeddings')
print(f'tie_word_embeddings = {tie} (expect False for full-scf8)')
\"
    CFG_CHECK_STATUS=\$?
    if [ \$CFG_CHECK_STATUS -ne 0 ]; then echo 'Config sanity check failed'; exit \$CFG_CHECK_STATUS; fi

    echo '### STEP 3: compare logits (random tokens, single forward pass) ###'
    python $CONV_DIR/compare_megatron_hf_logits.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        --megatron-extra-args $MEGATRON_EXTRA
    STEP3_STATUS=\$?
    if [ \$STEP3_STATUS -ne 0 ]; then echo 'Step 3 failed'; exit \$STEP3_STATUS; fi

    echo '### STEP 4: compare generation (real excerpts, multi-step decode loop) ###'
    python $CONV_DIR/compare_megatron_hf_generation.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        --data-folder $GUTENBERG_JSONL_DIR \
        --rep 0 --offset 0 --prefix-length 50 --suffix-length 50 --num-samples 5 \
        --megatron-extra-args $MEGATRON_EXTRA
    "
CONVERT_STATUS=$?

### Copy only the HF output to store (not the deletable intermediate torch/ format) ###
copy_scratch_to_store $CONVERT_STATUS "$SCRATCH_CONVERT_DIR/HF" "$STORE_HF_DIR"

echo "END TIME: $(date)"

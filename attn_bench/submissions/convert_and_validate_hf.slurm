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
#   MODEL      checkpoint to use, see llama_checkpoints.sh (default: full-scf8)
#   OVERWRITE  reconvert even if the HF checkpoint already exists in store
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
    # Architecture flags come from the checkpoint itself (--use-checkpoint-args, set in
    # convert_torch_dist_to_torch.py's args_defaults) -- no need to retype them here, and no
    # need to override any of them: the real checkpoint values are what Step 1 should use for
    # its own model construction. The only thing the checkpoint can't restore is rope scaling
    # (NVIDIA/Megatron-LM#6306), so \$MEGATRON_EXTRA supplies
    # --use-rope-scaling/--rope-scaling-factor per model tag.
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
    # Local copy (checkpoint_conversion/convert_megatron_to_hf.py), not PDM's: create_llama_config
    # needs the true rope-scaling base (always 8192 here, hardcoded in Megatron's own
    # rotary_pos_embedding.py) for rope_scaling.original_max_position_embeddings, which is a
    # different question from args.max_position_embeddings (this checkpoint's real buffer-size
    # setting, 131072) -- PDM's original reads the latter for both, which breaks once the two
    # diverge. See convert_megatron_to_hf.py's own docstring for the full explanation.
    # utils.py isn't forked -- it's unmodified, so reuse PDM's copy directly via PYTHONPATH
    # (scoped to this one command only, doesn't leak into steps 1/3/4).
    PYTHONPATH=\"$PDM_DIR/src/convert:\$PYTHONPATH\" python $CONV_DIR/convert_megatron_to_hf.py --experiment-path $SCRATCH_CONVERT_DIR
    STEP2_STATUS=\$?
    if [ \$STEP2_STATUS -ne 0 ]; then echo 'Step 2 failed'; exit \$STEP2_STATUS; fi

    echo '### Sanity check: does the written config.json actually have this checkpoints real rope-scaling-factor? ###'
    # Cheap, no-GPU check for exactly the failure mode that already bit this project once
    # (da76e69e1, silently dropped rope-scaling-factor) -- catch it here, not several GPU
    # minutes later as a subtle logits mismatch.
    python -c \"
import json
cfg = json.load(open('$SCRATCH_CONVERT_DIR/HF/config.json'))
factor = cfg.get('rope_scaling', {}).get('factor')
assert factor == 8, f'expected rope_scaling.factor == 8, got {factor!r} -- conversion likely dropped it silently'
print(f'rope_scaling.factor = {factor} (expected 8) -- OK')
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

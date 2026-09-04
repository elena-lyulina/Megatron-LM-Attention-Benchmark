#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:15:00
#SBATCH --job-name=debug-step4-generation
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Standalone re-run of convert_and_validate_hf.slurm's Step 4 (compare_megatron_hf_generation.py)
# only, generalized to any model and shorter --prefix-length/--suffix-length -- for fast
# iteration on a Step-4-only bug (e.g. the Qwen-hybrid Step-4 OOM, job 3296189) without
# re-running Steps 1-3.
#
# Does NOT redo Steps 1-3 -- reuses the existing scratch HF/ dir from convert_and_validate_hf.slurm
# (Step 1's output isn't even Step 4's dependency: compare_megatron_hf_generation.py's MegatronBackend
# loads the original torch_dist checkpoint directly via --ckpt-dir, never Step 1's converted copy --
# only Step 2/convert_megatron_to_hf.py consumes Step 1's output).
#
# DOES refresh $SCRATCH_CONVERT_DIR/HF/modeling_*.py from the live repo + clear the
# trust_remote_code module cache before every run: register_for_auto_class snapshots the file a
# family's *ForCausalLM class is DEFINED in (e.g. modeling_qwen_llama.py) into the checkpoint dir
# and ~/.cache/huggingface/modules/ at Step-2 conversion time, and never re-reads it from the repo
# afterward -- unlike files it only imports FROM (e.g. modeling_gated_llama.py's GatedLlamaAttention),
# which resolve live via PYTHONPATH every run. Without this refresh, edits to the *registered*
# file are silently invisible here (confirmed: job 3296451's QwenLlamaModel.forward debug print
# never fired, while the same-day modeling_gated_llama.py one did).
#
# Env vars:
#   MODEL           checkpoint tag (default: qwen), see llama_checkpoints.sh
#   PREFIX_LENGTH   prompt length (default: 500, matches convert_and_validate_hf.slurm's Step 4)
#   SUFFIX_LENGTH   tokens to generate (default: 20, vs Step 4's default 500 -- short for iteration)
#   NUM_SAMPLES     samples to run (default: 1)
#   WARMUP_SAMPLES  samples excluded from timing stats (default: 0 -- irrelevant for pure debugging)
#
# Usage: sbatch debug_step4_generation.slurm
#        sbatch --export=MODEL=qwen,SUFFIX_LENGTH=5 debug_step4_generation.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
source $MEGATRON_LM_DIR/attn_bench/scripts/llama_checkpoints.sh

MODEL=${MODEL:-qwen}
PREFIX_LENGTH=${PREFIX_LENGTH:-500}
SUFFIX_LENGTH=${SUFFIX_LENGTH:-20}
NUM_SAMPLES=${NUM_SAMPLES:-1}
WARMUP_SAMPLES=${WARMUP_SAMPLES:-0}
model_config "$MODEL"

MODEL_DIR=/users/$USER/store/pretrain-results/$CKPT_NAME
SCRATCH_CONVERT_DIR=/iopsstor/scratch/cscs/$USER/hf-checkpoints/$EXP_NAME
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
GUTENBERG_JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
CONV_DIR=$MEGATRON_LM_DIR/attn_bench/checkpoint_conversion

export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ ! -f "$SCRATCH_CONVERT_DIR/HF/config.json" ]; then
    echo "$SCRATCH_CONVERT_DIR/HF not found -- convert_and_validate_hf.slurm's scratch output"
    echo "must have been cleaned up. Re-run convert_and_validate_hf.slurm (Steps 1-2) first."
    exit 1
fi

TRITON_ENV=""
if [ "$NEEDS_TRITON" = "1" ]; then
    TRITON_ENV="export TRITON_CACHE_DIR=/tmp/triton_${USER}_\${SLURM_JOB_ID}_\$SLURM_PROCID; mkdir -p \$TRITON_CACHE_DIR;"
fi

# Same TE 2.15 + sink-token decode fix as convert_and_validate_hf.slurm's Step 4.
STEP4_MEGATRON_EXTRA="$MEGATRON_EXTRA"
STEP4_ENV=""
if [ "$NEEDS_UNFUSED_DECODE" = "1" ]; then
    STEP4_MEGATRON_EXTRA="$STEP4_MEGATRON_EXTRA --attention-backend unfused"
    STEP4_ENV="export NVTE_FUSED_ATTN=0; export CUDNN_LOGERR_DBG=1; export CUDNN_LOGDEST_DBG=stderr;"
fi

echo "RUN: model=$MODEL  exp=$EXP_NAME  prefix=$PREFIX_LENGTH  suffix=$SUFFIX_LENGTH  num_samples=$NUM_SAMPLES"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    $TRITON_ENV
    $STEP4_ENV

    echo '### refresh registered-class modeling_*.py snapshot(s) from the live repo ###'
    for f in $SCRATCH_CONVERT_DIR/HF/modeling_*.py; do
        [ -f \"\$f\" ] || continue
        repo_f=\"$CONV_DIR/attn_families/\$(basename \"\$f\")\"
        [ -f \"\$repo_f\" ] && cp \"\$repo_f\" \"\$f\" && echo \"Refreshed \$(basename \"\$f\") from repo\"
    done
    rm -rf ~/.cache/huggingface/modules/transformers_modules/HF

    echo '### compare_megatron_hf_generation.py (standalone, short suffix) ###'
    torchrun --nproc_per_node=1 $CONV_DIR/compare_megatron_hf_generation.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $SCRATCH_CONVERT_DIR/HF \
        --data-folder $GUTENBERG_JSONL_DIR \
        --rep 0 --offset 0 --prefix-length $PREFIX_LENGTH --suffix-length $SUFFIX_LENGTH \
        --num-samples $NUM_SAMPLES --warmup-samples $WARMUP_SAMPLES \
        --megatron-extra-args $STEP4_MEGATRON_EXTRA
    "

echo "END TIME: $(date)"

#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:10:00
#SBATCH --job-name=test-capture-attn-gated
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Quick smoke test for --capture-attention on the gated model.
# Uses --max-samples 4 and --suffix-length 50 to finish fast.
# Output goes to store/mem-results/SparseGutenberg/test/ so it doesn't pollute real results.
# Checks: hooks fire, the per-Rouge-L-bucket attn_scores / norm_attn maps and the gating
# histogram are written at run level with correct shapes.
#
# Usage:
#   sbatch attn_bench/submissions/test_capture_attn_gated.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26

### PATHS ###
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
PDM_DIR=/users/elyulina/scratch/PDM

EXP_NAME=llama3-1b-gated-attn-fineweb40B-gutenberg3B
MODEL_DIR=/users/elyulina/store/pretrain-results/$EXP_NAME
TEST_OUT=/users/elyulina/store/mem-results/SparseGutenberg/test/$EXP_NAME
GUTENBERG_JSONL_DIR=/users/elyulina/store/datasets/tokenized/gutenberg_rep_jsonl
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b

export PYTHONPATH=$MEGATRON_LM_DIR:$PDM_DIR:$PDM_DIR/src:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "RUN: exp=$EXP_NAME  prefix=50  suffix=50  max_samples=4  reps=0,16"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    -lu \
    bash -c "
    pip install numba -q
    export PYTHONPATH='${PYTHONPATH}'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID \
    torchrun \
        --nproc_per_node=4 \
        $MEGATRON_LM_DIR/attn_bench/evaluation/megatron_inference_sparse.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $TEST_OUT \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,16 \
        --offset 0 \
        --prefix-length 50 \
        --suffix-length 50 \
        --batch-size 4 \
        --max-samples 4 \
        --capture-attention \
        --megatron-extra-args --attention-output-gate

    ### SHAPE CHECK ###
    python - <<'EOF'
import numpy as np, pathlib, sys

# Capture is aggregated across all reps and written at the run level (not per-rep).
base = pathlib.Path('/users/elyulina/store/mem-results/SparseGutenberg/test/$EXP_NAME/inference/offset_0_prefix_50_suffix_50')
EXP_L, EXP_H = 16, 32          # llama3-1b gated
S = 50 + 50 - 1                # prompt_len + suffix - 1
BUCKETS = [f'{i:02d}-{i+1:02d}' for i in range(10)]
ok = True

for kind in ['attn_scores', 'norm_attn']:
    for label in BUCKETS:
        files = sorted(base.glob(f'{kind}_rouge_l_{label}_rank*.npz'))
        if not files:
            print(f'FAIL: no {kind}_rouge_l_{label}_rank*.npz in {base}')
            ok = False
            continue
        a = np.load(files[0])
        if a['mean'].shape != (EXP_L, EXP_H, S, S):
            print(f'FAIL {kind} {label}: mean {a[\"mean\"].shape} expect ({EXP_L},{EXP_H},{S},{S})')
            ok = False

print('Rouge-L bucket sample counts (attn_scores, summed over ranks):')
for label in BUCKETS:
    files = sorted(base.glob(f'attn_scores_rouge_l_{label}_rank*.npz'))
    total = sum(int(np.load(f)['count']) for f in files)
    print(f'  {label}: {total}')

gfiles = sorted(base.glob('gating_scores_rank*.npz'))
if not gfiles:
    print('FAIL: gating_scores_rank*.npz missing (gated model)')
    ok = False
else:
    g = np.load(gfiles[0])
    print(f'  gating hist: {g[\"hist\"].shape}  expect (10, {EXP_L}, {EXP_H}, 100)')
    if g['hist'].shape != (10, EXP_L, EXP_H, 100):
        print('FAIL: gating hist shape mismatch')
        ok = False
    if g['bin_edges'].shape != (101,):
        print('FAIL: gating bin_edges shape mismatch')
        ok = False

sys.exit(0 if ok else 1)
EOF
    echo \"Shape check exit code: \$?\""

echo "END TIME: $(date)"
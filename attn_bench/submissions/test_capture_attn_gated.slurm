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

# Quick smoke test for --capture-attention on the gated model.
# Uses --max-samples 4 and --suffix-length 50 to finish fast.
# Output goes to store/mem-results/SparseGutenberg/test/ so it doesn't pollute real results.
# Checks: hooks fire, .npz files are written with correct shapes, gate_matrix present.
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
GUTENBERG_JSONL_DIR=/users/elyulina/scratch/datasets/tokenized/gutenberg_rep_jsonl
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

base = pathlib.Path('/users/elyulina/store/mem-results/SparseGutenberg/test/$EXP_NAME/inference/offset_0_prefix_50_suffix_50')
ok = True
for rep in [0, 16]:
    d = base / f'rep_{rep}_greedy'
    stats_files = sorted(d.glob('attn_stats_rank*.npz'))
    exmpl_files = sorted(d.glob('attn_matrix_exmpl_rank*.npz'))
    if not stats_files or not exmpl_files:
        print(f'FAIL rep={rep}: missing .npz files in {d}')
        ok = False
        continue
    s = np.load(stats_files[0])
    m = np.load(exmpl_files[0])
    n_s = s['mean_attn'].shape[0]
    L, H = s['mean_attn'].shape[1], s['mean_attn'].shape[2]
    print(f'rep={rep}  n_samples={n_s}  layers={L}  heads={H}')
    print(f'  mean_attn:    {s[\"mean_attn\"].shape}   expect (*, {L}, {H}, 100)')
    print(f'  entropy:      {s[\"entropy\"].shape}    expect (*, {L}, {H}, 49)')
    print(f'  sink_mass:    {s[\"sink_mass\"].shape}  expect (*, {L}, {H}, 49)')
    print(f'  gate_mean:    {s[\"gate_mean\"].shape if \"gate_mean\" in s else \"MISSING\"}')
    print(f'  matrix:       {m[\"matrix\"].shape}    expect ({L}, {H}, 49, 100)')
    print(f'  gate_matrix:  {m[\"gate_matrix\"].shape if \"gate_matrix\" in m else \"MISSING\"}')
    if 'gate_mean' not in s:
        print(f'FAIL rep={rep}: gate_mean missing from attn_stats')
        ok = False
    if 'gate_matrix' not in m:
        print(f'FAIL rep={rep}: gate_matrix missing from attn_matrix_exmpl')
        ok = False
    if m['matrix'].shape != (L, H, 49, 100):
        print(f'FAIL rep={rep}: matrix shape mismatch')
        ok = False

sys.exit(0 if ok else 1)
EOF
    echo \"Shape check exit code: \$?\""

echo "END TIME: $(date)"
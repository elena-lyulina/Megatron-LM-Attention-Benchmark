#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:20:00
#SBATCH --job-name=check-sink-kernel
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Does flash_attn's cute-DSL module (FA4) run a real learnable_sink forward pass on this
# GH200 node? Uses a persistent, patched flash-attention checkout (same pattern as PDM --
# see attn_bench/results/docs/flash-attention-cute-workflow.md for one-time setup) rather
# than a fresh clone: main's own fmax() is missing a required nvvm.fmax arg, breaking every
# cute forward call on Hopper, patched at attn_bench/utils/flash_attention_patch.txt.
# See attn_bench/results/docs/hf_inference_kernel_support.md for the full investigation.
#
# Usage: sbatch check_sink_kernel.slurm

set -euo pipefail

CONTAINER_ENV=${CONTAINER_ENV:-nemo_26.04_te2.15}
FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention
SCRATCH_DIR=/users/$USER/scratch/tmp/check_sink_kernel_$SLURM_JOB_ID

mkdir -p "$SCRATCH_DIR"

cat > "$SCRATCH_DIR/check_sink_kernel.py" <<'PYEOF'
import inspect, sys, traceback, torch

def main(src_dir):
    import flash_attn
    print(f'installed flash_attn: {flash_attn.__version__} at {flash_attn.__file__}')
    flash_attn.__path__.append(f'{src_dir}/flash_attn')

    from flash_attn.cute.interface import flash_attn_func
    print('imported flash_attn.cute.interface OK')
    sink_param = 'learnable_sink'
    assert sink_param in inspect.signature(flash_attn_func).parameters

    B, S, H, D = 1, 128, 4, 64
    q = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, S, H, D, device='cuda', dtype=torch.bfloat16)
    sink = torch.randn(H, device='cuda', dtype=torch.bfloat16)

    result = flash_attn_func(q, k, v, causal=True, learnable_sink=sink)
    out = result[0] if isinstance(result, tuple) else result
    print(f'forward pass OK, output shape {tuple(out.shape)}, dtype {out.dtype}')
    print(f'RESULT: flash_attn_func(..., learnable_sink=...) works on {torch.cuda.get_device_name(0)}.')

if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except Exception:
        print('RESULT: FAILED')
        traceback.print_exc()
        sys.exit(1)
PYEOF

srun --environment=$CONTAINER_ENV bash -c "
set -e
pip install --quiet 'nvidia-cutlass-dsl==4.6.2'
python '$SCRATCH_DIR/check_sink_kernel.py' '$FLASH_ATTN_SRC_DIR'
"

rm -rf "$SCRATCH_DIR"

#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=0:20:00
#SBATCH --job-name=check-sink-kernel-correctness
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# check_sink_kernel.slurm only proved flash_attn.cute's learnable_sink runs without
# crashing (random tensors, shape/dtype check only -- no reference to compare against, so
# a numerically wrong kernel would have "passed" just as easily). This checks the actual
# numbers: same random q/k/v/sink fed into flash_attn.cute AND a plain-PyTorch reference
# implementing Megatron's exact sink formula (fused_softmax.py's SoftmaxOne -- scale+causal
# mask applied to real logits, sink appended as-is, softmax over the extended dim, sink
# column dropped), then diffed. No checkpoint needed -- this validates the kernel's math,
# not any specific trained weights.
# Runs two shapes: equal q/k/v heads, and sink-scf1's real GQA shape (32 query heads, 8
# key/value heads) -- the first version of this test only covered the equal-heads case and
# missed a real GQA-specific bug (job 3118223, sink-scf1 end-to-end conversion: 2.17%
# argmax agreement with Megatron, near-random, not just imprecise).
#
# Usage: sbatch check_sink_kernel_correctness.slurm

set -euo pipefail

CONTAINER_ENV=${CONTAINER_ENV:-nemo_26.04_te2.15}
FLASH_ATTN_SRC_DIR=/users/$USER/scratch/flash-attention
SCRATCH_DIR=/users/$USER/scratch/tmp/check_sink_kernel_correctness_$SLURM_JOB_ID

mkdir -p "$SCRATCH_DIR"

cat > "$SCRATCH_DIR/check_sink_kernel_correctness.py" <<'PYEOF'
import math
import sys
import torch

def reference_sink_attention(q, k, v, sink):
    """Megatron's exact formula (fused_softmax.py: forward_torch_softmax + SoftmaxOne).
    q: [B, S, Hq, D]. k/v: [B, S, Hkv, D], Hq a multiple of Hkv (GQA) or equal. sink: [Hq].
    Returns [B, S, Hq, D]."""
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    n_rep = Hq // Hkv
    qf, kf, vf = q.float(), k.float(), v.float()
    if n_rep > 1:
        # matches transformers' repeat_kv: query heads [g*n_rep : (g+1)*n_rep] share kv head g.
        kf = kf.repeat_interleave(n_rep, dim=2)
        vf = vf.repeat_interleave(n_rep, dim=2)
    scale = D ** -0.5
    scores = torch.einsum('bshd,bthd->bhst', qf, kf) * scale
    causal_mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    sink_col = sink.float().reshape(1, Hq, 1, 1).expand(B, Hq, S, 1)
    extended = torch.cat([scores, sink_col], dim=-1)
    probs = torch.softmax(extended, dim=-1)[..., :-1]
    out = torch.einsum('bhst,bthd->bshd', probs, vf)
    return out

def sanity_check_reference_formula(q, k, v):
    """Cross-checks reference_sink_attention (minus the sink term, which SDPA can't do)
    against torch's own scaled_dot_product_attention (enable_gqa handles unequal q/k/v
    head counts natively), so a mistake in the reference isn't mistaken for a kernel bug."""
    import torch.nn.functional as F
    Hq = q.shape[2]
    mine = reference_sink_attention(q, k, v, torch.full((Hq,), float('-inf'), device=q.device))
    q2, k2, v2 = (t.transpose(1, 2).float() for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True, enable_gqa=(q.shape[2] != k.shape[2])).transpose(1, 2)
    diff = (mine - ref).abs().max().item()
    print(f'reference formula sanity check vs SDPA (sink=-inf, i.e. no-op): max abs diff {diff:.6f}')
    # GQA's SDPA path uses a different (non-materializing) reduction order than my naive
    # repeat_interleave+einsum, so it picks up slightly more fp32 rounding noise than the
    # equal-heads case (0.0008) -- 3e-3 still catches a real indexing/logic bug (which would
    # show as a much larger diff) while tolerating that.
    assert diff < 3e-3, 'reference_sink_attention disagrees with SDPA even with the sink term neutralized -- fix the reference before trusting the kernel comparison'

def check_shape(flash_attn_func, label, B, S, Hq, Hkv, D):
    torch.manual_seed(0)
    q = torch.randn(B, S, Hq, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16)
    sink = torch.randn(Hq, device='cuda', dtype=torch.bfloat16)

    sanity_check_reference_formula(q, k, v)

    result = flash_attn_func(q, k, v, causal=True, learnable_sink=sink)
    kernel_out = (result[0] if isinstance(result, tuple) else result).float()
    ref_out = reference_sink_attention(q, k, v, sink)

    abs_diff = (kernel_out - ref_out).abs()
    atol, rtol = 1e-3, 0.02
    close = (abs_diff <= atol + rtol * ref_out.abs()).float().mean().item()
    print(f'[{label}] fraction close (atol={atol}, rtol={rtol}): {close:.4f}, max abs diff: {abs_diff.max().item():.6f}')
    threshold = 0.99
    if close >= threshold:
        print(f'[{label}] RESULT: CORRECT -- {100 * close:.2f}% of values close (need >= {100 * threshold:.0f}%).')
        return True
    print(f'[{label}] RESULT: MISMATCH -- only {100 * close:.2f}% of values close (need >= {100 * threshold:.0f}%).')
    return False

def main(src_dir):
    import flash_attn
    flash_attn.__path__.append(f'{src_dir}/flash_attn')
    from flash_attn.cute.interface import flash_attn_func

    ok_equal = check_shape(flash_attn_func, 'equal-heads', B=1, S=128, Hq=4, Hkv=4, D=64)
    ok_gqa = check_shape(flash_attn_func, 'gqa-sink-scf1-shape', B=1, S=128, Hq=32, Hkv=8, D=64)

    if not (ok_equal and ok_gqa):
        sys.exit(1)

if __name__ == '__main__':
    main(sys.argv[1])
PYEOF

srun --environment=$CONTAINER_ENV bash -c "
set -e
pip install --quiet 'nvidia-cutlass-dsl==4.6.2'
python '$SCRATCH_DIR/check_sink_kernel_correctness.py' '$FLASH_ATTN_SRC_DIR'
"

rm -rf "$SCRATCH_DIR"

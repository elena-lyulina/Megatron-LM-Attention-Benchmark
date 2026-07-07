"""Check that the state-norm segment-and-carry matches a single kernel call.

The wrapper in attn_bench/evaluation/gdn_state_norm.py runs the chunk kernel over slices
and carries the state across them. This asserts that gives the same output (all positions)
and same final state as one call over the whole sequence -- i.e. slicing does not change
the numbers. Runs on random tensors, no checkpoint needed.

fp32 is used so the tiny cross-tiling difference is the only signal; production runs bf16,
but that noise is unrelated to the property under test.

Usage (via torchrun, 1 GPU):
    torchrun --nproc_per_node=1 attn_bench/tests/test_gdn_state_norm.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from attn_bench.evaluation.gdn_state_norm import StateNormAccumulator, _make_wrapper

# Real GDN dims (num_value_heads, key_head_dim, value_head_dim) so we exercise shapes the
# kernel is known to support; batch 1 as in the eval.
NUM_HEADS = 8
KEY_HEAD_DIM = 192
VALUE_HEAD_DIM = 384
STATE_CHUNK = 64
TOL = 1e-2  # a broken carry resets the state each slice -> huge diff, so this is generous


def _random_inputs(seq_len, device):
    # Mimic what GatedDeltaNet hands the kernel: l2-normed q/k, plain v, log-space negative g,
    # sigmoid beta. Distribution does not matter for the equivalence, only validity.
    g_kwargs = dict(dtype=torch.float32, device=device)
    query = F.normalize(torch.randn(1, seq_len, NUM_HEADS, KEY_HEAD_DIM, **g_kwargs), dim=-1)
    key = F.normalize(torch.randn(1, seq_len, NUM_HEADS, KEY_HEAD_DIM, **g_kwargs), dim=-1)
    value = torch.randn(1, seq_len, NUM_HEADS, VALUE_HEAD_DIM, **g_kwargs)
    g = -F.softplus(torch.randn(1, seq_len, NUM_HEADS, **g_kwargs))
    beta = torch.rand(1, seq_len, NUM_HEADS, **g_kwargs)
    return query, key, value, g, beta


@torch.no_grad()
def _check_length(real_fn, seq_len, device):
    query, key, value, g, beta = _random_inputs(seq_len, device)
    common = dict(g=g, beta=beta, use_qk_l2norm_in_kernel=False, cu_seqlens=None)

    out_ref, state_ref = real_fn(query, key, value, initial_state=None, output_final_state=True, **common)

    accum = StateNormAccumulator(STATE_CHUNK, layer_ids=[0], num_heads=NUM_HEADS, device=device)
    wrapper = _make_wrapper(real_fn, STATE_CHUNK, accum, layer_number=0)
    out_seg, state_seg = wrapper(query, key, value, initial_state=None, output_final_state=True, **common)

    out_diff = (out_ref - out_seg).abs().max().item()
    state_diff = (state_ref - state_seg).abs().max().item()
    n_full = seq_len // STATE_CHUNK
    recorded = accum._seq[0].shape[0]

    ok = out_diff < TOL and state_diff < TOL and recorded == n_full
    print(f"  seq_len={seq_len}: max|dout|={out_diff:.2e}  max|dstate|={state_diff:.2e}  "
          f"boundaries logged={recorded} (expected {n_full})  -> {'ok' if ok else 'FAIL'}")
    return ok


def main():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.manual_seed(1234)

    print("\n### Test: state_norm_segmentation ###")
    clean_ok = _check_length(chunk_gated_delta_rule, 256, device)   # multiple of state_chunk
    tail_ok = _check_length(chunk_gated_delta_rule, 200, device)    # 3 full slices + a 8-token tail
    ok = clean_ok and tail_ok
    print(f"[{'PASS' if ok else 'FAIL'}] state_norm_segmentation: segment-and-carry == single call")

    print("\n### GDN state-norm — Summary ###")
    print(f"  {'PASS' if ok else 'FAIL'}   state_norm_segmentation")
    print(f"### Verdict: {'ALL PASS' if ok else 'FAILURES PRESENT'} ###")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
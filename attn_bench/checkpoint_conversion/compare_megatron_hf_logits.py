"""
Validates a Megatron -> HF conversion: compares logits on random tokens (data-agnostic --
tests only that HF computes the same thing as Megatron, not generation quality).

--self-comparison additionally forwards each backend twice against itself, as a kernel-
nondeterminism noise floor to compare the cross-backend divergence against.

Adapted from swiss-ai/Megatron-LM's tools/checkpoint/loader_core.py --test-logits and
tools/checkpoint/saver_swissai_hf.py, but runs both loads and the comparison in one script
instead of two.

Usage (see attn_bench/submissions/convert_and_validate_hf.slurm):
    python attn_bench/checkpoint_conversion/compare_megatron_hf_logits.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $HF_SAVE_DIR \
        --megatron-extra-args --use-rope-scaling --rope-scaling-factor 8

    # + noise-floor baselines (see attn_bench/submissions/debug_bf16_inference_backends.slurm):
    python attn_bench/checkpoint_conversion/compare_megatron_hf_logits.py --self-comparison ...
"""
from __future__ import annotations

import argparse

import torch

from attn_bench.evaluation.inference_backend import (HFBackend,
                                                     InferenceBackend,
                                                     MegatronBackend)


def _report(ref_output: torch.Tensor, output: torch.Tensor, seq_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnostic report for two [B, S, V] logits tensors (cross-backend, or two forward passes
    of the same backend for self-comparison). Returns (agree_frac, close_frac); self-comparison
    callers don't assert on these since some kernel-nondeterminism disagreement is expected."""
    assert output.size() == ref_output.size()

    preds_ref = torch.max(ref_output, dim=-1)[1]
    preds_new = torch.max(output, dim=-1)[1]
    disagree_mask = preds_ref != preds_new
    n_disagree = int(torch.sum(disagree_mask))
    n_total = preds_ref.numel()
    agree = 1 - n_disagree / n_total
    print(f"Agrees on {100 * agree:.2f}% of predictions "
          f"({n_total - n_disagree}/{n_total} positions agree, {n_disagree} disagree)")

    # Near-tie diagnostic: is disagreement a close race (bf16 tipping a near-tie) or one pass
    # confidently preferring a token the other ranks far down its list (a real divergence)?
    # Purely descriptive -- no external tolerance, each pass is only compared to itself/its own
    # ranking. Computed before check two flattens ref_output/output below.
    K = 10
    if disagree_mask.any():
        ref_topk_vals, ref_topk_idx = torch.topk(ref_output, K, dim=-1)   # [B, S, K], descending
        new_topk_vals, new_topk_idx = torch.topk(output, K, dim=-1)

        # 1. Consecutive-rank gaps within each pass's own top-K, at the SAME agree/disagree
        # split as above: rank i->i+1 is the gap between that pass's own i-th and (i+1)-th
        # ranked logit at a position -- large = confident, small = near-tie (bf16-flippable).
        # Flat across ranks = an N-way tie; peaked at rank1->2 only = a clean two-way near-tie.
        print(f"Median gap between consecutive top-{K} ranks (own ranking, own logits), "
              f"disagreeing positions (n={n_disagree}) vs agreeing (n={n_total - n_disagree}):")
        gaps = {}
        for label, vals in [("ref", ref_topk_vals), ("new", new_topk_vals)]:
            gaps[label] = vals[..., :-1] - vals[..., 1:]  # [B, S, K-1], rank_i - rank_{i+1}
        header = f"  {'rank pair':<10}{'ref disagree':>13}{'ref agree':>13}{'new disagree':>13}{'new agree':>13}"
        print(header)
        for i in range(K - 1):
            row = [f"  {f'{i + 1}->{i + 2}':<10}"]
            for label in ("ref", "new"):
                for mask in (disagree_mask, ~disagree_mask):
                    vals = gaps[label][..., i][mask]
                    row.append(f"{vals.median():>13.4f}" if vals.numel() > 0 else f"{'n/a':>13}")
            print("".join(row))

        # 2. Where does the other pass's pick fall in this pass's own top-K ranking, at
        # disagreeing positions? Rank-resolved (not just "is it #2"); "outside" = neither.
        def rank_in_topk(topk_idx: torch.Tensor, pick: torch.Tensor) -> torch.Tensor:
            match = topk_idx == pick.unsqueeze(-1)          # [B, S, K]
            found = match.any(-1)
            rank = match.float().argmax(-1)
            return torch.where(found, rank, torch.full_like(rank, -1))

        new_rank_in_ref = rank_in_topk(ref_topk_idx, preds_new)   # [B, S] -- full tensor, not masked
        ref_rank_in_new = rank_in_topk(new_topk_idx, preds_ref)   # [B, S]
        hist_new_in_ref = torch.bincount((new_rank_in_ref[disagree_mask] + 1).clamp(min=0), minlength=K + 1)
        hist_ref_in_new = torch.bincount((ref_rank_in_new[disagree_mask] + 1).clamp(min=0), minlength=K + 1)
        rank_cols = "  ".join(f"rank{i + 1}={hist_new_in_ref[i + 1].item()}" for i in range(K))
        print(f"Where the OTHER backend's pick lands in THIS backend's own top-{K} ranking "
              f"(disagreements only; a 'rank1' hit can also mean an exact tie at the top -- "
              f"see the note below the DISAGREE_TOKEN dump):")
        print(f"  new's pick in ref's ranking: outside_top{K}={hist_new_in_ref[0].item()}  {rank_cols}")
        rank_cols = "  ".join(f"rank{i + 1}={hist_ref_in_new[i + 1].item()}" for i in range(K))
        print(f"  ref's pick in new's ranking: outside_top{K}={hist_ref_in_new[0].item()}  {rank_cols}")

        # 3. Disagreement rate by absolute sequence position -- rising = accumulating error
        # (e.g. recurrent-state drift), flat = position-independent noise.
        n_buckets = min(16, seq_length)
        edges = torch.linspace(0, seq_length, n_buckets + 1).long()
        print(f"Disagreement rate by sequence position (0..{seq_length - 1}, {n_buckets} buckets):")
        for i in range(n_buckets):
            lo, hi = edges[i].item(), edges[i + 1].item()
            bucket = disagree_mask[:, lo:hi]
            print(f"  pos [{lo:5d},{hi:5d}): disagree={100 * bucket.float().mean():.2f}%  (n={bucket.numel()})")

        # 4. Per-disagreement dump. Header line ("DISAGREE_TOKEN batch=.. pos=..", greppable)
        # plus tab-indented detail lines: each backend's own top-K token ids, with a '*' marking
        # where the OTHER backend's pick actually sits in that ranking (no '*' at all = outside
        # the top-K entirely), and the logit gap after each rank.
        print(f"Per-disagreement detail ({n_disagree} tokens):")
        ref_gaps_full = ref_topk_vals[..., :-1] - ref_topk_vals[..., 1:]
        new_gaps_full = new_topk_vals[..., :-1] - new_topk_vals[..., 1:]
        for b, s in disagree_mask.nonzero(as_tuple=False).tolist():
            r_pick, n_pick = preds_ref[b, s].item(), preds_new[b, s].item()
            ref_ids_str = ", ".join(f"{t}*" if t == n_pick else str(t) for t in ref_topk_idx[b, s].tolist())
            new_ids_str = ", ".join(f"{t}*" if t == r_pick else str(t) for t in new_topk_idx[b, s].tolist())
            ref_gap_str = ", ".join(f"{g:.4f}" for g in ref_gaps_full[b, s].tolist())
            new_gap_str = ", ".join(f"{g:.4f}" for g in new_gaps_full[b, s].tolist())
            print(f"DISAGREE_TOKEN batch={b} pos={s}")
            print(f"\tref_pick_token_id={r_pick}\tnew_pick_token_id={n_pick}")
            print(f"\tref_top{K}_token_ids (*=where new's pick sits): {ref_ids_str}")
            print(f"\tref_top{K}_gaps_to_next_rank:                   {ref_gap_str}")
            print(f"\tnew_top{K}_token_ids (*=where ref's pick sits): {new_ids_str}")
            print(f"\tnew_top{K}_gaps_to_next_rank:                   {new_gap_str}")
        print("Note: a '*' at rank1 (first id in the list) whose gap-to-next-rank is 0.0000 "
              "means torch.max/torch.topk broke an exact tie differently, not that the other "
              "backend picked its confident #1.")

    # Check two: atol and rtol on all logits.
    atol = 1e-05
    rtol = 0.016
    output = torch.flatten(output).cpu()
    ref_output = torch.flatten(ref_output).cpu()
    abs_diff = torch.abs(output - ref_output)
    rel_diff = abs_diff / torch.abs(ref_output)
    rel_diff_inf_mask = torch.isinf(rel_diff)
    rel_diff_no_inf = rel_diff[~rel_diff_inf_mask]
    close_mask = abs_diff <= atol + rtol * torch.abs(ref_output)
    close = torch.sum(close_mask) / output.numel()
    print(f"Logits are close on {100 * close:.2f}% of values")
    print(f"Max absolute difference: {torch.max(abs_diff)}")
    print(f"Mean absolute difference: {torch.mean(abs_diff)}")
    print(f"Max relative difference: {torch.max(rel_diff)}")
    print(f"Mean relative difference (no inf): {torch.mean(rel_diff_no_inf)}")
    print(f"Relative difference inf proportion: {torch.mean(rel_diff_inf_mask.float())}")

    return agree, close


def compare_logits(ref_backend: InferenceBackend, new_backend: InferenceBackend, vocab_size: int,
                   seq_length: int, batch_size: int = 4, device: str = "cuda",
                   dtype: torch.dtype = torch.float32):
    """Forward two backends on the same random tokens, diff the logits, assert on the
    thresholds below. dtype defaults to fp32; pass bfloat16 for fused kernels (e.g. sink,
    GDN) that don't support fp32."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    ref_backend.model = ref_backend.model.to(device).to(dtype)
    with torch.no_grad():
        ref_output = ref_backend.forward_logits(tokens, position_ids)
    del ref_backend
    torch.cuda.empty_cache()

    new_backend.model = new_backend.model.to(device).to(dtype)
    with torch.no_grad():
        output = new_backend.forward_logits(tokens, position_ids)
    del new_backend
    torch.cuda.empty_cache()

    argmax_threshold = 0.99
    close_threshold = 0.95
    agree, close = _report(ref_output, output, seq_length)

    # Stats are always printed above regardless of pass/fail, so a failure on either one still
    # leaves the full picture in the log.
    assert agree >= argmax_threshold, f"Only {100 * agree:.2f}% argmax agreement (need >= {100 * argmax_threshold:.0f}%)"
    assert close >= close_threshold, f"Only {100 * close:.2f}% of logits close (need >= {100 * close_threshold:.0f}%)"


def self_compare_logits(backend: InferenceBackend, vocab_size: int, seq_length: int,
                        batch_size: int = 4, device: str = "cuda",
                        dtype: torch.dtype = torch.bfloat16):
    """Forward the same backend twice on the same input -- the kernel-nondeterminism noise
    floor. Dropout is 0 for these checkpoints, so the only source of difference is the
    kernel's own (non-deterministic) execution. No assertion -- diagnostic, not a gate."""
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)

    backend.model = backend.model.to(device).to(dtype)
    with torch.no_grad():
        ref_output = backend.forward_logits(tokens, position_ids)
        output = backend.forward_logits(tokens, position_ids)

    _report(ref_output, output, seq_length)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True, help="Original torch_dist Megatron checkpoint")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--hf-dir", required=True, help="Output of convert_megatron_to_hf.py")
    parser.add_argument("--self-comparison", action="store_true",
                       help="Also forward each backend twice against itself, as a "
                            "kernel-nondeterminism noise floor.")
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                       help="fp32 for precision (default); bfloat16 for fused kernels that don't support fp32 (e.g. sink)")
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = getattr(torch, args.dtype)

    megatron_backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    megatron_backend.load_model()
    hf_backend = HFBackend(args.hf_dir)
    hf_backend.load_model()
    vocab_size = hf_backend.model.config.vocab_size

    print("\n=== cross: megatron vs hf ===")
    try:
        compare_logits(megatron_backend, hf_backend, vocab_size, args.seq_length, args.batch_size, dtype=dtype)
        print("Logits check passed.")
        cross_failed = False
    except AssertionError as e:
        print(f"Logits check FAILED: {e}")
        cross_failed = True

    if args.self_comparison:
        print("\n=== self-comparison: megatron vs itself (kernel-nondeterminism noise floor) ===")
        # megatron_backend.model is a Float16Module wrapper, not the raw GPTModel, so it has no
        # .vocab_size of its own -- reuse the HF vocab_size (same tokenizer/padded vocab as the
        # megatron checkpoint it was converted from).
        self_compare_logits(megatron_backend, vocab_size, args.seq_length, args.batch_size, dtype=dtype)
        print("\n=== self-comparison: hf vs itself (kernel-nondeterminism noise floor) ===")
        self_compare_logits(hf_backend, vocab_size, args.seq_length, args.batch_size, dtype=dtype)

    if cross_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

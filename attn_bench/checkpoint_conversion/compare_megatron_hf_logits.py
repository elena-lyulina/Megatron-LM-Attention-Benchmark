"""
Validates a Megatron -> HF conversion: compares logits on random tokens (data-agnostic --
tests only that HF computes the same thing as Megatron, not generation quality).

Adapted from swiss-ai/Megatron-LM's tools/checkpoint/loader_core.py --test-logits and
tools/checkpoint/saver_swissai_hf.py, but runs both loads and the comparison in one script
instead of two.

Usage (see attn_bench/submissions/convert_and_validate_hf.slurm):
    python attn_bench/checkpoint_conversion/compare_megatron_hf_logits.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $HF_SAVE_DIR \
        --megatron-extra-args --use-rope-scaling --rope-scaling-factor 8
"""
from __future__ import annotations

import argparse

import torch

from attn_bench.evaluation.inference_backend import (HFBackend,
                                                     InferenceBackend,
                                                     MegatronBackend)


def compare_logits(ref_backend: InferenceBackend, new_backend: InferenceBackend, vocab_size: int,
                   seq_length: int, batch_size: int = 4, device: str = "cuda",
                   dtype: torch.dtype = torch.float32):
    """Forward both backends on the same random tokens and diff the logits. dtype defaults to
    fp32 for precision, but fused kernels like flash_attn.cute (sink) only support fp16/bf16
    -- pass bfloat16 for those."""
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

    assert output.size() == ref_output.size()

    # Check one: both models agree on next-token predictions in "most cases".
    threshold = 0.99
    preds_ref = torch.max(ref_output, dim=-1)[1]
    preds_new = torch.max(output, dim=-1)[1]
    agree = torch.sum(preds_ref == preds_new) / preds_ref.numel()
    print(f"Converted model agrees on {100 * agree:.2f}% of predictions")
    assert agree >= threshold, f"Only {100 * agree:.2f}% argmax agreement (need >= {100 * threshold:.0f}%)"

    # Check two: atol and rtol on all logits.
    threshold = 0.95
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
    print(f"Converted logits are close on {100 * close:.2f}% of values")
    print(f"Max absolute difference: {torch.max(abs_diff)}")
    print(f"Mean absolute difference: {torch.mean(abs_diff)}")
    print(f"Max relative difference: {torch.max(rel_diff)}")
    print(f"Mean relative difference (no inf): {torch.mean(rel_diff_no_inf)}")
    print(f"Relative difference inf proportion: {torch.mean(rel_diff_inf_mask.float())}")
    assert close >= threshold, f"Only {100 * close:.2f}% of logits close (need >= {100 * threshold:.0f}%)"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True, help="Original torch_dist Megatron checkpoint")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--hf-dir", required=True, help="Output of convert_megatron_to_hf.py")
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

    compare_logits(megatron_backend, hf_backend, vocab_size, args.seq_length, args.batch_size, dtype=dtype)
    print("Logits check passed.")


if __name__ == "__main__":
    main()

"""
Generation-level comparison: run the same real excerpts through greedy_generate (this
project's own KV-cache multi-step decode loop, inference_common.py) and HF's own
model.generate() (greedy), and print both generations side by side plus an exact-match count
and a wall-clock timing comparison. This exists to exercise the actual multi-step decode loop,
which compare_megatron_hf_logits.py's single forward pass on random tokens never touches.

Usage:
    python attn_bench/checkpoint_conversion/compare_megatron_hf_generation.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --hf-dir $HF_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --rep 0 --offset 0 --prefix-length 500 --suffix-length 500 --num-samples 5 \
        --megatron-extra-args --use-rope-scaling --rope-scaling-factor 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attn_bench.evaluation.inference_common import (BOS_TOKEN_ID,
                                                    greedy_generate,
                                                    load_megatron_model)
from attn_bench.evaluation.megatron_inference import load_rep_bucket


def generate_hf(model, prompt: torch.Tensor, suffix_length: int) -> torch.Tensor:
    output = model.generate(
        input_ids=prompt,
        max_new_tokens=suffix_length,
        min_new_tokens=suffix_length,  # matches greedy_generate: always exactly suffix_length tokens
        do_sample=False,
    )
    return output[:, prompt.shape[1]:]


def timed(fn, *fn_args):
    """Run fn(*fn_args) on GPU and return (result, elapsed_seconds), synced so the timing
    reflects actual kernel completion, not just launch latency."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn(*fn_args)
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--hf-dir", required=True, help="Output of convert_megatron_to_hf.py")
    parser.add_argument("--data-folder", required=True, help="Directory of rep_*_token.jsonl files")
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=50)
    parser.add_argument("--suffix-length", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup-samples", type=int, default=1,
                        help="First N samples run but excluded from timing stats (CUDA/cuDNN warmup)")
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda"

    rep_path = Path(args.data_folder) / f"rep_{args.rep}_token.jsonl"
    dataset = load_rep_bucket(rep_path, args.offset, args.prefix_length, args.suffix_length,
                              max_samples=args.num_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    megatron_model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    hf_model = AutoModelForCausalLM.from_pretrained(args.hf_dir, torch_dtype="auto").to(device)

    needs_bos = args.offset > 0  # offset == 0: excerpt already starts with BOS
    prompt_end = args.prefix_length + (1 if needs_bos else 0)

    n_match = 0
    megatron_times, hf_times = [], []
    for i, (sample_idx, excerpt) in enumerate(dataset):
        excerpt_t = torch.tensor([excerpt], dtype=torch.long, device=device)
        if needs_bos:
            bos = torch.full((1, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
            excerpt_t = torch.cat([bos, excerpt_t], dim=1)
        prompt = excerpt_t[:, :prompt_end]

        megatron_suffix, megatron_time = timed(greedy_generate, megatron_model, prompt, args.suffix_length)
        hf_suffix, hf_time = timed(generate_hf, hf_model, prompt, args.suffix_length)

        megatron_tokens = megatron_suffix[0].tolist()
        hf_tokens = hf_suffix[0].tolist()
        exact_match = megatron_tokens == hf_tokens
        n_match += exact_match

        is_warmup = i < args.warmup_samples
        if not is_warmup:
            megatron_times.append(megatron_time)
            hf_times.append(hf_time)

        print(f"\n=== sample_idx={sample_idx}{' (warmup, excluded from timing stats)' if is_warmup else ''} ===")
        print(f"prefix:      {tokenizer.decode(prompt[0], skip_special_tokens=True)!r}")
        print(f"megatron:    {tokenizer.decode(megatron_tokens, skip_special_tokens=True)!r}")
        print(f"hf:          {tokenizer.decode(hf_tokens, skip_special_tokens=True)!r}")
        print(f"exact match: {exact_match}")
        print(f"time:        megatron={megatron_time:.3f}s  hf={hf_time:.3f}s  "
              f"({args.suffix_length / megatron_time:.1f} vs {args.suffix_length / hf_time:.1f} tok/s)")

    print(f"\n{n_match}/{len(dataset)} samples generated identical tokens.")

    if megatron_times:
        megatron_mean = sum(megatron_times) / len(megatron_times)
        hf_mean = sum(hf_times) / len(hf_times)
        faster, ratio = ("megatron", hf_mean / megatron_mean) if megatron_mean < hf_mean \
            else ("hf", megatron_mean / hf_mean)
        print(f"\nmean generation time over {len(megatron_times)} timed samples "
              f"(suffix_length={args.suffix_length}):")
        print(f"  megatron: {megatron_mean:.3f}s  ({args.suffix_length / megatron_mean:.1f} tok/s)")
        print(f"  hf:       {hf_mean:.3f}s  ({args.suffix_length / hf_mean:.1f} tok/s)")
        print(f"  {faster} is {ratio:.2f}x faster")
    else:
        print("\nAll samples were warmup -- no timing stats. Increase --num-samples "
              "or lower --warmup-samples to get timing output.")


if __name__ == "__main__":
    main()

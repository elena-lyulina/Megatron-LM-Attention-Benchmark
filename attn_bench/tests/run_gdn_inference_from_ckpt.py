"""Run the gdn_inference suite on a real GDN checkpoint.

Loads the trained model via the exact same path as the memorization eval
(`megatron_inference_sparse.load_megatron_model`) and runs the gdn_inference test
functions on it, so the cached incremental decode is validated against the
quadratic oracle using the *actual trained weights* — not a tiny random model.
No memorization run, no checkpoint writes.

megatron_inference_sparse.py is imported, not modified.

Usage (via torchrun, TP=1):
    torchrun --nproc_per_node=1 attn_bench/tests/run_gdn_inference_from_ckpt.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --megatron-extra-args --experimental-attention-variant gated_delta_net ...
"""
from __future__ import annotations

import argparse
import sys

import torch.distributed as dist

from attn_bench.evaluation.megatron_inference_sparse import load_megatron_model
from attn_bench.tests.test_gdn_inference import register


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="torch_dist checkpoint directory")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                        help="Extra Megatron args forwarded to initialize_megatron "
                             "(e.g. the GDN architecture flags). Must be last.")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)

    # register(base_forward_step) -> [test_fn, ...]; base_forward_step is unused by the two
    # gdn_inference suites (they call the model directly), so None is fine. The functions print
    # their own PASS/FAIL via print_rank_0 and build tiny random prompts internally.
    tests = register(base_forward_step=None)
    results = {fn.__name__: bool(fn(model)) for fn in tests}

    all_ok = all(results.values())
    if dist.get_rank() == 0:
        print("\n### GDN inference from checkpoint — Summary ###")
        for name, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}   {name}")
        print(f"### Verdict: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ###")

    # nonzero exit on any failure so the slurm job is marked FAILED (deterministic across ranks:
    # the suites seed their RNG, so every rank computes the same verdict).
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Run an inference oracle suite on a real checkpoint via MegatronBackend.

`--model` picks the module guard + label (gdn | kda | qwen, plus the GDN carry aliases). Loads the
trained model the exact same way the memorization eval does (`MegatronBackend`) and runs
oracle_selfcheck + decode_matches_oracle on the *actual trained weights* -- not a tiny random model.
No memorization run, no checkpoint writes.

Usage (via torchrun, TP=1):
    torchrun --nproc_per_node=1 attn_bench/tests/run_inference_from_ckpt.py \
        --model qwen \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --megatron-extra-args <the mixer architecture flags> ...
"""
from __future__ import annotations

import argparse
import sys

import torch.distributed as dist

from attn_bench.evaluation.inference_backend import MegatronBackend
from attn_bench.tests.util.inference import make_oracle_suites
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.ssm.kimi_delta_attention import KimiDeltaAttention
from megatron.core.transformer.attention import SelfAttention

# model tag -> (required module classes, log label). Tags match llama_checkpoints.sh; the GDN carry
# variants are the same architecture as base GDN.
_MODELS = {
    "gdn": ([GatedDeltaNet], "GDN"),
    "carry-r0": ([GatedDeltaNet], "GDN"),
    "carry-r0.5": ([GatedDeltaNet], "GDN"),
    "carry-r1": ([GatedDeltaNet], "GDN"),
    "kda": ([KimiDeltaAttention], "KDA"),
    "qwen": ([GatedDeltaNet, SelfAttention], "Qwen hybrid"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(_MODELS),
                        help="which mixer family the checkpoint is (picks the module guard + label)")
    parser.add_argument("--ckpt-dir", required=True, help="torch_dist checkpoint directory")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                        help="Extra Megatron args forwarded to initialize_megatron "
                             "(e.g. the mixer architecture flags). Must be last.")
    return parser.parse_args()


def main():
    args = parse_args()
    require, label = _MODELS[args.model]

    backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    backend.load_model()

    # register(base_forward_step) -> [test_fn, ...]; base_forward_step is unused by the oracle suites
    # (they call the model directly), so None is fine. The functions print their own PASS/FAIL via
    # print_rank_0 and build tiny random prompts internally.
    tests = make_oracle_suites(require, label)(base_forward_step=None)
    results = {fn.__name__: bool(fn(backend.model)) for fn in tests}

    all_ok = all(results.values())
    if dist.get_rank() == 0:
        print(f"\n### {label} inference from checkpoint — Summary ###")
        for name, ok in results.items():
            print(f"  {'PASS' if ok else 'FAIL'}   {name}")
        print(f"### Verdict: {'ALL PASS' if all_ok else 'FAILURES PRESENT'} ###")

    # nonzero exit on any failure so the slurm job is marked FAILED (deterministic across ranks:
    # the suites seed their RNG, so every rank computes the same verdict).
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

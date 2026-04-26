"""
Cross-document attention masking test for pretrain_gpt's standard model path.

Wraps pretrain_gpt.py's forward_step to inject the two tests from xdoc_attn_kernels.py
on the first iteration, then delegates everything else to the real pretrain_gpt code.

Tests:
  1. mask_structure -- verifies _get_ltor_masks_and_position_ids produces the correct
                       block-diagonal mask (does not depend on the model/kernel).
  2. loss_isolation -- runs two forward passes with the same target doc but different
                       preceding docs; if masking is applied end-to-end, losses on the
                       target must be identical.  A FAIL means TE is ignoring the
                       explicit block-diagonal mask (attn_mask_type=causal overrides it).

Usage: see attn_bench/submissions/test_xdoc_attn_pretrain_gpt.slurm
"""

import time
_PROGRAM_START_TIME = time.time()

from functools import partial

from gpt_builders import gpt_builder
from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain, print_rank_0, set_startup_timestamps
from model_provider import model_provider

from pretrain_gpt import forward_step as _base_forward_step
from pretrain_gpt import get_embedding_ranks, train_valid_test_datasets_provider
from attn_bench.tests.xdoc_attn_kernels import test_loss_isolation, test_mask_structure

try:
    from megatron.post_training.arguments import add_modelopt_args
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

_TESTS_DONE = False


def forward_step(data_iterator, model, return_schedule_plan=False):
    global _TESTS_DONE
    if not _TESTS_DONE:
        _TESTS_DONE = True
        results = {
            "mask_structure": test_mask_structure(),
            "loss_isolation": test_loss_isolation(model),
        }
        print_rank_0("\n### Summary ###")
        for name, passed in results.items():
            print_rank_0(f"  {'PASS' if passed else 'FAIL'}: {name}")
        print_rank_0(f"\n{'All tests PASSED.' if all(results.values()) else 'Some tests FAILED.'}\n")

    return _base_forward_step(data_iterator, model, return_schedule_plan)

# Copied from [pretrain_gpt.py] to fully replicate the pipeline
if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
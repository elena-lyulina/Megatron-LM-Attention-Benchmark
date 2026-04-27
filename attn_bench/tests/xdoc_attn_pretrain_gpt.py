"""
Cross-document attention masking test for pretrain_gpt's standard model path.

Wraps pretrain_gpt.py's forward_step to inject three tests on the first iteration,
then delegates everything else to the real pretrain_gpt code.

Tests:
  1. mask_structure             -- verifies _get_ltor_masks_and_position_ids produces the
                                   correct block-diagonal mask (does not depend on kernel).
  2. loss_isolation_pretrain_gpt -- goes through the real forward_step with fake iterators;
                                   FAIL means cross-doc attn leaks end-to-end through pretrain_gpt.
                                   With --use-packed-seq-params this should PASS.

Usage: see attn_bench/submissions/test_xdoc_attn_pretrain_gpt.slurm
"""

import time
_PROGRAM_START_TIME = time.time()

from functools import partial

import torch

from gpt_builders import gpt_builder
from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.core.enums import ModelType
from megatron.training import get_args, get_tokenizer, inprocess_restart, pretrain, print_rank_0, set_startup_timestamps
from model_provider import model_provider

from pretrain_gpt import forward_step as _base_forward_step
from pretrain_gpt import get_embedding_ranks, train_valid_test_datasets_provider
from attn_bench.tests.xdoc_attn_kernels import test_mask_structure, _build_isolation_seqs

try:
    from megatron.post_training.arguments import add_modelopt_args
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

_TESTS_DONE = False


### Loss isolation test helpers ###

def _make_test_iter(token_seq, eos_id, args):
    """Wrap a 1D token sequence into a batch dict iterator matching micro_batch_size."""
    seq_len = args.seq_length
    tokens_1d = token_seq[:seq_len]
    labels_1d = token_seq[1:seq_len + 1]
    _, loss_mask_1d, pos_ids_1d = _get_ltor_masks_and_position_ids(
        data=tokens_1d,
        eod_token=eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=False,  # attention handled by packed_seq_params, not 2D mask
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=False,
    )
    mbs = args.micro_batch_size
    batch = {
        'tokens': tokens_1d.unsqueeze(0).repeat(mbs, 1),        # [mbs, seq_len]
        'labels': labels_1d.unsqueeze(0).repeat(mbs, 1),        # [mbs, seq_len]
        'loss_mask': loss_mask_1d.unsqueeze(0).repeat(mbs, 1),  # [mbs, seq_len]
        'position_ids': pos_ids_1d.unsqueeze(0).repeat(mbs, 1), # [mbs, seq_len]
    }
    return iter([batch])


def test_loss_isolation_pretrain_gpt(model):
    """
    Test that forward_step actually blocks cross-document attention end-to-end.

    Builds two sequences with the same target doc but different preceding docs.
    Runs each through the real forward_step (via fake data iterators) and compares
    per-token losses on the target doc.

    With --use-packed-seq-params: forward_step computes packed_seq_params and passes
    them to TE in THD format → cross-doc attention blocked → losses identical → PASS.

    Without --use-packed-seq-params: packed_seq_params=None, TE uses causal masking
    and ignores document boundaries → losses differ → FAIL.
    """
    print_rank_0("\n### Test 3: loss_isolation_pretrain_gpt ###")

    tokenizer = get_tokenizer()
    eos_id = tokenizer.eod

    args = get_args()
    seq_len = args.seq_length
    use_packed_seq_params = args.use_packed_seq_params

    print_rank_0(
        f"  use_packed_seq_params={use_packed_seq_params}"
        f"  transformer_impl={args.transformer_impl}"
    )

    seq_A, seq_B, prefix_A, prefix_B, target_start = _build_isolation_seqs(seq_len, tokenizer.bos, eos_id)

    was_training = model.training
    model.eval()
    with torch.no_grad(): # don't compute gradients
        out_A, _ = _base_forward_step(_make_test_iter(seq_A, eos_id, args), model)
        out_B, _ = _base_forward_step(_make_test_iter(seq_B, eos_id, args), model)
    if was_training:
        model.train()

    # output_tensor is per-token losses; slice to first copy only (positions 0..seq_len-1)
    # since get_batch flattens mbs copies, we must not compare beyond seq_len
    losses_A = out_A.view(-1).float()[target_start:seq_len]
    losses_B = out_B.view(-1).float()[target_start:seq_len]
    max_diff = (losses_A - losses_B).abs().max().item()
    mean_diff = (losses_A - losses_B).abs().mean().item()

    print_rank_0(f"  prefix_A[:4]={prefix_A[:4].tolist()}  prefix_B[:4]={prefix_B[:4].tolist()}")
    print_rank_0(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    diff_tolerance = 1e-4
    passed = max_diff < diff_tolerance
    if passed:
        print_rank_0("[PASS] loss_isolation_pretrain_gpt: losses identical → cross-doc masking applied end-to-end")
    else:
        if use_packed_seq_params:
            print_rank_0("[FAIL] loss_isolation_pretrain_gpt: losses differ despite use_packed_seq_params=True")
        else:
            print_rank_0("[FAIL] loss_isolation_pretrain_gpt: losses differ (cross-doc attn leaks through forward_step)")
            print_rank_0("  => expected without --use-packed-seq-params; add it to fix")

    return passed


### Wrapper forward_step with tests ###

def forward_step(data_iterator, model, return_schedule_plan=False):
    global _TESTS_DONE
    if not _TESTS_DONE:
        _TESTS_DONE = True
        results = {
            "mask_structure": test_mask_structure(),
            "loss_isolation_pretrain_gpt": test_loss_isolation_pretrain_gpt(model),
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
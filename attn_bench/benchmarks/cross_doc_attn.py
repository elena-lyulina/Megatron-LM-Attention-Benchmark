"""
Cross-document attention masking tests.

Two tests:
  1. mask_structure  — prints + verifies the attention mask, loss_mask, and position_ids
                       produced by _get_ltor_masks_and_position_ids for a sequence with
                       multiple documents. Always uses reset_attention_mask=True so the
                       expected block-diagonal structure is known.

  2. loss_isolation  — runs two forward passes with identical target docs but different
                       preceding docs; if reset_attention_mask is correctly applied end-to-end
                       (including inside TE), per-token losses on the target must be equal.
                       With attn_mask_type=causal in the GPT layer spec, TE may ignore the
                       explicit block-diagonal mask — this test will catch that.

Parameterisable via standard Megatron flags:
  --attn, --impl : attention mechanism / kernel (see attn_registry.py)
  --reset-attention-mask / (omit) : controls test 2 expectation
  --no-create-attention-mask-in-dataloader : controls whether mask tensor reaches TE at all
  --transformer-impl : transformer_engine|local

Usage: see attn_bench/submissions/cross_doc_attn.slurm
"""

from functools import partial

import torch

from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.core.enums import ModelType
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_tokenizer, pretrain, print_rank_0

from attn_bench.benchmarks.args import add_benchmark_args
from attn_bench.benchmarks.common import get_batch, loss_func, model_provider, train_valid_test_datasets_provider


# ─── Test 1: mask structure ───────────────────────────────────────────────────

def _assign_doc_ids(tokens, eos_id):
    """Return a tensor where each position holds its document index (increments after each EOS).

    Example: tokens=[BOS, t1, t2, EOS, BOS, t3, EOS, BOS, t4, t5, t6, EOS] => doc_ids=[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    """
    doc_ids = torch.zeros(len(tokens), dtype=torch.long)
    d = 0
    for i in range(len(tokens)):
        doc_ids[i] = d
        if tokens[i] == eos_id:
            d += 1
    return doc_ids


def _tok_label(pos, tokens, doc_ids, bos_id, eos_id):
    """Return a short label for a token position: 'doc<id>' with 'B' or 'E' for special symbols"""
    label = f"doc{doc_ids[pos].item()}"
    if tokens[pos] == eos_id:
        label += "-E"
    if tokens[pos] == bos_id:
        label += "-B"
    return label


def _print_attn_mask(mask_2d, tokens, doc_ids, bos_id, eos_id):
    """Print the seq_len x seq_len attention mask as ASCII (o = attends, . = blocked)."""
    seq_len = len(tokens)
    col_tens = "".join(str(j // 10) if j % 10 == 0 else " " for j in range(seq_len))
    col_ones = "".join(str(j % 10) for j in range(seq_len))

    labels = [_tok_label(i, tokens, doc_ids, bos_id, eos_id) for i in range(seq_len)]
    max_lw = max(len(l) for l in labels)
    # row format: "  {idx:3d} [{label:{max_lw}s}]:  {row}"
    # prefix width = 2 + 3 + 2 + max_lw + 4 = 11 + max_lw
    ruler_pad = " " * (9 + max_lw)  # 2 leading already in the f-string

    print_rank_0(f"\nAttention mask  (o = attends, . = blocked):")
    print_rank_0(f"  {ruler_pad}{col_tens}")
    print_rank_0(f"  {ruler_pad}{col_ones}")
    for i, label in enumerate(labels):
        row = "".join("o" if not mask_2d[i, j] else "." for j in range(seq_len))
        print_rank_0(f"  {i:3d} [{label:{max_lw}s}]:  {row}")
    print_rank_0(f"  {ruler_pad}{col_ones}")
    print_rank_0(f"  {ruler_pad}{col_tens}")


def _check_mask_errors(attn_mask, loss_mask, tokens, doc_ids, position_ids, bos_id, eos_id):
    """Return a list of error strings for any mask / loss_mask / position_ids violations."""
    seq_len = len(tokens)
    errors = []

    for i in range(seq_len):
        for j in range(i + 1):
            cross_doc = doc_ids[i] != doc_ids[j]
            blocked = attn_mask[i, j].item()
            li = _tok_label(i, tokens, doc_ids, bos_id, eos_id)
            lj = _tok_label(j, tokens, doc_ids, bos_id, eos_id)
            if cross_doc and not blocked:
                errors.append(f"  ({i}[{li}], {j}[{lj}]): cross-doc but NOT blocked")
            if not cross_doc and blocked and i != j:
                errors.append(f"  ({i}[{li}], {j}[{lj}]): same-doc but BLOCKED")

    for pos in (tokens == eos_id).nonzero(as_tuple=True)[0]:
        if loss_mask[pos] != 0.0:
            errors.append(f"  EOS at {pos.item()}: loss_mask={loss_mask[pos].item()}, expected 0")
        nxt = pos + 1
        if nxt < seq_len and position_ids[nxt] != 0:
            errors.append(f"  position_ids[{nxt.item()}]={position_ids[nxt].item()}, expected 0 after EOS")

    return errors


def test_mask_structure():
    """
    Verify _get_ltor_masks_and_position_ids produces a correct block-diagonal mask.

    Builds a 3-doc synthetic sequence, calls the function with reset_attention_mask=True,
    and prints the full attention mask, loss_mask, and position_ids. Also asserts the
    block-diagonal structure.

    This only tests Megatron's mask construction — not whether TE uses the mask.
    """
    print_rank_0("\n### Test 1: mask_structure ###")

    tokenizer = get_tokenizer()
    args = get_args()
    bos_id = tokenizer.bos
    eos_id = tokenizer.eod  # Megatron calls it eod; maps to EOS for HF tokenizers
    seq_len = args.seq_length

    # content tokens per doc (excludes BOS/EOS); 3 BOS + 3 EOS = 6 special positions total
    doc0_len = seq_len // 4
    doc1_len = seq_len // 4
    doc2_len = seq_len - doc0_len - doc1_len - 6

    # each doc: [BOS content EOS]
    tokens = torch.cat([
        torch.tensor([bos_id]),
        torch.randint(0, bos_id, (doc0_len,)),
        torch.tensor([eos_id]),
        torch.tensor([bos_id]),
        torch.randint(0, bos_id, (doc1_len,)),
        torch.tensor([eos_id]),
        torch.tensor([bos_id]),
        torch.randint(0, bos_id, (doc2_len,)),
        torch.tensor([eos_id]),
    ])

    attn_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=eos_id,
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        create_attention_mask=True,
    )
    # attn_mask: (1, seq_len, seq_len) bool -- True = blocked (cross-doc or future)
    # loss_mask: (seq_len,) float -- 0.0 at EOS positions, 1.0 elsewhere
    # position_ids: (seq_len,) int -- resets to 0 at the start of each new document
    attn_mask = attn_mask[0]

    doc_ids = _assign_doc_ids(tokens, eos_id)

    # eos0 = doc0_len + 1
    # eos1 = doc0_len + doc1_len + 3
    # print_rank_0(
    #     f"\nseq_len={seq_len}  bos={bos_id}  eos={eos_id}"
    #     f"  doc0=[0:{eos0}]  doc1=[{eos0+1}:{eos1}]  doc2=[{eos1+1}:{seq_len-1}]"
    # )

    print_rank_0(f"\nposition_ids:\n  {position_ids.tolist()}")
    _print_attn_mask(attn_mask, tokens, doc_ids, bos_id, eos_id)
    print_rank_0(f"\nloss_mask:\n  {loss_mask.tolist()}")

    errors = _check_mask_errors(attn_mask, tokens, doc_ids, loss_mask, position_ids, bos_id, eos_id)
    if errors:
        print_rank_0(f"\n[FAIL] mask_structure  ({len(errors)} errors):")
        for e in errors[:10]:
            print_rank_0(e)
        return False

    print_rank_0("\n[PASS] mask_structure")
    return True


# ─── Test 2: loss isolation ───────────────────────────────────────────────────

def _build_isolation_seqs(seq_len, bos_id, eos_id):
    """Build two length-(seq_len+1) sequences with the same target doc but different prefixes.

    Structure: [BOS | prefix | EOS | BOS | target_content | EOS | label_token]
    Returns (seq_A, seq_B, prefix_A, prefix_B, target_start) where target_start is the
    index of the first target-doc token inside seq_len (i.e. the BOS of the second doc).
    """
    prefix_len = seq_len // 2
    target_len = seq_len - prefix_len - 4  # -2 BOS, -2 EOS for the two docs
    target_start = prefix_len + 2  # BOS + prefix + EOS + BOS
    vocab_limit = bos_id  # content tokens drawn from [0, bos_id)

    gen = torch.Generator()
    gen.manual_seed(0); target_and_label = torch.randint(0, vocab_limit, (target_len + 1,), generator=gen)
    gen.manual_seed(1); prefix_A = torch.randint(0, vocab_limit, (prefix_len,), generator=gen)
    gen.manual_seed(2); prefix_B = torch.randint(0, vocab_limit, (prefix_len,), generator=gen)

    assert prefix_A.tolist() != prefix_B.tolist()

    target_content = target_and_label[:target_len]
    label = target_and_label[target_len:]  # 1 token: what the model predicts after the target EOS

    bos = torch.tensor([bos_id])
    eos = torch.tensor([eos_id])
    seq_A = torch.cat([bos, prefix_A, eos, bos, target_content, eos, label])
    seq_B = torch.cat([bos, prefix_B, eos, bos, target_content, eos, label])

    return seq_A, seq_B, prefix_A, prefix_B, target_start


def _make_inputs(token_seq, eos_id, args):
    """Build (tokens, labels, position_ids, attention_mask) from a 1D sequence of length seq_len+1."""
    seq_len = args.seq_length
    tokens = token_seq[:seq_len].unsqueeze(0)  # (1, seq_len)
    labels = token_seq[1:].unsqueeze(0)  # (1, seq_len)

    attn_mask, _, pos_ids = _get_ltor_masks_and_position_ids(
        data=token_seq[:seq_len],
        eod_token=eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )
    pos_ids = pos_ids.unsqueeze(0)  # (1, seq_len)
    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)  # (1, 1, seq_len, seq_len)

    dev = torch.cuda.current_device()
    return (
        tokens.to(dev),
        labels.to(dev),
        pos_ids.to(dev),
        attn_mask.to(dev) if attn_mask is not None else None,
    )


def _run_isolation_passes(model, seq_A, seq_B, target_start, eos_id, args):
    """Run two forward passes in eval mode; return (max_diff, mean_diff) of per-token losses on the target doc."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        tok_A, lab_A, pos_A, mask_A = _make_inputs(seq_A, eos_id, args)
        tok_B, lab_B, pos_B, mask_B = _make_inputs(seq_B, eos_id, args)

        out_A = model(tok_A, pos_A, mask_A, labels=lab_A)  # (1, seq_len)
        out_B = model(tok_B, pos_B, mask_B, labels=lab_B)

        # slice to target doc positions and compare
        losses_A = out_A[0, target_start:].float()
        losses_B = out_B[0, target_start:].float()
        max_diff = (losses_A - losses_B).abs().max().item()
        mean_diff = (losses_A - losses_B).abs().mean().item()

    if was_training:
        model.train()

    return max_diff, mean_diff


def _report_isolation_result(max_diff, mean_diff, prefix_A, prefix_B, args):
    """Print the pass/fail verdict for the loss_isolation test and return bool."""
    print_rank_0(f"  prefix_A[:4]={prefix_A[:4].tolist()}  prefix_B[:4]={prefix_B[:4].tolist()}")
    print_rank_0(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

    if args.reset_attention_mask:
        tol = 1e-4
        # passing only if the losses are identical
        passed = max_diff < tol
        if passed:
            print_rank_0("[PASS] loss_isolation: losses identical → mask is applied end-to-end")
        else:
            print_rank_0("[FAIL] loss_isolation: losses differ despite reset_attention_mask=True")
            print_rank_0("  => attn_mask_type=causal likely causes TE to ignore the block-diagonal mask")
    else:
        # passing either way, flagging if the losses are identical
        passed = True
        if max_diff < 1e-4:
            print_rank_0("[WARN] loss_isolation: losses identical even without reset_attention_mask — check prefix tokens")
        else:
            print_rank_0("[INFO] loss_isolation (no masking): losses differ as expected")

    return passed


def test_loss_isolation(model):
    """
    Check whether the applied attention actually respects document boundaries.

    Builds two sequences [BOS | prefix_A | EOS | BOS | target | EOS] and [BOS | prefix_B | EOS | BOS | target | EOS].
    The target is identical; only the prefix differs. Runs forward passes with frozen
    weights and compares per-token CE losses on the target positions.

    If reset_attention_mask works end-to-end (including inside TE):
        losses must be identical => PASS
    If TE ignores the block-diagonal mask (using only attn_mask_type=causal internally):
        losses will differ => FAIL
    """
    print_rank_0("\n### Test 2: loss_isolation ###")

    tokenizer = get_tokenizer()
    args = get_args()
    bos_id = tokenizer.bos
    eos_id = tokenizer.eod  # Megatron calls it eod; maps to EOS for HF tokenizers
    seq_len = args.seq_length

    print_rank_0(
        f"  reset_attention_mask={args.reset_attention_mask}"
        f"  create_attention_mask_in_dataloader={args.create_attention_mask_in_dataloader}"
        f"  transformer_impl={args.transformer_impl}"
    )

    # build two sequences with the same target doc but different prefixes
    seq_A, seq_B, prefix_A, prefix_B, target_start = _build_isolation_seqs(seq_len, bos_id, eos_id)

    # run forward passes; losses on the target must be identical iff masking is applied end-to-end
    max_diff, mean_diff = _run_isolation_passes(model, seq_A, seq_B, target_start, eos_id, args)

    return _report_isolation_result(max_diff, mean_diff, prefix_A, prefix_B, args)


# ─── Entry point ─────────────────────────────────────────────────────────────

_TESTS_DONE = False


def forward_step(data_iterator, model, return_schedule_plan: bool = False):
    global _TESTS_DONE

    vp_stage = get_attr_wrapped_model(model, "vp_stage", allow_none=True)
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)

    # only doing the tests on the first pass
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

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params)
    return output_tensor, partial(loss_func, loss_mask)


def main():
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_benchmark_args,
        args_defaults={},
    )


if __name__ == "__main__":
    main()
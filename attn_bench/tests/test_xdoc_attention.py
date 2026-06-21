# Cross-document attention tests: mask structure, position ids, and loss isolation.
# Direction-aware: the same suite confirms masking when --use-packed-seq-params is on,
# and confirms leakage when it is off.

import torch

from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training import get_args, get_tokenizer, print_rank_0

# end-to-end cross-doc influence = max per-token loss diff on the shared target doc
_ISOLATION_TOL = 1e-4   # below this → prefix left target-doc losses unchanged → masking applied
_LEAK_MIN = 1e-3        # above this → prefix changed target-doc losses → attention leaked across docs


# ── mask structure ────────────────────────────────────────────────────────────
# Tests that _get_ltor_masks_and_position_ids produces a correct block-diagonal causal mask
# for a packed 3-document sequence without running any model forward pass:
# constructs token sequences, calls the masking function with reset flags enabled, then checks that
# everything is alright
# Doesn't actually use the model / forward pass -- just calls the function being tested. The model is still passed to match the test signature

def _assign_doc_ids(tokens, eos_id):
    # assigns a document index to each token position by scanning left-to-right and incrementing after each EOS;
    # needed to identify cross-doc token pairs when checking the attention mask
    doc_ids = torch.zeros(len(tokens), dtype=torch.long)
    d = 0
    for i in range(len(tokens)):
        doc_ids[i] = d
        if tokens[i] == eos_id:
            d += 1
    return doc_ids


def _tok_label(pos, tokens, doc_ids, bos_id, eos_id):
    # builds a short label like "doc1-E" or "doc0-B" for position pos; used as row and column headers in the printed mask
    label = f"doc{doc_ids[pos].item()}"
    if tokens[pos] == eos_id:
        label += "-E"
    if tokens[pos] == bos_id:
        label += "-B"
    return label


def _print_attn_mask(mask_2d, tokens, doc_ids, bos_id, eos_id):
    # prints the seq_len x seq_len attention mask as ASCII with 'o' for attended and '.' for blocked;
    # helps visually confirm the expected block-diagonal structure
    seq_len = len(tokens)
    col_tens = "".join(str(j // 10) if j % 10 == 0 else " " for j in range(seq_len))
    col_ones = "".join(str(j % 10) for j in range(seq_len))
    labels = [_tok_label(i, tokens, doc_ids, bos_id, eos_id) for i in range(seq_len)]
    max_lw = max(len(l) for l in labels)
    ruler_pad = " " * (9 + max_lw)
    print_rank_0(f"\nAttention mask  (o = attends, . = blocked):")
    print_rank_0(f"  {ruler_pad}{col_tens}")
    print_rank_0(f"  {ruler_pad}{col_ones}")
    for i, label in enumerate(labels):
        row = "".join("o" if not mask_2d[i, j] else "." for j in range(seq_len))
        print_rank_0(f"  {i:3d} [{label:{max_lw}s}]:  {row}")
    print_rank_0(f"  {ruler_pad}{col_ones}")
    print_rank_0(f"  {ruler_pad}{col_tens}")


def _check_mask_errors(attn_mask, loss_mask, tokens, doc_ids, position_ids, bos_id, eos_id):
    # validates four invariants: cross-doc pairs are blocked, same-doc pairs follow causal (lower-triangular) access,
    # EOS tokens have zero loss, and position_ids reset to 0 at the start of each new document;
    # returns a list of error strings, empty if all invariants hold
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


_MAX_SEQ_FOR_MASK_TEST = 512  # O(n²) check + ASCII print are only feasible for short sequences

def test_mask_structure(model):
    print_rank_0("\n### Test: mask_structure ###")
    tokenizer = get_tokenizer()
    args = get_args()
    bos_id = tokenizer.bos
    eos_id = tokenizer.eod
    seq_len = args.seq_length

    if seq_len > _MAX_SEQ_FOR_MASK_TEST:
        print_rank_0(f"[SKIP] mask_structure: seq_len={seq_len} > {_MAX_SEQ_FOR_MASK_TEST} (O(n²) check + ASCII print not feasible; run with a short seq_len to verify mask structure)")
        return True

    doc0_len = seq_len // 4
    doc1_len = seq_len // 4
    doc2_len = seq_len - doc0_len - doc1_len - 6  # 6 = 3 BOS + 3 EOS

    tokens = torch.cat([
        torch.tensor([bos_id]), torch.randint(0, bos_id, (doc0_len,)), torch.tensor([eos_id]),
        torch.tensor([bos_id]), torch.randint(0, bos_id, (doc1_len,)), torch.tensor([eos_id]),
        torch.tensor([bos_id]), torch.randint(0, bos_id, (doc2_len,)), torch.tensor([eos_id]),
    ])

    attn_mask, loss_mask, position_ids = _get_ltor_masks_and_position_ids(
        data=tokens, eod_token=eos_id,
        reset_position_ids=True, reset_attention_mask=True, eod_mask_loss=True,
        create_attention_mask=True,
    )
    attn_mask = attn_mask[0]
    doc_ids = _assign_doc_ids(tokens, eos_id)

    print_rank_0(f"\nposition_ids:\n  {position_ids.tolist()}")
    _print_attn_mask(attn_mask, tokens, doc_ids, bos_id, eos_id)
    print_rank_0(f"\nloss_mask:\n  {loss_mask.tolist()}")

    errors = _check_mask_errors(attn_mask, loss_mask, tokens, doc_ids, position_ids, bos_id, eos_id)
    if errors:
        print_rank_0(f"\n[FAIL] mask_structure  ({len(errors)} errors):")
        for e in errors[:10]:
            print_rank_0(e)
        return False

    print_rank_0("\n[PASS] mask_structure")
    return True


# ── loss isolation ────────────────────────────────────────────────────────────
# tests that cross-doc masking prevents the prefix doc from influencing target doc losses end-to-end
# through the full model forward pass: runs forward_step on seq_A and seq_B, which share the same
# target doc tokens but have different random prefixes, then asserts per-token losses on the target doc
# are identical (max_diff < 1e-4). a nonzero diff means attention leaked across the document boundary.

def _build_isolation_seqs(seq_len, bos_id, eos_id):
    # builds two packed sequences [BOS prefix_A EOS BOS target EOS label] and [BOS prefix_B EOS BOS target EOS label]
    # where prefix_A and prefix_B are different random vectors but target and label tokens are identical;
    # returns both sequences and target_start so the caller can slice per-token losses on the shared target doc
    prefix_len = seq_len // 2
    target_len = seq_len - prefix_len - 4  # -2 BOS, -2 EOS
    target_start = prefix_len + 2          # BOS + prefix + EOS + BOS

    gen = torch.Generator()
    gen.manual_seed(0); target_and_label = torch.randint(0, bos_id, (target_len + 1,), generator=gen)
    gen.manual_seed(1); prefix_A = torch.randint(0, bos_id, (prefix_len,), generator=gen)
    gen.manual_seed(2); prefix_B = torch.randint(0, bos_id, (prefix_len,), generator=gen)

    assert prefix_A.tolist() != prefix_B.tolist()

    bos = torch.tensor([bos_id])
    eos = torch.tensor([eos_id])
    seq_A = torch.cat([bos, prefix_A, eos, bos, target_and_label[:target_len], eos, target_and_label[target_len:]])
    seq_B = torch.cat([bos, prefix_B, eos, bos, target_and_label[:target_len], eos, target_and_label[target_len:]])

    return seq_A, seq_B, prefix_A, prefix_B, target_start


def _make_test_iter(token_seq, eos_id, args):
    # builds a single-step data iterator from a 1D token sequence so forward_step can consume it;
    # computes loss_mask and position_ids via _get_ltor_masks_and_position_ids, then stacks the sequence
    # into a batch of micro_batch_size rows as forward_step expects
    seq_len = args.seq_length
    tokens_1d = token_seq[:seq_len]
    labels_1d = token_seq[1:seq_len + 1]
    _, loss_mask_1d, pos_ids_1d = _get_ltor_masks_and_position_ids(
        data=tokens_1d,
        eod_token=eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=False,  # cross-doc masking via packed_seq_params, not 2D mask
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=False,
    )
    mbs = args.micro_batch_size
    batch = {
        'tokens': tokens_1d.unsqueeze(0).repeat(mbs, 1),
        'labels': labels_1d.unsqueeze(0).repeat(mbs, 1),
        'loss_mask': loss_mask_1d.unsqueeze(0).repeat(mbs, 1),
        'position_ids': pos_ids_1d.unsqueeze(0).repeat(mbs, 1),
    }
    return iter([batch])


def _make_test_loss_isolation(base_forward_step):
    # returns test_loss_isolation as a (model)->bool function with base_forward_step captured in a closure.
    # test_loss_isolation needs base_forward_step from root pretrain_gpt.py, but this file cannot import it directly:
    # Python adds the script's directory to sys.path[0] when an entry point runs, which would shadow root pretrain_gpt.py.
    # the entry point instead imports base_forward_step and passes it into register(), which passes it here;
    # the closure captures it so test_loss_isolation keeps the expected (model)->bool registry signature.
    def test_loss_isolation(model):
        print_rank_0("\n### Test: loss_isolation ###")
        tokenizer = get_tokenizer()
        args = get_args()
        eos_id = tokenizer.eod

        print_rank_0(f"  use_packed_seq_params={args.use_packed_seq_params}  reset_position_ids={args.reset_position_ids}  transformer_impl={args.transformer_impl}")

        seq_A, seq_B, prefix_A, prefix_B, target_start = _build_isolation_seqs(
            args.seq_length, tokenizer.bos, eos_id
        )

        was_training = model.training
        model.eval()
        with torch.no_grad():
            out_A, _ = base_forward_step(_make_test_iter(seq_A, eos_id, args), model)
            out_B, _ = base_forward_step(_make_test_iter(seq_B, eos_id, args), model)
        if was_training:
            model.train()

        losses_A = out_A.view(-1).float()[target_start:args.seq_length]
        losses_B = out_B.view(-1).float()[target_start:args.seq_length]
        max_diff = (losses_A - losses_B).abs().max().item()
        mean_diff = (losses_A - losses_B).abs().mean().item()

        print_rank_0(f"  prefix_A[:4]={prefix_A[:4].tolist()}  prefix_B[:4]={prefix_B[:4].tolist()}")
        print_rank_0(f"  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

        # direction depends on the run config: with packing we expect isolation, without it we expect leakage
        expect_isolation = getattr(args, 'use_packed_seq_params', False)
        if expect_isolation:
            passed = max_diff < _ISOLATION_TOL
            if passed:
                print_rank_0(f"[PASS] loss_isolation: max_diff < {_ISOLATION_TOL:g} → cross-doc masking applied end-to-end")
            else:
                print_rank_0(f"[FAIL] loss_isolation: use_packed_seq_params=True but max_diff={max_diff:.6f} ≥ {_ISOLATION_TOL:g} → masking NOT applied")
        else:
            passed = max_diff > _LEAK_MIN
            if passed:
                print_rank_0(f"[PASS] loss_isolation: max_diff > {_LEAK_MIN:g} → cross-doc attention leaks (expected without --use-packed-seq-params)")
            else:
                print_rank_0(f"[FAIL] loss_isolation: expected leakage but max_diff={max_diff:.6f} ≤ {_LEAK_MIN:g} → prefix had no effect")
        return passed

    return test_loss_isolation


# ── position ids ──────────────────────────────────────────────────────────────
# Verifies position_ids match the run config, using the same dataset function the real run uses.
# With --reset-position-ids each packed document restarts at 0 (positions repeat across docs);
# without it, positions run continuously 0..seq_len-1 (all unique). Guards against accidentally
# leaving --reset-position-ids on for a leaking "one continuous document" run.

def test_position_ids(model):
    print_rank_0("\n### Test: position_ids ###")
    tokenizer = get_tokenizer()
    args = get_args()
    eos_id = tokenizer.eod
    seq_len = args.seq_length

    # three packed docs separated by EOD, padded/truncated to exactly seq_len
    doc_len = (seq_len - 3) // 3
    tokens = torch.cat([
        torch.randint(0, eos_id, (doc_len,)), torch.tensor([eos_id]),
        torch.randint(0, eos_id, (doc_len,)), torch.tensor([eos_id]),
        torch.randint(0, eos_id, (seq_len,)), torch.tensor([eos_id]),
    ])[:seq_len]

    _, _, position_ids = _get_ltor_masks_and_position_ids(
        data=tokens, eod_token=eos_id,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=False,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=False,
    )

    n_unique = position_ids.unique().numel()
    print_rank_0(f"  reset_position_ids={args.reset_position_ids}  unique={n_unique}/{seq_len}")
    print_rank_0(f"  position_ids[:24]={position_ids[:24].tolist()}")

    if args.reset_position_ids:
        # interior EOD boundaries must restart positions at 0 → positions repeat
        eod_pos = (tokens == eos_id).nonzero(as_tuple=True)[0]
        resets_ok = all(position_ids[p + 1].item() == 0 for p in eod_pos if p + 1 < seq_len)
        passed = resets_ok and n_unique < seq_len
        if passed:
            print_rank_0("[PASS] position_ids: reset to 0 at each document boundary")
        else:
            print_rank_0("[FAIL] position_ids: expected per-document reset to 0 but none found")
    else:
        passed = torch.equal(position_ids, torch.arange(seq_len, dtype=position_ids.dtype))
        if passed:
            print_rank_0(f"[PASS] position_ids: continuous 0..{seq_len - 1}, all unique → packed docs have distinct positions")
        else:
            print_rank_0("[FAIL] position_ids: expected continuous unique ids but found repeats/gaps")
    return passed


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve and return the test functions for the 'xdoc' suite;
    # each returned function has signature (model)->bool
    return [test_mask_structure, test_position_ids, _make_test_loss_isolation(base_forward_step)]

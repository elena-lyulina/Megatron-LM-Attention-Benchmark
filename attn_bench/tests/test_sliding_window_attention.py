# Sliding window attention tests: verify --window-size actually restricts each query's receptive
# field to the configured distance, rather than silently falling back to full causal attention.
#
# Single query position q = seq_len - 1 (window covers [window_edge, q] inclusive, window_edge = q -
# window). Starting from one random base sequence, flip one token at a time and compare the loss at q
# against the base:
#   - window_isolation:   flipping any token in [0, window_edge) (OUTSIDE the window) must leave the
#                          loss at q unchanged -> the window is actually cutting off distant context.
#   - window_sensitivity: flipping any token in [window_edge, q] (INSIDE the window) must change the
#                          loss at q -> the window isn't accidentally empty/dead. Catches an "always
#                          block everything before q" bug that would otherwise make window_isolation
#                          pass for the wrong reason.
#
# Each suite tests every position in its pool if the pool is small (<= _EXHAUSTIVE_THRESHOLD), else a
# random sample of _SAMPLE_SIZE positions -- exhaustive for a small seq_len/window (the tiny test),
# bounded-cost for a large one (the real 1B-model window=1024/seq_len=8192 test). A single mismatching
# position fails the suite.

import torch

from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training import get_args, get_tokenizer, print_rank_0

_ISOLATION_TOL = 1e-4    # below this -> flipping this token left the loss at q unchanged
_SENSITIVITY_TOL = 1e-4  # above this -> flipping this token changed the loss at q

_EXHAUSTIVE_THRESHOLD = 200  # pools up to this size are tested in full
_SAMPLE_SIZE = 100           # larger pools are randomly sampled down to this many positions


def _query_and_edge(seq_len, window):
    # q's window covers [window_edge, q] inclusive (window_size=(window, 0) is a causal, left-only
    # window -- see megatron/core/extensions/transformer_engine.py's window_size handling)
    q = seq_len - 1
    window_edge = q - window
    return q, window_edge


def _select_positions(pool, seed):
    if len(pool) <= _EXHAUSTIVE_THRESHOLD:
        return list(pool)
    gen = torch.Generator()
    gen.manual_seed(seed)
    idx = torch.randperm(len(pool), generator=gen)[:_SAMPLE_SIZE].tolist()
    return [pool[i] for i in idx]


def _build_base_seq(seq_len, vocab_hi, seed=1):
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randint(0, vocab_hi, (seq_len + 1,), generator=gen)  # +1: labels_1d needs seq[seq_len]


def _flip(seq, pos, vocab_hi):
    seq = seq.clone()
    seq[pos] = (seq[pos].item() + 1) % vocab_hi
    return seq


def _make_test_iter(seq, eos_id, args):
    # same construction as test_xdoc_attention._make_test_iter: drives loss_mask/position_ids through
    # the real dataset masking function instead of assuming values, using the run's actual
    # reset_position_ids/eod_mask_loss flags. With no EOS in seq this naturally yields loss_mask=1s
    # and position_ids=arange, but going through the real function keeps this test honest.
    seq_len = args.seq_length
    tokens_1d = seq[:seq_len]
    labels_1d = seq[1:seq_len + 1]
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


def _loss_at_q(base_forward_step, model, seq, eos_id, args, q):
    # --use-packed-seq-params flattens the batch to [1, mbs*seq_len]; reshape back to index row 0.
    out, _ = base_forward_step(_make_test_iter(seq, eos_id, args), model)
    mbs = args.micro_batch_size
    return out.reshape(mbs, args.seq_length)[0].float()[q].item()


def _diffs_at_positions(base_forward_step, model, args, eos_id, base_seq, q, positions):
    # positions: list of token positions to flip one at a time; returns {pos: |loss_pert - loss_base| at q}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss_base = _loss_at_q(base_forward_step, model, base_seq, eos_id, args, q)
        diffs = {}
        for pos in positions:
            pert_seq = _flip(base_seq, pos, eos_id)
            diffs[pos] = abs(_loss_at_q(base_forward_step, model, pert_seq, eos_id, args, q) - loss_base)
    if was_training:
        model.train()
    return diffs


def _verify_flip_effects(base_forward_step, model, args, eos_id, base_seq, q, pool, seed, tol, expect_change, label):
    # flips every position in `pool` (or a random sample of it, see _select_positions) one at a time
    # and checks whether the loss at q changed as `expect_change` says it should; fails on any
    # position that doesn't match, printing up to 10 mismatches for diagnosis
    positions = _select_positions(pool, seed)
    diffs = _diffs_at_positions(base_forward_step, model, args, eos_id, base_seq, q, positions)

    mismatches = [(p, d) for p, d in diffs.items() if (d <= tol) == expect_change]
    print_rank_0(f"  {label}: tested {len(diffs)}/{len(pool)} position(s), {len(mismatches)} mismatch(es)")
    if mismatches:
        shown = ", ".join(f"pos={p} diff={d:.6f}" for p, d in mismatches[:10])
        print_rank_0(f"    mismatches (up to 10): {shown}")
    return len(mismatches) == 0


def _make_window_tests(base_forward_step):
    # returns (test_window_isolation, test_window_sensitivity) as (model)->bool functions, with
    # base_forward_step captured in a closure -- same factory pattern as test_xdoc_attention /
    # test_sink_attention (this file can't import pretrain_gpt.forward_step directly, see those
    # files for why)

    def _setup(args, test_name):
        assert args.window_size is not None, "swa suite requires --window-size to be set"
        window = args.window_size[0]
        seq_len = args.seq_length
        q, window_edge = _query_and_edge(seq_len, window)
        if window_edge <= 0:
            print_rank_0(f"[SKIP] {test_name}: seq_len={seq_len} too small for window={window} "
                         f"(window_edge={window_edge} leaves no outside-window pool)")
            return None
        print_rank_0(f"  seq_len={seq_len}  window={window}  q={q}  window_edge={window_edge}")
        return q, window_edge

    def _make_property_test(name, pool_fn, seed, tol, expect_change, direction_label, ok_msg, bad_msg):
        # builds one (model)->bool test function; test_window_isolation and test_window_sensitivity
        # are just two instantiations of this with opposite pools/expectations
        def test_fn(model):
            print_rank_0(f"\n### Test: {name} ###")
            args = get_args()
            eos_id = get_tokenizer().eod

            setup = _setup(args, name)
            if setup is None:
                return True
            q, window_edge = setup

            base_seq = _build_base_seq(args.seq_length, eos_id)
            passed = _verify_flip_effects(base_forward_step, model, args, eos_id, base_seq, q,
                                           pool=pool_fn(q, window_edge), seed=seed,
                                           tol=tol, expect_change=expect_change, label=direction_label)

            verdict, msg = ("PASS", ok_msg) if passed else ("FAIL", bad_msg)
            print_rank_0(f"[{verdict}] {name}: {msg}")
            return passed

        test_fn.__name__ = f"test_{name}"
        return test_fn

    test_window_isolation = _make_property_test(
        "window_isolation", pool_fn=lambda q, we: list(range(we)), seed=1,
        tol=_ISOLATION_TOL, expect_change=False, direction_label="outside-window flips",
        ok_msg=f"all outside-window flips left the loss at q unchanged (< {_ISOLATION_TOL:g})",
        bad_msg="some outside-window flip changed the loss at q -> attention leaked past the window",
    )
    test_window_sensitivity = _make_property_test(
        "window_sensitivity", pool_fn=lambda q, we: list(range(we, q + 1)), seed=2,
        tol=_SENSITIVITY_TOL, expect_change=True, direction_label="inside-window flips",
        ok_msg=f"every inside-window flip changed the loss at q (> {_SENSITIVITY_TOL:g})",
        bad_msg="some inside-window flip had no effect on the loss at q -> in-window context is not wired in",
    )

    return test_window_isolation, test_window_sensitivity


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'swa' suite
    return list(_make_window_tests(base_forward_step))

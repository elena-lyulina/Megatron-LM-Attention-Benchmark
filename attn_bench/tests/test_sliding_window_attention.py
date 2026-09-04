# Sliding window attention tests: verify --window-size actually restricts each query's receptive
# field to the configured distance, rather than silently falling back to full causal attention.
#
# Single query position q = seq_len - 1. Starting from one random base sequence, flip one token at a
# time and compare the loss at q against the base:
#   - window_sensitivity: flipping any token in [q-window, q] (inside a SINGLE layer's window) must
#                          change the loss at q -> the window isn't accidentally empty/dead. Catches
#                          an "always block everything before q" bug that would otherwise make
#                          window_isolation pass for the wrong reason.
#   - window_isolation:   flipping any token in [0, q - num_layers*window) must leave the loss at q
#                          unchanged. This is NOT [0, q-window) -- windowed attention stacks across
#                          layers: layer 2's query at q reads layer-1 hidden states that already
#                          encode [q-window, q], and those hidden states themselves were computed from
#                          [q-2*window, q] at layer 1, etc. After num_layers layers, q's *effective*
#                          receptive field is num_layers*window tokens back, not window (this is the
#                          same "receptive span ~ window * num_layers" property that lets Mistral's
#                          32-layer, window=4096 model reach ~131K tokens deep -- see the earlier
#                          window-size-choice discussion). A single-layer window boundary would fail
#                          this suite on a correct implementation the moment num_layers > 1.
#
# Each suite tests every position in its pool if the pool is small (<= _EXHAUSTIVE_THRESHOLD), else a
# random sample of _SAMPLE_SIZE positions -- exhaustive for a small seq_len/window (the tiny test),
# bounded-cost for a large one (the real 1B-model window=1024/seq_len=8192 test). A single mismatching
# position fails the suite. Note that with num_layers*window >= seq_len (e.g. the real model's 16
# layers * window=4096 > seq_len=8192), NO position is provably outside the effective receptive field
# -- window_isolation then has an empty pool and reports [SKIP], which is the correct outcome, not a
# test gap.

import torch

from megatron.core.datasets.gpt_dataset import _get_ltor_masks_and_position_ids
from megatron.training import get_args, get_tokenizer, print_rank_0

_ISOLATION_TOL = 1e-4    # below this -> flipping this token left the loss at q unchanged
_SENSITIVITY_TOL = 1e-4  # above this -> flipping this token changed the loss at q

_EXHAUSTIVE_THRESHOLD = 200  # pools up to this size are tested in full
_SAMPLE_SIZE = 100           # larger pools are randomly sampled down to this many positions


def _query_and_edge(seq_len, window, depth_multiplier):
    # depth_multiplier=1 -> single-layer window edge (used by window_sensitivity, always a safe
    # "guaranteed inside" boundary regardless of num_layers); depth_multiplier=num_layers -> the
    # full-stack edge (used by window_isolation, see the module docstring for why)
    q = seq_len - 1
    edge = q - depth_multiplier * window
    return q, edge


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

    def _make_property_test(name, pool_fn, seed, tol, expect_change, direction_label, depth_multiplier_fn, ok_msg, bad_msg):
        # builds one (model)->bool test function; test_window_isolation and test_window_sensitivity
        # are just two instantiations of this with opposite pools/expectations/depth_multiplier
        def test_fn(model):
            print_rank_0(f"\n### Test: {name} ###")
            args = get_args()
            eos_id = get_tokenizer().eod
            assert args.window_size is not None, "swa suite requires --window-size to be set"
            window = args.window_size[0]

            q, edge = _query_and_edge(args.seq_length, window, depth_multiplier_fn(args))
            pool = pool_fn(q, edge)
            print_rank_0(f"  seq_len={args.seq_length}  window={window}  num_layers={args.num_layers}  "
                         f"q={q}  edge={edge}  pool_size={len(pool)}")
            if not pool:
                print_rank_0(f"[SKIP] {name}: empty pool -- no position provably matches this "
                             f"property for this seq_len/window/num_layers combination")
                return True

            base_seq = _build_base_seq(args.seq_length, eos_id)
            passed = _verify_flip_effects(base_forward_step, model, args, eos_id, base_seq, q,
                                           pool=pool, seed=seed,
                                           tol=tol, expect_change=expect_change, label=direction_label)

            verdict, msg = ("PASS", ok_msg) if passed else ("FAIL", bad_msg)
            print_rank_0(f"[{verdict}] {name}: {msg}")
            return passed

        test_fn.__name__ = f"test_{name}"
        return test_fn

    test_window_isolation = _make_property_test(
        "window_isolation", pool_fn=lambda q, edge: list(range(edge)), seed=1,
        tol=_ISOLATION_TOL, expect_change=False, direction_label="outside-effective-receptive-field flips",
        depth_multiplier_fn=lambda args: args.num_layers,
        ok_msg=f"all flips beyond num_layers*window left the loss at q unchanged (< {_ISOLATION_TOL:g})",
        bad_msg="some flip beyond num_layers*window changed the loss at q -> attention leaked past the full-stack receptive field",
    )
    test_window_sensitivity = _make_property_test(
        "window_sensitivity", pool_fn=lambda q, edge: list(range(max(0, edge), q + 1)), seed=2,
        tol=_SENSITIVITY_TOL, expect_change=True, direction_label="inside-window flips",
        depth_multiplier_fn=lambda args: 1,
        ok_msg=f"every inside-window flip changed the loss at q (> {_SENSITIVITY_TOL:g})",
        bad_msg="some inside-window flip had no effect on the loss at q -> in-window context is not wired in",
    )

    return test_window_isolation, test_window_sensitivity


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'swa' suite
    return list(_make_window_tests(base_forward_step))

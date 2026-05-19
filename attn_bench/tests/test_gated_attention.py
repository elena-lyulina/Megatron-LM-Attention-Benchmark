# Gated attention tests: verify --attention-output-gate adds a gate projection to linear_qkv.

import torch

from megatron.training import get_args, print_rank_0
from attn_bench.tests.util.common import make_simple_iter


# ── linear qkv shape ─────────────────────────────────────────────────────────
# tests that --attention-output-gate adds a gate projection column to linear_qkv,
# making its output dim 4 x (num_heads_per_tp x kv_channels) instead of 3x;
# then confirms the gate slice actually modulates the forward output.

def _find_linear_qkv_modules(model):
    # searches named_modules() for all modules whose name ends with 'linear_qkv';
    # returns list of (name, module) pairs
    return [(n, m) for n, m in model.named_modules() if n.endswith('linear_qkv')]


def _gate_dims(args):
    # computes per-TP-rank expected weight output dims for standard vs gated linear_qkv;
    # standard = (Q + K + V) / tp = (nh + 2*nkvh) * kvch / tp
    # gated    = (Q + K + V + gate) / tp = (2*nh + 2*nkvh) * kvch / tp  (gate has nh heads like Q)
    # nkvh == nh for MHA; nkvh < nh for GQA
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads
    nkvh = args.num_query_groups
    kvch = args.kv_channels
    standard_dim = (nh + 2 * nkvh) * kvch // tp
    gated_dim = (2 * nh + 2 * nkvh) * kvch // tp
    return nh, nkvh, kvch, standard_dim, gated_dim


def test_linear_qkv_shape(model):
    print_rank_0("\n### Test: linear_qkv_shape ###")
    modules = _find_linear_qkv_modules(model)
    if not modules:
        print_rank_0("[FAIL] linear_qkv_shape: no linear_qkv module found in model")
        return False

    args = get_args()
    nh, nkvh, kvch, standard_dim, gated_dim = _gate_dims(args)
    print_rank_0(f"  expected: gated={gated_dim}  standard={standard_dim}  (nh={nh}, nkvh={nkvh}, kv_channels={kvch}, tp={args.tensor_model_parallel_size})")

    errors = []
    for name, m in modules:
        actual = m.weight.shape[0]
        print_rank_0(f"  {name}: weight.shape[0]={actual}")
        if actual != gated_dim:
            errors.append(f"  {name}: got {actual}, expected {gated_dim} (standard would be {standard_dim})")

    if errors:
        print_rank_0(f"\n[FAIL] linear_qkv_shape ({len(errors)} errors):")
        for e in errors:
            print_rank_0(e)
        return False

    print_rank_0(f"[PASS] linear_qkv_shape: all {len(modules)} linear_qkv modules have 4× output dim")
    return True


def _make_test_gated_output_sensitivity(base_forward_step):
    # returns test_gated_output_sensitivity as (model)->bool with base_forward_step in a closure;
    # same factory pattern as test_xdoc_attention -- avoids sys.path shadowing of root pretrain_gpt.py
    def test_gated_output_sensitivity(model):
        # sets gate slice of linear_qkv.weight to +10 (sigmoid ~= 1, gate open) -> forward,
        # then to -10 (sigmoid ~= 0, gate closed) -> forward; asserts outputs differ,
        # confirming the gate projection is wired into the attention output and not dead code
        print_rank_0("\n### Test: gated_output_sensitivity ###")
        modules = _find_linear_qkv_modules(model)
        if not modules:
            print_rank_0("[FAIL] gated_output_sensitivity: no linear_qkv module found")
            return False

        args = get_args()
        _, _, _, gate_start, _ = _gate_dims(args)  # gate slice starts where standard Q/K/V ends

        originals = {name: m.weight.data[gate_start:].clone() for name, m in modules}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _, m in modules:
                m.weight.data[gate_start:].fill_(10.0)
            out_open, _ = base_forward_step(make_simple_iter(), model)

            for _, m in modules:
                m.weight.data[gate_start:].fill_(-10.0)
            out_closed, _ = base_forward_step(make_simple_iter(), model)
        if was_training:
            model.train()
        for name, m in modules:
            m.weight.data[gate_start:].copy_(originals[name])

        max_diff = (out_open.view(-1).float() - out_closed.view(-1).float()).abs().max().item()
        print_rank_0(f"  max_diff (gate_open=+10 vs gate_closed=-10): {max_diff:.6f}")

        passed = max_diff > 1e-4
        if passed:
            print_rank_0("[PASS] gated_output_sensitivity: gate affects output → wired into attention computation")
        else:
            print_rank_0("[FAIL] gated_output_sensitivity: gate has no effect → may be disconnected from computation")
        return passed

    return test_gated_output_sensitivity


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'gated' suite;
    # each returned function has signature (model)->bool
    return [test_linear_qkv_shape, _make_test_gated_output_sensitivity(base_forward_step)]

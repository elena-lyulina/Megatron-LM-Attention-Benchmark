# Sink attention tests: verify --softmax-type learnable wires a learnable softmax offset into the model.

import torch

from megatron.training import print_rank_0
from attn_bench.tests.util.common import make_simple_iter


# ── softmax offset param ──────────────────────────────────────────────────────
# tests that --softmax-type learnable correctly wires a learnable offset into the attention softmax:
# checks the parameter exists in the model and that perturbing it changes the forward output.

def _find_sink_params(model):
    # searches named_parameters() for the sink scalar; name depends on the implementation path:
    # 'softmax_offset' for --softmax-type learnable (native) and --impl te (TE creates it internally),
    # 'sink' for --impl torch (SinkTorchAttention registers self.sink as nn.Parameter)
    return [(n, p) for n, p in model.named_parameters()
            if 'softmax_offset' in n or n.split('.')[-1] == 'sink']


def test_softmax_offset_param(model):
    print_rank_0("\n### Test: softmax_offset_param ###")
    params = _find_sink_params(model)
    if not params:
        print_rank_0("[FAIL] softmax_offset_param: no sink parameter found (expected 'softmax_offset' or 'sink')")
        return False
    for name, p in params:
        print_rank_0(f"  found: {name}  shape={list(p.shape)}")
    print_rank_0(f"[PASS] softmax_offset_param: found {len(params)} parameter(s)")
    return True


def _make_test_sink_output_sensitivity(base_forward_step):
    # returns test_sink_output_sensitivity as (model)->bool with base_forward_step in a closure;
    # same factory pattern as test_xdoc_attention — avoids sys.path shadowing of root pretrain_gpt.py
    def test_sink_output_sensitivity(model):
        # zeros out all softmax_offset_param tensors → runs forward → fills with 5.0 → runs forward again;
        # asserts the two outputs differ, confirming the parameter is wired into the softmax and not a dead leaf
        print_rank_0("\n### Test: sink_output_sensitivity ###")
        params = _find_sink_params(model)
        if not params:
            print_rank_0("[FAIL] sink_output_sensitivity: no sink parameter found (expected 'softmax_offset' or 'sink')")
            return False

        originals = {name: p.data.clone() for name, p in params}
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for _, p in params:
                p.data.zero_()
            out_zero, _ = base_forward_step(make_simple_iter(), model)

            for _, p in params:
                p.data.fill_(5.0)
            out_five, _ = base_forward_step(make_simple_iter(), model)
        if was_training:
            model.train()
        for name, p in params:
            p.data.copy_(originals[name])

        max_diff = (out_zero.view(-1).float() - out_five.view(-1).float()).abs().max().item()
        print_rank_0(f"  max_diff (offset=0 vs offset=5.0): {max_diff:.6f}")

        passed = max_diff > 1e-4
        if passed:
            print_rank_0("[PASS] sink_output_sensitivity: offset affects output → wired into softmax computation")
        else:
            print_rank_0("[FAIL] sink_output_sensitivity: offset has no effect → may be disconnected from computation")
        return passed

    return test_sink_output_sensitivity


# ── init values ──────────────────────────────────────────────────────────────
# diagnose what TE initialises softmax_offset to; explains why params norm starts at 4k

def test_softmax_offset_init_values(model):
    print_rank_0("\n### Test: softmax_offset_init_values ###")
    params = _find_sink_params(model)
    if not params:
        print_rank_0("[FAIL] softmax_offset_init_values: no sink parameter found")
        return False
    all_vals = []
    for name, p in params:
        vals = p.data.float()
        all_vals.append(vals.flatten())
        print_rank_0(
            f"  {name}: shape={list(p.shape)}"
            f"  mean={vals.mean():.4f}  std={vals.std():.4f}"
            f"  min={vals.min():.4f}  max={vals.max():.4f}"
        )
    all_vals = torch.cat(all_vals)
    print_rank_0(
        f"  total: count={len(all_vals)}"
        f"  norm={all_vals.norm():.3f}"
        f"  mean_abs={all_vals.abs().mean():.4f}"
    )
    print_rank_0("[PASS] softmax_offset_init_values: diagnostic complete")
    return True


# ── norm decomposition ────────────────────────────────────────────────────────
# verify that the 4k logged params norm comes from softmax_offset, not a logging artifact

def test_norm_decomposition(model):
    print_rank_0("\n### Test: norm_decomposition ###")
    sink_params = _find_sink_params(model)
    sink_names = {n for n, _ in sink_params}
    other_params = [(n, p) for n, p in model.named_parameters() if n not in sink_names]

    def local_norm_sq(param_list):
        return sum(p.data.float().pow(2).sum().item() for _, p in param_list)

    sink_sq = local_norm_sq(sink_params)
    other_sq = local_norm_sq(other_params)
    combined = (sink_sq + other_sq) ** 0.5
    print_rank_0(f"  softmax_offset  (this rank): norm={sink_sq**0.5:.3f}")
    print_rank_0(f"  all other params (this rank): norm={other_sq**0.5:.3f}")
    print_rank_0(f"  combined         (this rank): norm={combined:.3f}")
    print_rank_0(
        "  note: Megatron's logged 'params norm' all-reduces across TP/PP groups"
        " and skips TP-duplicate params — compare rank 0 combined with the logged value"
    )
    print_rank_0("[PASS] norm_decomposition: diagnostic complete")
    return True


# ── gradient flows ────────────────────────────────────────────────────────────
# check whether TE's flash-attention training kernel backprops through softmax_offset;
# FAIL = grad is None / zero → kernel ignores the param (training is effectively full attention)
# PASS = grad is nonzero  → kernel trains the param (saturation may still prevent convergence)

def _make_test_gradient_flows(base_forward_step):
    def test_gradient_flows(model):
        print_rank_0("\n### Test: gradient_flows ###")
        params = _find_sink_params(model)
        if not params:
            print_rank_0("[FAIL] gradient_flows: no sink parameter found")
            return False

        was_training = model.training
        model.train()
        model.zero_grad()

        out, _ = base_forward_step(make_simple_iter(), model)
        if out is None:
            print_rank_0("[SKIP] gradient_flows: forward returned None (not a loss-computing pipeline stage)")
            if not was_training:
                model.eval()
            return True

        out.sum().backward()

        any_nonzero = False
        for name, p in params:
            if p.grad is None:
                print_rank_0(f"  {name}: grad=None → no gradient path through this parameter")
            else:
                gnorm = p.grad.float().norm().item()
                nonzero = gnorm > 1e-10
                any_nonzero = any_nonzero or nonzero
                print_rank_0(f"  {name}: grad_norm={gnorm:.6e}  ({'nonzero' if nonzero else 'ZERO'})")

        if not was_training:
            model.eval()

        if any_nonzero:
            print_rank_0("[PASS] gradient_flows: softmax_offset receives nonzero gradients → parameter is trainable")
        else:
            print_rank_0("[FAIL] gradient_flows: softmax_offset has zero/no gradient → TE flash-attn kernel ignores the parameter")
        return any_nonzero

    return test_gradient_flows


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'sink' suite;
    # each returned function has signature (model)->bool
    return [
        test_softmax_offset_param,
        _make_test_sink_output_sensitivity(base_forward_step),
        test_softmax_offset_init_values,
        test_norm_decomposition,
        _make_test_gradient_flows(base_forward_step),
    ]

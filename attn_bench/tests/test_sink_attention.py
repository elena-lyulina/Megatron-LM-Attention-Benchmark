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


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'sink' suite;
    # each returned function has signature (model)->bool
    return [test_softmax_offset_param, _make_test_sink_output_sensitivity(base_forward_step)]

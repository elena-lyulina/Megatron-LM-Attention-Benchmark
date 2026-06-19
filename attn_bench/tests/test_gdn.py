# Gated Delta Net tests: verify the GDN-specific parameters are wired into the forward pass.

import torch

from megatron.training import get_args, print_rank_0
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from attn_bench.tests.util.common import make_simple_iter


# ── parameter disturbance ──────────────────────────────────────────────────────
# perturbs each GDN-specific parameter (or in_proj row-slice) in turn and checks the output
# changes; if a target has no effect, it is not wired into the computation.
# in_proj output is laid out as [qkv | z | beta | alpha] (gated_delta_net.py forward split), so
# z / beta / alpha are tested via their row-blocks of in_proj.weight (they are activations, not
# standalone parameters). A_log / dt_bias / conv1d.weight are real parameters.

def _gdn_targets(name, m):
    # returns [(label, tensor, slice_or_None), ...] for one GDN module m named `name`;
    # tensor is the .data to perturb, slice restricts the perturbation to a row-block of in_proj.weight
    tp = m.tp_size
    qkv_len = (m.qk_dim * 2 + m.v_dim) // tp
    z_len = m.v_dim // tp
    b_len = m.num_value_heads // tp
    a_len = m.num_value_heads // tp
    z0 = qkv_len
    b0 = z0 + z_len
    a0 = b0 + b_len
    W = m.in_proj.weight.data
    return [
        (f"{name}.A_log", m.A_log.data, None),
        (f"{name}.dt_bias", m.dt_bias.data, None),
        (f"{name}.conv1d.weight", m.conv1d.weight.data, None),
        (f"{name}.in_proj[z]", W, (z0, z0 + z_len)),
        (f"{name}.in_proj[beta]", W, (b0, b0 + b_len)),
        (f"{name}.in_proj[alpha]", W, (a0, a0 + a_len)),
    ]


def _make_test_param_disturbance(base_forward_step):
    # returns test_param_disturbance as (model)->bool with base_forward_step in a closure;
    # same factory pattern as the xdoc/gated suites -- avoids sys.path shadowing of root pretrain_gpt.py
    def test_param_disturbance(model):
        print_rank_0("\n### Test: param_disturbance ###")
        gdn_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, GatedDeltaNet)]
        if not gdn_layers:
            print_rank_0("[FAIL] param_disturbance: no GatedDeltaNet module found in model")
            return False

        torch.manual_seed(1234)
        batch = next(make_simple_iter())
        was_training = model.training
        model.eval()

        def forward():
            with torch.no_grad():
                out, _ = base_forward_step(iter([batch]), model)
            return out.detach().float()

        out_ref = forward()

        targets = []
        for n, m in gdn_layers:
            targets.extend(_gdn_targets(n, m))

        errors = []
        for label, tensor, sl in targets:
            view = tensor if sl is None else tensor[sl[0]:sl[1]]
            saved = view.clone()
            # Deterministic, large, downward perturbation. Downward matters for the decay params:
            # g = -exp(A_log) * softplus(alpha + dt_bias); when the decay exp(g) saturates near 0,
            # pushing A_log/dt_bias UP leaves it ~0 (no output change), while pushing DOWN grows the
            # decay -> guaranteed effect. -2.0 is also a large change for every other target.
            # Deterministic (not randn) so the result doesn't depend on the RNG stream / TP shard shape.
            view.add_(-2.0)   # perturb in place
            out_pert = forward()
            view.copy_(saved)                          # restore

            max_diff = (out_ref - out_pert).abs().max().item()
            print_rank_0(f"  {label}: max_diff={max_diff:.6f}")
            if not (max_diff > 1e-4):
                errors.append(label)

        if was_training:
            model.train()

        if errors:
            print_rank_0(f"\n[FAIL] param_disturbance: no output change for {len(errors)} target(s):")
            for e in errors:
                print_rank_0(f"  {e}")
            return False

        print_rank_0(f"[PASS] param_disturbance: all {len(targets)} targets affect output → wired into forward")
        return True

    return test_param_disturbance


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'gdn' suite;
    # each returned function has signature (model)->bool
    return [_make_test_param_disturbance(base_forward_step)]
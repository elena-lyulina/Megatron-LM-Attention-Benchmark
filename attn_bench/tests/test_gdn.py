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


# ── state carry between batches (--gdn-state-carry-ratio) ───────────────────────
# Black-box suites: they observe whatever ratio the model was built with (never set it), and each
# slurm declares the expected verdict per ratio. They run in model.train() (carry is off in eval) and
# restore every GDN module afterwards so the real training that follows starts clean. A model built at
# ratio 0 has no carry slots, so every suite below fails there (nothing to verify) -- declared fail.

def _gdn_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, GatedDeltaNet)]


def _carry_enabled(layers):
    # the carry slots exist only when the model was built with a ratio > 0
    return all(hasattr(m, "_last_recurrent_state") for _, m in layers)


def _snapshot_carry(layers):
    # snapshot what the tests touch; the ratio is left alone
    return {n: (m._last_recurrent_state, m._last_conv_state, m._carry_step, m._carry_rank,
                m.layer_number) for n, m in layers}


def _restore_carry(layers, snap):
    for n, m in layers:
        (m._last_recurrent_state, m._last_conv_state, m._carry_step,
         m._carry_rank, m.layer_number) = snap[n]


def _clear_slots(layers):
    # empty the carried state so the next forward starts fresh; no-op for a ratio-0 model that has no
    # slots (so it never fabricates carry attributes)
    for _, m in layers:
        if hasattr(m, "_last_recurrent_state"):
            m._last_recurrent_state = None
            m._last_conv_state = None
            m._carry_step = 0


def _new_batch(seed):
    torch.manual_seed(seed)
    return next(make_simple_iter())


def _forward(base_forward_step, model, batch):
    # returns the per-token loss [b, s] (forward_step calls the model with labels, so output_tensor
    # is the cross-entropy loss) -- comparing these compares losses with vs without carried state
    with torch.no_grad():
        out, _ = base_forward_step(iter([batch]), model)
    return out.detach().float()


def _max_diff(a, b):
    return (a - b).abs().max().item()


# Statistical carry_effect: number of independent populate+carry trials. Each trial draws a fresh
# per-(layer, sequence) carry decision, so the assertion runs over CARRY_EFFECT_RUNS * mbs Bernoulli
# trials. At 2000 runs the worst-case (over any ratio) std of the measured rate is <= 0.5/sqrt(4000)
# ~ 0.008, so CARRY_EFFECT_TOL=0.03 is a ~4-sigma pass band (false-fail ~1e-4). ~4000 extra forwards
# need the 10-min job wall. Tunable.
CARRY_EFFECT_RUNS = 2000
CARRY_EFFECT_TOL = 0.03


def _make_test_carry_sample(base_forward_step):
    def test_carry_sample(model):
        # With the model's configured ratio: the per-sequence sampler produces the right carry
        # fraction, is reproducible, and gives independent streams per (rank, layer, step) -- which
        # for 0<ratio<1 means ~50% of decisions differ, and at ratio 0/1 means the seed changes
        # nothing (degenerate). Passes at every ratio.
        print_rank_0("\n### Test: carry_sample ###")
        layers = _gdn_layers(model)
        if not layers:
            print_rank_0("[FAIL] carry_sample: no GatedDeltaNet module found")
            return False
        if not _carry_enabled(layers):
            print_rank_0("  carry disabled (built at ratio 0); nothing to verify")
            print_rank_0("[FAIL] carry_sample")
            return False
        snap = _snapshot_carry(layers)
        _, m = layers[0]
        try:
            device = next(model.parameters()).device
            ratio = m.gdn_state_carry_ratio

            # carried fraction over a large sample matches the configured ratio
            frac = m._sample_state_carry(4000, device).float().mean().item()
            rate_ok = (frac == ratio) if ratio in (0.0, 1.0) else (abs(frac - ratio) < 0.05)

            def draw(rank, layer, step):
                m._carry_rank, m.layer_number, m._carry_step = rank, layer, step
                return m._sample_state_carry(256, device)

            def frac_differ(a, b):
                return (a != b).float().mean().item()

            base = draw(0, 1, 5)
            reproducible = torch.equal(base, draw(0, 1, 5))
            diffs = [frac_differ(base, draw(*c)) for c in [(1, 1, 5), (0, 2, 5), (0, 1, 6)]]
            if 0 < ratio < 1:
                decorrelated = all(abs(d - 0.5) < 0.1 for d in diffs)   # independent streams
            else:
                decorrelated = all(d == 0 for d in diffs)               # degenerate, seed-invariant

            ok = rate_ok and reproducible and decorrelated
            print_rank_0(f"  ratio={ratio} frac={frac:.3f} rate_ok={rate_ok} "
                         f"reproducible={reproducible} decorrelated={decorrelated} "
                         f"diffs={[round(d, 3) for d in diffs]}")
            print_rank_0(f"[{'PASS' if ok else 'FAIL'}] carry_sample")
            return ok
        finally:
            _restore_carry(layers, snap)
    return test_carry_sample


def _make_test_carry_effect(base_forward_step):
    def test_carry_effect(model):
        # Statistical: don't force carry on -- measure how OFTEN the carried state actually moves the
        # loss and check that rate against the probability implied by the built ratio (a single-shot
        # check would be flaky at 0<ratio<1, failing whenever the draw happened to skip the carry).
        # Each trial populates the state from batch_a then runs batch_b with a fresh per-sequence carry
        # draw. A sequence's loss is unchanged from fresh only when EVERY GDN layer independently
        # declined to carry it, so over many trials:
        #   unchanged_rate  -> (1-ratio)**L           (L = #GDN layers)
        #   both_unchanged  -> ((1-ratio)**L)**2       (the two sequences decide independently)
        # At ratio 1.0 -> 0 unchanged (always carries); ratio 0 has no carry slots (expected fail).
        print_rank_0("\n### Test: carry_effect ###")
        layers = _gdn_layers(model)
        if not layers:
            print_rank_0("[FAIL] carry_effect: no GatedDeltaNet module found")
            return False
        if not _carry_enabled(layers):
            print_rank_0("  carry disabled (built at ratio 0); nothing to verify")
            print_rank_0("[FAIL] carry_effect")
            return False
        try:
            model.train()
            ratio = layers[0][1].gdn_state_carry_ratio
            n_layers = len(layers)
            batch_a, batch_b = _new_batch(1), _new_batch(2)

            # Per-sequence noise floor: batch_b from empty slots, run twice (FLA kernel jitter).
            _clear_slots(layers)
            fresh1 = _forward(base_forward_step, model, batch_b)
            _clear_slots(layers)
            fresh2 = _forward(base_forward_step, model, batch_b)
            n_seq = fresh1.shape[0]
            thresh = [max(1e-4, 5 * _max_diff(fresh1[i], fresh2[i])) for i in range(n_seq)]

            # Each trial: populate from batch_a (starts from zeros, so its own draw is moot), then set
            # a unique _carry_step so batch_b draws an independent decision -- _clear_slots resets the
            # step to 0, which would otherwise freeze every trial to the same draw.
            changed = 0          # (trial, sequence) pairs whose loss moved past the noise floor
            both_unchanged = 0   # trials where neither sequence moved
            for r in range(CARRY_EFFECT_RUNS):
                _clear_slots(layers)
                _forward(base_forward_step, model, batch_a)
                for _, m in layers:
                    m._carry_step = r + 1
                out = _forward(base_forward_step, model, batch_b)
                moved = [_max_diff(out[i], fresh1[i]) > thresh[i] for i in range(n_seq)]
                changed += sum(moved)
                both_unchanged += int(not any(moved))

            n_trials = CARRY_EFFECT_RUNS * n_seq
            unchanged_rate = 1 - changed / n_trials
            exp_unchanged = (1 - ratio) ** n_layers
            both_unchanged_rate = both_unchanged / CARRY_EFFECT_RUNS

            ok = abs(unchanged_rate - exp_unchanged) < CARRY_EFFECT_TOL
            print_rank_0(f"  ratio={ratio} layers={n_layers} trials={n_trials} "
                         f"unchanged_rate={unchanged_rate:.3f} expected=(1-ratio)^{n_layers}={exp_unchanged:.3f} "
                         f"both_unchanged_rate={both_unchanged_rate:.3f} expected={exp_unchanged**2:.3f}")
            print_rank_0(f"[{'PASS' if ok else 'FAIL'}] carry_effect")
            return ok
        finally:
            _clear_slots(layers)
    return test_carry_effect


def _make_test_carry_mechanism(base_forward_step):
    def test_carry_mechanism(model):
        # The carry plumbing is correct: the stored state is detached (truncated BPTT) and survives a
        # checkpoint round-trip with the resume layout lock. Returns False when carry produced no
        # state (ratio 0) -- the expected verdict for the no-carry run.
        print_rank_0("\n### Test: carry_mechanism ###")
        import shutil
        import tempfile

        from megatron.training.checkpointing import load_gdn_states, save_gdn_states

        layers = _gdn_layers(model)
        if not layers:
            print_rank_0("[FAIL] carry_mechanism: no GatedDeltaNet module found")
            return False
        if not _carry_enabled(layers):
            # ratio 0: no carry slots exist (_snapshot_carry would AttributeError); nothing to verify
            print_rank_0("  carry disabled (built at ratio 0); nothing to verify")
            print_rank_0("[FAIL] carry_mechanism")
            return False
        args = get_args()
        snap = _snapshot_carry(layers)
        tmp = tempfile.mkdtemp()
        try:
            model.train()
            # Populate the slots with a grad-enabled forward, so we can also check they are detached.
            _clear_slots(layers)
            with torch.enable_grad():
                base_forward_step(iter([_new_batch(1)]), model)
            # At ratio 0 there are no slots (carry off); the first clause short-circuits the access.
            if not _carry_enabled(layers) or any(
                m._last_recurrent_state is None or m._last_conv_state is None for _, m in layers
            ):
                print_rank_0("  no carried state to verify (carry off)")
                print_rank_0("[FAIL] carry_mechanism")
                return False

            detached = all(not m._last_recurrent_state.requires_grad
                           and m._last_recurrent_state.grad_fn is None
                           and not m._last_conv_state.requires_grad
                           and m._last_conv_state.grad_fn is None for _, m in layers)

            # save -> corrupt -> load restores the exact state + step
            ref = {n: (m._last_recurrent_state.clone(), m._last_conv_state.clone(), m._carry_step)
                   for n, m in layers}
            save_gdn_states(model, tmp, 42)
            for _, m in layers:
                m._last_recurrent_state, m._last_conv_state, m._carry_step = None, None, 999
            load_gdn_states(model, tmp, 42)
            restored = all(torch.allclose(m._last_recurrent_state, ref[n][0])
                           and torch.allclose(m._last_conv_state, ref[n][1])
                           and m._carry_step == ref[n][2] for n, m in layers)

            # layout lock: a changed micro_batch_size must make load raise
            mbs0 = args.micro_batch_size
            args.micro_batch_size = mbs0 + 1
            raised = False
            try:
                load_gdn_states(model, tmp, 42)
            except AssertionError:
                raised = True
            finally:
                args.micro_batch_size = mbs0

            ok = detached and restored and raised
            print_rank_0(f"  detached={detached} restored={restored} layout_lock={raised}")
            print_rank_0(f"[{'PASS' if ok else 'FAIL'}] carry_mechanism")
            return ok
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            _restore_carry(layers, snap)
    return test_carry_mechanism


# ── registration ──────────────────────────────────────────────────────────────

def register(base_forward_step):
    # called by registry.py to resolve test functions for the 'gdn' suite;
    # each returned function has signature (model)->bool
    return [_make_test_param_disturbance(base_forward_step)]


def register_carry_sample(base_forward_step):
    return [_make_test_carry_sample(base_forward_step)]


def register_carry_effect(base_forward_step):
    return [_make_test_carry_effect(base_forward_step)]


def register_carry_mechanism(base_forward_step):
    return [_make_test_carry_mechanism(base_forward_step)]
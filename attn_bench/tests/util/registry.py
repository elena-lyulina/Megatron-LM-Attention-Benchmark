from attn_bench.tests.test_gated_attention import register as _register_gated
from attn_bench.tests.test_gdn import register as _register_gdn
from attn_bench.tests.test_gdn import register_carry_effect as _register_gdn_carry_effect
from attn_bench.tests.test_gdn import register_carry_mechanism as _register_gdn_carry_mechanism
from attn_bench.tests.test_gdn import register_carry_sample as _register_gdn_carry_sample
from attn_bench.tests.test_sink_attention import register as _register_sink
from attn_bench.tests.test_xdoc_attention import register_loss as _register_xdoc_loss
from attn_bench.tests.test_xdoc_attention import register_mask as _register_xdoc_mask
from attn_bench.tests.test_xdoc_attention import (
    register_position_ids as _register_xdoc_position_ids,
)
from megatron.training import get_args, print_rank_0

# maps --tests name → register(base_forward_step) → [test_fn, ...]
TEST_REGISTRY = {
    "xdoc_mask": _register_xdoc_mask,
    "xdoc_loss": _register_xdoc_loss,
    "xdoc_position_ids": _register_xdoc_position_ids,
    "sink": _register_sink,
    "gated": _register_gated,
    "gdn": _register_gdn,
    "gdn_carry_sample": _register_gdn_carry_sample,
    "gdn_carry_effect": _register_gdn_carry_effect,
    "gdn_carry_mechanism": _register_gdn_carry_mechanism,
}


def _parse_tests(test_items):
    # parses ["sink=fail", "xdoc_loss=pass"] -> {"sink": False, "xdoc_loss": True}, preserving order.
    # the expected verdict is REQUIRED: a bare suite name (no =pass|fail) is a parse error, so a run
    # can never report results without declaring what it expected.
    expect = {}
    for item in test_items:
        suite, sep, val = item.partition("=")
        val = val.lower()
        if not sep or val not in ("pass", "fail"):
            raise ValueError(f"--tests entry {item!r} must be SUITE=pass|fail (expected verdict required)")
        if suite not in TEST_REGISTRY:
            raise ValueError(f"Unknown test suite: {suite!r}. Available: {sorted(TEST_REGISTRY)}")
        expect[suite] = (val == "pass")
    return expect


def _verdict(v):
    # tri-state verdict renderer: True -> PASS, False -> FAIL, None -> '-' (suite had only diagnostics)
    return {True: "PASS", False: "FAIL", None: "-"}[v]


def _report(base_forward_step, model, expect):
    # runs every requested suite, prints a suite-grouped summary, and compares each suite's verdict
    # (AND of its non-diagnostic assertions) to the expected pass/fail declared in --tests.
    records = []  # (suite, fn_name, passed, is_diagnostic)
    for suite in expect:
        for fn in TEST_REGISTRY[suite](base_forward_step):
            passed = fn(model)
            records.append((suite, fn.__name__, passed, getattr(fn, "_is_diagnostic", False)))

    print_rank_0("\n### Summary ###")
    mismatches = 0
    for suite in expect:
        rows = [r for r in records if r[0] == suite]
        assertions = [r for r in rows if not r[3]]
        actual = all(r[2] for r in assertions) if assertions else None  # None => only diagnostics
        exp = expect[suite]
        ok = (actual == exp)
        mismatches += 0 if ok else 1
        print_rank_0(
            f"suite={suite}   expected={_verdict(exp)}  ->  {_verdict(actual)}  "
            f"{'[OK]' if ok else '[MISMATCH]'}"
        )
        for _, fn_name, passed, is_diag in rows:
            if is_diag:
                print_rank_0(f"    [diag] {fn_name}  ({'PASS' if passed else 'FAIL'}, not counted)")
            else:
                print_rank_0(f"    {'PASS' if passed else 'FAIL'}   {fn_name}")

    n = len(expect)
    if mismatches == 0:
        print_rank_0(f"\n### Verdict: ALL {n} SUITE(S) AS EXPECTED ###\n")
    else:
        print_rank_0(f"\n### Verdict: {mismatches}/{n} SUITE(S) UNEXPECTED ###\n")


def make_test_forward_step(base_forward_step):
    """Wraps base_forward_step to inject the suites listed in --tests on the first iteration."""
    _done = False

    def forward_step(data_iterator, model, return_schedule_plan=False):
        nonlocal _done
        if not _done:
            _done = True
            expect = _parse_tests(get_args().tests or [])
            if expect:
                _report(base_forward_step, model, expect)

        return base_forward_step(data_iterator, model, return_schedule_plan)

    return forward_step

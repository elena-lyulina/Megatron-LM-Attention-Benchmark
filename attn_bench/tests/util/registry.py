from megatron.training import get_args, print_rank_0

from attn_bench.tests.test_gated_attention import register as _register_gated
from attn_bench.tests.test_sink_attention import register as _register_sink
from attn_bench.tests.test_xdoc_attention import register as _register_xdoc

# maps --tests name → register(base_forward_step) → [test_fn, ...]
TEST_REGISTRY = {
    "xdoc": _register_xdoc,
    "sink": _register_sink,
    "gated": _register_gated,
}


def make_test_forward_step(base_forward_step):
    """Wraps base_forward_step to inject tests listed in --tests on the first iteration."""
    _done = False

    def forward_step(data_iterator, model, return_schedule_plan=False):
        nonlocal _done
        if not _done:
            _done = True
            args = get_args()
            test_fns = []
            for name in (args.tests or []):
                if name not in TEST_REGISTRY:
                    raise ValueError(f"Unknown test suite: {name!r}. Available: {sorted(TEST_REGISTRY)}")
                test_fns.extend(TEST_REGISTRY[name](base_forward_step))

            if test_fns:
                results = {fn.__name__: fn(model) for fn in test_fns}
                print_rank_0("\n### Summary ###")
                for name, passed in results.items():
                    print_rank_0(f"  {'PASS' if passed else 'FAIL'}: {name}")
                all_passed = all(results.values())
                print_rank_0(f"\n{'All tests PASSED.' if all_passed else 'Some tests FAILED.'}\n")

        return base_forward_step(data_iterator, model, return_schedule_plan)

    return forward_step

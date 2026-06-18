def add_test_args(parser):
    group = parser.add_argument_group("Attention Benchmark")
    group.add_argument("--tests", nargs='+', default=[], metavar="TEST",
        help="Test suites to inject into forward_step (e.g. xdoc). See util/registry.py.")
    return parser


def add_kernel_args(parser):
    group = parser.add_argument_group("Attention Kernel")
    group.add_argument("--attn", type=str, default="full",
        help="Attention type (e.g. full, sink).")
    group.add_argument("--impl", type=str, default="flash",
        help="Kernel implementation (e.g. flash, torch, te).")
    group.add_argument("--attn-kwargs", nargs='*', default=[], metavar="KEY=VALUE",
        help="Mechanism-specific kwargs as key=value pairs. See attn_bench/kernels/attn_registry.py.")
    return parser

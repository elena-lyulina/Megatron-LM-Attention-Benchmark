"""
Additional arguments for benchmarking, such as the name of the attention mechanism and its implementation.
Additional args for each implementation could be added here as well if needed.
"""


def add_benchmark_args(parser):
    """Register common benchmark CLI args. Pass as extra_args_provider to pretrain()."""
    group = parser.add_argument_group("Attention Benchmark")
    group.add_argument(
        "--attn",
        type=str,
        default="full",
        help="Attention type (e.g. full, sliding_window).",
    )
    group.add_argument(
        "--impl",
        type=str,
        default="flash",
        help="Kernel implementation (e.g. flash, torch).",
    )
    return parser
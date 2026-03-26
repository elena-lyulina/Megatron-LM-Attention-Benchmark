from attn_bench.kernels.full.specs import FullSpecProvider


class SinkSpecProvider(FullSpecProvider):
    # No structural difference from full attention here, sink is a parameter inside the kernel, so specs can stay the same
    pass
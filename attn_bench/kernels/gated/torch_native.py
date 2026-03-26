from attn_bench.kernels.full.torch_native import TorchFullAttention


class GatedTorchAttention(TorchFullAttention):
    # Gate is applied by GatedSelfAttention via _apply_output_gate, so here can inherit from full attention
    # Therefore can support any implementation the full attention supports
    pass
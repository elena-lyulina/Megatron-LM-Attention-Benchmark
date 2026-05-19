from megatron.core.extensions.transformer_engine import TEDotProductAttention


class GatedTEAttention(TEDotProductAttention):
    # Gate is applied by GatedSelfAttention via _apply_output_gate, so here can inherit from TE attention
    # Therefore can support any implementation TEDotProductAttention supports (packed seqs, TP, etc.)
    pass
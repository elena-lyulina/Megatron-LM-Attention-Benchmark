from typing import Optional

import torch.nn as nn
from torch import Tensor

try:
    from flash_attn import flash_attn_func

    HAVE_FLASH_ATTN = True
except ImportError:
    HAVE_FLASH_ATTN = False


class FlashFullAttention(nn.Module):
    """
    Full (standard) causal attention using the FlashAttention kernel.

    Format: sbhd (standard non-packed)
      query:      [sq, b, np, hn]
      key/value:  [sk, b, ng, hn]
        np = num query heads per TP rank
        ng = num KV heads per TP rank  (ng <= np for GQA)

    Output: [sq, b, np * hn]
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        softmax_scale: float,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert HAVE_FLASH_ATTN, (
            "flash_attn package is required"
        )
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p

    # The parameters come from the CoreAttention protocol that Megatron calls
    # Todo: could separate the protocol and "pure" kernel?
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        /,
        *,
        attn_mask_type,
        attention_bias: Optional[Tensor],
        packed_seq_params=None,
    ) -> Tensor:
        assert attention_bias is None, (
            "FlashFullAttention does not support attention_bias."
        )
        assert packed_seq_params is None, (
            "FlashFullAttention does not support packed sequences yet. "
            "Do not use --use-packed-seq-params with this kernel."
        )
        assert attn_mask_type.name == "causal", (
            f"FlashFullAttention only supports causal masking, got {attn_mask_type.name!r}."
        )

        sq, b, np, hn = query.shape

        # flash_attn expects [b, s, h, d] — GQA (ng < np) supported natively
        out = flash_attn_func(
            query.permute(1, 0, 2, 3),
            key.permute(1, 0, 2, 3),
            value.permute(1, 0, 2, 3),
            dropout_p=self.dropout_p if self.training else 0.0,
            softmax_scale=self.softmax_scale,
            causal=True,
        )

        # [b, sq, np, hn] -> [sq, b, np * hn]
        return out.permute(1, 0, 2, 3).reshape(sq, b, np * hn)
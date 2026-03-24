"""
Full (dense) causal attention using PyTorch's scaled_dot_product_attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TorchFullAttention(nn.Module):
    """
    Implemented via F.scaled_dot_product_attention.

    GQA (ng < np) is handled by repeating KV heads to match query heads.
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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p

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
            "TorchFullAttention does not support attention_bias."
        )
        assert packed_seq_params is None, (
            "TorchFullAttention does not support packed sequences. "
            "Do not use --use-packed-seq-params with this kernel."
        )
        assert attn_mask_type.name == "causal", (
            f"TorchFullAttention only supports causal masking, got {attn_mask_type.name!r}."
        )

        sq, b, np, hn = query.shape

        # GQA: repeat KV heads to match query heads
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        # [s, b, h, d] -> [b, h, s, d]  (SDPA expects batch-first)
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
            scale=self.softmax_scale,
        )

        # [b, np, sq, hn] -> [sq, b, np * hn]
        return out.permute(2, 0, 1, 3).reshape(sq, b, np * hn)

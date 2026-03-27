"""
Sink attention
gpt-oss-120b & gpt-oss-20b Model Card (Agarwal et al. 2025), https://arxiv.org/abs/2508.10925

Modifies the softmax denominator with a learnable per-head scalar `sink`:
    Softmax_sink(x)_{h,i} = exp(x_{h,i}) / (exp(sink_h) + sum_j exp(x_{h,j}))

Equivalent to appending a synthetic token with logit=sink_h and value=0 per head,
so the output is unaffected, only the normalization changes.
Matches TE's softmax_type='learnable': per-head parameter, torch.empty initialisation.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SinkTorchAttention(nn.Module):

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
        # Learnable sink logit per head, matching TE's softmax_offset shape [num_heads] and
        # initialisation: TE calls reset_parameters with get_default_init_method() = N(0, 0.023).
        self.sink = nn.Parameter(torch.empty(num_heads))
        nn.init.normal_(self.sink, mean=0.0, std=0.023)

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
            "SinkTorchAttention does not support attention_bias."
        )
        assert packed_seq_params is None, (
            "SinkTorchAttention does not support packed sequences."
        )
        assert attn_mask_type.name == "causal", (
            f"SinkTorchAttention only supports causal masking, got {attn_mask_type.name!r}."
        )

        sq, b, np, hn = query.shape

        # GQA: repeat KV heads to match query heads
        if self.num_kv_heads < self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        # [s, b, h, d] -> [b, h, s, d]
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.softmax_scale  # [b, h, sq, sk]

        # Causal mask
        sk = k.size(2)
        causal_mask = torch.ones(sq, sk, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Append sink logit as an extra column, softmax over it, then drop it
        # equivalent to the paper's formula but avoids any kernel modification.
        # Cast to activation dtype (matching TE's softmax_offset.to(dtype=matmul_result.dtype)).
        # self.sink: [h] -> [1, h, 1, 1] -> [b, h, sq, 1]
        sink_col = self.sink.to(q.dtype).reshape(1, self.num_heads, 1, 1).expand(b, self.num_heads, sq, 1)
        attn = F.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[:, :, :, :-1]

        if self.training and self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p)

        out = torch.matmul(attn, v)  # [b, h, sq, hn]

        # [b, h, sq, hn] -> [sq, b, np * hn]
        return out.permute(2, 0, 1, 3).reshape(sq, b, np * hn)
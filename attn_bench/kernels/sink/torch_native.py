"""
Sink attention
gpt-oss-120b & gpt-oss-20b Model Card (Agarwal et al. 2025), https://arxiv.org/abs/2508.10925

Modifies the softmax denominator with a learnable scalar `sink`:
    Softmax_sink(x)_i = exp(x_i) / (sum_j exp(x_j) + exp(sink))

Equivalent to appending a synthetic token with logit=sink and value=0, so the output is unaffected, only the normalization changes.
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
        chunk_size: int = 512,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p
        self.chunk_size = chunk_size
        # Learnable sink logit. exp(sink) is added to the softmax denominator,
        # acting as a zero-value synthetic token. init=0 → exp(sink)=1 at start.
        self.sink = nn.Parameter(torch.zeros(1))

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

        # Chunked attention over query dimension to avoid materialising the full [b, h, sq, sk]
        # score matrix (O(sq*sk) memory). Each chunk is [b, h, chunk_size, sk] instead.
        sk = k.size(2)
        out_chunks = []
        # it's alright that it's very inefficient, this implementation is for correctness only
        for q_start in range(0, sq, self.chunk_size):
            q_end = min(q_start + self.chunk_size, sq)
            q_chunk = q[:, :, q_start:q_end, :]  # [b, h, chunk_q, d]

            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.softmax_scale  # [b, h, chunk_q, sk]

            # Causal mask: query at position q_start+i can attend to keys 0..q_start+i
            q_pos = torch.arange(q_start, q_end, device=q.device).unsqueeze(1)  # [chunk_q, 1]
            k_pos = torch.arange(sk, device=k.device).unsqueeze(0)              # [1, sk]
            causal_mask = q_pos >= k_pos                                          # [chunk_q, sk]
            scores = scores.masked_fill(~causal_mask, float('-inf'))

            # Append sink logit as an extra column, softmax over it, then drop it
            sink_col = self.sink.expand(b, self.num_heads, q_end - q_start, 1)
            attn = F.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)[:, :, :, :-1]

            if self.training and self.dropout_p > 0.0:
                attn = F.dropout(attn, p=self.dropout_p)

            out_chunks.append(torch.matmul(attn, v))  # [b, h, chunk_q, hn]

        out = torch.cat(out_chunks, dim=2)  # [b, h, sq, hn]

        # [b, h, sq, hn] -> [sq, b, np * hn]
        return out.permute(2, 0, 1, 3).reshape(sq, b, np * hn)
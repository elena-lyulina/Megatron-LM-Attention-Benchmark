"""
Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
https://arxiv.org/abs/2505.06708

X = RMSNorm(residual_stream)
"""

from __future__ import annotations

from typing import Optional

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import CoreAttentionBuilder, SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from attn_bench.kernels.megatron_wrapper import make_megatron_wrapper


class GatedSelfAttention(SelfAttention):
    """
    Self-attention with a learned sigmoid output gate.

    Setting attention_output_gate=True before super().__init__ does three things
    automatically in the parent's forward:
      1. get_query_key_value_tensors(output_gate=True) — linear_qkv projects to
         [Q, gate, K, V] instead of [Q, K, V], with gate having the same shape as Q
      2. core_attention(Q, K, V, ...) — kernel is called unchanged
      3. _apply_output_gate(out, gate) — applies out * sigmoid(gate), dtype-safe

    Q = X W_q
    K = X W_k
    V = X W_v
    Y = SDPA(Q, K, V)
    gate = sigmoid(X W_θ)
    Y' = Y ⊙ gate
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.padding,
        cp_comm_type: str | None = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        pp_layer_offset: Optional[int] = None,
    ):
        # Must be set before super().__init__ so that linear_qkv_out_dim is expanded to include the gate slot
        # SelfAttention.forward() calls _apply_output_gate.
        config.attention_output_gate = True
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
        )


class GatedSpecProvider:
    # TODO: create a Base spec? to not duplicate
    def __init__(self, kernel_cls: type, base: BackendSpecProvider):
        self._kernel_wrapper_cls = make_megatron_wrapper(kernel_cls)
        self._base = base

    def core_attention(self) -> CoreAttentionBuilder:
        return self._kernel_wrapper_cls

    def attention_spec(self, qk_layernorm: bool = False) -> ModuleSpec:
        linear_qkv = (
            self.column_parallel_layer_norm_linear()
            if self.fuse_layernorm_and_linear()
            else self.column_parallel_linear()
        )
        qk_norm = self.layer_norm(for_qk=True) if qk_layernorm else IdentityOp

        # GatedSelfAttention sets config.attention_output_gate=True in __init__, which:
        #   1. expands linear_qkv to produce [Q, gate, K, V]
        #   2. wires gate through forward into _apply_output_gate(out * sigmoid(gate))
        return ModuleSpec(
            module=GatedSelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=linear_qkv,
                core_attention=self.core_attention(),
                linear_proj=self.row_parallel_linear(),
                q_layernorm=qk_norm,
                k_layernorm=qk_norm,
            ),
        )

    def __getattr__(self, name: str):
        return getattr(self._base, name)
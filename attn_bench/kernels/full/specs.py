"""
Spec provider for full (dense) attention.

One provider covers all full-attention kernel classes — the kernel and base
backend are injected at construction time. The Megatron wrapper is created
dynamically via make_megatron_wrapper, so adding a new impl (sdpa, triton)
requires no changes here.
"""

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.attention import (
    CoreAttentionBuilder,
    SelfAttention,
    SelfAttentionSubmodules,
)
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec

from attn_bench.kernels.megatron_wrapper import make_megatron_wrapper


class FullSpecProvider:
    """
    Spec provider for full (dense) attention.

    Wraps any pure full-attention kernel in a Megatron-compatible module and
    delegates all projection / layernorm choices to the base backend.
    """

    def __init__(self, kernel_cls: type, base: BackendSpecProvider):
        self._kernel_wrapper_cls = make_megatron_wrapper(kernel_cls)
        self._base = base

    def core_attention(self) -> CoreAttentionBuilder:
        return self._kernel_wrapper_cls

    def attention_spec(self, qk_layernorm: bool = False) -> ModuleSpec:
        """Build the full ModuleSpec for full attention."""
        linear_qkv = (
            self.column_parallel_layer_norm_linear()
            if self.fuse_layernorm_and_linear()
            else self.column_parallel_linear()
        )
        qk_norm = self.layer_norm(for_qk=True) if qk_layernorm else IdentityOp

        return ModuleSpec(
            module=SelfAttention,
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
        # Delegates all linear/layernorm/etc. questions to the base backend.
        return getattr(self._base, name)
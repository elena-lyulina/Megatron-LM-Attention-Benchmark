"""
Generic Megatron wrapper factory for pure attention kernels.

Any pure kernel class (zero Megatron imports, plain constructor args) can be
wrapped into a MegatronModule via make_megatron_wrapper(). The wrapper:
  - Accepts Megatron's standard CoreAttentionBuilder constructor signature
  - Extracts plain values from TransformerConfig
  - Passes mechanism-specific kwargs (e.g. window_size) to the kernel
  - Delegates all compute to the pure kernel

Shared across all mechanisms: each mechanism's specs.py calls this factory and contributes its own SpecProvider + attention_spec().
"""

import math
from typing import Optional

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


def make_megatron_wrapper(kernel_cls: type, **kernel_kwargs) -> type:
    """
    Dynamically create a MegatronModule wrapper for a pure attention kernel.

    Args:
    - kernel_cls: pure kernel class with signature:
         __init__(num_heads, num_kv_heads, head_dim, softmax_scale, dropout_p, **kernel_kwargs)
    - **kernel_kwargs: mechanism-specific kwargs forwarded to the kernel constructor (e.g. window_size=512 for sliding window)

    Returns:
        A MegatronModule subclass satisfying the CoreAttentionBuilder protocol.
    """

    class MegatronWrapper(MegatronModule):

        def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: Optional[float] = None,
            softmax_scale: Optional[float] = None,
            cp_comm_type: Optional[str] = None,  # CP is not supported yet
            pg_collection: Optional[ProcessGroupCollection] = None,
        ):
            super().__init__(config=config)

            if pg_collection is None:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(
                    required_pgs=['tp']
                )
            tp_size = pg_collection.tp.size()

            num_heads = divide(config.num_attention_heads, tp_size)
            num_kv_heads = divide(config.num_query_groups, tp_size)
            head_dim = config.kv_channels

            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(head_dim)

            dropout_p = (
                config.attention_dropout if attention_dropout is None else attention_dropout
            )

            self.kernel = kernel_cls(
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                **kernel_kwargs,
            )

        def forward(
            self,
            query,
            key,
            value,
            attention_mask,
            /,
            *,
            attn_mask_type: AttnMaskType,
            attention_bias,
            packed_seq_params: Optional[PackedSeqParams],
        ):
            return self.kernel(
                query, key, value, attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )

    MegatronWrapper.__name__ = f"{kernel_cls.__name__}Megatron"
    MegatronWrapper.__qualname__ = f"{kernel_cls.__qualname__}Megatron"
    return MegatronWrapper

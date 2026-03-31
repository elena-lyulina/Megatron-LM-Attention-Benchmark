import dataclasses
from typing import Optional

import torch.nn as nn
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig


class SinkTEAttention(TEDotProductAttention):
    """
    Sink attention via Transformer Engine's native learnable-softmax-offset support.

    Sets softmax_type='learnable' in the config copy before forwarding to TEDotProductAttention,
    which passes it to te.pytorch.DotProductAttention (requires TE >= 2.8.0).
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = "p2p",
        pg_collection: Optional[ProcessGroupCollection] = None,
        init_sink_zero: bool = False,
    ):
        sink_config = dataclasses.replace(config, softmax_type="learnable")
        super().__init__(
            config=sink_config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            softmax_scale=softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        if init_sink_zero:
            # changes it in-place, already after TE's initialized the params inside the super call
            nn.init.zeros_(self.softmax_offset)
"""HuggingFace config/model classes (subclassing transformers' Llama implementation) for
sink-attention checkpoints, needed so AutoModelForCausalLM can load them like any other HF
model. Custom (using flash_attn.cute (FA4)'s fused learnable_sink kernel) rather than
reusing gpt_oss's sink architecture -- gpt_oss's config hard-locks attention to kernels-community/vllm-flash-attn3,
which has no aarch64/Hopper build, so it would default to slow, non-fused fallback.
Everything except attention is stock Llama, subclassed rather than copied.
No eager/sdpa fallback: this architecture only exists for the fused-kernel speedup.
"""

import os

import flash_attn
import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (LlamaAttention,
                                                      LlamaConfig,
                                                      LlamaDecoderLayer,
                                                      LlamaForCausalLM,
                                                      LlamaModel)

if not hasattr(flash_attn, "cute"):
    flash_attn.__path__.append(f"{os.environ['FLASH_ATTN_SRC_DIR']}/flash_attn")
from flash_attn.cute.interface import flash_attn_func


def sink_flash_attention_forward(module, query, key, value, attention_mask, scaling,
                                 dropout=0.0, learnable_sink=None, **kwargs):
    """ALL_ATTENTION_FUNCTIONS entry point. query/key/value arrive as [B, H, S, D] (HF's
    internal convention); flash_attn.cute wants [B, S, H, D]."""
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    result = flash_attn_func(q, k, v, causal=True, softmax_scale=scaling, learnable_sink=learnable_sink)
    attn_output = result[0] if isinstance(result, tuple) else result
    return attn_output, None


ALL_ATTENTION_FUNCTIONS.register("sink_cute_attention", sink_flash_attention_forward)


class SinkLlamaConfig(LlamaConfig):
    model_type = "sink_llama"

    def __init__(self, **kwargs):
        # Defaults to our registered kernel rather than PreTrainedConfig's usual "sdpa" --
        # sdpa/eager would silently drop the learnable_sink kwarg instead of erroring, so
        # this can't be left to the normal fallback chain.
        kwargs.setdefault("attn_implementation", "sink_cute_attention")
        super().__init__(**kwargs)


class SinkLlamaAttention(LlamaAttention):
    def __init__(self, config: SinkLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.learnable_sink = nn.Parameter(torch.zeros(config.num_attention_heads))

    def forward(self, **kwargs):
        kwargs["learnable_sink"] = self.learnable_sink
        return super().forward(**kwargs)


class SinkLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: SinkLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = SinkLlamaAttention(config=config, layer_idx=layer_idx)


class SinkLlamaModel(LlamaModel):
    config: SinkLlamaConfig

    def __init__(self, config: SinkLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [SinkLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class SinkLlamaForCausalLM(LlamaForCausalLM):
    config: SinkLlamaConfig

    def __init__(self, config: SinkLlamaConfig):
        super().__init__(config)
        self.model = SinkLlamaModel(config)
        self.post_init()

"""Shared prepare_inputs_for_generation + _update_model_kwargs_for_generation for every custom
Mamba-cache-shaped family (GDN, KDA, the Qwen hybrid): each keeps per-layer conv/recurrent
state in a `cache_params` object instead of a standard `past_key_values` Cache
(transformers.generation.utils special-cases the `cache_params` name for this
Mamba2ForCausalLM-style convention).

The initial cache_position (first call) is seeded to the real prompt length, not a placeholder
-- GDN/KDA only check cache_position[0] > 0, so length never mattered for them, but Qwen's
softmax layers derive RoPE cos/sin from it too, and a short placeholder breaks
apply_rotary_pos_emb on prefill.

_update_model_kwargs_for_generation mirrors Mamba2ForCausalLM's own override (real
transformers.models.mamba.modeling_mamba.py) instead of falling back to GenerationMixin's
generic one via MRO: the generic version leaves cache_position uncollapsed across decode
steps for a cache_params-based model, growing every step instead of resetting to length 1 --
harmless for GDN/KDA (their mixers only ever check cache_position[0], never its length) but
fatal for Qwen's attention layers, whose KV cache then concatenates a never-truncated sequence
each step (quadratic memory -- the cause of the Qwen-hybrid Step-4 OOM without it).
"""

from typing import Optional

import torch


class CacheParamsGenerationMixin:
    """Mix in alongside a family's *PreTrainedModel + GenerationMixin bases (in that order,
    before GenerationMixin, so this method wins over GenerationMixin's own default) to get
    this prepare_inputs_for_generation for free -- e.g.
    class GDNLlamaForCausalLM(GDNLlamaPreTrainedModel, CacheParamsGenerationMixin, GenerationMixin).
    """

    def prepare_inputs_for_generation(
        self,
        input_ids,
        inputs_embeds=None,
        use_cache=None,
        cache_params=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        model_inputs = {"input_ids": input_ids.contiguous()}
        if use_cache and cache_params is None:
            cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)
            if inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}

        if use_cache and cache_position[0] > 0:
            model_inputs["input_ids"] = input_ids[:, -1].unsqueeze(-1).contiguous()

        if not use_cache and inputs_embeds is not None:
            model_inputs = {"inputs_embeds": inputs_embeds}

        model_inputs.update(
            {"cache_params": cache_params, "use_cache": use_cache, "cache_position": cache_position}
        )
        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, num_new_tokens=1, **kwargs):
        """See module docstring. **kwargs absorbs is_encoder_decoder (passed by keyword from
        GenerationMixin's _sample; none of GDN/KDA/Qwen are encoder-decoder)."""
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if (
            model_kwargs.get("use_cache", True)
            and "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

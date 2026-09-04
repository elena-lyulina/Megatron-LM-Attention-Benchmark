"""Shared prepare_inputs_for_generation for every custom Mamba-cache-shaped family (GDN, KDA,
the Qwen hybrid): each keeps per-layer conv/recurrent state in a `cache_params` object instead
of a standard `past_key_values` Cache (transformers.generation.utils special-cases the
`cache_params` name for exactly this Mamba2ForCausalLM-style convention).

The initial cache_position (first call) is seeded to the real prompt length, not a placeholder
-- GDN/KDA only check cache_position[0] > 0, so length never mattered for them, but Qwen's
softmax layers derive RoPE cos/sin from it too, and a short placeholder there breaks
apply_rotary_pos_emb on prefill (job 3296040). Correct for GDN/KDA either way.
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

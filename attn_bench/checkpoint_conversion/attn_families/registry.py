"""Detects which attention family a Megatron checkpoint belongs to from its own restored
args -- same principle MegatronBackend relies on: the checkpoint says what it is, no
user-facing flag needed -- and looks up that attn_family's config/state-dict builder module.
"""

from typing import Any

from attn_bench.checkpoint_conversion.attn_families import full, gated, mla, swa

ATTN_FAMILIES = ("full", "sink", "gated", "swa", "gdn", "kda", "mla")


def detect_attn_family(args: Any) -> str:
    if getattr(args, "experimental_attention_variant", None) == "gated_delta_net":
        return "gdn"
    if getattr(args, "experimental_attention_variant", None) == "kimi_delta_attention":
        return "kda"
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    if getattr(args, "attention_output_gate", False):
        return "gated"
    softmax_type = getattr(args, "softmax_type", "vanilla")
    if softmax_type == "learnable":
        return "sink"
    if softmax_type == "off-by-one":
        # off-by-one's softmax_offset is a fixed zero tensor, not an nn.Parameter -- never
        # lands in the checkpoint's state dict, so it needs its own conversion path.
        return "off-by-one"
    if getattr(args, "window_size", None) is not None:
        return "swa"
    return "full"


def get_attn_family_module(args: Any):
    attn_family = detect_attn_family(args)
    if attn_family == "full":
        return full
    if attn_family == "swa":
        return swa
    if attn_family == "gated":
        return gated
    if attn_family == "mla":
        return mla
    if attn_family == "sink":
        # lazy: pulls in modeling_sink_llama.py, which needs flash_attn.cute.
        from attn_bench.checkpoint_conversion.attn_families import sink
        return sink
    if attn_family == "gdn":
        # lazy: pulls in modeling_gdn_llama.py, which needs fla (flash-linear-attention).
        from attn_bench.checkpoint_conversion.attn_families import gdn
        return gdn
    if attn_family == "kda":
        # lazy: pulls in modeling_kda_llama.py, which needs fla >= 0.5.2 (chunk_kda / fused_recurrent_kda).
        from attn_bench.checkpoint_conversion.attn_families import kda
        return kda
    raise NotImplementedError(
        f"HF conversion for the '{attn_family}' attention family isn't implemented yet "
        f"(only {sorted(ATTN_FAMILIES)} are). See attn_bench/checkpoint_conversion/attn_families/."
    )

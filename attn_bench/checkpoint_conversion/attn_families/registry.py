"""Detects which attention family a Megatron checkpoint belongs to from its own restored
args -- same principle MegatronBackend relies on: the checkpoint says what it is, no
user-facing flag needed -- and looks up that attn_family's config/state-dict builder module.
"""

from typing import Any

from attn_bench.checkpoint_conversion.attn_families import full

ATTN_FAMILIES = {
    "full": full,
}


def detect_attn_family(args: Any) -> str:
    if getattr(args, "experimental_attention_variant", None) == "gated_delta_net":
        return "gdn"
    if getattr(args, "attention_output_gate", False):
        return "gated"
    softmax_type = getattr(args, "softmax_type", "vanilla")
    if softmax_type == "learnable":
        return "sink"
    if softmax_type == "off-by-one":
        # Distinct from "sink": off-by-one's softmax_offset is a fixed zero tensor, never
        # registered as an nn.Parameter, so it never lands in the checkpoint's state dict --
        # a real, separate conversion path, not just a variant of "sink".
        return "off-by-one"
    return "full"


def get_attn_family_module(args: Any):
    attn_family = detect_attn_family(args)
    if attn_family not in ATTN_FAMILIES:
        raise NotImplementedError(
            f"HF conversion for the '{attn_family}' attention family isn't implemented yet "
            f"(only {sorted(ATTN_FAMILIES)} are). See attn_bench/checkpoint_conversion/attn_families/."
        )
    return ATTN_FAMILIES[attn_family]

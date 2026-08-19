"""Detects which attention family a Megatron checkpoint belongs to from its own restored
args -- same principle MegatronBackend relies on: the checkpoint says what it is, no
user-facing flag needed -- and looks up that attn_family's config/state-dict builder module.
"""

from typing import Any

from attn_bench.checkpoint_conversion.attn_families import full

ATTN_FAMILIES = ("full", "sink", "gated", "swa")


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
    if getattr(args, "window_size", None) is not None:
        return "swa"
    return "full"


def get_attn_family_module(args: Any):
    attn_family = detect_attn_family(args)
    if attn_family not in ATTN_FAMILIES:
        raise NotImplementedError(
            f"HF conversion for the '{attn_family}' attention family isn't implemented yet "
            f"(only {sorted(ATTN_FAMILIES)} are). See attn_bench/checkpoint_conversion/attn_families/."
        )
    if attn_family == "full":
        return full
    if attn_family == "gated":
        raise NotImplementedError("TODO")
    if attn_family == "swa":
        raise NotImplementedError("TODO")
    # sink imported here, not at module top -- it pulls in modeling_sink_llama.py, which
    # needs flash_attn.cute available (see flash-attention-cute-workflow.md). Importing it
    # eagerly at module scope would force that cost onto every attn_family, not just sink.
    from attn_bench.checkpoint_conversion.attn_families import sink
    return sink

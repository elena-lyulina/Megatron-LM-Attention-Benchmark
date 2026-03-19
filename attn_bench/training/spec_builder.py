"""
Registry for attention specs.

Adding a new kernel:
  1. Implement the kernel in kernels/<mechanism>/<impl>.py
  2. Add one entry to _REGISTRY — reuse the existing SpecProvider for that
     mechanism if the module and projection structure are the same, or add a
     new SpecProvider in kernels/<mechanism>/specs.py if they differ
     (e.g. gated attention needs a different module and submodules dataclass).

Usage:
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    spec = make_spec("full", "flash", backend=TESpecProvider())
    # spec is a ModuleSpec, embed it directly in TransformerLayerSubmodules:
    TransformerLayerSubmodules(self_attention=spec, ...)
"""

from megatron.core.models.backends import BackendSpecProvider
from megatron.core.transformer.spec_utils import ModuleSpec

from attn_bench.kernels.full.flash import FlashFullAttention
from attn_bench.kernels.full.specs import FullSpecProvider

# Registry: (mechanism, impl) -> (SpecProvider class, kernel class)
# SpecProvider controls the module class, submodules dataclass, and projection
# layers. Multiple impls of the same mechanism share the same SpecProvider.
_REGISTRY: dict[tuple[str, str], tuple[type, type]] = {
    ("full", "flash"): (FullSpecProvider, FlashFullAttention),
    # ("full", "sdpa"):            (FullSpecProvider, SDPAFullAttention),
    # ("full", "triton"):          (FullSpecProvider, TritonFullAttention),
    # ("sliding_window", "flash"): (SlidingWindowSpecProvider, FlashSWAttention),
    # ("gated", "flash"):          (GatedSpecProvider, FlashGatedAttention),
}


def make_spec(
    mechanism: str,
    impl: str,
    backend: BackendSpecProvider,
    qk_layernorm: bool = False,
) -> ModuleSpec:
    """
    Build a ModuleSpec for the given (mechanism, impl) pair.

    The SpecProvider controls which attention module and submodules dataclass
    are used, for example, SelfAttention + SelfAttentionSubmodules for standard mechanisms,
    a custom module + submodules for structurally different ones (gated, linear).

    Args:
        mechanism:    attention mechanism, e.g. "full", "sliding_window"
        impl:         kernel implementation, e.g. "flash", "sdpa"
        backend:      base BackendSpecProvider for linear layers (TESpecProvider
                      or LocalSpecProvider) — controls everything except core_attention
        qk_layernorm: whether to apply per-head QK layer norms

    Returns:
        ModuleSpec describing the attention block and ready to use in TransformerLayerSubmodules.
    """
    key = (mechanism, impl)
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown (mechanism, impl)=({mechanism!r}, {impl!r}). "
            f"Available: {sorted(_REGISTRY)}"
        )

    provider_cls, kernel_cls = _REGISTRY[key]
    provider = provider_cls(kernel_cls, base=backend)
    return provider.attention_spec(qk_layernorm=qk_layernorm)
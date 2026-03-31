"""
Registry mapping (mechanism, impl) → (SpecProvider, kernel_cls, supported_kwargs).

To add a new attention kernel:
  1. Implement the kernel in kernels/<mechanism>/<impl>.py
  2. Reuse the existing SpecProvider for that mechanism if the module and projection structure are the same,
     or add a new SpecProvider in kernels/<mechanism>/specs.py if they differ
  2. Register it in ATTN_REGISTRY
  3. Declare any mechanism-specific kwargs in the registry entry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attn_bench.kernels.full.flash import FlashFullAttention
from attn_bench.kernels.full.specs import FullSpecProvider
from attn_bench.kernels.full.torch_native import TorchFullAttention
from attn_bench.kernels.gated.flash import GatedFlashAttention
from attn_bench.kernels.gated.specs import GatedSpecProvider
from attn_bench.kernels.gated.torch_native import GatedTorchAttention
from attn_bench.kernels.sink.specs import SinkSpecProvider
from attn_bench.kernels.sink.te import SinkTEAttention
from attn_bench.kernels.sink.torch_native import SinkTorchAttention


@dataclass
class AttnKwargSchema:
    """Schema for a single attention-mechanism-specific kwarg."""
    type: type
    default: Any
    help: str

# Registering additional attn arguments here for easier validation and overview.
# Different implementations of the same mechanism could have different parameters if the parameters are implementation-specific
# or share the same parameters if they are mechanism-specific (as in SinkAttention)
# TODO: if too much duplicate code due to repeating params across the same mechanism, move to SpecProvider?
ATTN_REGISTRY: dict[tuple[str, str], tuple[type, type, dict[str, AttnKwargSchema]]] = {
    ### FULL ATTENTION  ###
    ("full",  "flash"): (FullSpecProvider,  FlashFullAttention,  {}),
    ("full",  "torch"): (FullSpecProvider,  TorchFullAttention,  {}),
    ### GATED ATTENTION ###
    ("gated", "flash"): (GatedSpecProvider, GatedFlashAttention, {}),
    ("gated", "torch"): (GatedSpecProvider, GatedTorchAttention, {}),
    ### SINK ATTENTION ###
    # init_sink_zero=True: inits sinks with zeros (for reproducible comparison with sink/te).
    # init_sink_zero=False: inits sinks from N(0, 0.023), matching TE's get_default_init_method().
    #   However, TE's implementation uses its own rng tracker, so setting the same distribution won't match the TE's init values and could result in a different loss curve.
    #   Therefore, use init_sink_zero=True in both implementations (TE and Native) for a correctness run
    ("sink",  "torch"): (SinkSpecProvider,  SinkTorchAttention,  {
        "init_sink_zero": AttnKwargSchema(type=bool, default=False, help="initialise sink parameter to zeros instead of N(0, 0.023)"),
    }),
    ("sink",  "te"):    (SinkSpecProvider,  SinkTEAttention,     {
        "init_sink_zero": AttnKwargSchema(type=bool, default=False, help="initialise sink parameter to zeros instead of N(0, 0.023)"),
    }),
}


def parse_attn_kwargs(raw: list[str] | None) -> dict[str, str]:
    # convert ['key=value', ...] from CLI into {'key': 'value', ...}, '=' must separate the key and values
    if not raw:
        return {}
    result = {}
    for item in raw:
        if '=' not in item:
            raise ValueError(f"--attn-kwargs entries must be key=value, got: {item!r}")
        k, v = item.split('=', 1)
        result[k.strip()] = v.strip()
    return result


def validate_attn_kwargs(attn: str, impl: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    # Validate kwargs against the registry schema in ATTN_REGISTRY for (attn, impl), accepts both raw strings (from CLI) and already-typed values (from internal calls).
    # - returns a fully typed dict with defaults filled in for missing keys.
    # - raises ValueError for unknown (attn, impl), unknown keys, or type mismatches.

    key = (attn, impl)
    if key not in ATTN_REGISTRY:
        raise ValueError(
            f"Unknown (attn, impl)=({attn!r}, {impl!r}). Available: {sorted(ATTN_REGISTRY)}"
        )

    _, _, supported_kwargs = ATTN_REGISTRY[key]

    unknown_kwargs = set(kwargs) - set(supported_kwargs)
    if unknown_kwargs:
        raise ValueError(
            f"Unknown kwargs for ({attn!r}, {impl!r}): {unknown_kwargs}.\n"
            f"  Supported kwargs for this implementation: {set(supported_kwargs) or 'none'}.\n"
        )

    return {
        name: (cast_kwarg(name, kwargs[name], schema.type) if name in kwargs else schema.default)
        for name, schema in supported_kwargs.items()
    }


def cast_kwarg(name: str, value: Any, typ: type) -> Any:
    # Casts the kwarg value to its type

    # if already cast, return it (for when the args come not from CLI but we're still doing a validation)
    if isinstance(value, typ):
        return value

    if typ == bool:
        # bool("false") is always true since it's a non-empty string, checking the value explicitly
        if isinstance(value, str):
            if value.lower() == 'true':
                return True
            if value.lower() == 'false':
                return False
        raise ValueError(f"Cannot parse {name}={value!r} as bool. Use true/false (any case).")

    try:
        return typ(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot parse {name}={value!r} as {typ.__name__}: {e}")

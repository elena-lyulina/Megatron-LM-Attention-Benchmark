import functools

from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.backends import BackendSpecProvider

from attn_bench.kernels.full.specs import FullSpecProvider


class SinkSpecProvider(FullSpecProvider):
    def __init__(self, kernel_cls: type, base: BackendSpecProvider, **kernel_kwargs):
        # TEDotProductAttention subclasses already implement Megatron's CoreAttention interface
        # and must not be wrapped by make_megatron_wrapper (which expects a pure kernel signature).
        # Use functools.partial to bind any kernel_kwargs since Megatron instantiates them directly.
        if issubclass(kernel_cls, TEDotProductAttention):
            self._kernel_wrapper_cls = functools.partial(kernel_cls, **kernel_kwargs) if kernel_kwargs else kernel_cls
            self._base = base
        else:
            super().__init__(kernel_cls, base, **kernel_kwargs)
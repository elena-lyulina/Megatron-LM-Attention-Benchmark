from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.models.backends import BackendSpecProvider

from attn_bench.kernels.full.specs import FullSpecProvider
from attn_bench.kernels.megatron_wrapper import make_megatron_wrapper


class SinkSpecProvider(FullSpecProvider):
    def __init__(self, kernel_cls: type, base: BackendSpecProvider):
        # TEDotProductAttention subclasses already implement Megatron's CoreAttention interface
        # and must not be wrapped by make_megatron_wrapper (which expects a pure kernel signature).
        if issubclass(kernel_cls, TEDotProductAttention):
            self._kernel_wrapper_cls = kernel_cls
            self._base = base
        else:
            super().__init__(kernel_cls, base)
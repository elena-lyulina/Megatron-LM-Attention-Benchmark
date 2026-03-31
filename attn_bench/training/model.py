"""
GPT model builder with swappable attention.

Constructs a standard Megatron GPTModel but replaces the self_attention slot
in the transformer layer spec with our custom kernel, leaving all other
components (MLPs, layer norms, embeddings, parallelism) untouched.
"""
from __future__ import annotations

from typing import Any, Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from attn_bench.kernels.attn_registry import ATTN_REGISTRY


def build_attn_module_spec(
    attn: str,
    impl: str,
    backend: BackendSpecProvider,
    qk_layernorm: bool = False,
    attn_kwargs: dict[str, Any] | None = None,
) -> ModuleSpec:
    """
    Build a ModuleSpec for the given (attn, impl) pair.

    Args:
    - attn: attention mechanism, e.g. "full", "sink"
    - impl: kernel implementation, e.g. "flash", "torch", "te"
    - backend: base BackendSpecProvider for linear layers (TESpecProvider or LocalSpecProvider)
    - qk_layernorm: whether to apply per-head QK layer norms
    - attn_kwargs: mechanism-specific kwargs, already validated via validate_attn_kwargs

    Returns:
        ModuleSpec describing the attention block and ready to use in TransformerLayerSubmodules.
    """
    if attn_kwargs is None:
        attn_kwargs = {}

    provider_cls, kernel_cls, _ = ATTN_REGISTRY[(attn, impl)]
    provider = provider_cls(kernel_cls, base=backend, **attn_kwargs)
    return provider.attention_spec(qk_layernorm=qk_layernorm)


def build_model(
    args, # Megatron argparse namespace (from get_args())
    config: TransformerConfig, # TransformerConfig built from args
    attn: str, # attention type, e.g. "full"
    impl: str, # kernel implementation, e.g. "flash"
    attn_kwargs: dict | None = None, # mechanism-specific kwargs, validated against registry
    pre_process: bool = True, # whether this rank handles embeddings (pipeline parallel)
    post_process: bool = True, # whether this rank handles LM head (pipeline parallel)
    vp_stage: Optional[int] = None, # virtual pipeline stage index
    pg_collection: Optional[ProcessGroupCollection] = None, #process group collection
) -> GPTModel:
    # Start from the standard TE layer spec — same as a normal Megatron run.
    # We use the minimal set of config flags relevant to our benchmark so the spec stays simple.
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=getattr(config, 'qk_layernorm', False),
        multi_latent_attention=False,
        qk_l2_norm=getattr(config, 'qk_l2_norm', False),
    )

    # Replace only self_attention with our custom kernel.
    # All other slots (input_layernorm, mlp, mlp_bda, etc.) stay as-is.
    layer_spec.submodules.self_attention = build_attn_module_spec(
        attn=attn,
        impl=impl,
        backend=TESpecProvider(),
        qk_layernorm=getattr(config, 'qk_layernorm', False),
        attn_kwargs=attn_kwargs,
    )

    return GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )
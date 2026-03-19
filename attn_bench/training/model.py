"""
GPT model builder with swappable attention.

Constructs a standard Megatron GPTModel but replaces the self_attention slot
in the transformer layer spec with our custom kernel, leaving all other
components (MLPs, layer norms, embeddings, parallelism) untouched.
"""

from typing import Optional

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig

from attn_bench.training.spec_builder import make_spec


def build_model(
    args,
    config: TransformerConfig,
    mechanism: str,
    impl: str,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: Optional[int] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> GPTModel:
    """
    Build a GPTModel with a custom attention kernel.

    Starts from the standard Transformer Engine layer spec (same as a normal
    Megatron run with --transformer-impl transformer_engine) and replaces only
    the self_attention ModuleSpec with our custom kernel.

    Args:
        args:       Megatron argparse namespace (from get_args())
        config:     TransformerConfig built from args
        mechanism:  attention mechanism, e.g. "full", "sliding_window"
        impl:       kernel implementation, e.g. "flash", "sdpa"
        pre_process:  whether this rank handles embeddings (pipeline parallel)
        post_process: whether this rank handles LM head (pipeline parallel)
        vp_stage:   virtual pipeline stage index, or None
        pg_collection: process group collection, or None to use global state
    """
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
    layer_spec.submodules.self_attention = make_spec(
        mechanism=mechanism,
        impl=impl,
        backend=TESpecProvider(),
        qk_layernorm=getattr(config, 'qk_layernorm', False),
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
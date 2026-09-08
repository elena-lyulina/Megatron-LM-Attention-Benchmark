"""Config + state-dict conversion for Gemma-3-style hybrid checkpoints
(--window-size + --window-attn-skip-freq): sliding window on most layers, full attention
interleaved every SKIP_FREQ-th.

Mask-only change like swa.py, so no custom architecture -- but Mistral's sliding_window is
uniform across layers, which would drop the full-attention ones. Targets MinistralForCausalLM:
identical Llama module names, plus config.layer_types to pick the mask per layer.
"""

from typing import Any, List, Union

from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig

# build_state_dict imported (not defined here) so convert_megatron_to_hf.py can call
# attn_family.build_state_dict(...) uniformly across families -- gemma's is identical to full's.
from attn_bench.checkpoint_conversion.attn_families.full import (  # noqa: F401
    ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, build_state_dict)


def compute_layer_types(window_attn_skip_freq: Union[int, List[int]], num_hidden_layers: int) -> List[str]:
    """Mirrors Megatron's is_layer_window_attention (megatron/core/transformer/utils.py).

    freq as int N: full attention where layer_number % N == 0 (freq=6 over 16 layers -> full
    at layers 6 and 12). freq as a list: Megatron's own 1=SWA / 0=full convention, used as-is.
    Megatron's layer_number is 1-INDEXED, hence (i + 1) -- off by one here silently shifts
    which layers get full attention.
    """
    if isinstance(window_attn_skip_freq, int):
        pattern = [0 if ((i + 1) % window_attn_skip_freq == 0) else 1 for i in range(num_hidden_layers)]
    else:
        pattern = list(window_attn_skip_freq)
        assert len(pattern) == num_hidden_layers, (
            f"window_attn_skip_freq list length {len(pattern)} != num_hidden_layers {num_hidden_layers}"
        )
    return ["sliding_attention" if p else "full_attention" for p in pattern]


def build_config(args: Any) -> PretrainedConfig:
    """Build the HF config for a Gemma-3-style hybrid checkpoint. Same fields as full.py's
    build_config (see its docstring for the two args-naming fixes this fork needs), plus
    sliding_window from --window-size and layer_types from --window-attn-skip-freq."""
    return AutoConfig.for_model(
        model_type="ministral",
        architectures=["MinistralForCausalLM"],
        attention_dropout=args.attention_dropout,
        bos_token_id=128000,
        eos_token_id=128001,
        head_dim=int(args.hidden_size / args.num_attention_heads),
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        # Without this MinistralConfig defaults every layer to sliding -- i.e. plain swa.
        layer_types=compute_layer_types(args.window_attn_skip_freq, args.num_layers),
        max_position_embeddings=131072,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_query_groups,
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see full.py's module docstring.
        rms_norm_eps=args.layernorm_epsilon,
        # rope_scaling, not rope_parameters: transformers 5.x maps the old name onto the new
        # one, so this spelling works on 4.x and 5.x alike. Same as full/swa.
        rope_scaling={
            "factor": args.rope_scaling_factor,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "rope_type": "llama3"
        },
        rope_theta=args.rotary_base,
        # +1 as in swa.py: flash-attn's window is EXCLUSIVE of self (1024 -> 1025 positions),
        # HF's sliding_window is INCLUSIVE. Without it the converted window is one token
        # narrower than trained. Sliding layers only -- HF nulls it on the full-attention ones.
        sliding_window=args.window_size[0] + 1,
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        torch_dtype=args.params_dtype,
        use_cache=True,
        vocab_size=args.padded_vocab_size
    )


def build_model(config: PretrainedConfig) -> AutoModelForCausalLM:
    """Construct an uninitialized model from config -- standard registered architecture,
    no auto_map/trust_remote_code needed (unlike custom families e.g. sink/gated)."""
    return AutoModelForCausalLM.from_config(config)

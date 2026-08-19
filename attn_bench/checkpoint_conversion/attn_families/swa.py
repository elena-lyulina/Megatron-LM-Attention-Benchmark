"""Config + state-dict conversion for sliding-window-attention (--window-size) checkpoints.

Sliding-window attention changes only the causal mask, not the weights -- so unlike sink/gated
this doesn't need a custom architecture at all. Targets HF's built-in MistralForCausalLM: its
module names (q_proj/k_proj/v_proj/o_proj, gate_proj/up_proj/down_proj, RMSNorm, GQA) are
identical to Llama's, and it already wires config.sliding_window into transformers' generic
masking utils (create_sliding_window_causal_mask), uniform across all layers -- exactly the
--window-size behavior this project trains (see window_size=(window, 0) in
megatron/core/extensions/transformer_engine.py), not Gemma2-style alternating local/global.
Everything else (build_state_dict) is reused unchanged from full.py's Llama conversion.
"""

from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig

# build_state_dict imported (not defined here) so convert_megatron_to_hf.py can call
# attn_family.build_state_dict(...) uniformly across families -- swa's is identical to full's.
from attn_bench.checkpoint_conversion.attn_families.full import (  # noqa: F401
    ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS, build_state_dict)


def build_config(args: Any) -> MistralConfig:
    """Build the HF config for a sliding-window-attention checkpoint. Same fields as full.py's
    build_config (see its docstring for the two args-naming fixes this fork needs), plus
    sliding_window from --window-size."""
    return AutoConfig.for_model(
        model_type="mistral",
        architectures=["MistralForCausalLM"],
        attention_dropout=args.attention_dropout,
        bos_token_id=128000,
        eos_token_id=128001,
        head_dim=int(args.hidden_size / args.num_attention_heads),
        hidden_act="silu",
        hidden_size=args.hidden_size,
        initializer_range=0.01,
        intermediate_size=args.ffn_hidden_size,
        max_position_embeddings=131072,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_layers,
        num_key_value_heads=args.num_query_groups,
        # --norm-epsilon (still the CLI flag name) sets args.layernorm_epsilon in this fork,
        # not args.norm_epsilon -- see full.py's module docstring.
        rms_norm_eps=args.layernorm_epsilon,
        rope_scaling={
            "factor": args.rope_scaling_factor,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "rope_type": "llama3"
        },
        rope_theta=args.rotary_base,
        # args.window_size is a (window, 0) tuple -- see megatron/core/transformer/
        # transformer_config.py; the 0 is the (unused, right-side) part of a causal window.
        sliding_window=args.window_size[0],
        tie_word_embeddings=not args.untie_embeddings_and_output_weights,
        torch_dtype=args.params_dtype,
        use_cache=True,
        vocab_size=args.padded_vocab_size
    )


def build_model(config: MistralConfig) -> AutoModelForCausalLM:
    """Construct an uninitialized model from config -- standard registered architecture,
    no auto_map/trust_remote_code needed (unlike custom families e.g. sink/gated)."""
    return AutoModelForCausalLM.from_config(config)

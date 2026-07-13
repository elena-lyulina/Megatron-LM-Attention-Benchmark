"""Shared helpers for the Megatron-native inference scripts (no metric dependencies).

Kept free of verbatim_eval/PDM imports so scripts that only need the checkpoint loader
don't pull in the Rouge/LCS stack.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch

BOS_TOKEN_ID = 128000  # Llama-3 beginning-of-sequence token


def load_megatron_model(ckpt_dir: str, tokenizer_path: str, extra_megatron_args: list | None = None,
                        tensor_parallel: int = 1):
    """Load model from a torch_dist checkpoint using --use-checkpoint-args.

    Checkpoint TP shards are resharded transparently by DCP to tensor_parallel (no
    pre-conversion needed), so tensor_parallel can differ from the training TP. Sharding the
    heads across ranks cuts per-rank attention memory, which lets the unfused-attention
    models run at longer sequence lengths. Architecture flags are read from the checkpoint;
    extra_megatron_args allows passing boolean store_true flags (e.g. --attention-output-gate)
    that --use-checkpoint-args may not restore correctly.
    """
    from gpt_builders import gpt_builder
    from megatron.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from model_provider import model_provider

    saved_argv = sys.argv[:]
    sys.argv = [
        'megatron_inference_sparse',
        '--use-checkpoint-args',
        '--tensor-model-parallel-size', str(tensor_parallel),
        '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1',
        '--micro-batch-size', '1',
        '--global-batch-size', '4',
        '--train-iters', '1',
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--tokenizer-model', tokenizer_path,
        '--load', ckpt_dir,
        '--no-load-optim',
        '--no-load-rng',
        '--ckpt-format', 'torch_dist',
        '--dist-ckpt-strictness', 'assume_ok_unexpected',
        '--finetune',
        '--bf16',
        '--transformer-impl', 'transformer_engine',
        '--main-grads-dtype', 'fp32',
        *(extra_megatron_args or []),
    ]
    try:
        from megatron.training.arguments import parse_and_validate_args

        # reads arguments directly and exclusively through sys.argv -- so we're swapping them beforehand.
        # PR #4225 moved arg parsing out of initialize_megatron; launch scripts must parse + set globals first.
        parse_and_validate_args()
        initialize_megatron()
        model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)
        model = model[0]
        model.eval()
        return model
    finally:
        sys.argv = saved_argv


def find_rep_paths(data_folder: Path, repetitions: set) -> list:
    return sorted(
        (p for p in data_folder.glob("rep_[0-9]*_token.jsonl")
         if int(p.stem.split("_")[1]) in repetitions and "_swaps_" not in p.name),
        key=lambda p: int(p.stem.split("_")[1]),
    )


@torch.no_grad()
def greedy_generate(model, prompt_ids: torch.Tensor, suffix_length: int,
                    prefill_callback=None, decode_step_callback=None):
    """Greedy generation with StaticInferenceContext KV cache.

    prompt_ids: [B, prompt_len]   — prefix tokens (BOS already included as token 0)
    prefill_callback: optional callable() invoked right after the prefill forward
                      (before any decode forward overwrites the attention buffers).
    decode_step_callback: optional callable(t: int) called after each decode step with the
                   0-indexed step number (t=0 for first decode step, etc.).
                   n_steps total = suffix_length - 1; prefill is not a step.
    Returns:    [B, suffix_length] — generated tokens
    """
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.utils import InferenceMode

    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    max_seq_len = prompt_len + suffix_length

    ctx = StaticInferenceContext(max_batch_size=B, max_sequence_length=max_seq_len)
    ctx.reset()
    ctx.enable_prefill_mode()

    # The model only slices the logits down to the last prompt token when InferenceMode
    # is active (upstream gates this on InferenceMode.is_active(), not on inference_context
    # being set). Without it the prefill returns logits for every prompt position and we
    # would pick position 0 below, generating from the wrong token and losing all recall.
    with InferenceMode.active():
        # Prefill: process all prompt tokens, get logit for the first generated token
        pos = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
        logits = model(prompt_ids, pos, attention_mask=None, inference_context=ctx,
                       runtime_gather_output=True)
        # logits: [B, 1, V]  (materialize_only_last_token_logits gives the last-token slice)
        ctx.sequence_len_offset = prompt_len
        ctx.enable_decode_mode()

        if prefill_callback is not None:
            prefill_callback()

        next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)  # [B, 1]
        generated = [next_token]

        # Decode: one new token per step, KV of previous tokens served from cache
        for step_t in range(suffix_length - 1):
            pos = torch.full((B, 1), ctx.sequence_len_offset, dtype=torch.long, device=device)
            logits = model(next_token, pos, attention_mask=None, inference_context=ctx,
                           runtime_gather_output=True)
            ctx.sequence_len_offset += 1
            next_token = logits[:, 0, :].argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if decode_step_callback is not None:
                decode_step_callback(step_t)

    return torch.cat(generated, dim=1)  # [B, suffix_length]
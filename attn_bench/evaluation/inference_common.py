"""Shared helpers for the Megatron-native inference scripts (no metric dependencies).

Kept free of verbatim_eval/PDM imports so scripts that only need the checkpoint loader
don't pull in the Rouge/LCS stack.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import torch

BOS_TOKEN_ID = 128000  # Llama-3 beginning-of-sequence token


def load_megatron_model(ckpt_dir: str, tokenizer_path: str, extra_megatron_args: list | None = None,
                        tensor_parallel: int = 1):
    """Load model from a torch_dist checkpoint using --use-checkpoint-args.

    Checkpoint TP shards are resharded transparently by DCP to tensor_parallel, so it can differ
    from the training TP. extra_megatron_args re-passes flags --use-checkpoint-args doesn't
    restore (e.g. --attention-output-gate, --use-rope-scaling/--rope-scaling-factor) -- sourced
    per model tag from MEGATRON_EXTRA in llama_checkpoints.sh.
    """
    from gpt_builders import gpt_builder
    from megatron.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from model_provider import model_provider

    saved_argv = sys.argv[:]
    sys.argv = [
        'megatron_inference',
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


def discover_all_offset_prefix_suffix_dirs(experiment_path: Path) -> list:
    """Every existing offset_O_prefix_P_suffix_S dir under experiment_path/inference,
    unfiltered. Callers that already know a specific (offset, prefix_length) should
    filter this down rather than re-scanning the directory themselves."""
    inference_root = experiment_path / "inference"
    if not inference_root.exists():
        return []
    return sorted(d for d in inference_root.iterdir() if d.is_dir() and d.name.startswith("offset_"))


def sample_idx_per_rank(world_size: int, dataset_len: int) -> list[list[int]]:
    """Reconstruct which original dataset indices DistributedSampler(shuffle=False) gave
    each rank, matching torch's own algorithm: pad range(dataset_len) up to a multiple of
    world_size by wrapping from the start, then take every world_size-th index starting
    at each rank. Returns one list of original indices per rank, in rank order -- reading
    rank{r}.jsonl's records in file order and zipping them with per_rank[r] recovers each
    record's original sample_idx, with no other information needed.
    """
    total_size = ((dataset_len + world_size - 1) // world_size) * world_size # a multiple of world_size
    padding = total_size - dataset_len
    # padding repeats indices from the beginning
    indices = list(range(dataset_len)) + list(range(dataset_len))[:padding]
    return [indices[r::world_size] for r in range(world_size)]


def load_records_by_sample_idx(rep_dir: Path, dataset_len: int | None = None) -> dict:
    """Read every rank*.jsonl under rep_dir, keyed by sample_idx. Recovers sample_idx on
    the fly for records that predate it (via sample_idx_per_rank), given the dataset
    length that produced them -- required by the caller in that case, since world_size
    alone isn't enough to know the padding.
    """
    rank_files = sorted(rep_dir.glob("rank*.jsonl"))
    records = {}
    needs_recovery = []
    for rank, rank_file in enumerate(rank_files):
        with open(rank_file) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        # position is the record's true index in rank_file, not a count of only the
        # untagged ones -- a mixed file would otherwise drift off the correct index.
        for position, rec in enumerate(lines):
            if "sample_idx" in rec:
                records[rec["sample_idx"]] = rec
            else:
                needs_recovery.append((rank, position, rec))
    if needs_recovery:
        if dataset_len is None:
            raise ValueError(
                f"{rep_dir}: records without sample_idx found, but no dataset_len given "
                "to recover it -- pass the source rep_*_token.jsonl's line count."
            )
        per_rank = sample_idx_per_rank(len(rank_files), dataset_len)
        for rank, position, rec in needs_recovery:
            records[per_rank[rank][position]] = rec
    return records


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
"""
Megatron-native sparse Gutenberg inference for memorization measurement.

Loads the model directly from a torch_dist checkpoint (no HF conversion),
supporting all attention variants (full, gated, sink, off-by-one).
The HF conversion doesn't support custom attentions out of the box,
hence we need to do it this way.

Writes one rank{N}.jsonl per GPU in PDM-compatible format, ready for
verbatim_eval/main.py to consume.

Usage (via torchrun):
    torchrun --nproc_per_node=4 attn_bench/evaluation/megatron_inference_sparse.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --offset 0 \
        --prefix-length 500 \
        --suffix-length 500 \
        --batch-size 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from verbatim_eval.LCS import find_longest_common_substrings
from verbatim_eval.my_rouge import _compute_dp_matrix_2d, compute_rouge_l_2d



### MODEL ###

def load_megatron_model(ckpt_dir: str, tokenizer_path: str, extra_megatron_args: list | None = None):
    """Load model from a torch_dist checkpoint using --use-checkpoint-args.

    TP=2 shards are merged transparently by DCP resharding (no pre-conversion needed).
    Architecture flags are read from the checkpoint; extra_megatron_args allows passing
    boolean store_true flags (e.g. --attention-output-gate) that --use-checkpoint-args
    may not restore correctly.
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
        '--tensor-model-parallel-size', '1',
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
        # reads arguments directly and exclusively through sys.argv -- so we're swapping them beforehand
        initialize_megatron()
        model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)
        model = model[0]
        model.eval()
        return model
    finally:
        sys.argv = saved_argv


def patch_sink_scale(model, sink_scale: float) -> list:
    """Scale the virtual sink weight at inference: offset_new = offset_trained + log(sink_scale).

    Equivalently: exp(offset_new) = sink_scale × exp(offset_trained).
    sink_scale=1 is identity; >1 strengthens the sink, <1 weakens it.
    Supports off-by-one (trained offset=0, so offset_new=log(sink_scale)) and learnable.
    Raises for vanilla attention (no softmax_offset). Returns original per-layer
    per-head values as list of lists for metadata.
    """
    from megatron.core.transformer.dot_product_attention import DotProductAttention as MegatronDPA
    try:
        import transformer_engine.pytorch as te
        TE_DPA = te.DotProductAttention
    except ImportError:
        TE_DPA = None

    if sink_scale < 0:
        raise ValueError(f"sink_scale must be >= 0, got {sink_scale}")
    log_scale = math.log(sink_scale) if sink_scale > 0 else float("-inf")
    originals = []
    count = 0
    for module in model.modules():
        if isinstance(module, MegatronDPA) and module.softmax_offset is not None:
            assert module.config.softmax_type in ("off-by-one", "learnable"), (
                f"patch_sink_scale only supports off-by-one and learnable attention, "
                f"got softmax_type='{module.config.softmax_type}'"
            )
            originals.append(module.softmax_offset.detach().cpu().tolist())
            module.softmax_offset.data.add_(log_scale)
            count += 1
        elif TE_DPA is not None and isinstance(module, TE_DPA) and module.softmax_offset is not None:
            assert module.softmax_type in ("off-by-one", "learnable"), (
                f"patch_sink_scale only supports off-by-one and learnable attention, "
                f"got softmax_type='{module.softmax_type}'"
            )
            originals.append(module.softmax_offset.detach().cpu().tolist())
            module.softmax_offset.data.add_(log_scale)
            count += 1

    if count == 0:
        raise RuntimeError(
            "patch_sink_scale: no patchable attention layers found "
            "(neither MegatronDPA nor TE DPA with softmax_offset != None)."
        )
    print(f"Patched softmax_offset += log({sink_scale}) = {log_scale:.4f} in {count} attention layers")
    return originals


### INFERENCE ###

@torch.no_grad()
def compute_nll(model, input_ids: torch.Tensor, suffix_length: int):
    """Single forward pass; return NLL stats for the last suffix_length tokens.

    input_ids: [B, S]  — BOS + prefix + suffix (or BOS + prefix + generated)
    Returns: (mean [B], std [B], ppl [B])
    """
    B, S = input_ids.shape
    device = input_ids.device
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    position_ids = torch.arange(S - 1, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    logits = model(inputs, position_ids, attention_mask=None)  # [B, S-1, V]

    token_nlls = -F.log_softmax(logits.float(), dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    del logits

    suffix_nlls = token_nlls[:, -suffix_length:]
    mean = suffix_nlls.mean(dim=1)
    std = suffix_nlls.std(dim=1)
    return mean, std, mean.exp()


@torch.no_grad()
def greedy_generate(model, prompt_ids: torch.Tensor, suffix_length: int,
                    step_callback=None):
    """Greedy generation with StaticInferenceContext KV cache.

    prompt_ids: [B, prompt_len]   — prefix tokens (BOS already included as token 0)
    step_callback: optional callable(t: int) called after each decode step with the
                   0-indexed step number (t=0 for first decode step, etc.).
                   n_steps total = suffix_length - 1; prefill is not a step.
    Returns:    [B, suffix_length] — generated tokens
    """
    from megatron.core.inference.contexts import StaticInferenceContext

    B, prompt_len = prompt_ids.shape
    device = prompt_ids.device
    max_seq_len = prompt_len + suffix_length

    ctx = StaticInferenceContext(max_batch_size=B, max_sequence_length=max_seq_len)
    ctx.reset()
    ctx.enable_prefill_mode()

    # Prefill: process all prompt tokens, get logit for the first generated token
    pos = torch.arange(prompt_len, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)
    logits = model(prompt_ids, pos, attention_mask=None, inference_context=ctx,
                   runtime_gather_output=True)
    # logits: [B, 1, V]  (StaticInferenceContext sets materialize_only_last_token_logits=True)
    ctx.sequence_len_offset = prompt_len
    ctx.enable_decode_mode()

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
        if step_callback is not None:
            step_callback(step_t)

    return torch.cat(generated, dim=1)  # [B, suffix_length]


### METRICS ###

def text_metrics(true_seq: list, gen_seq: list) -> dict:
    import numpy as np
    dp = _compute_dp_matrix_2d(np.array(true_seq, dtype=np.int32), np.array(gen_seq, dtype=np.int32))
    rouge_l = compute_rouge_l_2d(dp)
    # type-token ration
    ttr_ref = len(set(true_seq)) / len(true_seq) if true_seq else 0.0
    ttr_gen = len(set(gen_seq)) / len(gen_seq) if gen_seq else 0.0
    return {"TTR_ref": ttr_ref, "TTR_gen": ttr_gen, "Rouge-L": rouge_l}


### MAIN LOOP ###

BOS_TOKEN_ID = 128000  # Llama-3 BOS; present at token 0 when offset=0, must be prepended when offset>0

def run_bucket(model, dataset, prefix_length, suffix_length, batch_size, inference_dir,
               rank, world_size, needs_bos: bool, capture=None):
    """Run inference for one repetition bucket.

    needs_bos: True when offset > 0, i.e. the excerpts don't start with BOS and we must prepend it.
    capture: AttentionCapture instance (or None to skip attention capture).
    """
    device = next(model.parameters()).device

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda b: b)

    inference_dir.mkdir(parents=True, exist_ok=True)

    sample_start = 0  # tracks first global sample index of each batch (within this rank's slice)

    with open(inference_dir / f"rank{rank}.jsonl", "w") as f:
        for batch in loader:
            torch.cuda.empty_cache()

            batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)  # [B, prefix+suffix]
            B = batch_tensor.shape[0]

            if needs_bos:
                # offset > 0: excerpt starts mid-document, prepend BOS so model sees proper start
                bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
                seq = torch.cat([bos, batch_tensor], dim=1)  # [B, 1+prefix+suffix]
                prompt_end = 1 + prefix_length
            else:
                # offset == 0: BOS is already token 0 of every excerpt
                seq = batch_tensor   # [B, prefix+suffix]
                prompt_end = prefix_length

            # Reference NLL on the gold suffix
            ref_mean, ref_std, ref_ppl = compute_nll(model, seq, suffix_length)

            # Greedy generation with KV cache
            prompt = seq[:, :prompt_end]                               # [B, prompt_end]

            if capture is not None:
                batch_slice = slice(sample_start, sample_start + B)
                is_sample_0 = (rank == 0 and sample_start == 0)
                step_cb = lambda t, _bs=batch_slice, _s0=is_sample_0: (
                    capture.collect_step(t, _bs, _s0)
                )
                generated = greedy_generate(model, prompt, suffix_length, step_callback=step_cb)
            else:
                generated = greedy_generate(model, prompt, suffix_length)

            # NLL on the generated suffix
            gen_full = torch.cat([prompt, generated], dim=1)
            gen_mean, gen_std, gen_ppl = compute_nll(model, gen_full, suffix_length)

            # Raw prefix/suffix from the original excerpt (for output, no BOS management)
            prefixes = batch_tensor[:, :prefix_length].cpu().tolist()
            true_suffixes = batch_tensor[:, prefix_length:].cpu().tolist()
            gen_suffixes = generated.cpu().tolist()

            lcs = find_longest_common_substrings(true_suffixes, gen_suffixes)
            lcs_norm = (lcs['max_length'].to_numpy() / suffix_length).tolist()

            ref_mean_l = ref_mean.tolist()
            ref_std_l = ref_std.tolist()
            ref_ppl_l = ref_ppl.tolist()
            gen_mean_l = gen_mean.tolist()
            gen_std_l = gen_std.tolist()
            gen_ppl_l = gen_ppl.tolist()

            for i in range(B):
                record = {
                    "prefix": prefixes[i],
                    "true_suffix": true_suffixes[i],
                    "generated_suffix": gen_suffixes[i],
                    "nll_mean": gen_mean_l[i],
                    "nll_std": gen_std_l[i],
                    "perplexity": gen_ppl_l[i],
                    "ref_nll_mean": ref_mean_l[i],
                    "ref_nll_std": ref_std_l[i],
                    "ref_perplexity": ref_ppl_l[i],
                    "lcs_norm": lcs_norm[i],
                    **text_metrics(true_suffixes[i], gen_suffixes[i]),
                }
                json.dump(record, f)
                f.write("\n")
                f.flush()

            sample_start += B
            del batch_tensor, seq, prompt, generated, gen_full
            del ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl
            torch.cuda.empty_cache()

    if capture is not None:
        capture.save(inference_dir, rank)
        capture.remove()

    dist.barrier()


### CLI HELPERS ###

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True, help="torch_dist checkpoint directory")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--experiment-path", required=True, help="Output root (MEM_DIR)")
    parser.add_argument("--data-folder", required=True, help="Directory of rep_*_token.jsonl files")
    parser.add_argument("--repetitions", required=True, help="Comma-separated, e.g. 0,1,2,4,8,16,32,64,128,256")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=500)
    parser.add_argument("--suffix-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap sequences per repetition bucket (for testing)")
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                        help="Extra Megatron args forwarded verbatim to initialize_megatron "
                             "(e.g. --megatron-extra-args --attention-output-gate)")
    parser.add_argument("--sink-scale", type=float, default=None,
                        help="Scale the virtual sink weight at inference: offset_new = offset_trained + log(sink_scale). "
                             "sink_scale=1 is identity, >1 strengthens the sink, <1 weakens it. "
                             "Supports off-by-one and learnable attention. "
                             "Original per-head values saved to sink_scale_metadata.json. "
                             "Appends _sscale{X} to experiment path.")
    parser.add_argument("--capture-attention", action="store_true",
                        help="Capture decode-time attention weights for visualization. "
                             "Writes attn_stats_rank{N}.npz and attn_matrix_exmpl_rank{N}.npz "
                             "alongside each rep_*_greedy/rank*.jsonl.")
    return parser.parse_args()


def find_rep_paths(data_folder: Path, repetitions: set) -> list:
    return sorted(
        (p for p in data_folder.glob("rep_[0-9]*_token.jsonl")
         if int(p.stem.split("_")[1]) in repetitions and "_swaps_" not in p.name),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def load_rep_bucket(path: Path, offset: int, prefix_length: int, suffix_length: int,
                    max_samples: int | None = None) -> list:
    dataset = []
    with open(path) as f:
        for line in f:
            if max_samples is not None and len(dataset) >= max_samples:
                break
            ids = json.loads(line)["input_ids"]
            excerpt = ids[offset: offset + prefix_length + suffix_length]
            assert len(excerpt) == prefix_length + suffix_length, (
                f"{path.name}: sequence too short ({len(ids)} tokens)"
            )
            dataset.append(excerpt)
    return dataset


def _make_capture(model, args, n_samples_on_rank, rank):
    """Instantiate and register an AttentionCapture for one bucket."""
    from attn_bench.evaluation.attn_capture import AttentionCapture

    cfg = model.config
    n_layers = cfg.num_layers
    n_heads  = cfg.num_attention_heads
    n_steps  = args.suffix_length - 1
    max_seq  = args.prefix_length + args.suffix_length
    is_gated = getattr(cfg, 'attention_output_gate', False)

    capture = AttentionCapture(
        n_samples=n_samples_on_rank,
        n_layers=n_layers,
        n_heads=n_heads,
        n_steps=n_steps,
        max_seq_len=max_seq,
        is_gated=is_gated,
        is_rank0=(rank == 0),
        prompt_len=args.prefix_length,
    )
    capture.register(model)
    return capture


def run_inference(model, args, rank, world_size):
    output_path = (
        Path(args.experiment_path)
        / "inference"
        / f"offset_{args.offset}_prefix_{args.prefix_length}_suffix_{args.suffix_length}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    needs_bos = args.offset > 0  # offset==0: BOS already at token 0; offset>0: must prepend

    for path in paths:
        rep = int(path.stem.split("_")[1])
        inference_dir = output_path / f"rep_{rep}_greedy"

        rank0_file = inference_dir / "rank0.jsonl"
        attn_done  = (inference_dir / f"attn_stats_rank{rank}.npz").exists()
        jsonl_done = rank0_file.exists() and rank0_file.stat().st_size > 0

        if jsonl_done and (not args.capture_attention or attn_done):
            if rank == 0:
                print(f"Skipping rep={rep} (already done)")
            continue

        if rank == 0:
            print(f"\nProcessing rep={rep}")

        dataset = load_rep_bucket(path, args.offset, args.prefix_length, args.suffix_length,
                                   max_samples=args.max_samples)

        if rank == 0:
            print(f"  {len(dataset)} sequences")

        # Build capture per-bucket (fresh state each repetition level)
        capture = None
        if args.capture_attention and not attn_done:
            sampler_size = math.ceil(len(dataset) / world_size)
            capture = _make_capture(model, args, sampler_size, rank)

        run_bucket(
            model, dataset,
            args.prefix_length, args.suffix_length,
            args.batch_size, inference_dir,
            rank, world_size,
            needs_bos=needs_bos,
            capture=capture,
        )

        if rank == 0:
            print(f"  Done rep={rep}")
        torch.cuda.empty_cache()

    if rank == 0:
        print(f"\nAll repetitions done. Results in: {output_path}")


def main():
    args = parse_args()

    if args.sink_scale is not None:
        args.experiment_path = args.experiment_path.rstrip('/') + f"_sscale{args.sink_scale:g}"

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)

    if args.sink_scale is not None:
        originals = patch_sink_scale(model, args.sink_scale)
        if dist.get_rank() == 0:
            meta_path = Path(args.experiment_path) / "sink_scale_metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump({"sink_scale": args.sink_scale, "original_softmax_offset": originals}, f, indent=2)
            print(f"Saved sink scale metadata to {meta_path}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    run_inference(model, args, rank, world_size)


if __name__ == "__main__":
    main()

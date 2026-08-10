"""
Megatron-native sparse Gutenberg inference for memorization measurement (Step 1).

Loads the model directly from a torch_dist checkpoint (no HF conversion -- HF doesn't
support the custom attention variants). GPU/model-dependent work only: greedy generation,
NLL, and p_z (Hayes et al. 2025). Text metrics live in compute_memorization_metrics.py.

Writes one rank{N}.jsonl per GPU under offset_O_prefix_P_suffix_S/rep_R_greedy/. Reuses
a sibling suffix' >= the request if one exists (nothing to compute), or extends from a
smaller suffix' via teacher-forced prefill instead of regenerating. Smaller suffix dirs
are kept, not deleted, as a check that extending reproduces a fresh run's tokens.

Usage (via torchrun):
    torchrun --nproc_per_node=4 attn_bench/evaluation/megatron_inference.py \
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
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from attn_bench.evaluation.inference_common import (
    BOS_TOKEN_ID, discover_all_offset_prefix_suffix_dirs, find_rep_paths,
    greedy_generate, load_megatron_model, load_records_by_sample_idx)

P_Z_TOP_K = 40
P_Z_TEMPERATURE = 1.0

### MODEL ###

def patch_sink_scale(model, sink_scale: float) -> list:
    """Scale the virtual sink weight at inference: offset_new = offset_trained + log(sink_scale).

    Equivalently: exp(offset_new) = sink_scale × exp(offset_trained).
    sink_scale=1 is identity; >1 strengthens the sink, <1 weakens it.
    Supports off-by-one (trained offset=0, so offset_new=log(sink_scale)) and learnable.
    Raises for vanilla attention (no softmax_offset). Returns original per-layer
    per-head values as list of lists for metadata.
    """
    from megatron.core.transformer.dot_product_attention import \
        DotProductAttention as MegatronDPA
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


### NLL / p_z ###

@torch.no_grad()
def compute_nll(model, input_ids: torch.Tensor, suffix_length: int):
    """One forward pass; per-position NLL for the last suffix_length tokens.

    Also returns the logits/labels that produced it, so callers (p_z_log_probs) can
    reuse the same forward pass instead of paying for another one.

    input_ids: [B, S]  — BOS + prefix + suffix (or BOS + prefix + generated)
    Returns: per_position_nll [B, suffix_length], suffix_logits [B, suffix_length, V],
             suffix_labels [B, suffix_length]
    """
    B, S = input_ids.shape
    device = input_ids.device
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    position_ids = torch.arange(S - 1, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    logits = model(inputs, position_ids, attention_mask=None)  # [B, S-1, V]

    # Only the suffix positions' NLL is used below -- computing log_softmax (and upcasting
    # to fp32) over the full prefix+suffix span wastes memory that grows with prefix length
    # for no benefit (this is what OOM'd at prefix>=5000, see job 3036124). Slice to the
    # last suffix_length positions first instead.
    suffix_logits = logits[:, -suffix_length:, :].float()
    suffix_labels = labels[:, -suffix_length:]
    del logits

    per_position_nll = -F.log_softmax(suffix_logits, dim=-1).gather(2, suffix_labels.unsqueeze(-1)).squeeze(-1)
    return per_position_nll, suffix_logits, suffix_labels


def nll_stats(per_position_nll: torch.Tensor):
    """mean/std/ppl per sample, for the flat top-level PDM-compatible fields."""
    mean = per_position_nll.mean(dim=1)
    std = per_position_nll.std(dim=1)
    return mean, std, mean.exp()


def p_z_log_probs(suffix_logits: torch.Tensor, suffix_labels: torch.Tensor,
                  top_k: int = P_Z_TOP_K, temperature: float = P_Z_TEMPERATURE) -> torch.Tensor:
    """Per-position log-probability of the true token under top-k/temperature sampling
    (Hayes et al. 2025's p_z). Computed from the same forward pass as NLL, no generation involved.
     -inf at positions where the true token falls outside the top-k set, since that scheme could never sample it there.
    """
    scaled = suffix_logits / temperature
    topk_vals, topk_idx = scaled.topk(top_k, dim=-1)
    topk_log_probs = F.log_softmax(topk_vals, dim=-1)

    matches = topk_idx == suffix_labels.unsqueeze(-1)  # [B, suffix_length, top_k]
    in_top_k = matches.any(dim=-1)
    log_prob = (topk_log_probs * matches).sum(dim=-1)
    return torch.where(in_top_k, log_prob, torch.full_like(log_prob, float("-inf")))


def compute_nll_pz_stats(model, true_full_sequence: torch.Tensor, gen_full_sequence: torch.Tensor,
                         suffix_length: int):
    """ref/gen NLL, p_z, and their mean/std/ppl summaries for one batch."""
    ref_nll, ref_logits, ref_labels = compute_nll(model, true_full_sequence, suffix_length)
    p_z = p_z_log_probs(ref_logits, ref_labels)
    del ref_logits, ref_labels
    gen_nll, gen_logits, gen_labels = compute_nll(model, gen_full_sequence, suffix_length)
    del gen_logits, gen_labels
    ref_mean, ref_std, ref_ppl = nll_stats(ref_nll)
    gen_mean, gen_std, gen_ppl = nll_stats(gen_nll)
    return ref_nll, gen_nll, p_z, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl


### CROSS-SUFFIX LOOKUP ###

def find_suffix_dirs(experiment_path: Path, offset: int, prefix_length: int) -> list:
    """All existing offset_O_prefix_P_suffix_* dirs for this (offset, prefix), as
    (suffix_length, path) pairs."""
    prefix_str = f"offset_{offset}_prefix_{prefix_length}_suffix_"
    found = []
    for d in discover_all_offset_prefix_suffix_dirs(experiment_path):
        if d.name.startswith(prefix_str):
            try:
                found.append((int(d.name[len(prefix_str):]), d))
            except ValueError:
                continue
    return found


def find_rep_source(experiment_path: Path, offset: int, prefix_length: int,
                    suffix_length: int, rep: int):
    """Among existing suffix dirs for (offset, prefix), find one whose rep_{rep}_greedy
    is complete: the smallest suffix' >= suffix_length if any (nothing to compute), else
    the largest suffix' < suffix_length (extend from here). None if nothing usable exists.

    Returns (suffix', rep_dir_path) or None.
    """
    usable = []
    for suffix_prime, d in find_suffix_dirs(experiment_path, offset, prefix_length):
        rank0 = d / f"rep_{rep}_greedy" / "rank0.jsonl"
        if rank0.exists() and rank0.stat().st_size > 0:
            usable.append((suffix_prime, d / f"rep_{rep}_greedy"))
    if not usable:
        return None
    at_least = [c for c in usable if c[0] >= suffix_length]
    if at_least:
        return min(at_least, key=lambda c: c[0])
    below = [c for c in usable if c[0] < suffix_length]
    return max(below, key=lambda c: c[0]) if below else None


### MAIN LOOP ###

def _capture_rouge_l(true_suffixes: list, gen_suffixes: list) -> list:
    """Per-sample Rouge-L, computed locally only to route attention maps into buckets
    for --capture-attention. Not persisted to the jsonl -- that's Step 2's job for
    everything else. Deferred import so the common (non-capture) path stays free of
    PDM/verbatim_eval deps.
    """
    import numpy as np
    from verbatim_eval.my_rouge import (_compute_dp_matrix_2d,
                                        compute_rouge_l_2d)

    scores = []
    for true_seq, gen_seq in zip(true_suffixes, gen_suffixes):
        dp = _compute_dp_matrix_2d(np.array(true_seq, dtype=np.int32), np.array(gen_seq, dtype=np.int32))
        scores.append(compute_rouge_l_2d(dp))
    return scores


def run_bucket(model, dataset, prefix_length, suffix_length, batch_size, inference_dir,
               rank, world_size, needs_bos: bool, capture=None,
               extend_records: dict | None = None, extend_from_suffix: int | None = None) -> float:
    """Run inference for one repetition bucket.

    dataset: list of (sample_idx, excerpt_tokens) -- sample_idx tagged before this
             DistributedSampler split, so it survives regardless of --batch-size/world
             size across separate job invocations.
    needs_bos: True when offset > 0, i.e. the excerpts don't start with BOS and we must prepend it.
    extend_records: {sample_idx: old_record} from a smaller suffix' run to extend from
                    (old_record must have "generated_suffix" of length extend_from_suffix),
                    or None to generate fresh.
    capture: shared AttentionCapture instance (or None). Always regenerates from scratch
             (extend_records is ignored when capture is set) -- the maps need the actual
             forward passes.

    Returns: total wall time spent in greedy_generate, isolated from everything else in
             this loop (checkpoint load happened before this function is ever called).
    """
    device = next(model.parameters()).device

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda b: b)

    inference_dir.mkdir(parents=True, exist_ok=True)

    use_extend = extend_records is not None and capture is None
    generation_time = 0.0

    with open(inference_dir / f"rank{rank}.jsonl", "w") as f:
        for batch in loader:
            torch.cuda.empty_cache()

            sample_indices = [item[0] for item in batch]
            excerpts = [item[1] for item in batch]
            batch_tensor = torch.tensor(excerpts, dtype=torch.long, device=device)  # [B, prefix+suffix]
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

            prompt = seq[:, :prompt_end]                               # [B, prompt_end]

            if use_extend:
                old_generated = torch.tensor(
                    [extend_records[idx]["generated_suffix"] for idx in sample_indices],
                    dtype=torch.long, device=device,
                )
                decode_prompt = torch.cat([prompt, old_generated], dim=1)
                new_steps = suffix_length - extend_from_suffix
                t0 = time.monotonic()
                new_tokens = greedy_generate(model, decode_prompt, new_steps)
                generation_time += time.monotonic() - t0
                generated = torch.cat([old_generated, new_tokens], dim=1)
            elif capture is not None:
                capture.begin_batch(B)
                t0 = time.monotonic()
                generated = greedy_generate(
                    model, prompt, suffix_length,
                    prefill_callback=capture.collect_prefill,
                    decode_step_callback=capture.collect_decode,
                )
                generation_time += time.monotonic() - t0
            else:
                t0 = time.monotonic()
                generated = greedy_generate(model, prompt, suffix_length)
                generation_time += time.monotonic() - t0

            gen_full = torch.cat([prompt, generated], dim=1)
            ref_nll, gen_nll, p_z, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl = \
                compute_nll_pz_stats(model, seq, gen_full, suffix_length)

            # Raw prefix/suffix from the original excerpt (for output, no BOS management)
            prefixes = batch_tensor[:, :prefix_length].cpu().tolist()
            true_suffixes = batch_tensor[:, prefix_length:].cpu().tolist()
            gen_suffixes = generated.cpu().tolist()

            ref_nll_l = ref_nll.cpu().tolist()
            gen_nll_l = gen_nll.cpu().tolist()
            p_z_l = p_z.cpu().tolist()
            ref_mean_l, ref_std_l, ref_ppl_l = ref_mean.tolist(), ref_std.tolist(), ref_ppl.tolist()
            gen_mean_l, gen_std_l, gen_ppl_l = gen_mean.tolist(), gen_std.tolist(), gen_ppl.tolist()

            for i in range(B):
                record = {
                    "sample_idx": sample_indices[i],
                    "prefix": prefixes[i],
                    "true_suffix": true_suffixes[i],
                    "generated_suffix": gen_suffixes[i],
                    "ref_nll": ref_nll_l[i],
                    "gen_nll": gen_nll_l[i],
                    "p_z_logprob": p_z_l[i],
                    "nll_mean": gen_mean_l[i],
                    "nll_std": gen_std_l[i],
                    "perplexity": gen_ppl_l[i],
                    "ref_nll_mean": ref_mean_l[i],
                    "ref_nll_std": ref_std_l[i],
                    "ref_perplexity": ref_ppl_l[i],
                }
                json.dump(record, f)
                f.write("\n")
                f.flush()

            if capture is not None:
                capture.flush_batch(_capture_rouge_l(true_suffixes, gen_suffixes))

            del batch_tensor, seq, prompt, generated, gen_full
            del ref_nll, gen_nll, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl
            torch.cuda.empty_cache()

    dist.barrier()
    return generation_time


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
    parser.add_argument("--container-env", default=None,
                        help="Container/environment name this run executed in (e.g. nemo_26.04_te2.15). "
                             "Recorded verbatim in run_metadata.json for provenance.")
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
                        help="Capture full causal attention maps (prefill + decode), averaged into "
                             "Rouge-L buckets across all repetition buckets. Writes "
                             "attn_scores_rouge_l_{NN-MM}_rank{N}.npz, norm_attn_rouge_l_{NN-MM}_rank{N}.npz "
                             "and (gated only) gating_scores_rank{N}.npz at the run-level inference dir. "
                             "Requires prefix+suffix <= 600 (maps are O((prefix+suffix)^2) per layer/head). "
                             "Always regenerates from scratch, ignoring any reusable smaller-suffix run.")
    return parser.parse_args()


def load_rep_bucket(path: Path, offset: int, prefix_length: int, suffix_length: int,
                    max_samples: int | None = None) -> list:
    """Returns [(sample_idx, excerpt_tokens), ...] -- sample_idx is the 0-based line
    index in the source file, tagged here before DistributedSampler ever splits the
    dataset, so it's stable across --batch-size/world size in later runs.
    """
    dataset = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples is not None and len(dataset) >= max_samples:
                break
            ids = json.loads(line)["input_ids"]
            excerpt = ids[offset: offset + prefix_length + suffix_length]
            assert len(excerpt) == prefix_length + suffix_length, (
                f"{path.name}: sequence too short ({len(ids)} tokens)"
            )
            dataset.append((i, excerpt))
    return dataset


def _make_capture(model, args, needs_bos: bool):
    """Instantiate and register a shared AttentionCapture spanning all repetition buckets."""
    from attn_bench.evaluation.attn_capture import AttentionCapture

    cfg = model.config
    prompt_len = args.prefix_length + (1 if needs_bos else 0)

    capture = AttentionCapture(
        n_layers=cfg.num_layers,
        n_heads=cfg.num_attention_heads,
        prompt_len=prompt_len,
        suffix_length=args.suffix_length,
        is_gated=getattr(cfg, 'attention_output_gate', False),
    )
    capture.register(model)
    return capture


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_run_metadata(output_path: Path, args, world_size: int, action: str,
                       extended_from_suffix: int | None = None) -> None:
    """Append one entry to run_metadata.json -- generate/extend/backfill/metrics jobs can
    all touch the same directory over time, so the history is what's useful, not just the
    last write.
    """
    meta_path = output_path / "run_metadata.json"
    history = []
    if meta_path.exists():
        with open(meta_path) as f:
            history = json.load(f)
    history.append({
        "action": action,
        "extended_from_suffix": extended_from_suffix,
        "container_env": args.container_env,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "ckpt_dir": args.ckpt_dir,
        "world_size": world_size,
        "max_samples": args.max_samples,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with open(meta_path, "w") as f:
        json.dump(history, f, indent=2)


def _setup_capture(model, args, output_path: Path, rank: int, needs_bos: bool):
    """Attention capture (when requested) aggregates full maps across ALL repetition
    buckets into Rouge-L buckets, written once at run level. A run-level marker decides
    resume."""
    from attn_bench.evaluation.attn_capture import N_BUCKETS, bucket_label

    last_bucket = bucket_label(N_BUCKETS - 1)
    capture_marker = output_path / f"attn_scores_rouge_l_{last_bucket}_rank{rank}.npz"
    do_capture = args.capture_attention and not capture_marker.exists()
    if args.capture_attention and not do_capture and rank == 0:
        print("Attention capture already done — skipping capture (jsonl still processed as needed).")
    return _make_capture(model, args, needs_bos) if do_capture else None


def _process_rep(model, args, experiment_path: Path, output_path: Path, path: Path,
                 rank: int, world_size: int, needs_bos: bool, capture):
    """Runs (or skips, or extends) one repetition bucket. Returns None if skipped,
    else (extend_from_suffix_or_None, generation_time)."""
    rep = int(path.stem.split("_")[1])
    inference_dir = output_path / f"rep_{rep}_greedy"

    rank0_file = inference_dir / "rank0.jsonl"
    jsonl_done = rank0_file.exists() and rank0_file.stat().st_size > 0

    source = None if capture is not None else find_rep_source(
        experiment_path, args.offset, args.prefix_length, args.suffix_length, rep,
    )

    if (jsonl_done or (source is not None and source[0] >= args.suffix_length)) and capture is None:
        if rank == 0:
            print(f"Skipping rep={rep} (already covered by an existing suffix >= {args.suffix_length})")
        return None

    if rank == 0:
        print(f"\nProcessing rep={rep}")

    dataset = load_rep_bucket(path, args.offset, args.prefix_length, args.suffix_length,
                               max_samples=args.max_samples)

    if rank == 0:
        print(f"  {len(dataset)} sequences")

    extend_records = None
    extend_from_suffix = None
    if source is not None and source[0] < args.suffix_length:
        extend_from_suffix = source[0]
        extend_records = load_records_by_sample_idx(source[1], dataset_len=len(dataset))
        if rank == 0:
            print(f"  Extending from suffix={extend_from_suffix}")

    generation_time = run_bucket(
        model, dataset,
        args.prefix_length, args.suffix_length,
        args.batch_size, inference_dir,
        rank, world_size,
        needs_bos=needs_bos,
        capture=capture,
        extend_records=extend_records,
        extend_from_suffix=extend_from_suffix,
    )

    if rank == 0:
        print(f"  Done rep={rep} (generation: {generation_time:.1f}s)")
    torch.cuda.empty_cache()

    return extend_from_suffix, generation_time


def _save_capture(output_path: Path, rank: int, capture) -> None:
    if capture is not None:
        capture.save(output_path, rank)
        capture.remove()


def _write_run_summary(output_path: Path, args, world_size: int, rank: int,
                       did_generate: bool, did_extend: bool, extended_from_suffixes: set,
                       total_generation_time: float) -> None:
    """Appends a run_metadata.json entry (skipped if nothing was generated/extended this
    run) and prints the completion message."""
    if rank != 0:
        return
    if did_extend or did_generate:
        action = "extend" if did_extend and not did_generate else (
            "generate" if did_generate and not did_extend else "generate+extend"
        )
        write_run_metadata(
            output_path, args, world_size, action=action,
            extended_from_suffix=(next(iter(extended_from_suffixes)) if len(extended_from_suffixes) == 1
                                  else sorted(extended_from_suffixes) or None),
        )
        print(f"Total generation time this run: {total_generation_time:.1f}s")
    print(f"\nAll repetitions done. Results in: {output_path}")


def run_inference(model, args, rank: int, world_size: int) -> None:
    experiment_path = Path(args.experiment_path)
    output_path = (
        experiment_path
        / "inference"
        / f"offset_{args.offset}_prefix_{args.prefix_length}_suffix_{args.suffix_length}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    needs_bos = args.offset > 0  # offset==0: BOS already at token 0; offset>0: must prepend
    capture = _setup_capture(model, args, output_path, rank, needs_bos)

    did_generate = False
    did_extend = False
    extended_from_suffixes = set()
    total_generation_time = 0.0

    for path in paths:
        result = _process_rep(model, args, experiment_path, output_path, path,
                              rank, world_size, needs_bos, capture)
        if result is None:
            continue
        extend_from_suffix, generation_time = result
        total_generation_time += generation_time
        if extend_from_suffix is not None:
            did_extend = True
            extended_from_suffixes.add(extend_from_suffix)
        else:
            did_generate = True

    _save_capture(output_path, rank, capture)
    _write_run_summary(output_path, args, world_size, rank,
                       did_generate, did_extend, extended_from_suffixes, total_generation_time)


def results_already_complete(args, world_size: int) -> bool:
    """True if every requested rep already has a suffix' >= args.suffix_length on disk
    (and, when capturing, every rank's capture file too) — i.e. there is nothing left to
    compute. Checked before the expensive checkpoint load.

    Checked from env (WORLD_SIZE), not torch.distributed, so it can run *before* the
    process group is initialized — no barrier to deadlock on an early exit.
    """
    from attn_bench.evaluation.attn_capture import N_BUCKETS, bucket_label

    experiment_path = Path(args.experiment_path)
    output_path = (
        experiment_path
        / "inference"
        / f"offset_{args.offset}_prefix_{args.prefix_length}_suffix_{args.suffix_length}"
    )

    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    if not paths:
        return False  # no input data found — let the normal path no-op/report

    if args.capture_attention:
        # Capture always regenerates every rep (the maps need the forward passes), so
        # cross-suffix reuse never applies here -- fall back to the exact-match check.
        for path in paths:
            rep = int(path.stem.split("_")[1])
            rank0_file = output_path / f"rep_{rep}_greedy" / "rank0.jsonl"
            if not (rank0_file.exists() and rank0_file.stat().st_size > 0):
                return False
        last_bucket = bucket_label(N_BUCKETS - 1)
        for r in range(world_size):
            if not (output_path / f"attn_scores_rouge_l_{last_bucket}_rank{r}.npz").exists():
                return False
        return True

    for path in paths:
        rep = int(path.stem.split("_")[1])
        source = find_rep_source(experiment_path, args.offset, args.prefix_length, args.suffix_length, rep)
        if source is None or source[0] < args.suffix_length:
            return False

    return True


def main():
    args = parse_args()

    if args.capture_attention and (args.prefix_length + args.suffix_length) > 600:
        raise ValueError(
            f"--capture-attention requires prefix+suffix <= 600 (full attention maps are "
            f"O((prefix+suffix)^2) per layer/head); got "
            f"{args.prefix_length}+{args.suffix_length}={args.prefix_length + args.suffix_length}."
        )

    if args.sink_scale is not None:
        args.experiment_path = args.experiment_path.rstrip('/') + f"_sscale{args.sink_scale:g}"

    # Check results before loading the checkpoint: loading the model is the
    # expensive part, so if everything is already on disk we skip it entirely.
    # (run_inference still does a finer per-rep skip for the partially-done case.)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if results_already_complete(args, world_size):
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"All results already present for offset={args.offset} "
                f"prefix={args.prefix_length} suffix={args.suffix_length} "
                f"(capture={args.capture_attention}) — skipping checkpoint load."
            )
        return

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

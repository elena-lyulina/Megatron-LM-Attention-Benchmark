"""
Sparse Gutenberg inference for memorization measurement (Step 1): greedy-generate a suffix
from a prefix at increasing repetition counts and score it. Backend-agnostic via
InferenceBackend -- Megatron or an already-converted HF checkpoint. GPU-dependent work only
(generation, NLL, p_z); text metrics live in compute_memorization_metrics.py.

Writes one rank{N}.jsonl per GPU under offset_O_prefix_P_suffix_S/rep_R_greedy/. Reuses a
sibling suffix' >= the request, or extends a smaller one via teacher-forced prefill.

--points takes one or more offset:prefix_length pairs sharing a single checkpoint load, e.g.
--points 0:500 50:450 -- a single pair behaves exactly like the old --offset/--prefix-length.
--suffix-length is a single value shared by every point in the run.

Usage (via torchrun):
    torchrun --nproc_per_node=4 attn_bench/evaluation/prefix_extraction_inference.py \
        --checkpoint-backend megatron \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --points 0:500 \
        --suffix-length 500 \
        --batch-size 20

    # or, against an already HF-converted checkpoint (faster generate(), full-attention only):
    # swap --checkpoint-backend megatron --ckpt-dir ... --tokenizer-path ... for
    # --checkpoint-backend hf --hf-dir $HF_DIR, everything else unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from attn_bench.evaluation.inference_backend import (HFBackend,
                                                     InferenceBackend,
                                                     MegatronBackend)
from attn_bench.evaluation.inference_common import (BOS_TOKEN_ID,
                                                    find_rep_paths,
                                                    find_suffix_dirs,
                                                    load_records_by_sample_idx,
                                                    parse_points)

P_Z_TOP_K = 40
P_Z_TEMPERATURE = 1.0

### NLL / p_z ###

@torch.no_grad()
def compute_nll(backend: InferenceBackend, input_ids: torch.Tensor, suffix_length: int):
    """One forward pass; per-position NLL for the last suffix_length tokens. Also returns
    the logits/labels so callers (p_z_log_probs) can reuse this pass instead of paying for
    another one.

    input_ids: [B, S] -- BOS + prefix + suffix (or BOS + prefix + generated)
    Returns: per_position_nll [B, suffix_length], suffix_logits [B, suffix_length, V], suffix_labels [B, suffix_length]
    """
    B, S = input_ids.shape
    device = input_ids.device
    inputs = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    position_ids = torch.arange(S - 1, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    logits = backend.forward_logits(inputs, position_ids)  # [B, S-1, V]

    # Slice to suffix_length first -- log_softmax in fp32 over the full prefix+suffix span
    # wasted memory that grew with prefix length for no benefit (OOM'd at prefix>=5000, job 3036124).
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
    (Hayes et al. 2025's p_z), from the same forward pass as NLL -- no generation involved.
    -inf where the true token falls outside the top-k set (that scheme could never sample it there)."""
    scaled = suffix_logits / temperature
    topk_vals, topk_idx = scaled.topk(top_k, dim=-1)
    topk_log_probs = F.log_softmax(topk_vals, dim=-1)

    matches = topk_idx == suffix_labels.unsqueeze(-1)  # [B, suffix_length, top_k]
    in_top_k = matches.any(dim=-1)
    log_prob = (topk_log_probs * matches).sum(dim=-1)
    return torch.where(in_top_k, log_prob, torch.full_like(log_prob, float("-inf")))


def compute_nll_pz_stats(backend: InferenceBackend, true_full_sequence: torch.Tensor,
                         gen_full_sequence: torch.Tensor, suffix_length: int):
    """ref/gen NLL, p_z, and their mean/std/ppl summaries for one batch."""
    ref_nll, ref_logits, ref_labels = compute_nll(backend, true_full_sequence, suffix_length)
    p_z = p_z_log_probs(ref_logits, ref_labels)
    del ref_logits, ref_labels
    gen_nll, gen_logits, gen_labels = compute_nll(backend, gen_full_sequence, suffix_length)
    del gen_logits, gen_labels
    ref_mean, ref_std, ref_ppl = nll_stats(ref_nll)
    gen_mean, gen_std, gen_ppl = nll_stats(gen_nll)
    return ref_nll, gen_nll, p_z, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl


### MULTI-LOCATION LOOKUP ###

def file_exists_in_locations(locations, file: str, require_nonempty: bool = False) -> bool:
    """True if location/file exists under any of the given locations, checked in order
    (None entries skipped). require_nonempty also checks size > 0, for markers a crashed
    write can leave present but empty."""
    for location in locations:
        if location is None:
            continue
        p = location / file
        if p.exists() and (not require_nonempty or p.stat().st_size > 0):
            return True
    return False


### CROSS-SUFFIX LOOKUP ###

def find_rep_source(experiment_path: Path, offset: int, prefix_length: int,
                    suffix_length: int, rep: int, persistent_storage_path: Path | None = None):
    """Among existing suffix dirs for (offset, prefix) -- experiment_path and, if given,
    persistent_storage_path -- find one whose rep_{rep}_greedy is complete: the smallest suffix' >=
    suffix_length if any (nothing to compute), else the largest suffix' < suffix_length
    (extend from here). None if nothing usable exists.

    Returns (suffix', rep_dir_path) or None.
    """
    usable = []
    for suffix_prime, d in find_suffix_dirs(experiment_path, offset, prefix_length, persistent_storage_path):
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
    """Per-sample Rouge-L, computed locally just to route attention maps into buckets for
    --capture-attention -- not persisted (that's Step 2's job). Deferred import keeps the
    common non-capture path free of PDM/verbatim_eval deps."""
    import numpy as np
    from verbatim_eval.my_rouge import (_compute_dp_matrix_2d,
                                        compute_rouge_l_2d)

    scores = []
    for true_seq, gen_seq in zip(true_suffixes, gen_suffixes):
        dp = _compute_dp_matrix_2d(np.array(true_seq, dtype=np.int32), np.array(gen_seq, dtype=np.int32))
        scores.append(compute_rouge_l_2d(dp))
    return scores


def run_bucket(backend: InferenceBackend, dataset, prefix_length, suffix_length, batch_size,
              inference_dir, rank, world_size, needs_bos: bool, capture=None,
              extend_records: dict | None = None, extend_from_suffix: int | None = None) -> float:
    """Run inference for one repetition bucket.

    dataset: (sample_idx, excerpt_tokens) list -- sample_idx tagged pre-split, so it's
             stable across --batch-size/world size in later runs.
    needs_bos: True when offset > 0 (excerpts don't start with BOS, so we prepend it).
    extend_records: {sample_idx: old_record} to extend from a smaller suffix' run, or None
                    to generate fresh (old_record needs "generated_suffix" of length
                    extend_from_suffix).
    capture: shared AttentionCapture, or None. Always regenerates from scratch when set
             (extend_records is ignored) -- the maps need real forward passes.

    Returns: wall time spent generating (checkpoint load happens before this is called).
    """
    device = backend.device

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
                bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
                seq = torch.cat([bos, batch_tensor], dim=1)  # [B, 1+prefix+suffix]
                prompt_end = 1 + prefix_length
            else:
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
                new_tokens = backend.generate(decode_prompt, new_steps)
                generation_time += time.monotonic() - t0
                generated = torch.cat([old_generated, new_tokens], dim=1)
            elif capture is not None:
                capture.begin_batch(B)
                t0 = time.monotonic()
                generated = backend.generate_with_capture(
                    prompt, suffix_length,
                    prefill_callback=capture.collect_prefill,
                    decode_step_callback=capture.collect_decode,
                )
                generation_time += time.monotonic() - t0
            else:
                t0 = time.monotonic()
                generated = backend.generate(prompt, suffix_length)
                generation_time += time.monotonic() - t0

            gen_full = torch.cat([prompt, generated], dim=1)
            ref_nll, gen_nll, p_z, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl = \
                compute_nll_pz_stats(backend, seq, gen_full, suffix_length)

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
    parser.add_argument("--checkpoint-backend", choices=["megatron", "hf"], default="megatron",
                        help="megatron: load the torch_dist checkpoint directly (all attention "
                             "variants). hf: load an already-converted HF checkpoint (faster "
                             "generate(), full-attention models only). Requires --hf-dir; "
                             "incompatible with --sink-scale/--capture-attention.")
    parser.add_argument("--ckpt-dir", default=None, help="torch_dist checkpoint directory (megatron backend)")
    parser.add_argument("--tokenizer-path", default=None, help="(megatron backend)")
    parser.add_argument("--hf-dir", default=None,
                        help="Output of checkpoint_conversion/convert_megatron_to_hf.py (hf backend)")
    parser.add_argument("--experiment-path", required=True, help="Output root (MEM_DIR)")
    parser.add_argument("--persistent-storage-path", default=None,
                        help="Secondary mirror of --experiment-path, checked as a fallback "
                             "wherever existing results are looked up -- e.g. if "
                             "--experiment-path has since lost them. Never written to.")
    parser.add_argument("--data-folder", required=True, help="Directory of rep_*_token.jsonl files")
    parser.add_argument("--repetitions", required=True, help="Comma-separated, e.g. 0,1,2,4,8,16,32,64,128,256")
    parser.add_argument("--points", nargs="+", required=True,
                        help="One or more offset:prefix_length pairs sharing one checkpoint "
                             "load, e.g. --points 0:500 50:450 100:400. A single pair behaves "
                             "exactly like the old single-point --offset/--prefix-length.")
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
                             "Supports off-by-one and learnable attention (megatron backend only). "
                             "Original per-head values saved to sink_scale_metadata.json. "
                             "Appends _sscale{X} to experiment path.")
    parser.add_argument("--capture-attention", action="store_true",
                        help="Capture full causal attention maps (prefill + decode), averaged into "
                             "Rouge-L buckets across all repetition buckets (megatron backend only). "
                             "Writes attn_scores_rouge_l_{NN-MM}_rank{N}.npz, "
                             "norm_attn_rouge_l_{NN-MM}_rank{N}.npz and (gated only) "
                             "gating_scores_rank{N}.npz at the run-level inference dir. "
                             "Requires prefix+suffix <= 600 (maps are O((prefix+suffix)^2) per layer/head). "
                             "Always regenerates from scratch, ignoring any reusable smaller-suffix run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report which points are already complete, write nothing, skip "
                             "checkpoint load entirely.")
    return parser.parse_args()


def build_backend(args) -> InferenceBackend:
    """Constructs the requested backend (raises if required args are missing/invalid), then
    checks any optional capability flags against what it actually implements -- fails before
    load_model()'s GPU/checkpoint work, without main() hardcoding which backend supports what."""
    if args.checkpoint_backend == "hf":
        backend = HFBackend(args.hf_dir)
    else:
        backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args,
                                  sink_scale=args.sink_scale)

    if args.sink_scale is not None and type(backend).patch_sink_scale is InferenceBackend.patch_sink_scale:
        raise ValueError(f"--sink-scale: {backend.name} backend does not implement patch_sink_scale.")
    if args.capture_attention and type(backend).setup_attention_capture is InferenceBackend.setup_attention_capture:
        raise ValueError(f"--capture-attention: {backend.name} backend does not implement setup_attention_capture.")
    return backend


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


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def write_run_metadata(output_path: Path, args, backend: InferenceBackend, world_size: int,
                       action: str, extended_from_suffix: int | None = None) -> None:
    """Append one entry to run_metadata.json -- generate/extend/backfill/metrics jobs can
    all touch the same directory over time, so the history is what's useful, not just the
    last write.
    """
    meta_path = output_path / "run_metadata.json"
    history = []
    if meta_path.exists():
        with open(meta_path) as f:
            loaded = json.load(f)
        # Legacy dirs from before this history became a list carry a single flat dict --
        # keep it as the first entry instead of discarding it.
        history = loaded if isinstance(loaded, list) else [loaded]
    history.append({
        "action": action,
        "extended_from_suffix": extended_from_suffix,
        "container_env": args.container_env,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_backend": backend.name,
        "ckpt_dir": getattr(args, "ckpt_dir", None),
        "hf_dir": getattr(args, "hf_dir", None),
        "world_size": world_size,
        "max_samples": args.max_samples,
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    with open(meta_path, "w") as f:
        json.dump(history, f, indent=2)


def _process_rep(backend: InferenceBackend, args, experiment_path: Path, output_path: Path,
                 path: Path, rank: int, world_size: int, offset: int, prefix_length: int,
                 needs_bos: bool, capture, persistent_storage_path: Path | None = None):
    """Runs (or skips, or extends) one repetition bucket. Returns None if skipped,
    else (extend_from_suffix_or_None, generation_time)."""
    rep = int(path.stem.split("_")[1])
    inference_dir = output_path / f"rep_{rep}_greedy"

    persistent_inference_dir = (
        persistent_storage_path / "inference" /
        f"offset_{offset}_prefix_{prefix_length}_suffix_{args.suffix_length}" / f"rep_{rep}_greedy"
        if persistent_storage_path is not None else None
    )
    jsonl_done = file_exists_in_locations([inference_dir, persistent_inference_dir],
                                          "rank0.jsonl", require_nonempty=True)

    source = None if capture is not None else find_rep_source(
        experiment_path, offset, prefix_length, args.suffix_length, rep, persistent_storage_path,
    )

    if (jsonl_done or (source is not None and source[0] >= args.suffix_length)) and capture is None:
        if rank == 0:
            print(f"Skipping rep={rep} (already covered by an existing suffix >= {args.suffix_length})")
        return None

    if rank == 0:
        print(f"\nProcessing rep={rep}")

    dataset = load_rep_bucket(path, offset, prefix_length, args.suffix_length,
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
        backend, dataset,
        prefix_length, args.suffix_length,
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


def _write_run_summary(output_path: Path, args, backend: InferenceBackend, world_size: int, rank: int,
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
            output_path, args, backend, world_size, action=action,
            extended_from_suffix=(next(iter(extended_from_suffixes)) if len(extended_from_suffixes) == 1
                                  else sorted(extended_from_suffixes) or None),
        )
        print(f"Total generation time this run: {total_generation_time:.1f}s")
    print(f"\nAll repetitions done. Results in: {output_path}")


def run_inference(backend: InferenceBackend, args, rank: int, world_size: int,
                  offset: int, prefix_length: int, persistent_storage_path: Path | None = None) -> None:
    experiment_path = Path(args.experiment_path)
    output_path = (
        experiment_path
        / "inference"
        / f"offset_{offset}_prefix_{prefix_length}_suffix_{args.suffix_length}"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    needs_bos = offset > 0  # offset==0: BOS already at token 0; offset>0: must prepend
    capture = backend.setup_attention_capture(args, output_path, rank, needs_bos) if args.capture_attention else None

    did_generate = False
    did_extend = False
    extended_from_suffixes = set()
    total_generation_time = 0.0

    for path in paths:
        result = _process_rep(backend, args, experiment_path, output_path, path,
                              rank, world_size, offset, prefix_length, needs_bos, capture,
                              persistent_storage_path)
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
    _write_run_summary(output_path, args, backend, world_size, rank,
                       did_generate, did_extend, extended_from_suffixes, total_generation_time)


def results_already_complete(args, world_size: int, offset: int, prefix_length: int,
                             persistent_storage_path: Path | None = None) -> bool:
    """True if every requested rep already has a suffix' >= args.suffix_length (checked on
    args.experiment_path, then persistent_storage_path; capture mode also needs every
    rank's capture file). Uses env WORLD_SIZE, not torch.distributed, so it can run -- and
    exit early -- before the process group is initialized, with no barrier to deadlock on."""
    from attn_bench.evaluation.attn_capture import N_BUCKETS, bucket_label

    experiment_path = Path(args.experiment_path)
    suffix_dir_name = f"offset_{offset}_prefix_{prefix_length}_suffix_{args.suffix_length}"
    output_path = experiment_path / "inference" / suffix_dir_name
    persistent_output_path = (
        persistent_storage_path / "inference" / suffix_dir_name
        if persistent_storage_path is not None else None
    )

    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    if not paths:
        return False  # no input data found — let the normal path no-op/report

    if args.capture_attention:
        # Capture always regenerates (see run_bucket) -- exact-match check only, no reuse.
        for path in paths:
            rep = int(path.stem.split("_")[1])
            if not file_exists_in_locations([output_path, persistent_output_path],
                                            f"rep_{rep}_greedy/rank0.jsonl", require_nonempty=True):
                return False
        last_bucket = bucket_label(N_BUCKETS - 1)
        for r in range(world_size):
            if not file_exists_in_locations([output_path, persistent_output_path],
                                            f"attn_scores_rouge_l_{last_bucket}_rank{r}.npz"):
                return False
        return True

    for path in paths:
        rep = int(path.stem.split("_")[1])
        source = find_rep_source(experiment_path, offset, prefix_length, args.suffix_length, rep,
                                 persistent_storage_path)
        if source is None or source[0] < args.suffix_length:
            return False

    return True


def print_dry_run_report(points: list, status: dict) -> None:
    """status: {(offset, prefix): bool}, True meaning already complete. Report to stderr,
    space-separated offset:prefix needed-list to stdout -- same convention Stage 2 uses."""
    done = [pt for pt in points if status[pt]]
    needed = [pt for pt in points if not status[pt]]
    print(f"{len(done)}/{len(points)} points already complete:", file=sys.stderr)
    for offset, prefix_length in done:
        print(f"  done:   offset={offset} prefix={prefix_length}", file=sys.stderr)
    for offset, prefix_length in needed:
        print(f"  needed: offset={offset} prefix={prefix_length}", file=sys.stderr)
    print(" ".join(f"{o}:{p}" for o, p in needed))


def main():
    args = parse_args()
    points = parse_points(args.points)

    if args.capture_attention and len(points) > 1:
        # Capture always regenerates (see run_bucket), so multi-point sharing gains
        # nothing here. TODO: fix capture's skip-check to look at existing output per rep,
        # then this restriction can go away.
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "--capture-attention doesn't support multi-point runs yet -- turning it off "
                "for this run. Pass a single --points pair to actually capture attention maps."
            )
        args.capture_attention = False

    if args.capture_attention:
        offset, prefix_length = points[0]
        if (prefix_length + args.suffix_length) > 600:
            raise ValueError(
                f"--capture-attention requires prefix+suffix <= 600 (full attention maps are "
                f"O((prefix+suffix)^2) per layer/head); got "
                f"{prefix_length}+{args.suffix_length}={prefix_length + args.suffix_length}."
            )

    backend = build_backend(args)
    args.experiment_path = args.experiment_path.rstrip('/') + backend.experiment_path_suffix()
    persistent_storage_path = None
    if args.persistent_storage_path:
        persistent_storage_path = Path(
            args.persistent_storage_path.rstrip('/') + backend.experiment_path_suffix()
        )

    # Skip checkpoint load entirely if EVERY point is already done; run_inference does a
    # finer per-rep skip for points only partially done.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    status = {(o, p): results_already_complete(args, world_size, o, p, persistent_storage_path)
             for o, p in points}

    if args.dry_run:
        print_dry_run_report(points, status)
        return

    remaining = [pt for pt in points if not status[pt]]
    if not remaining:
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"All results already present for every requested point "
                f"(suffix={args.suffix_length}, capture={args.capture_attention}) "
                f"— skipping checkpoint load."
            )
        return

    backend.load_model()

    if args.sink_scale is not None:
        originals = backend.patch_sink_scale()
        if dist.get_rank() == 0:
            meta_path = Path(args.experiment_path) / "sink_scale_metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(meta_path, "w") as f:
                json.dump({"sink_scale": args.sink_scale, "original_softmax_offset": originals}, f, indent=2)
            print(f"Saved sink scale metadata to {meta_path}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    for offset, prefix_length in remaining:
        if rank == 0:
            print(f"\n=== POINT offset={offset} prefix={prefix_length} suffix={args.suffix_length} ===")
        run_inference(backend, args, rank, world_size, offset, prefix_length, persistent_storage_path)


if __name__ == "__main__":
    main()

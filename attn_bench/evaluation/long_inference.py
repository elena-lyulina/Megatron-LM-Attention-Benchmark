"""
Shared engine for per-position long-sequence loss inference: one teacher-forced
forward pass per sequence, recording per-token NLL past the training seq_len (where
attention is expected to degrade but state-carrying models might not).

Dataset-agnostic: takes a `dataset` of (token-id list, seq_id) pairs and a `key` string
naming the output file. Callers (long_gutenberg_inference.py, long_fineweb_inference.py)
only differ in how they build token-id sequences and what `key`s/loader args they pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from attn_bench.evaluation.gdn_state_norm import install_state_norm_hooks
from attn_bench.evaluation.inference_backend import (HFBackend,
                                                     InferenceBackend,
                                                     MegatronBackend)
from megatron.core import parallel_state as mpu

SEQ_LEN = 8192  # training sequence length; suffix (position >= sample_len) is the extrapolation region
SAMPLE_SEED = 42  # fixed seed for --max-samples subsampling, so calibration runs are reproducible


### DATA ###

def sample_lines(path: Path, max_samples: int | None) -> list[str]:
    """Read every line in a jsonl file; if max_samples is given, randomly subsample without
    replacement (fixed seed) instead of just taking the first N records."""
    with open(path) as f:
        lines = f.readlines()
    if max_samples is not None and len(lines) > max_samples:
        lines = random.Random(SAMPLE_SEED).sample(lines, max_samples)
    return lines


### FORWARD ###

@torch.no_grad()
def per_position_nll(backend: InferenceBackend, seq_ids: torch.Tensor, softmax_chunk: int,
                     store_individual: bool = False):
    """One forward over the whole sequence; return per-position NLL [S-1] (float32), plus,
    when store_individual, the argmax predicted token and the true token's rank in the
    predicted distribution (both [S-1] long, else None).

    nll[k] is the loss for predicting token at index k+1, i.e. position k measured from
    the sample start (index 0 = BOS, so position 0 = first sample token c0).
    Logits/softmax are computed chunked over the sequence dim: materializing the full
    float logit tensor at 30k+ tokens x 128k vocab would OOM.
    """
    inputs = seq_ids[:, :-1]
    labels = seq_ids[:, 1:]
    S1 = inputs.shape[1]
    device = seq_ids.device
    pos = torch.arange(S1, dtype=torch.long, device=device).unsqueeze(0)

    logits = backend.forward_logits(inputs, pos, fp32_output=False)  # [1, S1, V] (bf16)

    nll = torch.empty(S1, dtype=torch.float32, device=device)
    argmax_token = torch.empty(S1, dtype=torch.long, device=device) if store_individual else None
    true_rank = torch.empty(S1, dtype=torch.long, device=device) if store_individual else None
    for c0 in range(0, S1, softmax_chunk):
        c1 = min(c0 + softmax_chunk, S1)
        chunk_logits = logits[:, c0:c1, :].float()
        label_chunk = labels[:, c0:c1].unsqueeze(-1)
        logp = -F.log_softmax(chunk_logits, dim=-1)
        nll[c0:c1] = logp.gather(2, label_chunk).squeeze(-1).squeeze(0)
        if store_individual:
            argmax_token[c0:c1] = chunk_logits.argmax(dim=-1).squeeze(0)
            true_logit = chunk_logits.gather(2, label_chunk)
            # Rank 0 = the true token was the model's top pick; counts strictly-higher logits.
            true_rank[c0:c1] = (chunk_logits > true_logit).sum(dim=-1).squeeze(0)
    del logits
    return nll, argmax_token, true_rank


class IndividualCollector:
    """Collects raw per-sequence, per-position records (NLL, argmax token, true-token rank,
    true token) for a bucket.

    Kept separate from the sum/sqsum/count aggregates above: those are additive across DP
    ranks (plain all_reduce), but a per-sequence record lives entirely on whichever rank
    processed that sequence, so it needs an explicit gather instead.
    """

    def __init__(self):
        self.records = []

    def reset_bucket(self):
        self.records = []

    def record(self, idx, seq_id, nll, argmax_token, true_rank, true_token):
        self.records.append({
            "idx": idx,
            "seq_id": seq_id,
            "length": nll.shape[0],
            "nll": nll.cpu().tolist(),
            "argmax_token": argmax_token.cpu().tolist(),
            "true_token_rank": true_rank.cpu().tolist(),
            "true_token": true_token.cpu().tolist(),
        })

    def gather(self, dp_group):
        # Gathers onto the DP group's source rank (== global rank 0 for the TP=1 jobs this
        # is used with). Merges every rank's records and sorts by idx for a deterministic
        # file regardless of how the shard striding split the work.
        dst = mpu.get_data_parallel_src_rank()
        world = dist.get_world_size(dp_group)
        gathered = [None] * world if dist.get_rank() == dst else None
        dist.gather_object(self.records, gathered, dst=dst, group=dp_group)
        if gathered is None:
            self.records = []
            return
        self.records = sorted((r for rank_records in gathered for r in rank_records),
                              key=lambda r: r["idx"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")


### RUN ###

def run_inference(backend: InferenceBackend, dataset, maxpos, rank, softmax_chunk, device, desc="",
                  accum=None, individual=None):
    """Accumulate per-position NLL sum / sqsum / count over this rank's shard.

    Sharding and pooling are over the data-parallel group (== WORLD, TP is always 1 here).
    If accum is given, the GDN state norms are pooled the same way. If individual is given,
    raw per-sequence records are gathered onto the DP source rank (see IndividualCollector).

    dataset holds (tokens, seq_id) pairs -- seq_id is only used when individual is set.
    """
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    dp_group = mpu.get_data_parallel_group()

    pos_sum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_sqsum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_cnt = torch.zeros(maxpos, dtype=torch.float64, device=device)
    if accum is not None:
        accum.reset_bucket(maxpos)
    if individual is not None:
        individual.reset_bucket()

    # Each DP rank strides its own shard; only global rank 0 prints the bar.
    shard = range(dp_rank, len(dataset), dp_size)
    for idx in tqdm(shard, desc=desc, disable=(rank != 0), mininterval=5.0):
        tokens, seq_id = dataset[idx]
        seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        if accum is not None:
            accum.reset_sequence()
        nll, argmax_token, true_rank = per_position_nll(
            backend, seq, softmax_chunk, store_individual=individual is not None)
        if accum is not None:
            accum.accumulate()  # reads the norms the GDN wrappers recorded during the forward
        if individual is not None:
            individual.record(idx, seq_id, nll, argmax_token, true_rank, seq[0, 1:])
        nll = nll.double()
        L = nll.shape[0]
        pos_sum[:L] += nll
        pos_sqsum[:L] += nll * nll
        pos_cnt[:L] += 1.0
        del seq, nll
        torch.cuda.empty_cache()

    for t in (pos_sum, pos_sqsum, pos_cnt):
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=dp_group)
    if accum is not None:
        accum.reduce(dp_group)
    if individual is not None:
        individual.gather(dp_group)
    return pos_sum, pos_sqsum, pos_cnt


def save_npz(path: Path, pos_sum, pos_sqsum, pos_cnt, seq_len: int = SEQ_LEN):
    cnt = pos_cnt.cpu().numpy()
    keep = cnt > 0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        position=np.arange(len(cnt))[keep],
        nll_sum=pos_sum.cpu().numpy()[keep],
        nll_sqsum=pos_sqsum.cpu().numpy()[keep],
        count=cnt[keep],
        seq_len=seq_len,
    )


### CONFIG / OUTPUT PATHS ###

def config_name(max_length: int | None, max_samples: int | None) -> str:
    # Encodes the params that change the numbers, so different runs land in different
    # folders and the "already done" check doubles as a validity check.
    samples = max_samples if max_samples is not None else "all"
    length = max_length if max_length is not None else "full"
    return f"{samples}_samples_{length}_tokens"


def config_dir(experiment_path: str, max_length: int | None, max_samples: int | None) -> Path:
    return Path(experiment_path) / config_name(max_length, max_samples)


def npz_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"{key}.npz"


def state_npz_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"{key}_state.npz"


def individual_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"{key}_individual.jsonl"


def file_exists_in_locations(locations, file: str) -> bool:
    """True if location/file exists under any of the given locations (None entries skipped)."""
    return any(location is not None and (location / file).exists() for location in locations)


def result_done(output_dir: Path, key: str, log_state_norm: bool, store_individual: bool = False,
                persistent_output_dir: Path | None = None) -> bool:
    # Done when the NLL file exists -- and, when logging state norms / individual records,
    # those files too (so turning either flag on for a finished model re-runs to fill it in).
    # persistent_output_dir is checked as a fallback for results output_dir no longer has.
    locations = (output_dir, persistent_output_dir)
    if not file_exists_in_locations(locations, f"{key}.npz"):
        return False
    if log_state_norm and not file_exists_in_locations(locations, f"{key}_state.npz"):
        return False
    if store_individual and not file_exists_in_locations(locations, f"{key}_individual.jsonl"):
        return False
    return True


def write_run_metadata(output_dir: Path, extra: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }, f, indent=2)


### CLI ###

def add_common_args(p: argparse.ArgumentParser, max_samples_default: int | None = None) -> None:
    """Flags shared by every long-sequence inference script (dataset selection is caller-specific)."""
    p.add_argument("--checkpoint-backend", choices=["megatron", "hf"], default="megatron",
                   help="megatron: load the torch_dist checkpoint directly (--ckpt-dir/"
                        "--tokenizer-path, all attention variants). hf: load an already-"
                        "converted HF checkpoint (--hf-dir).")
    p.add_argument("--ckpt-dir", default=None, help="torch_dist checkpoint directory (megatron backend)")
    p.add_argument("--tokenizer-path", default=None, help="(megatron backend)")
    p.add_argument("--hf-dir", default=None,
                   help="Output of checkpoint_conversion/convert_megatron_to_hf.py (hf backend)")
    p.add_argument("--experiment-path", required=True, help="Output root")
    p.add_argument("--persistent-storage-path", default=None,
                   help="Secondary mirror of --experiment-path, checked as a fallback for the "
                        "done-check -- e.g. if --experiment-path has since lost them. Never written to.")
    p.add_argument("--max-length", type=int, default=None, help="Cap each sequence to this many tokens.")
    p.add_argument("--max-samples", type=int, default=max_samples_default,
                   help="Randomly subsample to this many sequences (fixed seed, for testing/calibration "
                        f"or to avoid running on all available data). Default: {max_samples_default or 'no cap'}.")
    p.add_argument("--softmax-chunk", type=int, default=4096,
                   help="How many positions to run log_softmax over at once. Smaller uses less memory, "
                        "same result. Does not change the NLL values.")
    p.add_argument("--log-state-norm", action="store_true",
                   help="For GDN models, also write recurrent-state norms to <key>_state.npz. "
                        "No effect on attention models (they have no state to log).")
    p.add_argument("--state-chunk", type=int, default=128,
                   help="Read the GDN state every this many tokens (only used with --log-state-norm).")
    p.add_argument("--store-individual", action="store_true",
                   help="Also write raw per-sequence, per-position records (NLL, argmax predicted "
                        "token, true-token rank, true token) to <key>_individual.jsonl -- one line "
                        "per sequence. Off by default; the aggregated <key>.npz (mean/std/count) is "
                        "unaffected either way.")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if <key>.npz already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Report which keys are already complete, write nothing, skip checkpoint "
                        "load entirely.")
    p.add_argument("--container-env", default=None, help="Container/env name, recorded for provenance")
    p.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                   help="Extra Megatron args forwarded to the checkpoint loader (e.g. --attention-output-gate)")


### MAIN ###

def build_backend(args) -> InferenceBackend:
    """Constructs the requested backend (raises if required args are missing/invalid). Cheap --
    no checkpoint/GPU work happens until backend.load_model()."""
    if args.checkpoint_backend == "hf":
        return HFBackend(args.hf_dir)
    return MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)


def print_dry_run_report(items: list, status: dict) -> None:
    """status: {key: bool}, True meaning already complete. Report to stderr, space-separated
    needed-key list to stdout -- same convention the mem-results scripts use."""
    done = [key for key, _ in items if status[key]]
    needed = [key for key, _ in items if not status[key]]
    print(f"{len(done)}/{len(items)} keys already complete:", file=sys.stderr)
    for key in done:
        print(f"  done:   {key}", file=sys.stderr)
    for key in needed:
        print(f"  needed: {key}", file=sys.stderr)
    print(" ".join(needed))


def run_main(args, items: list[tuple[str, object]], load_dataset: Callable, metadata_extra: dict) -> None:
    """Shared driver for a long-sequence inference script's main().

    `items` is a list of (key, loader_arg) pairs -- one per output file this invocation may
    produce (Gutenberg: one per requested repetition; fineweb: a single item). `load_dataset`
    is called as `load_dataset(loader_arg, args.max_length, args.max_samples)
    -> list[tuple[list[int], str]]` -- (tokens, seq_id) pairs; seq_id is only used when
    --store-individual is set (Gutenberg's book_id, fineweb's doc_id).
    """
    backend = build_backend(args)
    args.experiment_path = args.experiment_path.rstrip('/') + backend.experiment_path_suffix()
    if getattr(args, "persistent_storage_path", None):
        args.persistent_storage_path = (
            args.persistent_storage_path.rstrip('/') + backend.experiment_path_suffix()
        )

    output_dir = config_dir(args.experiment_path, args.max_length, args.max_samples)
    persistent_output_dir = (
        config_dir(args.persistent_storage_path, args.max_length, args.max_samples)
        if getattr(args, "persistent_storage_path", None) else None
    )

    status = {
        key: result_done(output_dir, key, args.log_state_norm, args.store_individual, persistent_output_dir)
        for key, _ in items
    }

    if args.dry_run:
        print_dry_run_report(items, status)
        return

    if not args.overwrite and items and all(status.values()):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"All requested results already present in {output_dir} -- skipping checkpoint load.")
        return

    backend.load_model()
    rank = dist.get_rank()
    device = backend.device

    accum = None
    if args.log_state_norm:
        accum = install_state_norm_hooks(backend.model, args.state_chunk, device)
        if accum is None and rank == 0:
            print("--log-state-norm set but model has no GatedDeltaNet layers; skipping state norms.")

    individual = IndividualCollector() if args.store_individual else None

    if rank == 0:
        write_run_metadata(output_dir, {
            "container_env": args.container_env,
            "checkpoint_backend": args.checkpoint_backend,
            "ckpt_dir": args.ckpt_dir,
            "hf_dir": args.hf_dir,
            "max_length": args.max_length,
            "max_samples": args.max_samples,
            **metadata_extra,
        })

    for key, loader_arg in items:
        out_path = npz_path(output_dir, key)

        if result_done(output_dir, key, accum is not None, args.store_individual,
                      persistent_output_dir) and not args.overwrite:
            if rank == 0:
                print(f"Skipping {key} (already done)")
            continue

        dataset = load_dataset(loader_arg, args.max_length, args.max_samples)
        maxpos = max(len(tokens) for tokens, _ in dataset) - 1  # same on every rank (all load the full list)
        if rank == 0:
            print(f"{key}: {len(dataset)} sequences, maxpos={maxpos}")

        pos_sum, pos_sqsum, pos_cnt = run_inference(
            backend, dataset, maxpos, rank, args.softmax_chunk, device, desc=key, accum=accum,
            individual=individual,
        )

        if rank == 0:
            save_npz(out_path, pos_sum, pos_sqsum, pos_cnt)
            print(f"  done {key} -> {out_path}")
            if accum is not None:
                state_file = state_npz_path(output_dir, key)
                accum.save(state_file, SEQ_LEN)
                print(f"  done {key} state -> {state_file}")
            if individual is not None:
                indiv_file = individual_path(output_dir, key)
                individual.save(indiv_file)
                print(f"  done {key} individual -> {indiv_file}")
        dist.barrier()

    if rank == 0:
        print(f"\nAll done. Results in: {output_dir}")
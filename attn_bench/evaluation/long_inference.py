"""
Shared engine for per-position long-sequence loss inference: one teacher-forced
forward pass per sequence, recording per-token NLL past the training seq_len (where
attention is expected to degrade but state-carrying models might not).

Dataset-agnostic: takes a `dataset` of plain token-id lists and a `key` string
naming the output file. Callers (long_gutenberg_inference.py, long_fineweb_inference.py)
only differ in how they build token-id sequences and what `key`s/loader args they pass.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from attn_bench.evaluation.gdn_state_norm import install_state_norm_hooks
from attn_bench.evaluation.inference_common import load_megatron_model
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
def per_position_nll(model, seq_ids: torch.Tensor, softmax_chunk: int) -> torch.Tensor:
    """One forward over the whole sequence; return per-position NLL [S-1] (float32).

    nll[k] is the loss for predicting token at index k+1, i.e. position k measured from
    the sample start (index 0 = BOS, so position 0 = first sample token c0).
    The -log_softmax is computed chunked over the sequence dim: materializing the full
    float logit tensor at 30k+ tokens x 128k vocab would OOM.
    """
    inputs = seq_ids[:, :-1]
    labels = seq_ids[:, 1:]
    S1 = inputs.shape[1]
    device = seq_ids.device
    pos = torch.arange(S1, dtype=torch.long, device=device).unsqueeze(0)

    # runtime_gather_output gathers the vocab-parallel logits under TP (no-op at TP=1).
    logits = model(inputs, pos, attention_mask=None, runtime_gather_output=True)  # [1, S1, V] (bf16)

    nll = torch.empty(S1, dtype=torch.float32, device=device)
    for c0 in range(0, S1, softmax_chunk):
        c1 = min(c0 + softmax_chunk, S1)
        logp = -F.log_softmax(logits[:, c0:c1, :].float(), dim=-1)
        nll[c0:c1] = logp.gather(2, labels[:, c0:c1].unsqueeze(-1)).squeeze(-1).squeeze(0)
    del logits
    return nll


### RUN ###

def run_inference(model, dataset, maxpos, rank, softmax_chunk, device, desc="", accum=None):
    """Accumulate per-position NLL sum / sqsum / count over this rank's shard.

    Sharding and pooling are over the *data-parallel* group, not WORLD, so it is correct at
    any tensor-parallel size. At TP=1 the DP group is WORLD (plain stride over all ranks); at
    TP=world_size the DP group is a single rank (no stride, no cross-rank reduce -- each rank
    already holds the full loss because TP gathers it). If accum is given, the GDN state norms
    are pooled the same way.
    """
    dp_rank = mpu.get_data_parallel_rank()
    dp_size = mpu.get_data_parallel_world_size()
    dp_group = mpu.get_data_parallel_group()

    pos_sum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_sqsum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_cnt = torch.zeros(maxpos, dtype=torch.float64, device=device)
    if accum is not None:
        accum.reset_bucket(maxpos)

    # Each DP rank strides its own shard; only global rank 0 prints the bar.
    shard = range(dp_rank, len(dataset), dp_size)
    for idx in tqdm(shard, desc=desc, disable=(rank != 0), mininterval=5.0):
        seq = torch.tensor(dataset[idx], dtype=torch.long, device=device).unsqueeze(0)
        if accum is not None:
            accum.reset_sequence()
        nll = per_position_nll(model, seq, softmax_chunk).double()
        if accum is not None:
            accum.accumulate()  # reads the norms the GDN wrappers recorded during the forward
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

def config_name(max_length: int | None, max_samples: int | None, tensor_parallel: int = 1) -> str:
    # Encodes the params that change the numbers, so different runs land in different
    # folders and the "already done" check doubles as a validity check. tp>1 gets a suffix
    # so a TP run does not clash with the data-parallel (TP=1) run at the same length.
    samples = max_samples if max_samples is not None else "all"
    length = max_length if max_length is not None else "full"
    tp = f"_tp{tensor_parallel}" if tensor_parallel > 1 else ""
    return f"{samples}_samples_{length}_tokens{tp}"


def config_dir(experiment_path: str, max_length: int | None, max_samples: int | None,
               tensor_parallel: int = 1) -> Path:
    return Path(experiment_path) / config_name(max_length, max_samples, tensor_parallel)


def npz_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"{key}.npz"


def state_npz_path(output_dir: Path, key: str) -> Path:
    return output_dir / f"{key}_state.npz"


def result_done(output_dir: Path, key: str, log_state_norm: bool) -> bool:
    # Done when the NLL file exists -- and, when logging state norms, its state file too
    # (so turning --log-state-norm on for a finished model re-runs to fill it in).
    if not npz_path(output_dir, key).exists():
        return False
    if log_state_norm and not state_npz_path(output_dir, key).exists():
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
    p.add_argument("--ckpt-dir", required=True, help="torch_dist checkpoint directory")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--experiment-path", required=True, help="Output root")
    p.add_argument("--max-length", type=int, default=None, help="Cap each sequence to this many tokens.")
    p.add_argument("--max-samples", type=int, default=max_samples_default,
                   help="Randomly subsample to this many sequences (fixed seed, for testing/calibration "
                        f"or to avoid running on all available data). Default: {max_samples_default or 'no cap'}.")
    p.add_argument("--tensor-parallel", type=int, default=1,
                   help="Tensor-parallel size. >1 shards the attention heads across GPUs (less memory "
                        "per GPU, no data-parallel throughput), letting unfused-attention models run longer.")
    p.add_argument("--softmax-chunk", type=int, default=4096,
                   help="How many positions to run log_softmax over at once. Smaller uses less memory, "
                        "same result. Does not change the NLL values.")
    p.add_argument("--log-state-norm", action="store_true",
                   help="For GDN models, also write recurrent-state norms to <key>_state.npz. "
                        "No effect on attention models (they have no state to log).")
    p.add_argument("--state-chunk", type=int, default=128,
                   help="Read the GDN state every this many tokens (only used with --log-state-norm).")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if <key>.npz already exists")
    p.add_argument("--container-env", default=None, help="Container/env name, recorded for provenance")
    p.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                   help="Extra Megatron args forwarded to the checkpoint loader (e.g. --attention-output-gate)")


### MAIN ###

def run_main(args, items: list[tuple[str, object]], load_dataset: Callable, metadata_extra: dict) -> None:
    """Shared driver for a long-sequence inference script's main().

    `items` is a list of (key, loader_arg) pairs -- one per output file this invocation may
    produce (Gutenberg: one per requested repetition; fineweb: a single item). `load_dataset`
    is called as `load_dataset(loader_arg, args.max_length, args.max_samples) -> list[list[int]]`.
    """
    output_dir = config_dir(args.experiment_path, args.max_length, args.max_samples, args.tensor_parallel)

    if not args.overwrite and items and all(
        result_done(output_dir, key, args.log_state_norm) for key, _ in items
    ):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"All requested results already present in {output_dir} -- skipping checkpoint load.")
        return

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args,
                                tensor_parallel=args.tensor_parallel)
    rank = dist.get_rank()
    device = next(model.parameters()).device

    accum = None
    if args.log_state_norm:
        accum = install_state_norm_hooks(model, args.state_chunk, device)
        if accum is None and rank == 0:
            print("--log-state-norm set but model has no GatedDeltaNet layers; skipping state norms.")

    if rank == 0:
        write_run_metadata(output_dir, {
            "container_env": args.container_env,
            "ckpt_dir": args.ckpt_dir,
            "max_length": args.max_length,
            "max_samples": args.max_samples,
            "tensor_parallel": args.tensor_parallel,
            **metadata_extra,
        })

    for key, loader_arg in items:
        out_path = npz_path(output_dir, key)

        if result_done(output_dir, key, accum is not None) and not args.overwrite:
            if rank == 0:
                print(f"Skipping {key} (already done)")
            continue

        dataset = load_dataset(loader_arg, args.max_length, args.max_samples)
        maxpos = max(len(s) for s in dataset) - 1  # same on every rank (all load the full list)
        if rank == 0:
            print(f"{key}: {len(dataset)} sequences, maxpos={maxpos}")

        pos_sum, pos_sqsum, pos_cnt = run_inference(
            model, dataset, maxpos, rank, args.softmax_chunk, device, desc=key, accum=accum,
        )

        if rank == 0:
            save_npz(out_path, pos_sum, pos_sqsum, pos_cnt)
            print(f"  done {key} -> {out_path}")
            if accum is not None:
                state_file = state_npz_path(output_dir, key)
                accum.save(state_file, SEQ_LEN)
                print(f"  done {key} state -> {state_file}")
        dist.barrier()

    if rank == 0:
        print(f"\nAll done. Results in: {output_dir}")
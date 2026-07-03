"""
Long Gutenberg inference: per-position loss on sequences past the training seq_len.

Feeds each long Gutenberg excerpt as [BOS, sample, suffix] (no prefix) in a single
teacher-forced forward pass and records the per-position NLL (= per-token cross-entropy
loss), averaged over sequences within each repetition bucket. The region past the
training seq_len (the suffix, position >= sample_len) is what we want to look at:
attention models are expected to blow up there, state-carrying GDN maybe not.

Usage (via torchrun; work is split across ranks by index striding):
    torchrun --nproc_per_node=4 attn_bench/evaluation/long_gutenberg_inference.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $OUT_DIR \
        --data-folder $GUTENBERG_LONG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 \
        --max-length 32768
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from attn_bench.evaluation.inference_common import BOS_TOKEN_ID, find_rep_paths, load_megatron_model

SEQ_LEN = 8192  # training sequence length; suffix (position >= sample_len) is the extrapolation region


### DATA ###

def load_long_sequence(path: Path, max_length: int | None, max_samples: int | None) -> list:
    """Build [BOS, sample, suffix] sequences from a long-context bucket file.

    The record stores input_ids = [BOS] + prefix + sample + suffix + [EOS] with
    sample_offset marking the first sample token. We drop the prefix (start at the
    sample) and the trailing EOS, prepend a fresh BOS, then cap at max_length.
    """
    sequences = []
    with open(path) as f:
        for line in f:
            if max_samples is not None and len(sequences) >= max_samples:
                break
            record = json.loads(line)
            sample_start_index = record["sample_offset"] 
            long_sample_len = record["sample_len"] + record["extra_suffix_len"]
            # input_ids = [bos_id] + extra_prefix + content + extra_suffix + [eos_id]
            long_sample_tokens = record["input_ids"][sample_start_index: sample_start_index + long_sample_len]
            assert len(long_sample_tokens) == long_sample_len, (
                f"{path.name}: short record ({len(long_sample_tokens)} != {long_sample_len})"
            )
            sequence_tokens = [BOS_TOKEN_ID] + long_sample_tokens
            if max_length is not None:
                sequence_tokens = sequence_tokens[:max_length]
            sequences.append(sequence_tokens)
    return sequences


### FORWARD ###

@torch.no_grad()
def per_position_nll(model, seq_ids: torch.Tensor, seq_chunk: int) -> torch.Tensor:
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

    logits = model(inputs, pos, attention_mask=None)  # [1, S1, V] (bf16)

    nll = torch.empty(S1, dtype=torch.float32, device=device)
    for c0 in range(0, S1, seq_chunk):
        c1 = min(c0 + seq_chunk, S1)
        logp = -F.log_softmax(logits[:, c0:c1, :].float(), dim=-1)
        nll[c0:c1] = logp.gather(2, labels[:, c0:c1].unsqueeze(-1)).squeeze(-1).squeeze(0)
    del logits
    return nll


### RUN ###

def run_rep(model, dataset, maxpos, rank, world_size, seq_chunk, device, desc=""):
    """Accumulate per-position NLL sum / sqsum / count over this rank's shard.

    Sharding is a plain stride (rank, rank+world_size, ...): no padding, no duplicate
    samples, so the all_reduce below is an exact pooled average.
    """
    pos_sum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_sqsum = torch.zeros(maxpos, dtype=torch.float64, device=device)
    pos_cnt = torch.zeros(maxpos, dtype=torch.float64, device=device)

    # Every rank strides its own shard; only rank 0 prints the bar (they run in lockstep).
    shard = range(rank, len(dataset), world_size)
    for idx in tqdm(shard, desc=desc, disable=(rank != 0), mininterval=5.0):
        seq = torch.tensor(dataset[idx], dtype=torch.long, device=device).unsqueeze(0)
        nll = per_position_nll(model, seq, seq_chunk).double()
        L = nll.shape[0]
        pos_sum[:L] += nll
        pos_sqsum[:L] += nll * nll
        pos_cnt[:L] += 1.0
        del seq, nll
        torch.cuda.empty_cache()

    for t in (pos_sum, pos_sqsum, pos_cnt):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return pos_sum, pos_sqsum, pos_cnt


def save_rep(rep_file: Path, pos_sum, pos_sqsum, pos_cnt):
    cnt = pos_cnt.cpu().numpy()
    keep = cnt > 0
    rep_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        rep_file,
        position=np.arange(len(cnt))[keep],
        nll_sum=pos_sum.cpu().numpy()[keep],
        nll_sqsum=pos_sqsum.cpu().numpy()[keep],
        count=cnt[keep],
        seq_len=SEQ_LEN,
    )


### CLI ###

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True, help="torch_dist checkpoint directory")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--experiment-path", required=True, help="Output root")
    p.add_argument("--data-folder", required=True, help="Directory of long-context rep_*_token.jsonl files")
    p.add_argument("--repetitions", default="0", help="Comma-separated buckets, e.g. 0,1,2,4,8,16,32,64,128,256")
    p.add_argument("--max-length", type=int, default=None,
                   help="Cap each sequence to this many tokens (BOS+sample+suffix). Default: no cap (full suffix).")
    p.add_argument("--max-samples", type=int, default=None, help="Cap sequences per bucket (for testing/calibration)")
    p.add_argument("--seq-chunk", type=int, default=4096, help="Sequence chunk for the chunked log_softmax (memory knob)")
    p.add_argument("--overwrite", action="store_true", help="Recompute buckets whose rep_{R}.npz already exists")
    p.add_argument("--container-env", default=None, help="Container/env name, recorded for provenance")
    p.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None,
                   help="Extra Megatron args forwarded to the checkpoint loader (e.g. --attention-output-gate)")
    return p.parse_args()


def write_run_metadata(output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "container_env": args.container_env,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "ckpt_dir": args.ckpt_dir,
            "max_length": args.max_length,
            "max_samples": args.max_samples,
            "repetitions": args.repetitions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


def config_name(max_length: int | None, max_samples: int | None) -> str:
    # Encodes the params that change the numbers, so different runs land in different
    # folders and the "already done" check doubles as a validity check.
    samples = max_samples if max_samples is not None else "all"
    length = max_length if max_length is not None else "full"
    return f"{samples}_samples_{length}_tokens"


def config_dir(experiment_path: str, max_length: int | None, max_samples: int | None) -> Path:
    return Path(experiment_path) / config_name(max_length, max_samples)


def rep_npz_path(output_dir: Path, rep: int) -> Path:
    return output_dir / f"rep_{rep}.npz"


def all_reps_done(args) -> bool:
    """True if every requested bucket's rep_{R}.npz is already on disk.

    Checked before the (expensive) checkpoint load, from the filesystem only, so every
    rank reaches the same verdict independently -- no process group, no barrier to
    deadlock. Lets a redundantly-submitted job no-op cheaply.
    """
    output_dir = config_dir(args.experiment_path, args.max_length, args.max_samples)
    paths = find_rep_paths(Path(args.data_folder), {int(r) for r in args.repetitions.split(",")})
    if not paths:
        return False
    return all(rep_npz_path(output_dir, int(p.stem.split("_")[1])).exists() for p in paths)


def main():
    args = parse_args()

    if not args.overwrite and all_reps_done(args):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"All requested buckets already present in {config_dir(args.experiment_path, args.max_length, args.max_samples)} -- skipping checkpoint load.")
        return

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = next(model.parameters()).device

    output_dir = config_dir(args.experiment_path, args.max_length, args.max_samples)
    if rank == 0:
        write_run_metadata(output_dir, args)

    reps = {int(r) for r in args.repetitions.split(",")}
    paths = find_rep_paths(Path(args.data_folder), reps)

    for path in paths:
        rep = int(path.stem.split("_")[1])
        rep_file = rep_npz_path(output_dir, rep)

        if rep_file.exists() and not args.overwrite:
            if rank == 0:
                print(f"Skipping rep={rep} (already done)")
            continue

        dataset = load_long_sequence(path, args.max_length, args.max_samples)
        maxpos = max(len(s) for s in dataset) - 1  # same on every rank (all load the full list)
        if rank == 0:
            print(f"rep={rep}: {len(dataset)} sequences, maxpos={maxpos}")

        pos_sum, pos_sqsum, pos_cnt = run_rep(
            model, dataset, maxpos, rank, world_size, args.seq_chunk, device,
            desc=f"rep={rep}",
        )

        if rank == 0:
            save_rep(rep_file, pos_sum, pos_sqsum, pos_cnt)
            print(f"  done rep={rep} -> {rep_file}")
        dist.barrier()

    if rank == 0:
        print(f"\nAll buckets done. Results in: {output_dir}")


if __name__ == "__main__":
    main()

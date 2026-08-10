"""
Backfills existing inference results with sample_idx and full per-position
ref_nll/gen_nll/p_z, matching Step 1's (megatron_inference_sparse.py) output format.
One-time migration, kept separate from the live path. Doesn't compute Rouge-L/lcs_norm/
TTR/token_acc/divergence_point -- that's compute_memorization_metrics.py's job.

Usage (via torchrun, one model/experiment per invocation -- see
megatron_inference_backfill_all.sh for selecting several):
    torchrun --nproc_per_node=4 attn_bench/evaluation/megatron_inference_backfill.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $MEM_DIR \
        --data-folder $GUTENBERG_JSONL_DIR \
        --batch-size 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist

from attn_bench.evaluation.inference_common import (
    BOS_TOKEN_ID, discover_all_offset_prefix_suffix_dirs, load_megatron_model,
    sample_idx_per_rank)
from attn_bench.evaluation.megatron_inference import (compute_nll_pz_stats,
                                                      write_run_metadata)


def parse_offset_prefix_suffix(dirname: str) -> tuple:
    parts = dirname.split("_")
    return int(parts[1]), int(parts[3]), int(parts[5])


def record_needs_backfill(record: dict) -> bool:
    return "sample_idx" not in record or "ref_nll" not in record


def _filter_suffix_dirs(experiment_path: Path, offset: int | None, prefix_length: int | None) -> list:
    dirs = discover_all_offset_prefix_suffix_dirs(experiment_path)
    if offset is None and prefix_length is None:
        return dirs
    kept = []
    for suffix_dir in dirs:
        dir_offset, dir_prefix, _ = parse_offset_prefix_suffix(suffix_dir.name)
        if offset is not None and dir_offset != offset:
            continue
        if prefix_length is not None and dir_prefix != prefix_length:
            continue
        kept.append(suffix_dir)
    return kept


def already_backfilled(experiment_path: Path, offset: int | None = None, prefix_length: int | None = None) -> bool:
    """Cheap check before loading the checkpoint: peek at the first record of each rep
    dir's rank0.jsonl (records within one file are always backfilled together as a unit
    by this script, so the first record is representative)."""
    for suffix_dir in _filter_suffix_dirs(experiment_path, offset, prefix_length):
        for rep_dir in sorted(d for d in suffix_dir.iterdir() if d.is_dir() and d.name.startswith("rep_")):
            rank0 = rep_dir / "rank0.jsonl"
            if not rank0.exists():
                continue
            with open(rank0) as f:
                first_line = f.readline()
            if first_line.strip() and record_needs_backfill(json.loads(first_line)):
                return False
    return True


def backfill_rep_dir(model, rep_dir: Path, offset: int, prefix_length: int, suffix_length: int,
                     data_folder: Path, batch_size: int) -> bool:
    """Rewrites rep_dir's rank*.jsonl in place. Returns False if nothing needed backfilling."""
    rank_files = sorted(rep_dir.glob("rank*.jsonl"))
    if not rank_files:
        return False

    all_records = []
    for rank_file in rank_files:
        with open(rank_file) as f:
            all_records.append([json.loads(line) for line in f if line.strip()])

    if not any(record_needs_backfill(r) for recs in all_records for r in recs):
        return False

    world_size = len(rank_files)
    rep = int(rep_dir.name.split("_")[1])
    rep_path = data_folder / f"rep_{rep}_token.jsonl"
    with open(rep_path) as f:
        dataset_len = sum(1 for _ in f)
    per_rank_indices = sample_idx_per_rank(world_size, dataset_len)

    needs_bos = offset > 0
    device = next(model.parameters()).device

    for rank, recs in enumerate(all_records):
        indices = per_rank_indices[rank]
        for start in range(0, len(recs), batch_size):
            batch = recs[start:start + batch_size]
            batch_indices = indices[start:start + batch_size]
            B = len(batch)

            prefix_t = torch.tensor([r["prefix"] for r in batch], dtype=torch.long, device=device)
            true_t = torch.tensor([r["true_suffix"] for r in batch], dtype=torch.long, device=device)
            gen_t = torch.tensor([r["generated_suffix"] for r in batch], dtype=torch.long, device=device)

            true_full = torch.cat([prefix_t, true_t], dim=1)
            gen_full = torch.cat([prefix_t, gen_t], dim=1)
            if needs_bos:
                bos = torch.full((B, 1), BOS_TOKEN_ID, dtype=torch.long, device=device)
                true_full = torch.cat([bos, true_full], dim=1)
                gen_full = torch.cat([bos, gen_full], dim=1)

            ref_nll, gen_nll, p_z, ref_mean, ref_std, ref_ppl, gen_mean, gen_std, gen_ppl = \
                compute_nll_pz_stats(model, true_full, gen_full, suffix_length)

            for i, rec in enumerate(batch):
                rec["sample_idx"] = batch_indices[i]
                rec["ref_nll"] = ref_nll[i].cpu().tolist()
                rec["gen_nll"] = gen_nll[i].cpu().tolist()
                rec["p_z_logprob"] = p_z[i].cpu().tolist()
                # Existing flat fields (already computed by the original run) stay as-is --
                # only fill them in if somehow missing.
                rec.setdefault("nll_mean", gen_mean[i].item())
                rec.setdefault("nll_std", gen_std[i].item())
                rec.setdefault("perplexity", gen_ppl[i].item())
                rec.setdefault("ref_nll_mean", ref_mean[i].item())
                rec.setdefault("ref_nll_std", ref_std[i].item())
                rec.setdefault("ref_perplexity", ref_ppl[i].item())

            torch.cuda.empty_cache()

    for rank_file, recs in zip(rank_files, all_records):
        # Write to a temp file and rename into place, so a crash mid-write can never leave
        # rank_file truncated -- readers only ever see the fully-old or fully-new content.
        tmp_file = rank_file.with_suffix(rank_file.suffix + ".tmp")
        with open(tmp_file, "w") as f:
            for r in recs:
                json.dump(r, f)
                f.write("\n")
        tmp_file.replace(rank_file)

    return True


def run_backfill(model, args, rank: int, world_size: int):
    """Each rank has its own full model copy (tensor_parallel=1, same as Step 1) --
    no cross-rank collectives needed for the forward passes, so work is split by rep_dir
    across ranks (round-robin) rather than every rank redundantly processing everything.
    Different ranks always handle disjoint rep_dirs, so there's no write race.
    """
    experiment_path = Path(args.experiment_path)
    data_folder = Path(args.data_folder)

    tasks = []
    for suffix_dir in _filter_suffix_dirs(experiment_path, args.offset, args.prefix_length):
        offset, prefix_length, suffix_length = parse_offset_prefix_suffix(suffix_dir.name)
        for rep_dir in sorted(d for d in suffix_dir.iterdir() if d.is_dir() and d.name.startswith("rep_")):
            tasks.append((rep_dir, offset, prefix_length, suffix_length))

    touched_dirs = set()
    for rep_dir, offset, prefix_length, suffix_length in tasks[rank::world_size]:
        print(f"[rank {rank}] Backfilling {rep_dir} ...")
        changed = backfill_rep_dir(model, rep_dir, offset, prefix_length, suffix_length,
                                   data_folder, args.batch_size)
        if changed:
            touched_dirs.add(rep_dir.parent)  # the offset_..._suffix_... dir, where metadata lives
        print(f"[rank {rank}]   {'backfilled' if changed else 'already up to date'}")

    # Metadata lives per offset_..._suffix_... dir, same place Step 1 writes it -- gather
    # which dirs each rank actually changed so rank 0 can log one entry per dir touched.
    if dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, touched_dirs)
        all_touched = set().union(*gathered)
    else:
        all_touched = touched_dirs

    if rank == 0:
        for suffix_dir in sorted(all_touched):
            write_run_metadata(suffix_dir, args, world_size=world_size, action="backfill")
        print("Backfill pass complete." if all_touched else "Nothing needed backfilling.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--experiment-path", required=True, help="Same root as the live inference runs (MEM_DIR)")
    parser.add_argument("--data-folder", required=True, help="Directory of rep_*_token.jsonl files")
    parser.add_argument("--offset", type=int, default=None, help="Restrict to one offset (default: all)")
    parser.add_argument("--prefix-length", type=int, default=None, help="Restrict to one prefix length (default: all)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--container-env", default=None)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Unused by backfill itself; kept so write_run_metadata's shape matches Step 1's.")
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if already_backfilled(Path(args.experiment_path), args.offset, args.prefix_length):
        print(f"{args.experiment_path}: already fully backfilled -- skipping checkpoint load.")
        return

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    run_backfill(model, args, rank, world_size)


if __name__ == "__main__":
    main()

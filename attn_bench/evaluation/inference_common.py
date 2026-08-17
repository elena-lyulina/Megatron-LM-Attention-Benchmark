"""Shared dataset/path helpers for the Megatron-native inference scripts (no metric or
model-loading dependencies -- see inference_backend.py for that).

Kept free of verbatim_eval/PDM imports so scripts that only need these don't pull in the
Rouge/LCS stack.
"""

from __future__ import annotations

import json
from pathlib import Path

BOS_TOKEN_ID = 128000  # Llama-3 beginning-of-sequence token


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
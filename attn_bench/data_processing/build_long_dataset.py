"""
Build a token-budget-capped training dataset out of the longest documents in a
tokenized Megatron dataset pool.

Ranks every document in --data-folder by token length, descending, and keeps
documents -- longest first -- until their cumulative length reaches --budget
tokens. Kept documents are written verbatim (token ids + document boundaries
intact) to a single Megatron .bin/.idx shard under --output-folder, so the
result drops straight into a pretrain blend like any other tokenized dataset.

Usage:
    python build_long_dataset.py \
        --data-folder /path/to/tokenized/fineweb-edu-dedup-160B-datatrove \
        --budget 40000000000 \
        --output-folder /path/to/tokenized/fineweb-edu-dedup-160B-datatrove_long40B
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from attn_bench.data_processing.dataset_doc_stats import shard_prefixes
from megatron.core.datasets.indexed_dataset import (IndexedDataset,
                                                    IndexedDatasetBuilder)

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def doc_lengths_for_prefix(prefix: str):
    """Per-document token lengths and dtype from a single shard's index."""
    ds = IndexedDataset(prefix, mmap=True)
    seq_lengths = ds.index.sequence_lengths.astype(np.int64)
    doc_indices = ds.index.document_indices
    cumsum = np.concatenate([[0], np.cumsum(seq_lengths)])
    lengths = cumsum[doc_indices[1:]] - cumsum[doc_indices[:-1]]
    dtype = ds.index.dtype
    del ds
    return lengths, dtype


def select_top_docs(prefixes: list, budget: int):
    """Rank every document across all shards by length, descending, and keep the
    longest ones until their cumulative length reaches budget.

    Returns (prefix_idx, local_doc_id, lengths, dtype), all sorted by length descending,
    restricted to the kept documents.
    """
    all_lengths, all_prefix_idx, all_local_id = [], [], []
    dtype = None
    for p_idx, prefix in enumerate(prefixes):
        lengths, d = doc_lengths_for_prefix(prefix)
        if dtype is None:
            dtype = d
        else:
            assert d == dtype, f"dtype mismatch across shards: {dtype} vs {d} ({prefix})"
        all_lengths.append(lengths)
        all_prefix_idx.append(np.full(len(lengths), p_idx, dtype=np.int32))
        all_local_id.append(np.arange(len(lengths), dtype=np.int64))

    lengths = np.concatenate(all_lengths)
    prefix_idx = np.concatenate(all_prefix_idx)
    local_id = np.concatenate(all_local_id)

    order = np.argsort(lengths, kind="stable")[::-1]
    lengths, prefix_idx, local_id = lengths[order], prefix_idx[order], local_id[order]

    cum = np.cumsum(lengths)
    total_available = int(cum[-1]) if len(cum) else 0
    if total_available < budget:
        raise ValueError(
            f"budget ({budget:,}) exceeds the total tokens available in {len(lengths):,} "
            f"documents ({total_available:,})"
        )
    cutoff = int(np.searchsorted(cum, budget))  # first index with cum >= budget

    keep = slice(0, cutoff + 1)
    return prefix_idx[keep], local_id[keep], lengths[keep], dtype


def write_dataset(prefixes: list, prefix_idx: np.ndarray, local_id: np.ndarray, dtype, out_prefix: str) -> int:
    """Write the selected documents to out_prefix.bin/.idx, one add_document call each."""
    builder = IndexedDatasetBuilder(out_prefix + ".bin", dtype=dtype)

    # group by shard, sorted by local doc id within each shard, for on-disk locality
    order = np.lexsort((local_id, prefix_idx))
    prefix_idx, local_id = prefix_idx[order], local_id[order]

    total_written = 0
    cur_p_idx, ds = None, None
    for p_idx, doc_id in zip(prefix_idx.tolist(), local_id.tolist()):
        if p_idx != cur_p_idx:
            del ds
            ds = IndexedDataset(prefixes[p_idx], mmap=True)
            cur_p_idx = p_idx
        start, end = int(ds.index.document_indices[doc_id]), int(ds.index.document_indices[doc_id + 1])
        tokens = np.concatenate(ds[start:end])
        item_lengths = ds.index.sequence_lengths[start:end].tolist()
        builder.add_document(tokens, item_lengths)
        total_written += 1
    del ds

    builder.finalize(out_prefix + ".idx")
    return total_written


def stats_from_lengths(lengths: np.ndarray) -> dict:
    pct = np.percentile(lengths, PERCENTILES)
    return {
        "documents": int(len(lengths)),
        "total_tokens": int(lengths.sum()),
        "mean": float(lengths.mean()),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "percentiles": {str(p): int(v) for p, v in zip(PERCENTILES, pct)},
    }


def print_stats(stats: dict) -> None:
    print(f"  documents:    {stats['documents']:,}")
    print(f"  total tokens: {stats['total_tokens']:,}")
    print(f"  mean:         {stats['mean']:,.1f}")
    print(f"  min / max:    {stats['min']:,} / {stats['max']:,}")
    print("  percentiles:  " + "  ".join(f"p{p}={v:,}" for p, v in stats["percentiles"].items()))


def write_run_metadata(output_folder: str, args: argparse.Namespace, stats: dict) -> None:
    with open(os.path.join(output_folder, "run_metadata.json"), "w") as f:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_folder": args.data_folder,
            "budget": args.budget,
            "stats": stats,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", required=True, help="Tokenized Megatron dataset dir to select documents from")
    parser.add_argument("--budget", type=int, required=True, help="Target total tokens, e.g. 40000000000 for 40B")
    parser.add_argument("--output-folder", required=True, help="Where to write dataset.bin/.idx + run_metadata.json")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    prefixes = shard_prefixes(args.data_folder)
    if not prefixes:
        raise FileNotFoundError(f"No .bin shards found under {args.data_folder}")
    print(f"{len(prefixes)} shard(s) under {args.data_folder}")
    print(f"budget: {args.budget:,} tokens")

    prefix_idx, local_id, lengths, dtype = select_top_docs(prefixes, args.budget)
    print(f"selected {len(lengths):,} documents, length threshold {int(lengths.min()):,} tokens")

    out_prefix = os.path.join(args.output_folder, "dataset")
    total_written = write_dataset(prefixes, prefix_idx, local_id, dtype, out_prefix)
    print(f"wrote {total_written:,} documents -> {out_prefix}.bin/.idx")

    stats = stats_from_lengths(lengths)
    print_stats(stats)
    write_run_metadata(args.output_folder, args, stats)
    print(f"run_metadata.json -> {os.path.join(args.output_folder, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
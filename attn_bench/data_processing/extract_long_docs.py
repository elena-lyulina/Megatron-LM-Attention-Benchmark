"""
Extract long documents from a tokenized Megatron dataset.

Pulls every document whose own token length falls in [--min-length, --max-length)
directly off the .idx/.bin index and writes it verbatim to a single jsonl file. 

Usage:
    python attn_bench/data_processing/extract_long_docs.py \
        --data-folder /path/to/tokenized/fineweb-edu-dedup-160B-datatrove_0.5_unseen \
        --min-length 24576 --max-length 32768
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from attn_bench.data_processing.dataset_doc_stats import shard_prefixes
from attn_bench.evaluation.inference_common import BOS_TOKEN_ID
from megatron.core.datasets.indexed_dataset import IndexedDataset

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def extract_from_prefix(prefix: str, lo: int, hi: int):
    """Yield (doc_id, input_ids, length, has_bos) for every document in [lo, hi) tokens in one shard."""
    ds = IndexedDataset(prefix, mmap=True)
    seq_lengths = ds.index.sequence_lengths.astype(np.int64)
    doc_indices = ds.index.document_indices
    cumsum = np.concatenate([[0], np.cumsum(seq_lengths)])
    lengths = cumsum[doc_indices[1:]] - cumsum[doc_indices[:-1]]
    keep = np.nonzero((lengths >= lo) & (lengths < hi))[0]
    for d in keep:
        start, end = int(doc_indices[d]), int(doc_indices[d + 1])
        input_ids = np.concatenate(ds[start:end]).tolist()
        has_bos = bool(input_ids) and input_ids[0] == BOS_TOKEN_ID
        yield int(d), input_ids, int(lengths[d]), has_bos
    del ds


def print_stats(lengths: np.ndarray) -> None:
    if len(lengths) == 0:
        print("  no documents matched")
        return
    pct = np.percentile(lengths, PERCENTILES)
    print(f"  documents:    {len(lengths):,}")
    print(f"  total tokens: {int(lengths.sum()):,}")
    print(f"  mean:         {lengths.mean():,.1f}")
    print(f"  min / max:    {lengths.min():,} / {lengths.max():,}")
    print("  percentiles:  " + "  ".join(f"p{p}={int(v):,}" for p, v in zip(PERCENTILES, pct)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", required=True, help="Tokenized Megatron dataset dir (or .bin/prefix path)")
    parser.add_argument("--min-length", type=int, required=True, help="Keep docs with length >= this (inclusive)")
    parser.add_argument("--max-length", type=int, required=True, help="Keep docs with length < this (exclusive)")
    parser.add_argument("--output-dir", default=None, help="Default: <data-folder>_long/ next to the source")
    args = parser.parse_args()

    data_folder = args.data_folder.rstrip("/")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{data_folder}_long")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = output_dir / f"long_{args.min_length}_{args.max_length}.jsonl"
    out_stats = output_dir / f"long_{args.min_length}_{args.max_length}_stats.json"

    prefixes = shard_prefixes(data_folder)
    if not prefixes:
        raise FileNotFoundError(f"No .bin shards found under {data_folder}")
    print(f"{len(prefixes)} shard(s) under {data_folder}")
    print(f"range: [{args.min_length:,}, {args.max_length:,})")

    lengths = []
    no_bos = 0
    with open(out_jsonl, "w") as f:
        for prefix in prefixes:
            shard_n = 0
            for doc_id, input_ids, length, has_bos in extract_from_prefix(prefix, args.min_length, args.max_length):
                f.write(json.dumps({
                    "doc_id": f"{os.path.basename(prefix)}:{doc_id}",
                    "input_ids": input_ids,
                    "length": length,
                }) + "\n")
                lengths.append(length)
                no_bos += not has_bos
                shard_n += 1
            print(f"  {os.path.basename(prefix)}: {shard_n} docs")

    lengths = np.array(lengths, dtype=np.int64)
    print(f"\nTotal: {len(lengths):,} documents -> {out_jsonl}")
    print_stats(lengths)
    if no_bos:
        print(f"  WARNING: {no_bos:,} / {len(lengths):,} documents do not start with BOS ({BOS_TOKEN_ID})")

    stats = {
        "data_folder": str(data_folder),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "documents": int(len(lengths)),
        "no_bos": no_bos,
        "total_tokens": int(lengths.sum()) if len(lengths) else 0,
        "mean": float(lengths.mean()) if len(lengths) else None,
        "min": int(lengths.min()) if len(lengths) else None,
        "max": int(lengths.max()) if len(lengths) else None,
        "percentiles": (
            {str(p): int(v) for p, v in zip(PERCENTILES, np.percentile(lengths, PERCENTILES))}
            if len(lengths) else {}
        ),
    }
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"stats -> {out_stats}")


if __name__ == "__main__":
    main()
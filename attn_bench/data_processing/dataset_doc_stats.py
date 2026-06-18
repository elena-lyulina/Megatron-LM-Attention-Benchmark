"""
Compute per-document token-length distributions for tokenized Megatron datasets.

Reads only the .idx files (no .bin data, no GPU) via the Megatron IndexedDataset
index, so it is cheap enough to run on a login node. For each dataset directory it
globs all .bin/.idx shards, computes every document's length in tokens, saves the
raw lengths to a .npy file, and prints summary stats.

The .npy files are meant to be transferred locally and plotted in
attn_bench/notebooks/doc_lengths.ipynb.

Usage:
    python dataset_doc_stats.py
    python dataset_doc_stats.py --datasets /path/to/dataset_dir_a /path/to/dataset_dir_b
    python dataset_doc_stats.py --out-dir /users/$USER/store/datasets/analysis/doc-lengths
"""

import argparse
import glob
import os

import numpy as np

from megatron.core.datasets.indexed_dataset import IndexedDataset

USER = os.environ.get("USER", "")

DEFAULT_DATASETS = [
    f"/users/{USER}/store/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25",
    f"/users/{USER}/store/datasets/tokenized/gutenberg_rep_1_256",
]
DEFAULT_OUT_DIR = f"/users/{USER}/store/datasets/analysis/doc-lengths"

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def shard_prefixes(path: str) -> list:
    """Return sorted .bin/.idx prefixes under a dataset dir (or the prefix itself)."""
    # mirror the slurm: find -L <dir> -name '*.bin' | sed 's/\.bin$//' | sort
    # os.walk(followlinks=True) replicates find -L (glob '**' does not follow symlinks)
    if os.path.isdir(path):
        bins = []
        for root, _, files in os.walk(path, followlinks=True):
            bins += [os.path.join(root, f) for f in files if f.endswith(".bin")]
    elif path.endswith(".bin"):
        bins = [path]
    else:
        # treat as a prefix
        bins = [path + ".bin"]
    return sorted(p[: -len(".bin")] for p in bins)


def doc_lengths_for_prefix(prefix: str) -> np.ndarray:
    """Per-document token lengths from a single shard's index."""
    ds = IndexedDataset(prefix, mmap=True)
    seq_lengths = ds.index.sequence_lengths.astype(np.int64)
    doc_indices = ds.index.document_indices  # sequence boundaries per document
    # cumsum-diff handles documents that span multiple sequences (and empty docs)
    cumsum = np.concatenate([[0], np.cumsum(seq_lengths)])
    lengths = cumsum[doc_indices[1:]] - cumsum[doc_indices[:-1]]
    del ds
    return lengths


def collect_doc_lengths(path: str) -> np.ndarray:
    prefixes = shard_prefixes(path)
    if not prefixes:
        raise FileNotFoundError(f"No .bin shards found under {path}")
    print(f"  {len(prefixes)} shard(s)")
    parts = []
    for prefix in prefixes:
        lengths = doc_lengths_for_prefix(prefix)
        parts.append(lengths)
    lengths = np.concatenate(parts)
    assert lengths.max() < 2**31, "document longer than int32 max; widen dtype"
    return lengths.astype(np.int32)


def print_stats(name: str, lengths: np.ndarray) -> None:
    pct = np.percentile(lengths, PERCENTILES)
    print(f"  documents:    {len(lengths):,}")
    print(f"  total tokens: {int(lengths.sum()):,}")
    print(f"  mean:         {lengths.mean():,.1f}")
    print(f"  std:          {lengths.std():,.1f}")
    print(f"  min / max:    {lengths.min():,} / {lengths.max():,}")
    print("  percentiles:  " + "  ".join(f"p{p}={int(v):,}" for p, v in zip(PERCENTILES, pct)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Dataset dirs (or .bin / prefix paths)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="Where to write <dataset_name>.npy raw length arrays")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.datasets:
        name = os.path.basename(path.rstrip("/")).removesuffix(".bin")
        print(f"\n### {name} ###")
        print(f"  {path}")
        lengths = collect_doc_lengths(path)
        print_stats(name, lengths)
        out_path = os.path.join(args.out_dir, f"{name}.npy")
        np.save(out_path, lengths)
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
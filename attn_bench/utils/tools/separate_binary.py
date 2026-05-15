#!/usr/bin/env python3
"""
Copied from https://github.com/swiss-ai/data-pipeline-pretrain (branch yxu/separate_binary)

Split MMap indexed datasets (.idx/.bin pairs) into N random splits by documents.

For each subdataset under --input-dir, randomly assigns documents to splits
using Megatron's greedy error sampling (build_blending_indices) to match
target ratios, then writes new .idx/.bin pairs using Megatron's IndexedDataset
and IndexedDatasetBuilder.

Splits whose expected token count falls below --min-tokens are dropped and
their share is redistributed (via renormalized ratios) to the remaining splits,
so no data is lost.

Supports two directory layouts:
  - Nested (data-pipeline-pretrain format):
        input-dir/dataset_name/dump-N/*_tokens.{idx,bin}
  - Flat (Megatron tokenization format):
        input-dir/dump_N/*_tokens.{idx,bin}

Usage:
    python separate_binary.py \
        --input-dir /path/to/fineweb-edu-dedup-160B-datatrove \
        --output-dir /path/to/tokenized \
        --ratios 0.25 0.75 \
        --seed 42 \
        --workers 32 \
        --min-tokens 8193

Given flat layout (input-dir/dump_N/):
    Creates:
        output-dir/fineweb-edu-dedup-160B-datatrove_0.25/dump_0/*.{idx,bin}
        output-dir/fineweb-edu-dedup-160B-datatrove_0.75/dump_0/*.{idx,bin}

Original: https://github.com/swiss-ai/data-pipeline-pretrain (branch yxu/separate_binary)
"""

import argparse
import logging
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from megatron.core.datasets import helpers
from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder

logger = logging.getLogger(__name__)


def _iter_dump_dirs(parent_dir):
    """Yield dump subdirectories (dump-N or dump_N), sorted by index."""
    return sorted(
        (d for d in parent_dir.iterdir() if d.is_dir() and re.fullmatch(r'dump[-_]\d+', d.name)),
        key=lambda d: int(re.search(r'\d+$', d.name).group()),
    )


def split_documents(n_docs, ratios, seed):
    """Randomly assign documents to splits using greedy error sampling.

    1. Shuffle document indices for randomness.
    2. Use Megatron's build_blending_indices to assign each position to a
       split, minimizing the maximum deviation from target ratios.
    3. Sort within each split to preserve original index order (spatial
       locality for sequential I/O on the source .bin).

    Returns:
        list[np.ndarray]: One sorted array of document indices per split.
    """
    rng = np.random.default_rng(seed)
    doc_perm = rng.permutation(n_docs)

    n_splits = len(ratios)
    weights = np.array(ratios, dtype=np.float64)
    split_assignment = np.zeros(n_docs, dtype=np.int16)
    _unused = np.zeros(n_docs, dtype=np.int64)  # required by C++ API
    helpers.build_blending_indices(
        split_assignment, _unused, weights, n_splits, n_docs, False
    )

    return [np.sort(doc_perm[split_assignment == s]) for s in range(n_splits)]


def write_split(dataset, doc_idx, doc_ids, prefix_out, dtype):
    """Write a subset of documents to a new .idx/.bin pair."""
    os.makedirs(os.path.dirname(prefix_out), exist_ok=True)
    builder = IndexedDatasetBuilder(prefix_out + ".bin", dtype=dtype)

    for d in doc_ids:
        start, end = int(doc_idx[d]), int(doc_idx[d + 1])
        sequences = dataset[start:end]
        lengths = [len(s) for s in sequences]
        builder.add_document(np.concatenate(sequences), lengths)

    builder.finalize(prefix_out + ".idx")


def process_shard(prefix_in, prefixes_out, ratios, seed, min_tokens):
    """Split one .idx/.bin shard into N splits by documents."""
    name = os.path.basename(prefix_in)

    if not IndexedDataset.exists(prefix_in):
        logger.warning("%s: missing .idx or .bin, skipping", name)
        return

    dataset = IndexedDataset(prefix_in)
    doc_idx = dataset.document_indices
    seq_lengths = dataset.sequence_lengths
    dtype = dataset.index.dtype
    n_docs = len(doc_idx) - 1
    n_rows = len(seq_lengths)
    total_tokens = int(seq_lengths.sum())

    if n_rows == 0:
        del dataset
        logger.warning("%s: empty, skipping", name)
        return

    # Precompute per-document token counts and sequence counts (vectorized)
    doc_tokens = np.add.reduceat(seq_lengths, doc_idx[:-1].astype(int))
    doc_seqs = np.diff(doc_idx)

    # Determine which splits are viable (expected tokens >= min_tokens)
    viable = []
    for i, r in enumerate(ratios):
        if total_tokens * r >= min_tokens:
            viable.append(i)
        else:
            logger.warning("%s: split %.3f would get ~%d tokens < %d, dropping",
                           name, r, int(total_tokens * r), min_tokens)

    if not viable:
        # Entire shard too small for any split — write to the largest ratio
        logger.warning("%s: %d tokens too small for any split, writing to largest ratio",
                       name, total_tokens)
        viable = [int(np.argmax(ratios))]

    # Split using only viable ratios (renormalized to sum to 1)
    viable_ratios = [ratios[i] for i in viable]
    ratio_sum = sum(viable_ratios)
    viable_ratios = [r / ratio_sum for r in viable_ratios]

    viable_splits = split_documents(n_docs, viable_ratios, seed)
    split_map = dict(zip(viable, viable_splits))

    # Write each split
    parts = []
    split_tokens = []
    for i, prefix_out in enumerate(prefixes_out):
        doc_ids = split_map.get(i)
        if doc_ids is None:
            parts.append("[dropped]")
            split_tokens.append(0)
            continue
        split_tok = int(doc_tokens[doc_ids].sum())
        split_seqs = int(doc_seqs[doc_ids].sum())
        write_split(dataset, doc_idx, doc_ids, prefix_out, dtype)
        parts.append(f"{len(doc_ids)} docs ({split_seqs} rows, {split_tok} tokens)")
        split_tokens.append(split_tok)

    del dataset
    logger.info("%s: %d docs (%d rows, %d tokens) -> %s",
                name, n_docs, n_rows, total_tokens, " + ".join(parts))
    return split_tokens


def discover_shards(input_dir):
    """Find all subdatasets and their shards.

    Supports two layouts:
    - Flat   (Megatron tokenization format):  input_dir/dump_N/*_tokens.idx
    - Nested (data-pipeline-pretrain format): input_dir/dataset_name/dump-N/*_tokens.idx

    Yields (subdataset_name, dump_name, stem, prefix) for each .idx/.bin shard.
    """
    # If input_dir directly contains dump dirs, treat it as the dataset root (flat layout).
    if _iter_dump_dirs(input_dir):
        subdirs = [(input_dir.name, input_dir)]
    else:
        subdirs = [(d.name, d) for d in sorted(input_dir.iterdir()) if d.is_dir()]

    for ds_name, ds_dir in subdirs:
        dump_dirs = _iter_dump_dirs(ds_dir)
        if not dump_dirs:
            continue
        for dump_dir in dump_dirs:
            for idx_file in sorted(dump_dir.glob("*_tokens.idx")):
                prefix = str(idx_file)[:-4]
                yield ds_name, dump_dir.name, idx_file.stem, prefix


def main():
    parser = argparse.ArgumentParser(
        description="Split MMap indexed datasets into N random splits by document."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Dataset root (flat layout: contains dump_N/) or parent of multiple datasets (nested layout)")
    parser.add_argument("--output-dir", required=True,
                        help="Parent directory where split folders will be created")
    parser.add_argument("--ratios", type=float, nargs="+", required=True,
                        help="Split ratios (must sum to 1), e.g. --ratios 0.25 0.75")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--min-tokens", type=int, default=8193,
                        help="Skip output splits with fewer tokens than this (default: 8193 = seq_len 8192 + 1)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: number of CPUs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        logger.error("--ratios must sum to 1.0, got %.6f", sum(args.ratios))
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    logger.info("Input:      %s", input_dir)
    logger.info("Output:     %s", output_dir)
    logger.info("Ratios:     %s", args.ratios)
    logger.info("Seed:       %d", args.seed)
    logger.info("Min tokens: %d (per output split)", args.min_tokens)

    master_rng = np.random.default_rng(args.seed)
    tasks = []

    for ds_name, dump_name, stem, prefix in discover_shards(input_dir):
        out_dirs = [output_dir / f"{ds_name}_{r}" for r in args.ratios]
        prefixes_out = [str(d / dump_name / stem) for d in out_dirs]
        shard_seed = int(master_rng.integers(0, 2**63))
        tasks.append((prefix, prefixes_out, args.ratios, shard_seed, args.min_tokens))

    if not tasks:
        logger.error(
            "No shards found under %s. Expected dump_N/*_tokens.idx files.\n"
            "Check --input-dir points to either:\n"
            "  - A dataset root containing dump_N/ directories (flat layout), or\n"
            "  - A parent directory of datasets, each containing dump-N/ (nested layout).",
            input_dir,
        )
        sys.exit(1)

    n_workers = min(args.workers or os.cpu_count(), len(tasks))
    logger.info("Found %d shards to process with %d workers", len(tasks), n_workers)

    total_tokens = [0] * len(args.ratios)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(process_shard, *t): t[0] for t in tasks}
        for future in as_completed(futures):
            split_tokens = future.result()
            if split_tokens:
                for i, tok in enumerate(split_tokens):
                    total_tokens[i] += tok

    logger.info("### Total tokens per split ###")
    for r, tok in zip(args.ratios, total_tokens):
        logger.info("  ratio %.2f -> %d tokens (%.3f B)", r, tok, tok / 1e9)
    logger.info("  total      -> %d tokens (%.3f B)", sum(total_tokens), sum(total_tokens) / 1e9)
    logger.info("Done!")


if __name__ == "__main__":
    main()
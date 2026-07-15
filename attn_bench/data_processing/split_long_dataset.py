"""
Split each document in a tokenized Megatron dataset into fixed-length pieces, to build
the "short chunk" comparison arm for the long-doc dataset (build_long_dataset.py).

Every source document is already <BOS> content <EOS> wrapped (the datatrove tokenization
pipeline's default). Each one is cut into pieces of --chunk-size tokens TOTAL (content +
its own BOS + EOS, so content is chunked at chunk_size - 2); a trailing remainder shorter
than --tail-merge-threshold is merged into the preceding piece instead of becoming its
own tiny document. Megatron's packed-seq / reset-position-ids / eod-mask-loss
key off the literal EOD token in the stream, not the .idx document boundaries alone
(pretrain_gpt.py's pack_batch_with_cu_seqlens, gpt_dataset.py's _get_ltor_masks_and_position_ids)
-- so every new interior boundary gets a fresh BOS/EOS inserted; the original document's
outer BOS/EOS are kept in place on the first/last piece.

Source documents are partitioned into --num-workers contiguous ranges (by document count,
not length -- source doc order isn't length-sorted, so this balances work evenly). Each
worker writes its own independent shard_NNNNN.bin/.idx, same convention as
tokenize_fineweb_edu_native.py -- no merge step needed, Megatron blends across however
many .bin/.idx pairs live in a directory.

Usage:
    python split_long_dataset.py \
        --data-folder /path/to/tokenized/fineweb-edu-dedup-160B-datatrove_long_40B \
        --tokenizer-path /path/to/tokenizer \
        --chunk-size 1024 \
        --tail-merge-threshold 256 \
        --num-workers 20 \
        --output-folder /path/to/tokenized/fineweb-edu-dedup-160B-datatrove_long_40B_split1024
"""

import argparse
import json
import os
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from attn_bench.data_processing.dataset_doc_stats import shard_prefixes
from megatron.core.datasets.indexed_dataset import (IndexedDataset,
                                                    IndexedDatasetBuilder)

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]


def chunk_sizes(content_len: int, chunk_size: int, tail_merge_threshold: int) -> list:
    """Piece sizes (summing to content_len) for one document's inner content."""
    if content_len <= chunk_size:
        return [content_len]
    num_full, remainder = divmod(content_len, chunk_size)
    if remainder == 0:
        return [chunk_size] * num_full
    if remainder < tail_merge_threshold:
        return [chunk_size] * (num_full - 1) + [chunk_size + remainder]
    return [chunk_size] * num_full + [remainder]


def split_document(tokens: np.ndarray, bos_id: int, eos_id: int, chunk_size: int, tail_merge_threshold: int) -> list:
    """Split one <BOS>...<EOS> document into pieces, each itself <BOS>...<EOS> wrapped, so
    each piece's total length (content + 2 markers) is chunk_size."""
    assert tokens[0] == bos_id and tokens[-1] == eos_id, "source document is not BOS/EOS wrapped"
    content = tokens[1:-1]
    sizes = chunk_sizes(len(content), chunk_size - 2, tail_merge_threshold)
    pieces, offset, n = [], 0, len(sizes)
    for i, size in enumerate(sizes):
        piece_content = content[offset:offset + size]
        offset += size
        start_token = tokens[0] if i == 0 else bos_id
        end_token = tokens[-1] if i == n - 1 else eos_id
        pieces.append(np.concatenate([[start_token], piece_content, [end_token]]))
    return pieces


def shard_doc_counts(prefixes: list) -> list:
    """(prefix, num_docs) for each shard, in order -- index-only, no .bin reads."""
    counts = []
    for prefix in prefixes:
        ds = IndexedDataset(prefix, mmap=True)
        counts.append(len(ds.index.document_indices) - 1)
        del ds
    return counts


def partition_work(prefixes: list, counts: list, num_workers: int) -> list:
    """Split the flat (shard, local_doc_id) sequence into num_workers contiguous chunks.

    Returns a list of length num_workers, each a list of (prefix, local_start, local_end)
    tuples describing the doc ranges that worker owns (may span multiple shards)."""
    total_docs = sum(counts)
    boundaries = np.linspace(0, total_docs, num_workers + 1).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(counts)])

    work = []
    for w in range(num_workers):
        lo, hi = int(boundaries[w]), int(boundaries[w + 1])
        items = []
        for s, prefix in enumerate(prefixes):
            shard_lo, shard_hi = int(cum[s]), int(cum[s + 1])
            start, end = max(lo, shard_lo) - shard_lo, min(hi, shard_hi) - shard_lo
            if end > start:
                items.append((prefix, start, end))
        work.append(items)
    return work


def worker_fn(worker_id: int, work_items: list, bos_id: int, eos_id: int, chunk_size: int,
              tail_merge_threshold: int, out_prefix: str) -> tuple:
    dtype, builder = None, None
    total_docs_in, total_docs_out = 0, 0
    piece_lengths = []

    total_assigned = sum(end - start for _, start, end in work_items)
    pbar = tqdm(total=total_assigned, desc=f"worker-{worker_id:02d}", mininterval=10, position=worker_id)

    for prefix, start, end in work_items:
        ds = IndexedDataset(prefix, mmap=True)
        if dtype is None:
            dtype = ds.index.dtype
            builder = IndexedDatasetBuilder(out_prefix + ".bin", dtype=dtype)
        else:
            assert ds.index.dtype == dtype, f"dtype mismatch across shards: {dtype} vs {ds.index.dtype} ({prefix})"

        doc_indices = ds.index.document_indices
        for d in range(start, end):
            s, e = int(doc_indices[d]), int(doc_indices[d + 1])
            tokens = np.concatenate(ds[s:e])
            total_docs_in += 1
            for piece in split_document(tokens, bos_id, eos_id, chunk_size, tail_merge_threshold):
                builder.add_document(piece, [len(piece)])
                piece_lengths.append(len(piece))
                total_docs_out += 1
            pbar.update(1)
        del ds
    pbar.close()

    builder.finalize(out_prefix + ".idx")
    lengths_path = out_prefix + "_lengths.npy"
    np.save(lengths_path, np.array(piece_lengths, dtype=np.int64))
    return total_docs_in, total_docs_out, lengths_path


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


def write_run_metadata(output_folder: str, args: argparse.Namespace, documents_in: int, stats: dict) -> None:
    with open(os.path.join(output_folder, "run_metadata.json"), "w") as f:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data_folder": args.data_folder,
            "chunk_size": args.chunk_size,
            "tail_merge_threshold": args.tail_merge_threshold,
            "num_workers": args.num_workers,
            "documents_in": documents_in,
            "stats": stats,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-folder", required=True, help="Tokenized Megatron dataset dir to split")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--tail-merge-threshold", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=20)
    parser.add_argument("--output-folder", required=True, help="Where to write shard_NNNNN.bin/.idx + run_metadata.json")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    print(f"BOS id: {bos_id} | EOS id: {eos_id}")

    prefixes = shard_prefixes(args.data_folder)
    if not prefixes:
        raise FileNotFoundError(f"No .bin shards found under {args.data_folder}")
    counts = shard_doc_counts(prefixes)
    print(f"{len(prefixes)} shard(s), {sum(counts):,} documents under {args.data_folder}")
    print(f"chunk_size={args.chunk_size}  tail_merge_threshold={args.tail_merge_threshold}  num_workers={args.num_workers}")

    work = partition_work(prefixes, counts, args.num_workers)
    worker_args = [
        (w, work[w], bos_id, eos_id, args.chunk_size, args.tail_merge_threshold,
         os.path.join(args.output_folder, f"shard_{w:05d}"))
        for w in range(args.num_workers)
    ]

    with Pool(args.num_workers) as pool:
        results = pool.starmap(worker_fn, worker_args)

    total_docs_in = sum(r[0] for r in results)
    total_docs_out = sum(r[1] for r in results)
    print(f"{total_docs_in:,} documents -> {total_docs_out:,} pieces -> {args.output_folder}/shard_*.bin/.idx")

    lengths = np.concatenate([np.load(r[2]) for r in results])
    for r in results:
        os.remove(r[2])

    stats = stats_from_lengths(lengths)
    print_stats(stats)
    write_run_metadata(args.output_folder, args, total_docs_in, stats)
    print(f"run_metadata.json -> {os.path.join(args.output_folder, 'run_metadata.json')}")


if __name__ == "__main__":
    main()
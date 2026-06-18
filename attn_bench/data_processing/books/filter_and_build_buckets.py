"""
Filter Gutenberg books and partition into repetition buckets with equalized
mean perplexity across buckets.

Filters: ppl in [p{ppl_lo_q}, p{ppl_hi_q}], fineweb_max_ngram_hits == 0.

Perplexity equalization: books sorted by ppl, excess trimmed symmetrically
from both ends, then for each group of n_buckets consecutive books the
assignment to buckets is randomized — every bucket covers the full ppl range.

Usage:
    python -m attn_bench.data_processing.books.filter_and_build_buckets \\
        --input     /path/to/sampled_containment.jsonl \\
        --output    /path/to/filtered_buckets.jsonl \\
        --stats-dir /path/to/stats/buckets/
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


REPETITIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
N_SAMPLE = 300
SNIPPET_CHARS = 5000


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")
    return records


def apply_filters(records: list[dict], ppl_lo_q: float, ppl_hi_q: float) -> list[dict]:
    ppls = np.array([r.get('perplexity') or float('nan') for r in records])
    valid = ~np.isnan(ppls)
    ppl_lo = float(np.nanpercentile(ppls, ppl_lo_q))
    ppl_hi = float(np.nanpercentile(ppls, ppl_hi_q))

    ppl_mask = valid & (ppls >= ppl_lo) & (ppls <= ppl_hi)
    hit_mask = np.array([r.get('fineweb_max_ngram_hits', 0) == 0 for r in records])
    combined = ppl_mask & hit_mask

    n = len(records)
    print(f"\nFilter thresholds:")
    print(f"  perplexity:  p{ppl_lo_q:.0f}/p{ppl_hi_q:.0f}  ({ppl_lo:.1f} – {ppl_hi:.1f})")
    print(f"  max_hit:     == 0  (implies coverage == 0)")
    print(f"\nDrop breakdown (not mutually exclusive):")
    print(f"  no perplexity score:  {int((~valid).sum()):>6,}  ({(~valid).sum() / n * 100:.1f}%)")
    print(f"  ppl out of range:     {int((valid & ~ppl_mask).sum()):>6,}  ({(valid & ~ppl_mask).sum() / n * 100:.1f}%)")
    print(f"  max_hit > 0:          {int((~hit_mask).sum()):>6,}  ({(~hit_mask).sum() / n * 100:.1f}%)")
    print(f"\nKept: {int(combined.sum()):,} / {n:,}  ({combined.sum() / n * 100:.1f}%)")

    return [r for r, keep in zip(records, combined) if keep]


def write_sample_stats(filtered: list[dict], stats_dir: Path):
    """300 books stratified across ppl deciles, sorted by ppl, saved as text for inspection."""
    per_decile = N_SAMPLE // 10
    sorted_books = sorted(filtered, key=lambda r: r['perplexity'])
    n = len(sorted_books)

    rng = random.Random(0)
    sample = []
    for d in range(10):
        start, end = d * n // 10, (d + 1) * n // 10
        sample.extend(rng.sample(sorted_books[start:end], min(per_decile, end - start)))
    sample.sort(key=lambda r: r['perplexity'])

    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / 'filtered_sample_300.txt'
    sep = '=' * 80
    with open(path, 'w') as f:
        f.write(f"Sample: {len(sample)} books stratified by perplexity  (full filtered set: {n:,})\n\n")
        for r in sample:
            f.write(f"{sep}\n")
            f.write(f"ppl={r['perplexity']:.1f}  id={r.get('book_id', '?')}  title={r.get('book_title', '?')!r}\n\n")
            excerpt = r.get('text_excerpt', '')
            f.write(excerpt[:SNIPPET_CHARS])
            if len(excerpt) > SNIPPET_CHARS:
                f.write('...')
            f.write('\n\n')
    print(f"Sample -> {path}")


def build_buckets(filtered: list[dict], repetitions: list[int], seed: int = 42) -> tuple[list[dict], list[dict]]:
    n_buckets = len(repetitions) + 1  # +1 for unseen bucket

    # sort by ppl, trim symmetrically so len is divisible by n_buckets
    sorted_books = sorted(filtered, key=lambda r: r['perplexity'])
    excess = len(sorted_books) % n_buckets
    trim_lo = excess // 2
    trim_hi = len(sorted_books) - (excess - trim_lo)
    sorted_books = sorted_books[trim_lo:trim_hi]
    bucket_size = len(sorted_books) // n_buckets
    print(f"\nTrimmed {trim_lo} low + {excess - trim_lo} high ppl books  →  {len(sorted_books):,} books  ({bucket_size} per bucket, {n_buckets - 1} training + 1 unseen)")

    # bucket_books[i] = list of books assigned to repetitions[i]; bucket_books[-1] = unseen
    rng = random.Random(seed)
    bucket_books: list[list[dict]] = [[] for _ in range(n_buckets)]
    for i in range(0, len(sorted_books), n_buckets):
        book_group = sorted_books[i:i + n_buckets]
        # indices = [0, .., n_buckets) shuffled -> maps each book in group to a random bucket
        bucket_indices = list(range(n_buckets))
        rng.shuffle(bucket_indices)
        for bucket_idx, book in zip(bucket_indices, book_group):
            bucket_books[bucket_idx].append(book)

    training = []
    # first, add repetitions
    for rep, books in zip(repetitions, bucket_books):
        # shuffle the books within the bucket
        rng.shuffle(books)
        for book in books:
            # each book is a dict loaded from json, making a shallow copy here
            book = dict(book)
            book['bucket_rep'] = rep  # training repetition label; actual copying done in write_megatron step
            training.append(book)

    unseen = []
    # then, add unseen books
    rng.shuffle(bucket_books[-1])
    for book in bucket_books[-1]:
        book = dict(book)
        book['bucket_rep'] = 1
        unseen.append(book)

    return training, unseen


def print_bucket_stats(training: list[dict], unseen: list[dict], repetitions: list[int]):
    print(f"\nBucket summary:")
    print(f"  {'rep':>6}  {'n':>6}  {'mean_ppl':>10}  {'std_ppl':>9}  {'min_ppl':>9}  {'max_ppl':>9}")
    for rep in repetitions:
        ppls = [r['perplexity'] for r in training if r['bucket_rep'] == rep]
        print(f"  {rep:>6}  {len(ppls):>6,}  {np.mean(ppls):>10.1f}  "
              f"{np.std(ppls):>9.1f}  {min(ppls):>9.1f}  {max(ppls):>9.1f}")
    ppls = [r['perplexity'] for r in unseen]
    print(f"  {'unseen':>6}  {len(ppls):>6,}  {np.mean(ppls):>10.1f}  "
          f"{np.std(ppls):>9.1f}  {min(ppls):>9.1f}  {max(ppls):>9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--unseen-output', required=True)
    parser.add_argument('--stats-dir', required=True)
    parser.add_argument('--ppl-lo-q', type=float, default=10)
    parser.add_argument('--ppl-hi-q', type=float, default=90)
    parser.add_argument('--repetitions', type=str, default=None,
                        help='JSON list of repetition labels, e.g. "[1,2,4,8,16,32,64,128,256]"')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    repetitions = json.loads(args.repetitions) if args.repetitions else REPETITIONS

    records = load_records(args.input)
    filtered = apply_filters(records, args.ppl_lo_q, args.ppl_hi_q)

    write_sample_stats(filtered, Path(args.stats_dir))

    training, unseen = build_buckets(filtered, repetitions, seed=args.seed)
    print_bucket_stats(training, unseen, repetitions)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for r in training:
            f.write(json.dumps(r) + '\n')
    print(f"\nTraining output -> {out_path}  ({len(training):,} records)")

    unseen_path = Path(args.unseen_output)
    unseen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(unseen_path, 'w') as f:
        for r in unseen:
            f.write(json.dumps(r) + '\n')
    print(f"Unseen output   -> {unseen_path}  ({len(unseen):,} records)")


if __name__ == '__main__':
    main()
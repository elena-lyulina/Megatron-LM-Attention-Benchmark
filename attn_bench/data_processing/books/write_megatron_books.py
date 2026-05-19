"""
Write filtered_buckets.jsonl to Megatron bin/idx format.

Produces one bin/idx pair per repetition level, matching the FineWeb format
(MegatronTokenizedFile, token_size=4 / int32). Tokens are written as-is —
they already contain BOS and EOS from the Gutenberg pipeline.

Ordering matches PDM: [book1, book2, ..., bookN] repeated rep times.

Output:
    <output_dir>/rep_1_tokens.bin + .idx
    <output_dir>/rep_2_tokens.bin + .idx
    ...

Usage:
    python -m attn_bench.data_processing.books.write_megatron_books \\
        --input     /path/to/filtered_buckets.jsonl \\
        --output-dir /path/to/gutenberg_megatron/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from attn_bench.data_processing.tokenization.megatron_tokenizer_budgeted import MegatronTokenizedFile


TOKEN_SIZE = 4  # int32, matches FineWeb tokenization
SEQ_LEN = 8192


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")
    return records


def group_by_rep(records: list[dict]) -> dict[int, list[dict]]:
    # bucket_rep -> list of books assigned to that repetition level
    buckets: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        buckets[r['bucket_rep']].append(r)
    return buckets


def write_bucket(rep: int, books: list[dict], output_dir: str):
    filename = f"rep_{rep}_tokens"
    f = MegatronTokenizedFile(output_dir, filename, token_size=TOKEN_SIZE)

    # write [book1..bookN, book1..bookN, ...] rep times, matches PDM
    for _ in range(rep):
        for book in books:
            f.write(book['token_ids'])  # already BOS + content + EOS, exactly SEQ_LEN tokens

    f.close()
    n_seqs = rep * len(books)
    print(f"  rep={rep:>4}  books={len(books):>6,}  sequences={n_seqs:>8,}  "
          f"tokens={n_seqs * SEQ_LEN:>12,}  ->  {filename}.bin/.idx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='filtered_buckets.jsonl with bucket_rep and token_ids fields')
    parser.add_argument('--output-dir', required=True,
                        help='directory to write rep_N_tokens.bin/.idx files')
    args = parser.parse_args()

    records = load_records(args.input)
    buckets = group_by_rep(records)

    output_dir = str(Path(args.output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {len(buckets)} repetition buckets to {output_dir}")
    print(f"  {'rep':>4}  {'books':>6}  {'sequences':>9}  {'tokens':>12}")

    total_tokens = 0
    for rep in sorted(buckets):
        write_bucket(rep, buckets[rep], output_dir)
        total_tokens += rep * len(buckets[rep]) * SEQ_LEN

    print(f"\nTotal tokens written: {total_tokens:,}  ({total_tokens / 1e9:.2f}B)")


if __name__ == '__main__':
    main()
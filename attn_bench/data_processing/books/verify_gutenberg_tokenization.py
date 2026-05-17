"""
Verify Gutenberg repetition bucket bin files.

For each rep_N_tokens.bin/.idx in output_dir:
  - Reports actual sequence length (asserts all == 8192)
  - Checks BOS at position 0, EOS at last position, each appearing exactly once
  - Detects unique books and repetition counts (expects each unique book N times in rep_N)
  - Prints a few example sequences so you can eyeball BOS / EOS yourself

Writes a summary.txt to --stats-dir (per-bucket stats + totals, no token examples).

Usage:
    python verify_gutenberg_tokenization.py \\
        --tokenized-dir /path/to/gutenberg_rep_1_256 \\
        --tokenizer-path /path/to/llama-3.2-1b \\
        --stats-dir /path/to/preprocessing/stats/megatron-tokenization
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))
from megatron.core.datasets.indexed_dataset import IndexedDataset

SEQ_LEN = 8192
NUM_EXAMPLES = 3
PREVIEW_TOKENS = 6


def parse_rep(filename: str) -> int | None:
    m = re.search(r"rep_(\d+)_tokens", filename)
    return int(m.group(1)) if m else None


def verify_bucket(prefix: str, rep: int, bos_id: int, eos_id: int):
    dataset = IndexedDataset(prefix)
    n = len(dataset)

    lengths = []
    no_bos_start = 0    # BOS not at position 0
    no_eos_end = 0      # EOS not at last position
    multi_bos = 0       # BOS appears more than once
    multi_eos = 0       # EOS appears more than once
    hashes: list[int] = []

    for i in range(n):
        tokens = dataset[i]
        tlist = tokens.tolist()
        lengths.append(len(tlist))
        bos_count = tlist.count(bos_id)
        eos_count = tlist.count(eos_id)
        if tlist[0] != bos_id:
            no_bos_start += 1
        if tlist[-1] != eos_id:
            no_eos_end += 1
        if bos_count != 1:
            multi_bos += 1
        if eos_count != 1:
            multi_eos += 1
        hashes.append(hash(tokens.tobytes()))

    counts = Counter(hashes)
    unique = len(counts)
    rep_dist = Counter(counts.values())
    wrong_reps = sum(1 for c in counts.values() if c != rep)

    min_len = min(lengths)
    max_len = max(lengths)
    len_ok = min_len == max_len == SEQ_LEN

    examples = [dataset[i].tolist() for i in range(min(NUM_EXAMPLES, n))]

    return {
        'n': n,
        'min_len': min_len,
        'max_len': max_len,
        'len_ok': len_ok,
        'no_bos_start': no_bos_start,
        'no_eos_end': no_eos_end,
        'multi_bos': multi_bos,
        'multi_eos': multi_eos,
        'unique': unique,
        'rep_dist': rep_dist,
        'wrong_reps': wrong_reps,
        'examples': examples,
    }


def print_bucket(lines: list[str], r: dict):
    for line in lines:
        print(line)
    print(f"\n  Examples (first {len(r['examples'])} sequences):")
    for i, tokens in enumerate(r['examples']):
        first = tokens[:PREVIEW_TOKENS]
        last = tokens[-PREVIEW_TOKENS:]
        print(f"    seq {i} : first {PREVIEW_TOKENS}: {first}   last {PREVIEW_TOKENS}: {last}")


def bucket_summary_lines(name: str, rep: int, r: dict) -> list[str]:
    ok = lambda v: "OK" if v else "FAIL"
    lines = [
        f"\n### {name} ###",
        f"  sequences    : {r['n']:,}",
        f"  unique books : {r['unique']:,}",
    ]
    if r['len_ok']:
        lines.append(f"  seq length   : {r['min_len']}  (expected {SEQ_LEN})  {ok(r['len_ok'])}")
    else:
        lines.append(f"  seq length   : min={r['min_len']}  max={r['max_len']}  (expected {SEQ_LEN})  {ok(r['len_ok'])}")
    lines += [
        f"  BOS at start : {r['n'] - r['no_bos_start']:,}/{r['n']:,}  {ok(r['no_bos_start'] == 0)}",
        f"  EOS at end   : {r['n'] - r['no_eos_end']:,}/{r['n']:,}  {ok(r['no_eos_end'] == 0)}",
        f"  BOS exactly 1: {r['n'] - r['multi_bos']:,}/{r['n']:,}  {ok(r['multi_bos'] == 0)}",
        f"  EOS exactly 1: {r['n'] - r['multi_eos']:,}/{r['n']:,}  {ok(r['multi_eos'] == 0)}",
    ]
    rep_ok = r['wrong_reps'] == 0
    lines.append(
        f"  rep dist     : {dict(sorted(r['rep_dist'].items()))}  "
        f"{'all repeat exactly ' + str(rep) + 'x' if rep_ok else str(r['wrong_reps']) + ' books with wrong repeat count'}  {ok(rep_ok)}"
    )
    return lines


def write_summary(stats_dir: Path, lines: list[str]):
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / 'summary.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSummary written to {path}")


def verify_all(prefixes: list, bos_id: int, eos_id: int) -> tuple[bool, int, int, list[str]]:
    all_ok = True
    total_seqs = 0
    total_unique = 0
    summary_lines: list[str] = []

    for prefix, rep in prefixes:
        name = Path(prefix).name
        if rep is None:
            print(f"  WARNING: could not parse rep level from {name}, skipping")
            continue
        r = verify_bucket(prefix, rep, bos_id, eos_id)
        lines = bucket_summary_lines(name, rep, r)
        print_bucket(lines, r)
        summary_lines.extend(lines)
        total_seqs += r['n']
        total_unique += r['unique']
        if not r['len_ok'] or r['no_bos_start'] or r['no_eos_end'] or r['multi_bos'] or r['multi_eos'] or r['wrong_reps']:
            all_ok = False

    return all_ok, total_seqs, total_unique, summary_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenized-dir', required=True)
    parser.add_argument('--tokenizer-path', required=True)
    parser.add_argument('--stats-dir', required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    bos_id: int = tok.bos_token_id
    eos_id: int = tok.eos_token_id
    print(f"BOS id: {bos_id}  |  EOS id: {eos_id}")

    tokenized_dir = Path(args.tokenized_dir)
    prefixes = sorted(
        (str(p.with_suffix('')), parse_rep(p.stem))
        for p in tokenized_dir.glob("rep_*_tokens.bin")
    )
    if not prefixes:
        print(f"No rep_N_tokens.bin files found in {tokenized_dir}")
        sys.exit(1)

    print(f"\nFound {len(prefixes)} bucket(s) in {tokenized_dir}")

    all_ok, total_seqs, total_unique, summary_lines = verify_all(prefixes, bos_id, eos_id)

    result = 'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'
    totals = [
        f"\n{'=' * 60}",
        f"SUMMARY",
        f"  total sequences  : {total_seqs:,}",
        f"  total unique books (across all buckets): {total_unique:,}",
        f"  total tokens     : {total_seqs * SEQ_LEN:,}  ({total_seqs * SEQ_LEN / 1e9:.2f}B)",
        f"  {result}",
    ]
    for line in totals:
        print(line)

    write_summary(Path(args.stats_dir), summary_lines + totals)


if __name__ == '__main__':
    main()
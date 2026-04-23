"""
Verify tokenization output by decoding a few sequences from a Megatron binary shard.

Usage:
    # Single shard:
    python verify_tokenization.py \
        --shard-path /path/to/00000_tokens \
        --tokenizer-path /path/to/tokenizer

    # All shards in a directory:
    python verify_tokenization.py \
        --tokenized-dir /path/to/tokenized_dataset \
        --tokenizer-path /path/to/tokenizer
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[3]))
from megatron.core.datasets.indexed_dataset import IndexedDataset


def check_shard(shard_path: str, bos_id: int, eos_id: int):
    dataset = IndexedDataset(shard_path)
    n = len(dataset)
    fails = {"no_bos": 0, "no_eos": 0, "multi_bos": 0, "multi_eos": 0}
    for i in range(n):
        token_ids = dataset[i].tolist()
        bos_count = token_ids.count(bos_id)
        eos_count = token_ids.count(eos_id)
        if token_ids[0] != bos_id:
            fails["no_bos"] += 1
        if token_ids[-1] != eos_id:
            fails["no_eos"] += 1
        if bos_count > 1:
            fails["multi_bos"] += 1
        if eos_count > 1:
            fails["multi_eos"] += 1
    return n, fails


def print_detailed(shard_path: str, tokenizer, bos_id: int, eos_id: int, num_sequences: int) -> int:
    """Returns number of sequences with non-zero round-trip delta."""
    dataset = IndexedDataset(shard_path)
    n = len(dataset)
    print(f"\n{'=' * 80}")
    print(f"Detailed check: {Path(shard_path).name} ({n:,} sequences, showing {min(num_sequences, n)})")
    print("=" * 80)
    nonzero_delta = 0
    for i in range(min(num_sequences, n)):
        token_ids = dataset[i].tolist()
        bos_count = token_ids.count(bos_id)
        eos_count = token_ids.count(eos_id)
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        reenc = tokenizer.encode(text, add_special_tokens=False)
        roundtrip_delta = len(reenc) - len(token_ids)
        if roundtrip_delta != 0:
            nonzero_delta += 1
        print(f"\n--- Sequence {i} | {len(token_ids)} tokens ---")
        print(f"First tokens: {token_ids[:10]}")
        print(f"Last tokens:  {token_ids[-5:]}")
        print(f"BOS at start: {token_ids[0] == bos_id} | count: {bos_count} {'OK' if bos_count == 1 else 'WARN'}")
        print(f"EOS at end:   {token_ids[-1] == eos_id} | count: {eos_count} {'OK' if eos_count == 1 else 'WARN'}")
        print(f"Round-trip:   delta={roundtrip_delta:+d} tokens {'OK' if roundtrip_delta == 0 else 'WARN'}")
        print(f"Text preview: {text[:300]!r}")
        print(f"Text ending:  {text[-100:]!r}")
    return nonzero_delta


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    print(f"BOS id: {bos_id} | EOS id: {eos_id}")

    if args.tokenized_dir:
        shard_paths = sorted(
            str(p.with_suffix("")) for p in Path(args.tokenized_dir).rglob("*.bin")
        )
    else:
        shard_paths = [args.shard_path]

    print(f"\nFound {len(shard_paths)} shard(s)\n")

    # --- Fast pass: stats for all shards ---
    total_seqs = 0
    total_fails = {"no_bos": 0, "no_eos": 0, "multi_bos": 0, "multi_eos": 0}
    shard_results = []

    for shard_path in shard_paths:
        n, fails = check_shard(shard_path, bos_id, eos_id)
        total_seqs += n
        for k in total_fails:
            total_fails[k] += fails[k]
        shard_results.append((shard_path, n, fails))
        all_ok = all(v == 0 for v in fails.values())
        print(f"  {Path(shard_path).name} | {n:,} seqs | {'OK' if all_ok else f'ISSUES: {fails}'}")

    print(f"\nTotal: {total_seqs:,} sequences across {len(shard_paths)} shards")
    print(f"  BOS missing at start : {total_fails['no_bos']:,}")
    print(f"  EOS missing at end   : {total_fails['no_eos']:,}")
    print(f"  Multiple BOS         : {total_fails['multi_bos']:,}")
    print(f"  Multiple EOS         : {total_fails['multi_eos']:,}")

    if any(v > 0 for v in total_fails.values()):
        raise ValueError(f"Tokenization checks failed: {total_fails}")
    print("  ALL OK")

    # --- Detailed preview for each shard ---
    total_nonzero_delta = 0
    total_checked = 0
    for shard_path, _, _ in shard_results:
        total_nonzero_delta += print_detailed(shard_path, tokenizer, bos_id, eos_id, args.num_sequences)
        total_checked += min(args.num_sequences, len(IndexedDataset(shard_path)))

    print(f"\nRound-trip summary: {total_nonzero_delta} / {total_checked} sequences had non-zero delta")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Megatron binary tokenization")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard-path", type=str, help="Path prefix to a single shard (without .bin/.idx)")
    group.add_argument("--tokenized-dir", type=str, help="Directory to search for all shards recursively")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=5)
    args = parser.parse_args()
    main(args)
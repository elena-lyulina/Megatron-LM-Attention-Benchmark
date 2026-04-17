"""
Verify tokenization output by decoding a few sequences from a Megatron binary shard.

Usage:
    python verify_tokenization.py \
        --shard-path /path/to/shard_00000 \
        --tokenizer-path /path/to/tokenizer \
        --num-sequences 5
"""

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.datasets.indexed_dataset import IndexedDataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    dataset = IndexedDataset(args.shard_path)
    print(f"Shard: {args.shard_path}")
    print(f"Total sequences: {len(dataset):,}")
    print("=" * 80)

    for i in range(min(args.num_sequences, len(dataset))):
        token_ids = dataset[i].tolist()
        text = tokenizer.decode(token_ids)
        print(f"\n--- Sequence {i} | {len(token_ids)} tokens ---")
        print(f"First tokens: {token_ids[:10]}")
        print(f"Last tokens:  {token_ids[-5:]}")
        print(f"EOS at end:   {token_ids[-1] == tokenizer.eos_token_id} (eos={tokenizer.eos_token_id})")
        print(f"Text preview: {text[:300]!r}")
        print(f"Text ending:  {text[-100:]!r}")
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Megatron binary tokenization")
    parser.add_argument("--shard-path", type=str, required=True, help="Path prefix (without .bin/.idx)")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--num-sequences", type=int, default=5)
    args = parser.parse_args()
    main(args)
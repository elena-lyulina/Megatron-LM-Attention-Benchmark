"""
Stream FineWeb-Edu-dedup from HuggingFace and write directly to Megatron binary format.

Writes output in shards so that completed shards are safe if the job is interrupted.
On restart, reads progress.json and resumes from the last completed shard.

Output layout:
    <output_dir>/
        shard_00000.bin / shard_00000.idx
        shard_00001.bin / shard_00001.idx
        ...
        progress.json

Usage:
    # Full run
    python tokenize_fineweb_edu.py \
        --tokenizer-path /path/to/tokenizer \
        --output-dir /path/to/dedup-fineweb-edu-160B \
        --token-budget 160_000_000_000

    # Quick test to measure throughput
    python tokenize_fineweb_edu.py \
        --tokenizer-path /path/to/tokenizer \
        --output-dir /path/to/dedup-fineweb-edu-test \
        --token-budget 10_000_000 \
        --shard-size 5_000_000
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, DType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

LOG_INTERVAL = 100_000  # log every N documents

DATASET_NAME = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"


def load_progress(output_dir: Path) -> dict:
    progress_file = output_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        logger.info(
            f"Resuming from progress.json: "
            f"{progress['completed_shards']} shards done, "
            f"{progress['total_docs']:,} docs, "
            f"{progress['total_tokens'] / 1e9:.2f}B tokens"
        )
        return progress
    return {"completed_shards": 0, "total_docs": 0, "total_tokens": 0}


def save_progress(output_dir: Path, completed_shards: int, total_docs: int, total_tokens: int):
    progress_file = output_dir / "progress.json"
    with open(progress_file, "w") as f:
        json.dump(
            {"completed_shards": completed_shards, "total_docs": total_docs, "total_tokens": total_tokens},
            f, indent=2,
        )


def write_shard(shard_path_prefix: str, docs: list[list[int]], dtype) -> None:
    """Write a list of token-id lists to a shard."""
    builder = IndexedDatasetBuilder(shard_path_prefix + ".bin", dtype=dtype)
    for token_ids in docs:
        builder.add_document(torch.tensor(token_ids, dtype=torch.int32), [len(token_ids)])
    builder.finalize(shard_path_prefix + ".idx")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    dtype = DType.optimal_dtype(tokenizer.vocab_size)
    logger.info(f"EOS token id: {eos_token_id}, vocab size: {tokenizer.vocab_size}, dtype: {dtype}")

    progress = load_progress(output_dir)
    completed_shards = progress["completed_shards"]
    total_docs = progress["total_docs"]
    total_tokens = progress["total_tokens"]

    if total_tokens >= args.token_budget:
        logger.info("Token budget already reached. Nothing to do.")
        return

    logger.info(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}) ...")
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    if total_docs > 0:
        logger.info(f"Skipping {total_docs:,} already-processed docs ...")
        dataset = dataset.skip(total_docs)

    t_start = time.time()
    current_shard_docs = []
    current_shard_tokens = 0

    for batch in dataset.iter(batch_size=1000):
        all_token_ids = tokenizer(batch["text"], add_special_tokens=False).input_ids

        for token_ids in all_token_ids:
            token_ids.append(eos_token_id)
            current_shard_docs.append(token_ids)
            current_shard_tokens += len(token_ids)
            total_tokens += len(token_ids)
            total_docs += 1

        if total_docs % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Docs: {total_docs:,} | Tokens: {total_tokens:,} "
                f"({total_tokens / 1e9:.2f}B / {args.token_budget / 1e9:.0f}B) | "
                f"Elapsed: {elapsed / 3600:.2f}h"
            )

        if current_shard_tokens >= args.shard_size:
            shard_prefix = str(output_dir / f"shard_{completed_shards:05d}")
            logger.info(f"Writing shard {completed_shards} ({current_shard_tokens / 1e9:.2f}B tokens) ...")
            write_shard(shard_prefix, current_shard_docs, dtype)
            completed_shards += 1
            save_progress(output_dir, completed_shards, total_docs, total_tokens)
            current_shard_docs = []
            current_shard_tokens = 0

        if total_tokens >= args.token_budget:
            logger.info(f"Reached token budget ({args.token_budget / 1e9:.0f}B). Stopping.")
            break

    # write final partial shard (if anything left)
    if current_shard_docs:
        shard_prefix = str(output_dir / f"shard_{completed_shards:05d}")
        logger.info(f"Writing final shard {completed_shards} ({current_shard_tokens / 1e9:.2f}B tokens) ...")
        write_shard(shard_prefix, current_shard_docs, dtype)
        completed_shards += 1
        save_progress(output_dir, completed_shards, total_docs, total_tokens)

    elapsed = time.time() - t_start
    logger.info(
        f"Done. Shards: {completed_shards} | Docs: {total_docs:,} | "
        f"Tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B) | "
        f"Time: {elapsed / 3600:.2f}h"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-Edu-dedup to sharded Megatron binary format"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to the HuggingFace tokenizer directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for shards and progress.json",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=160_000_000_000,
        help="Stop after this many tokens total (default: 160B)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000_000_000,
        help="Tokens per shard (default: 10B)",
    )
    args = parser.parse_args()
    main(args)
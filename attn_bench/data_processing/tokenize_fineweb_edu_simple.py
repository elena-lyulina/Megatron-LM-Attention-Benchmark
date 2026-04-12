"""
Simple (no sharding) version of tokenize_fineweb_edu.py for testing throughput.

Streams FineWeb-Edu-dedup, tokenizes with LLaMA 3, and writes a single Megatron
binary. Use this to measure speed before committing to the full sharded run.

Output: <output_path>.bin and <output_path>.idx

Usage:
    python tokenize_fineweb_edu_simple.py \
        --tokenizer-path /path/to/tokenizer \
        --output-path /path/to/output/fineweb_edu_text_document \
        --token-budget 10_000_000
"""

import argparse
import logging
import sys
import time

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

LOG_INTERVAL = 10_000

DATASET_NAME = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"


def main(args):
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    dtype = DType.optimal_dtype(tokenizer.vocab_size)
    logger.info(f"EOS token id: {eos_token_id}, vocab size: {tokenizer.vocab_size}, dtype: {dtype}")

    logger.info(f"Streaming {DATASET_NAME} ({DATASET_CONFIG}) ...")
    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    builder = IndexedDatasetBuilder(args.output_path + ".bin", dtype=dtype)

    total_tokens = 0
    total_docs = 0
    t_start = time.time()

    for batch in dataset.iter(batch_size=1000):
        all_token_ids = tokenizer(batch["text"], add_special_tokens=False).input_ids

        for token_ids in all_token_ids:
            token_ids.append(eos_token_id)
            builder.add_document(torch.tensor(token_ids, dtype=torch.int32), [len(token_ids)])
            total_tokens += len(token_ids)
            total_docs += 1

        if total_docs % LOG_INTERVAL == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Docs: {total_docs:,} | Tokens: {total_tokens:,} "
                f"({total_tokens / 1e9:.2f}B) | "
                f"Speed: {total_tokens / elapsed / 1e6:.1f}M tok/s | "
                f"Elapsed: {elapsed:.1f}s"
            )

        if total_tokens >= args.token_budget:
            logger.info(f"Reached token budget ({args.token_budget / 1e6:.0f}M). Stopping.")
            break

    builder.finalize(args.output_path + ".idx")

    elapsed = time.time() - t_start
    logger.info(
        f"Done. Docs: {total_docs:,} | Tokens: {total_tokens:,} | "
        f"Speed: {total_tokens / elapsed / 1e6:.1f}M tok/s | "
        f"Time: {elapsed:.1f}s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple FineWeb-Edu tokenization for throughput testing")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True, help=".bin and .idx will be appended")
    parser.add_argument("--token-budget", type=int, default=10_000_000, help="default: 10M tokens")
    args = parser.parse_args()
    main(args)
"""
Parallel tokenization of FineWeb-Edu-dedup from local parquet files.

Each worker processes its own subset of parquet files and writes a Megatron binary shard.
A shared counter stops all workers once the global token budget is reached.

Download parquet files first with download_fineweb_edu.py, then run this script.

Output layout:
    <output_dir>/
        shard_00000.bin / shard_00000.idx   ← worker 0
        shard_00001.bin / shard_00001.idx   ← worker 1
        ...

Usage:
    python tokenize_fineweb_edu_native.py \
        --tokenizer-path /path/to/tokenizer \
        --raw-dir /path/to/raw \
        --output-dir /path/to/dedup-fineweb-edu-160B \
        --token-budget 160000000000 \
        --num-workers 128
"""

import argparse
import logging
import multiprocessing
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, DType

DATASET_CONFIG = "fineweb-edu-dedup"
BATCH_SIZE = 1000
LOG_INTERVAL = 50_000  # docs

# Shared counter injected into each worker via Pool initializer
_shared_counter = None


def _init_worker(shared_counter):
    global _shared_counter
    _shared_counter = shared_counter


def get_parquet_paths(raw_dir: str) -> list[str]:
    parquet_dir = Path(raw_dir)
    paths = sorted(str(p) for p in parquet_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {parquet_dir}")
    return paths


def worker_fn(worker_id: int, parquet_urls: list, output_dir: str, tokenizer_path: str, token_budget: int, n_workers: int) -> tuple[int, int]:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker-{worker_id:02d}] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger()

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger.info(f"Starting — {len(parquet_urls)} parquet files assigned")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    dtype = DType.optimal_dtype(tokenizer.vocab_size)

    shard_prefix = str(Path(output_dir) / f"shard_{worker_id:05d}")
    builder = IndexedDatasetBuilder(shard_prefix + ".bin", dtype=dtype)

    total_tokens = 0
    total_docs = 0
    t_start = time.time()
    done = False

    for parquet_url in parquet_urls:
        logger.info(f"Processing {Path(parquet_url).name}")
        ds = load_dataset("parquet", data_files={"train": parquet_url}, split="train", streaming=True)

        for batch in ds.iter(batch_size=BATCH_SIZE):
            all_token_ids = tokenizer(batch["text"], add_special_tokens=False).input_ids

            batch_tokens = 0
            for token_ids in all_token_ids:
                token_ids.append(eos_token_id)
                builder.add_document(torch.tensor(token_ids, dtype=torch.int32), [len(token_ids)])
                batch_tokens += len(token_ids)
                total_docs += 1
            total_tokens += batch_tokens

            with _shared_counter.get_lock():
                _shared_counter.value += batch_tokens
                global_total = _shared_counter.value

            if global_total >= token_budget:
                logger.info("Global token budget reached. Stopping.")
                done = True
                break

        elapsed = time.time() - t_start
        speed = total_tokens / elapsed if elapsed > 0 else 0
        remaining = max(token_budget - global_total, 0)
        eta_seconds = remaining / (speed * n_workers) if speed > 0 else float("inf")
        logger.info(
            f"Finished {Path(parquet_url).name} | "
            f"Docs: {total_docs:,} | Tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M) | "
            f"Speed: {speed / 1e6:.1f}M tok/s | "
            f"ETA: {eta_seconds / 60:.1f}min"
        )

        if done:
            break

    builder.finalize(shard_prefix + ".idx")

    elapsed = time.time() - t_start
    logger.info(
        f"Done. Docs: {total_docs:,} | Tokens: {total_tokens:,} ({total_tokens / 1e6:.1f}M) | "
        f"Speed: {total_tokens / elapsed / 1e6:.1f}M tok/s"
    )
    return total_tokens, total_docs


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [main] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("main")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading parquet files from {args.raw_dir} ...")
    parquet_urls = get_parquet_paths(args.raw_dir)
    logger.info(f"Found {len(parquet_urls)} parquet files for {DATASET_CONFIG}")

    # Split files round-robin across workers
    n_workers = min(args.num_workers, len(parquet_urls))
    worker_files = [[] for _ in range(n_workers)]
    for i, url in enumerate(parquet_urls):
        worker_files[i % n_workers].append(url)

    logger.info(
        f"Splitting across {n_workers} workers | "
        f"Budget: {args.token_budget / 1e9:.1f}B tokens"
    )

    shared_counter = multiprocessing.Value("l", 0)  # signed long, counts tokens globally

    worker_args = [
        (i, worker_files[i], str(output_dir), args.tokenizer_path, args.token_budget, n_workers)
        for i in range(n_workers)
    ]

    t_start = time.time()
    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(shared_counter,),
    ) as pool:
        results = pool.starmap(worker_fn, worker_args)

    total_tokens = sum(r[0] for r in results)
    total_docs = sum(r[1] for r in results)
    elapsed = time.time() - t_start
    logger.info(
        f"All workers done. "
        f"Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B) | "
        f"Total docs: {total_docs:,} | "
        f"Time: {elapsed / 3600:.2f}h | "
        f"Combined speed: {total_tokens / elapsed / 1e6:.1f}M tok/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parallel tokenization of FineWeb-Edu-dedup to Megatron binary format"
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--token-budget",
        type=int,
        default=160_000_000_000,
        help="Global token budget across all workers (default: 160B)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Number of parallel worker processes (default: 20)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Path to local directory containing downloaded parquet files.",
    )
    args = parser.parse_args()
    main(args)
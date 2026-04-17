"""
Tokenize FineWeb-Edu-dedup parquet files to Megatron binary format using datatrove.

Each task writes one shard (shard_XXXXX.bin / shard_XXXXX.idx).
Datatrove handles format correctness, checkpointing, and stats logging.

Usage:
    python tokenize_fineweb_edu_datatrove.py \
        --tokenizer-path /path/to/tokenizer \
        --raw-dir /path/to/raw/fineweb-edu-dedup \
        --output-dir /path/to/tokenized/fineweb-edu-dedup-160B \
        --token-budget 160000000000 \
        --num-workers 128
"""

import argparse
import logging
import multiprocessing
import struct
import time
from pathlib import Path

import numpy as np

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader

from attn_bench.data_processing.tokenization.budgeted_tokenizer import BudgetedMegatronTokenizer

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def read_idx_stats(idx_path: Path) -> tuple[int, int]:
    """Return (total_tokens, total_docs) from a Megatron .idx file."""
    with open(idx_path, "rb") as f:
        assert f.read(9) == _INDEX_HEADER
        f.read(8)  # version
        f.read(1)  # dtype code
        sequence_count = struct.unpack("<Q", f.read(8))[0]
        document_count = struct.unpack("<Q", f.read(8))[0]
        lengths = np.frombuffer(f.read(sequence_count * 4), dtype=np.int32)
    return int(lengths.sum()), document_count - 1  # document_count is always n_docs + 1


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [main] %(message)s")
    logger = logging.getLogger("main")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    node_budget = args.token_budget // args.num_nodes
    shared_counter = multiprocessing.Value('q', 0)

    pipeline = [
        ParquetReader(
            data_folder=args.raw_dir,
            text_key="text",
            glob_pattern="*.parquet",
        ),
        BudgetedMegatronTokenizer(
            shared_counter=shared_counter,
            node_budget=node_budget,
            output_folder=str(output_dir),
            tokenizer_name_or_path=args.tokenizer_path,
            save_filename="shard",
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.num_workers,
        workers=args.num_workers,
        logging_dir=str(output_dir / "logs"),
    )

    t_start = time.time()
    executor.run()
    elapsed = time.time() - t_start

    # Summarise across all shards
    total_tokens, total_docs = 0, 0
    for idx_path in sorted(output_dir.glob("*.idx")):
        tokens, docs = read_idx_stats(idx_path)
        total_tokens += tokens
        total_docs += docs

    logger.info(
        f"Done. "
        f"Total tokens: {total_tokens:,} ({total_tokens / 1e9:.2f}B) | "
        f"Total docs: {total_docs:,} | "
        f"Speed: {total_tokens / elapsed / 1e6:.1f}M tok/s | "
        f"Time: {elapsed / 3600:.2f}h"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-Edu-dedup to Megatron binary format using datatrove"
    )
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--raw-dir", type=str, required=True,
                        help="Directory containing downloaded parquet files")
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
        default=128,
        help="Number of parallel tasks / output shards (default: 128)",
    )
    args = parser.parse_args()
    main(args)
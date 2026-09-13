"""
Lay out a long-document jsonl (extract_long_docs.py output) as a rep_0_token.jsonl bucket,
so prefix_extraction_inference.py / measure_mem.slurm can generate from unseen documents
via JSONL_DIR. Subsamples with the same fixed seed and count as the long-context perplexity
runs (long_inference.sample_lines, SAMPLE_SEED = 42), so both use the same documents.

Usage:
    python attn_bench/data_processing/make_unseen_rep_jsonl.py \
        --src /users/$USER/store/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.75_unseen_long/long_24576_32768.jsonl \
        --dst-dir /users/$USER/store/datasets/tokenized/fineweb_unseen_rep_jsonl \
        --max-samples 660
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SAMPLE_SEED = 42  # keep equal to attn_bench.evaluation.long_inference.SAMPLE_SEED
BOS_TOKEN_ID = 128000


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", required=True, type=Path, help="long_<min>_<max>.jsonl from extract_long_docs.py")
    parser.add_argument("--dst-dir", required=True, type=Path, help="folder to hold rep_0_token.jsonl (pass as JSONL_DIR)")
    parser.add_argument("--max-samples", type=int, default=660,
                        help="seeded random subsample, as the perplexity runs (default 660)")
    parser.add_argument("--rep", type=int, default=0, help="bucket name to write (default rep_0)")
    args = parser.parse_args()

    with open(args.src) as f:
        lines = f.readlines()
    total = len(lines)
    if args.max_samples is not None and total > args.max_samples:
        lines = random.Random(SAMPLE_SEED).sample(lines, args.max_samples)

    shortest = min(len(json.loads(line)["input_ids"]) for line in lines)
    no_bos = sum(json.loads(line)["input_ids"][0] != BOS_TOKEN_ID for line in lines)

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    dst = args.dst_dir / f"rep_{args.rep}_token.jsonl"
    dst.write_text("".join(lines))
    print(f"{len(lines)} of {total} documents -> {dst}")
    print(f"shortest document {shortest} tokens; documents not starting with BOS: {no_bos}")


if __name__ == "__main__":
    main()

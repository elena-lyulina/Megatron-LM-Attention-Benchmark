"""
Convert repetition_buckets.jsonl and unseen_buckets.jsonl into per-bucket
JSONL files for the PDM sparse memorization inference.

Output files: rep_0_token.jsonl (unseen), rep_1_token.jsonl, ..., rep_256_token.jsonl
Each line: {"input_ids": [...]}

Usage:
    python attn_bench/data_processing/books/build_rep_jsonl.py \
        --repetition-buckets /path/to/repetition_buckets.jsonl \
        --unseen-buckets     /path/to/unseen_buckets.jsonl \
        --output-dir         /path/to/gutenberg_rep_jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def write_bucket(records, path):
    with open(path, "w") as f:
        for token_ids in records:
            f.write(json.dumps({"input_ids": token_ids}) + "\n")
    print(f"{path.name}: {len(records)} sequences → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetition-buckets", required=True)
    parser.add_argument("--unseen-buckets", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # repetition_buckets.jsonl → rep_{bucket_rep}_token.jsonl
    buckets = defaultdict(list)
    with open(args.repetition_buckets) as f:
        for line in f:
            r = json.loads(line)
            buckets[r["bucket_rep"]].append(r["token_ids"])

    for rep, records in sorted(buckets.items()):
        write_bucket(records, out_dir / f"rep_{rep}_token.jsonl")

    # unseen_buckets.jsonl → rep_0_token.jsonl
    unseen = []
    with open(args.unseen_buckets) as f:
        for line in f:
            unseen.append(json.loads(line)["token_ids"])

    write_bucket(unseen, out_dir / "rep_0_token.jsonl")


if __name__ == "__main__":
    main()
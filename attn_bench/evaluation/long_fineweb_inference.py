"""
Long FineWeb-Edu inference: per-position NLL on one jsonl of naturally-long docs
(from extract_long_docs.py), used verbatim -- no prefix/suffix framing, no repetition
axis. See long_inference.py for the shared engine.

Usage (via torchrun):
    torchrun --nproc_per_node=4 attn_bench/evaluation/long_fineweb_inference.py \
        --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $OUT_DIR --data-file $FINEWEB_LONG_DIR/long_24576_32768.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attn_bench.evaluation.long_inference import (add_common_args, run_main,
                                                  sample_lines)

# Comparable in size to one Gutenberg repetition bucket -- enough to see the trend
# without running on every long doc in the partition.
DEFAULT_MAX_SAMPLES = 660

### DATA ###

def load_flat_docs(path: Path, max_length: int | None, max_samples: int | None) -> list:
    """Read extract_long_docs.py's jsonl output: input_ids used as-is (already has BOS).

    Returns (tokens, doc_id) pairs -- doc_id is only consumed by --store-individual.
    """
    sequences = []
    for line in sample_lines(path, max_samples):
        record = json.loads(line)
        tokens = record["input_ids"]
        if max_length is not None:
            tokens = tokens[:max_length]
        sequences.append((tokens, record["doc_id"]))
    return sequences


### CLI ###

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", required=True, help="jsonl file from extract_long_docs.py")
    add_common_args(p, max_samples_default=DEFAULT_MAX_SAMPLES)
    return p.parse_args()


def main():
    args = parse_args()
    key = Path(args.data_file).stem
    items = [(key, Path(args.data_file))]
    run_main(args, items, load_flat_docs, {"data_file": args.data_file})


if __name__ == "__main__":
    main()

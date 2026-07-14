"""
Long Gutenberg inference: per-position NLL on [BOS, sample, suffix] excerpts, one
repetition bucket per --repetitions value. See long_inference.py for the shared engine.

Usage (via torchrun):
    torchrun --nproc_per_node=4 attn_bench/evaluation/long_gutenberg_inference.py \
        --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $OUT_DIR --data-folder $GUTENBERG_LONG_JSONL_DIR \
        --repetitions 0,1,2,4,8,16,32,64,128,256 --max-length 32768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from attn_bench.evaluation.inference_common import BOS_TOKEN_ID, find_rep_paths
from attn_bench.evaluation.long_inference import (add_common_args, run_main,
                                                  sample_lines)

### DATA ###

def load_long_sequence(path: Path, max_length: int | None, max_samples: int | None) -> list:
    """Build [BOS, sample, suffix] sequences from a long-context bucket file.

    The record stores input_ids = [BOS] + prefix + sample + suffix + [EOS] with
    sample_offset marking the first sample token. We drop the prefix (start at the
    sample) and the trailing EOS, prepend a fresh BOS, then cap at max_length.

    Returns (tokens, book_id) pairs -- book_id is only consumed by --store-individual.
    """
    sequences = []
    for line in sample_lines(path, max_samples):
        record = json.loads(line)
        sample_start_index = record["sample_offset"]
        long_sample_len = record["sample_len"] + record["extra_suffix_len"]
        # input_ids = [bos_id] + extra_prefix + content + extra_suffix + [eos_id]
        long_sample_tokens = record["input_ids"][sample_start_index: sample_start_index + long_sample_len]
        assert len(long_sample_tokens) == long_sample_len, (
            f"{path.name}: short record ({len(long_sample_tokens)} != {long_sample_len})"
        )
        sequence_tokens = [BOS_TOKEN_ID] + long_sample_tokens
        if max_length is not None:
            sequence_tokens = sequence_tokens[:max_length]
        sequences.append((sequence_tokens, str(record["book_id"])))
    return sequences


### CLI ###

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-folder", required=True, help="Directory of long-context rep_*_token.jsonl files")
    p.add_argument("--repetitions", default="0", help="Comma-separated buckets, e.g. 0,1,2,4,8,16,32,64,128,256")
    add_common_args(p)
    return p.parse_args()


def rep_key(rep: int) -> str:
    return f"rep_{rep}"


def main():
    args = parse_args()
    reps = {int(r) for r in args.repetitions.split(",")}
    paths = find_rep_paths(Path(args.data_folder), reps)
    items = [(rep_key(int(p.stem.split("_")[1])), p) for p in paths]
    run_main(args, items, load_long_sequence, {"repetitions": args.repetitions})


if __name__ == "__main__":
    main()

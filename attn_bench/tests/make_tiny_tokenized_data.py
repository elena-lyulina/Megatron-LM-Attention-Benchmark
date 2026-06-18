#!/usr/bin/env python3
"""Generate a tiny tokenized dataset (2 .bin/.idx prefixes) for the data-exhaustion test.

The goal is a small NON-repeating blend: when passed via `--data-path prefix0 prefix1`
with no weights, Megatron caps the blended dataset at the sum of the per-file one-epoch
sample counts. With ~20k tokens per file this is only a few dozen samples, so training
exhausts the data in a handful of iterations and exercises the clean-exit + final-save
path in checkpoint_and_decide_exit.

We don't need real tokens here, only valid ids (< vocab) with document boundaries, so we
write random ids and an EOD id at the end of each document.
"""

import argparse
import os

import numpy as np
import torch

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder, get_bin_path, get_idx_path


def build_file(prefix, num_tokens, min_doc, max_doc, eod_id, vocab_cap, rng):
    """Write one .bin/.idx prefix with random-length docs until num_tokens is reached."""
    builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=np.int32)
    written = 0
    num_docs = 0
    while written < num_tokens:
        doc_len = int(rng.integers(min_doc, max_doc + 1))
        # random ids in [1, vocab_cap), last token is EOD to mark the document end
        ids = rng.integers(1, vocab_cap, size=doc_len - 1, dtype=np.int32)
        doc = np.concatenate([ids, np.array([eod_id], dtype=np.int32)])
        builder.add_item(torch.from_numpy(doc))
        builder.end_document()
        written += doc_len
        num_docs += 1
    builder.finalize(get_idx_path(prefix))
    print(f"  {prefix}: {num_docs} docs, {written} tokens")
    return num_docs, written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-files", type=int, default=2)
    p.add_argument("--tokens-per-file", type=int, default=20000)
    p.add_argument("--min-doc", type=int, default=50)
    p.add_argument("--max-doc", type=int, default=400)
    p.add_argument("--eod-id", type=int, default=0)
    p.add_argument("--vocab-cap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    total_tokens = 0
    print(f"Writing {args.num_files} tiny tokenized files to {args.out_dir}")
    for i in range(args.num_files):
        prefix = os.path.join(args.out_dir, f"tiny_{i}")
        _, written = build_file(
            prefix, args.tokens_per_file, args.min_doc, args.max_doc,
            args.eod_id, args.vocab_cap, rng,
        )
        total_tokens += written
    print(f"Done. {total_tokens} total tokens across {args.num_files} files.")


if __name__ == "__main__":
    main()

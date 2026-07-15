"""
Compute the packed-segment ("chunk") length histogram for a single tokenized Megatron
dataset, as it would be seen during training: one full epoch, packed by concatenating
all documents and chopping into fixed seq-len windows (the same way GPTDataset feeds
training samples).

Builds the real GPTDataset/BlendedDataset objects (CPU-only, no GPU needed) via
BlendedMegatronDatasetBuilder, requesting a full epoch (num_samples=None) so the result
is an intrinsic property of the dataset + seq_len, independent of any specific training
run's blend or sample count -- safe to compute once per dataset and reuse across
whichever blends that dataset later appears in.

A "chunk" is one contiguous document segment within one seq-len-token training sample --
it ends at whichever comes first: the document's end, or the sample window's end. Since
every chunk length is an integer in [1, seq_len], the result is stored as a histogram
(one array of length seq_len) rather than raw per-chunk lengths -- lossless for the
distribution, and tiny (a few tens of KB) regardless of corpus size.

Usage:
    python packed_chunk_stats.py \
        --dataset-dir /path/to/fineweb-edu-dedup-160B-datatrove_0.25 \
        --tokenizer-path /path/to/tokenizer \
        --cache-path /path/to/dataset/cache \
        --seq-len 8192 \
        --out-dir /path/to/output

Writes <name>_chunk_len_hist.npy and <name>_run_metadata.json directly into --out-dir
(name = basename of --dataset-dir) -- flat files, not a per-dataset subfolder, so --out-dir
can be shared across many datasets.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
from dataset_doc_stats import shard_prefixes

from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import \
    BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.core.tokenizers.megatron_tokenizer import MegatronTokenizer

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]
SPLIT = "100,0,0"  # only the (single, full-epoch) train split is used


def build_full_epoch_dataset(dataset_dir, tokenizer_path, cache_path, seq_len, seed):
    """Returns the top-level dataset for one full epoch of dataset_dir (BlendedDataset if the
    dir has multiple .bin shards, plain GPTDataset if it's a single shard)."""
    prefixes = sorted(shard_prefixes(dataset_dir))
    tokenizer = MegatronTokenizer.from_pretrained(tokenizer_path=tokenizer_path)
    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_len,
        blend=(prefixes, None),
        split=SPLIT,
        path_to_cache=cache_path,
        tokenizer=tokenizer,
        reset_position_ids=True,
        reset_attention_mask=False,
        eod_mask_loss=True,
        create_attention_mask=False,
        num_dataset_builder_threads=4,
    )
    # num_samples=None -> one full epoch (every document used exactly once), an intrinsic
    # property of the dataset rather than of any particular training run's sample count.
    train_ds, _, _ = BlendedMegatronDatasetBuilder(
        GPTDataset, [None, None, None], lambda: True, config
    ).build()
    return train_ds


def chunk_length_histogram(ds: GPTDataset) -> np.ndarray:
    """Histogram (length seq_len, index i = count of length-(i+1) chunks) of packed segment
    lengths over this GPTDataset's full epoch."""
    seq_len = ds.config.sequence_length
    num_samples = ds.sample_index.shape[0] - 1

    seq_lengths = ds.dataset.sequence_lengths.astype(np.int64)
    doc_indices = ds.dataset.document_indices
    doc_cumsum = np.concatenate([[0], np.cumsum(seq_lengths)])
    doc_lengths_by_id = doc_cumsum[doc_indices[1:]] - doc_cumsum[doc_indices[:-1]]
    flat_doc_lengths = doc_lengths_by_id[np.asarray(ds.document_index)]

    stream_len = num_samples * seq_len
    doc_boundaries = np.cumsum(flat_doc_lengths)
    doc_boundaries = doc_boundaries[doc_boundaries < stream_len]
    sample_boundaries = np.arange(0, num_samples + 1) * seq_len
    lengths = np.diff(np.union1d(doc_boundaries, sample_boundaries))

    return np.bincount(lengths, minlength=seq_len + 1)[1 : seq_len + 1]


def dataset_histogram(train_ds) -> np.ndarray:
    """Sums per-shard histograms into one histogram for the whole dataset dir."""
    shards = train_ds.datasets if isinstance(train_ds, BlendedDataset) else [train_ds]
    seq_len = shards[0].config.sequence_length
    hist = np.zeros(seq_len, dtype=np.int64)
    for shard in shards:
        hist += chunk_length_histogram(shard)
    return hist


def stats_from_histogram(hist: np.ndarray) -> dict:
    lengths = np.arange(1, len(hist) + 1)
    total = int(hist.sum())
    mean = float((lengths * hist).sum() / total)
    variance = float((((lengths - mean) ** 2) * hist).sum() / total)
    cum = np.cumsum(hist)
    nonzero = np.flatnonzero(hist)
    percentiles = {}
    for p in PERCENTILES:
        target = p / 100 * total
        percentiles[p] = int(lengths[np.searchsorted(cum, target)])
    return {
        "chunks": total,
        "mean": mean,
        "std": variance**0.5,
        "min": int(lengths[nonzero[0]]),
        "max": int(lengths[nonzero[-1]]),
        "percentiles": percentiles,
    }


def print_stats(stats: dict) -> None:
    print(f"  chunks:       {stats['chunks']:,}")
    print(f"  mean:         {stats['mean']:,.1f}")
    print(f"  std:          {stats['std']:,.1f}")
    print(f"  min / max:    {stats['min']:,} / {stats['max']:,}")
    print("  percentiles:  " + "  ".join(f"p{p}={v:,}" for p, v in stats["percentiles"].items()))


def write_run_metadata(out_dir: str, name: str, args: argparse.Namespace, stats: dict) -> None:
    with open(os.path.join(out_dir, f"{name}_run_metadata.json"), "w") as f:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_dir": args.dataset_dir,
            "tokenizer_path": args.tokenizer_path,
            "cache_path": args.cache_path,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "split": SPLIT,
            "stats": stats,
        }, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--out-dir", required=True,
                         help="Shared output dir; writes <name>_chunk_len_hist.npy + "
                              "<name>_run_metadata.json (name = basename of --dataset-dir)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    name = os.path.basename(args.dataset_dir.rstrip("/"))
    print(f"### {name} ###\n  {args.dataset_dir}")

    train_ds = build_full_epoch_dataset(
        args.dataset_dir, args.tokenizer_path, args.cache_path, args.seq_len, args.seed
    )
    hist = dataset_histogram(train_ds)
    stats = stats_from_histogram(hist)
    print_stats(stats)

    hist_path = os.path.join(args.out_dir, f"{name}_chunk_len_hist.npy")
    np.save(hist_path, hist)
    print(f"  saved -> {hist_path}")

    write_run_metadata(args.out_dir, name, args, stats)
    print(f"  saved -> {os.path.join(args.out_dir, f'{name}_run_metadata.json')}")


if __name__ == "__main__":
    main()
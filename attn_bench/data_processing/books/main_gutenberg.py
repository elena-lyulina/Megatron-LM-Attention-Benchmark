"""
Sample one 8192-token excerpt per Gutenberg book.

Pipeline:
  1.  init_columns              — initialise all fields with defaults (parallel)
  2.  dedup_id_and_title        — mark duplicates by ID and normalized title (sequential)
  3.  cluster_titles            — keep only one book per title cluster (sequential)
  4.  normalize_text            — fix encoding, collapse whitespace (parallel)
  5.  set_content_bounds        — compute content start/end (parallel)
  6.  mark_too_short            — skip books below minimum length (parallel)
  7.  sample_window             — pick a random WINDOW_CHARS window (parallel)
  8.  split_and_stat_paragraphs — split window into paragraphs, compute line stats (parallel)
  9.  classify_paragraphs       — classify each paragraph type with NLI model (GPU)
  10. find_excerpt_start        — align window to sentence boundary (parallel)
  11. tokenize_excerpt          — tokenize and trim to SEQ_LEN tokens (parallel)
  12. verify_token_count        — drop books with wrong token count (parallel)
  13. minhash_dedup             — mark near-duplicate excerpts by text similarity (sequential LSH)
  14. write_stats               — aggregate stats + write stats/ folder
  15. filter + write            — keep only keep=True books

Usage:
  python prepare_gutenberg.py \
      --raw-dir /path/to/raw/gutenberg-en \
      --output-dir /path/to/gutenberg-en/
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from pathlib import Path
import multiprocessing as mp

from datasets import load_dataset, load_from_disk

from .dedup_cluster_titles import add_title_embeddings, build_title_clusters, \
    write_clusters_stats, get_encode_devices
from .columns import Col, init_columns
from .dedup_id_title import dedup_id, dedup_title
from .checkpoint import run_step, dir_size_gb
from .find_excerpt_start import load_punkt
from .normalize import normalize_text
from .sample_excerpt import sample_window
from .set_content_bounds import set_content_bounds, mark_too_short
from .strip_gutenberg import strip_gutenberg_markers, verify_no_project_gutenberg, \
    write_gutenberg_occurrences
from .tokenize_excerpts import TOKENIZER_ID, load_tokenizer
from .unwrap_new_lines import split_and_stat_paragraphs, write_paragraph_line_stats, \
    vote_paragraph_unwrap, write_unwrap_stats


# could have a stale cache
HF_CACHE_RESULTS = True


### STATS ###

def print_stats(ds):
    counts = Counter(ds[Col.SKIP_REASON])
    kept = counts.pop(None, 0)
    total = kept + sum(counts.values())
    print(f"\nSTATS  total={total:,}  kept={kept:,}  {'  '.join(f'{k}={v:,}' for k, v in counts.items())}")


### I/O ###

def load_ds(raw_dir: str):
    gz_files = sorted(Path(raw_dir).glob("*.gz"))
    if not gz_files:
        raise FileNotFoundError(f"No gz files found in {raw_dir}")
    print(f"Found {len(gz_files)} gz files in {raw_dir}")
    return load_dataset(
        "json",
        data_files={"train": [str(p) for p in gz_files]},
        split="train",
    )

#  todo: remove? i think it just dumps everything and now we have checkpoints?
def write_processing_results(ds, output_dir: Path):
    path = output_dir / "processed.parquet"
    if path.exists():
        print(f"Overwriting existing processed.parquet")
    print(f"Writing {len(ds):,} rows -> {path}")
    ds.remove_columns(["text", Col.TOKEN_IDS, Col.TEXT_EXCERPT]).to_parquet(str(path))
    print(f"Done")


def write_tokenized_excerpts(ds, output_dir: Path, num_proc: int):
    path = output_dir / "sampled.jsonl"
    if path.exists():
        print(f"Overwriting existing sampled.jsonl")
    ds_kept = ds.filter(lambda x: x[Col.KEEP], num_proc=num_proc)
    print(f"Writing {len(ds_kept):,} kept books -> {path}")
    ds_kept.select_columns([Col.BOOK_ID, Col.TOKEN_IDS, Col.TEXT_EXCERPT]).to_json(str(path), lines=True)
    print(f"Done")


### CHECKPOINTING ###

def gz_size_gb(raw_dir: str) -> float:
    return sum(p.stat().st_size for p in Path(raw_dir).glob("*.gz")) / 1e9


def print_size_summary(raw_dir: str, step_sizes: dict):
    raw_gb = gz_size_gb(raw_dir)
    saved = {name: size for name, size in step_sizes.items() if size is not None}
    total_gb = sum(saved.values())
    n = len(saved)
    avg = total_gb / n if n else 0
    print(f"\nraw gz:       {raw_gb:.2f} GB")
    print(f"checkpoints:  {total_gb:.2f} GB over {n} steps (avg {avg:.2f} GB/step)")
    for name, size in step_sizes.items():
        size_str = f"{size:.2f} GB" if size is not None else "skipped"
        print(f"  {name:<35} {size_str}")


### MAIN PIPELINE ###






def pipeline(raw_dir: str, tokenizer_path: str, output_dir: Path, ckpt_dir: Path | None, stats_dir: Path | None):
    t_start = time.time()
    num_proc = os.cpu_count()
    punkt = load_punkt()
    tokenizer, bos_id, eos_id = load_tokenizer(tokenizer_path)

    step_ckpt_sizes = {}
    ds = load_ds(raw_dir)
    print(f"Loaded: {len(ds):,} books")

    def step(step_name, step_fn, stats_fn=None, save_ckpt=True):
        nonlocal ds
        ckpt_path = ckpt_dir / step_name if (ckpt_dir and save_ckpt) else None
        ds, size = run_step(step_fn, ds, ckpt_path)
        if ckpt_dir:
            step_ckpt_sizes[step_name] = size
        if stats_dir and stats_fn:
            stats_path = stats_dir / step_name
            stats_path.mkdir(parents=True, exist_ok=True)
            stats_fn(ds, stats_path)

    step("01_init_columns", lambda ds: ds.map(init_columns, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="init columns"), save_ckpt=False)
    step("02_dedup_id", dedup_id, save_ckpt=False)
    step("03_dedup_title", dedup_title, save_ckpt=False)
    step("04_add_title_embeddings", add_title_embeddings)  # could be parallelized but it's pretty fast already
    step("05_build_title_clusters", build_title_clusters, stats_fn=write_clusters_stats, save_ckpt=False)
    step("06_strip_gutenberg_markers", lambda ds: ds.map(strip_gutenberg_markers, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="stripping gutenberg markers"), save_ckpt=False)
    #  add new line removal but nor for poetry, ask the model to detect poetry
    #  use a model for text normalization?
    step("07_normalize_text", lambda ds: ds.map(normalize_text, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="normalizing text"))
    step("08_set_content_bounds", lambda ds: ds.map(set_content_bounds, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="setting content bounds"), save_ckpt=False)
    step("09_mark_too_short", lambda ds: ds.map(mark_too_short, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="marking too short"), save_ckpt=False)
    step("10_verify_no_gutenberg", lambda ds: ds.map(verify_no_project_gutenberg, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="verifying no gutenberg"), stats_fn=write_gutenberg_occurrences, save_ckpt=False)

    # step("10_compute_chunk_sigs", lambda ds: ds.map(partial(compute_chunk_signatures, chunk_size=CHUNK_WORDS, num_perm=NUM_PERM), num_proc=num_proc, load_from_cache_file=CACHE_RESULTS, desc="computing chunk signatures"))
    # step("11_dedup_chunks_minhash", dedup_chunks_minhash)
    #
    step("12_sample_window", lambda ds: ds.map(sample_window, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="sampling window"))
    step("13_split_and_stat_paragraphs", lambda ds: ds.map(split_and_stat_paragraphs, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="computing paragraph stats"), stats_fn=write_paragraph_line_stats)
    # step("14_classify_paragraphs", classify_paragraphs, stats_fn=write_classification_stats)
    step("14_vote_paragraph_unwrap", lambda ds: ds.map(vote_paragraph_unwrap, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="voting paragraph unwrap"), stats_fn=write_unwrap_stats)

    # step("15_find_excerpt_start", lambda ds: ds.map(partial(find_excerpt_start, punkt=punkt), num_proc=num_proc, load_from_cache_file=CACHE_RESULTS, desc="finding excerpt start"), stats_fn=write_no_excerpt_start_stats)
    # step("16_tokenize_excerpt", lambda ds: ds.map(partial(tokenize_excerpt, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id), num_proc=num_proc, load_from_cache_file=CACHE_RESULTS, desc="tokenizing excerpt"))
    # step("17_verify_tokenization", lambda ds: ds.map(partial(verify_tokenization, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id), num_proc=num_proc, load_from_cache_file=CACHE_RESULTS, desc="verifying tokenization"))
    #
    # step("18_compute_excerpt_sigs", lambda ds: ds.map(compute_excerpt_signatures, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="computing excerpt minhash signatures"))
    # step("19_dedup_excerpts_minhash", dedup_excerpts_minhash, stats_fn=write_similar_excerpts)

    # output_dir.mkdir(parents=True, exist_ok=True)
    # write_tokenized_excerpts(ds, output_dir, num_proc)

    print_stats(ds)

    if stats_dir:
        stats_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_dir and step_ckpt_sizes:
        print_size_summary(raw_dir, step_ckpt_sizes)

    print(f"\nTime: {(time.time() - t_start) / 60:.1f}min  output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sample one 8192-token excerpt per Gutenberg book")
    parser.add_argument("--tokenizer-path", type=str, default=TOKENIZER_ID)
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--checkpoint", action="store_true", help="save/load per-step checkpoints under output-dir/checkpoints/")
    parser.add_argument("--no-stats", action="store_true", help="skip writing stats")
    args = parser.parse_args()
    ckpt_dir = Path(args.output_dir) / "checkpoints" if args.checkpoint else None
    stats_dir = Path(args.output_dir) / "stats" if not args.no_stats else None
    pipeline(
        raw_dir=args.raw_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=Path(args.output_dir),
        ckpt_dir=ckpt_dir,
        stats_dir=stats_dir,
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
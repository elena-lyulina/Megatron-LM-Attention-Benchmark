"""
Full processing pipeline for laion/Project-Gutenberg.

Pipeline:
  1.  init_columns           — normalise schema to Col fields, filter non-English
  2.  extract_text           — epub bytes -> plain text via inscriptis
  3.  dedup_id               — mark duplicates by ID (sequential)
  4.  dedup_title            — mark duplicates by normalized title (sequential)
  5.  add_title_embeddings   — embed titles with sentence-transformers (GPU)
  6.  build_title_clusters   — keep only one book per title cluster (sequential)
  7.  strip_gutenberg         — strip any Project Gutenberg preamble/footer
  8.  normalize_text         — fix encoding, collapse whitespace
  9.  set_content_bounds     — compute content start/end
  10. mark_too_short         — skip books below minimum length
  11. verify_no_gutenberg    — drop books with residual Gutenberg text
  12. compute_content_chunk_sigs — compute per-chunk minhash signatures over book content
  13. dedup_content_minhash      — mark near-duplicate books by chunk similarity (sequential LSH)
  14. sample_window          — pick a random WINDOW_CHARS window
  15. find_excerpt_start     — align window to sentence boundary
  16. tokenize_excerpt       — tokenize and trim to SEQ_LEN tokens
  17. verify_tokenization    — verify round-trip tokenization
  18. compute_excerpt_chunk_sigs — compute per-chunk minhash signatures over TEXT_EXCERPT
  19. dedup_excerpts_minhash     — mark near-duplicate excerpts by exact chunk Jaccard (sequential LSH)
  20. score_perplexity       — score each excerpt with FineWeb-Edu LLaMA 1B (GPU, optional)

Usage:
  python -m attn_bench.data_processing.books.main_gutenberg_laion \
      --dataset-dir /path/to/gutenberg-laion/raw-hf-7000 \
      --output-dir /path/to/gutenberg-laion/processed

  # with perplexity scoring (requires GPU + Megatron env):
  python -m attn_bench.data_processing.books.main_gutenberg_laion \
      --dataset-dir /path/to/gutenberg-laion/raw-hf-7000 \
      --output-dir /path/to/gutenberg-laion/processed \
      --checkpoint \
      --megatron-ckpt-dir /path/to/fineweb-160B/llama3-1b-fineweb160B/checkpoints
"""
from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import os
import re
import time
import zipfile
from collections import Counter
from functools import partial
from pathlib import Path

from inscriptis import get_text as html_to_text
from datasets import load_dataset

from .checkpoint import run_step
from .columns import Col, DEFAULTS
from .dedup_cluster_titles import add_title_embeddings, build_title_clusters, write_clusters_stats
from .dedup_id_title import dedup_id, dedup_title, write_dedup_id_stats, write_dedup_title_stats
from .dedup_minhash import compute_content_chunk_signatures, dedup_content_minhash, compute_excerpt_chunk_signatures, dedup_excerpts_minhash
from .find_excerpt_start import load_punkt, find_excerpt_start, write_no_excerpt_start_stats
from .normalize import normalize_text
from .sample_excerpt import sample_window
from .score_perplexity import load_model, score_perplexity, write_perplexity_stats
from .set_content_bounds import set_content_bounds, mark_too_short, write_content_bounds_samples, write_too_short_stats
from .strip_gutenberg import strip_gutenberg_markers, verify_no_project_gutenberg, write_gutenberg_occurrences, write_gutenberg_strip_stats
from .tokenize_excerpts import TOKENIZER_ID, load_tokenizer, tokenize_excerpt, verify_tokenization, write_tokenize_stats, write_verify_stats


HF_CACHE_RESULTS = False


### STATS ###

def print_stats(ds):
    counts = Counter(ds[Col.SKIP_REASON])
    kept = counts.pop(None, 0)
    total = kept + sum(counts.values())
    print(f"\nSTATS  total={total:,}  kept={kept:,}  {'  '.join(f'{k}={v:,}' for k, v in counts.most_common())}")


### INIT COLUMNS ###

def init_columns_laion(book):
    for col, default in DEFAULTS.items():
        book.setdefault(col, default)

    book[Col.BOOK_ID] = str(book.get("gutenberg_id") or book.get("id") or "unknown")

    title = (book.get("title") or "").strip()
    if not title:
        book[Col.BOOK_TITLE] = "Unknown Title"
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "no_title"
    else:
        book[Col.BOOK_TITLE] = title

    lang = (book.get("language") or "").strip().lower()
    if book[Col.KEEP] and lang and lang not in ("en", "english", "eng"):
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "non_english"

    return book


def write_init_columns_stats(ds, stats_dir: Path):
    no_title = [r for r in ds if r[Col.SKIP_REASON] == "no_title"]
    non_english = [r for r in ds if r[Col.SKIP_REASON] == "non_english"]
    if not no_title and not non_english:
        return
    lang_counts = Counter((r.get("language") or "unknown").strip().lower() for r in non_english)
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "init_columns.txt"
    with open(path, "w") as f:
        f.write(f"no_title: {len(no_title):,}\n")
        f.write(f"non_english: {len(non_english):,}\n\n")
        f.write("language breakdown:\n")
        for lang, count in lang_counts.most_common():
            f.write(f"  {lang}: {count:,}\n")
        if no_title:
            f.write("\nno_title books:\n")
            for row in no_title:
                f.write(f"  book_id={row[Col.BOOK_ID]}\n")
    print(f"Init columns stats -> {path}")


### EXTRACT TEXT ###

def _epub_to_text(epub_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(epub_bytes)) as z:
            container = z.read("META-INF/container.xml").decode("utf-8", errors="replace")
            opf_match = re.search(r'full-path="([^"]+\.opf)"', container)
            if not opf_match:
                return ""
            opf_path = opf_match.group(1)
            opf_dir = str(Path(opf_path).parent)
            if opf_dir == ".":
                opf_dir = ""
            opf = z.read(opf_path).decode("utf-8", errors="replace")

            # manifest: id -> href (attributes may appear in any order)
            manifest = {}
            for item in re.finditer(r'<item\s([^>]+)/>', opf):
                attrs = item.group(1)
                id_m = re.search(r'\bid="([^"]+)"', attrs)
                href_m = re.search(r'\bhref="([^"]+)"', attrs)
                if id_m and href_m:
                    manifest[id_m.group(1)] = href_m.group(1)

            # spine order
            spine_idrefs = re.findall(r'<itemref\s[^>]*\bidref="([^"]+)"', opf)

            parts = []
            for idref in spine_idrefs:
                href = manifest.get(idref)
                if not href:
                    continue
                full_path = f"{opf_dir}/{href}" if opf_dir else href
                try:
                    html = z.read(full_path).decode("utf-8", errors="replace")
                    parts.append(html_to_text(html))
                except Exception:
                    pass

            return "\n\n".join(parts)
    except Exception:
        return ""


def extract_text(book):
    if not book[Col.KEEP]:
        book["text"] = ""
        return book
    text = _epub_to_text(bytes(book["epub"]))
    if not text.strip():
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "empty_text"
        book["text"] = ""
        return book
    book["text"] = text
    return book


def write_extract_text_stats(ds, stats_dir: Path):
    dropped = [r for r in ds if r[Col.SKIP_REASON] == "empty_text"]
    if not dropped:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "empty_text.txt"
    with open(path, "w") as f:
        f.write(f"total dropped: {len(dropped):,}\n\n")
        for row in dropped:
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
    print(f"Empty text ({len(dropped)}) -> {path}")


### I/O ###

def write_tokenized_excerpts(ds, output_dir: Path, num_proc: int):
    path = output_dir / "sampled.jsonl"
    if path.exists():
        print(f"Overwriting existing sampled.jsonl")
    ds_kept = ds.filter(lambda x: x[Col.KEEP], num_proc=num_proc)
    print(f"Writing {len(ds_kept):,} kept books -> {path}")
    cols = [Col.BOOK_ID, Col.TOKEN_IDS, Col.TEXT_EXCERPT]
    if any(r[Col.PERPLEXITY] is not None for r in ds_kept):
        cols.append(Col.PERPLEXITY)
    ds_kept.select_columns(cols).to_json(str(path), lines=True)
    print("Done")


### PIPELINE ###

def _clear_stats(stats_dir: Path, ckpt_dir: Path | None):
    import shutil
    # steps 13 and 18 write stats internally (not via stats_fn) so they must be preserved
    # when their checkpoint exists — otherwise stats are lost and never regenerated
    preserved = set()
    for step_name in ("13_dedup_content_minhash", "19_dedup_excerpts_minhash"):
        ckpt = ckpt_dir / step_name if ckpt_dir else None
        if ckpt and ckpt.exists():
            preserved.add(step_name)
    if preserved:
        for child in stats_dir.iterdir():
            if child.name not in preserved:
                (shutil.rmtree if child.is_dir() else child.unlink)(child)
    else:
        shutil.rmtree(stats_dir)


def pipeline(dataset_dir: Path, output_dir: Path, tokenizer_path: str, ckpt_dir: Path | None, stats_dir: Path | None, megatron_ckpt_dir: str | None = None, num_workers: int | None = None):
    t_start = time.time()
    num_proc = num_workers or os.cpu_count()

    if stats_dir and stats_dir.exists():
        _clear_stats(stats_dir, ckpt_dir)

    punkt = load_punkt()
    tokenizer, bos_id, eos_id = load_tokenizer(tokenizer_path)

    print(f"Loading dataset from {dataset_dir}...")
    arrow_files = sorted(str(f) for f in dataset_dir.glob("*.arrow") if not f.name.startswith("cache"))
    ds = load_dataset("arrow", data_files={"train": arrow_files}, split="train")
    print(f"Loaded: {len(ds):,} books, columns: {ds.column_names}")

    step_ckpt_sizes = {}

    def step(step_name, step_fn, stats_fn=None, save_ckpt=True):
        nonlocal ds
        ckpt_path = ckpt_dir / step_name if (ckpt_dir and save_ckpt) else None
        ds, size = run_step(step_fn, ds, ckpt_path)
        if ckpt_dir:
            step_ckpt_sizes[step_name] = size
        if stats_dir and stats_fn:
            stats_path = stats_dir / step_name
            stats_fn(ds, stats_path)

    step("01_init_columns", lambda d: d.map(init_columns_laion, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="init columns"), stats_fn=write_init_columns_stats, save_ckpt=False)
    step("02_extract_text", lambda d: d.map(extract_text, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="extract text"), stats_fn=write_extract_text_stats)
    step("03_dedup_id", dedup_id, stats_fn=write_dedup_id_stats, save_ckpt=False)
    step("04_dedup_title", dedup_title, stats_fn=write_dedup_title_stats, save_ckpt=False)
    step("05_add_title_embeddings", add_title_embeddings)
    step("06_build_title_clusters", build_title_clusters, stats_fn=write_clusters_stats, save_ckpt=False)
    step("07_strip_gutenberg", lambda d: d.map(strip_gutenberg_markers, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="strip gutenberg"), stats_fn=write_gutenberg_strip_stats, save_ckpt=False)
    step("08_normalize_text", lambda d: d.map(normalize_text, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="normalize text"))
    step("09_set_content_bounds", lambda d: d.map(set_content_bounds, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="set content bounds"), stats_fn=write_content_bounds_samples, save_ckpt=False)
    step("10_mark_too_short", lambda d: d.map(mark_too_short, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="mark too short"), stats_fn=write_too_short_stats, save_ckpt=False)
    step("11_verify_no_gutenberg", lambda d: d.map(verify_no_project_gutenberg, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="verify no gutenberg"), stats_fn=write_gutenberg_occurrences, save_ckpt=False)
    step("12_compute_content_chunk_sigs", lambda d: d.map(compute_content_chunk_signatures, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="compute content chunk sigs"))
    step("13_dedup_content_minhash", lambda d: dedup_content_minhash(d, stats_dir=stats_dir / "13_dedup_content_minhash" if stats_dir else None))
    step("14_sample_window", lambda d: d.map(sample_window, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="sample window"))
    step("15_find_excerpt_start", lambda d: d.map(partial(find_excerpt_start, punkt=punkt), num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="find excerpt start"), stats_fn=write_no_excerpt_start_stats)
    step("16_tokenize_excerpt", lambda d: d.map(partial(tokenize_excerpt, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id), num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="tokenize excerpt"), stats_fn=write_tokenize_stats)
    step("17_verify_tokenization", lambda d: d.map(partial(verify_tokenization, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id), num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="verify tokenization"), stats_fn=partial(write_verify_stats, tokenizer=tokenizer), save_ckpt=False)
    step("18_compute_excerpt_chunk_sigs", lambda d: d.map(compute_excerpt_chunk_signatures, num_proc=num_proc, load_from_cache_file=HF_CACHE_RESULTS, desc="compute excerpt chunk sigs"))
    step("19_dedup_excerpts_minhash", lambda d: dedup_excerpts_minhash(d, stats_dir=stats_dir / "19_dedup_excerpts_minhash" if stats_dir else None))

    if megatron_ckpt_dir:
        ppl_model = load_model(megatron_ckpt_dir, tokenizer_path)
        step("20_score_perplexity", lambda d: score_perplexity(d, ppl_model), stats_fn=write_perplexity_stats)

    print_stats(ds)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tokenized_excerpts(ds, output_dir, num_proc)

    print(f"\nTime: {(time.time() - t_start) / 60:.1f}min  output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full processing pipeline for laion/Project-Gutenberg")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="path to HF dataset directory (output of download_gutenberg_laion.py)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="path to write sampled.jsonl and stats")
    parser.add_argument("--tokenizer-path", type=str, default=TOKENIZER_ID)
    parser.add_argument("--checkpoint", action="store_true",
                        help="save/load per-step checkpoints under <output-dir>/checkpoints/")
    parser.add_argument("--no-stats", action="store_true", help="skip writing stats")
    parser.add_argument("--megatron-ckpt-dir", type=str, default=None,
                        help="path to FineWeb LLaMA 1B Megatron checkpoint dir for step 20 perplexity scoring (requires GPU)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="number of parallel workers for dataset.map() steps (default: os.cpu_count())")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints" if args.checkpoint else None
    stats_dir = output_dir / "stats" if not args.no_stats else None
    pipeline(
        dataset_dir=Path(args.dataset_dir),
        output_dir=output_dir,
        tokenizer_path=args.tokenizer_path,
        ckpt_dir=ckpt_dir,
        stats_dir=stats_dir,
        megatron_ckpt_dir=args.megatron_ckpt_dir,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
"""
Exploration pipeline for the Standard Ebooks Kaggle dataset (metadata only, no text).

Pipeline:
  1. download_and_load  — download books.csv via kagglehub, print columns + raw stats
  2. init_columns       — map to standard Col fields, mark non-English originals
  3. dedup_id           — mark duplicate download URLs
  4. dedup_title        — mark duplicate normalised titles
  5. print_stats        — kept/skipped counts by reason

Usage:
  python -m attn_bench.data_processing.books.main_standard_ebooks
  python -m attn_bench.data_processing.books.main_standard_ebooks --download-path /path/to/already/downloaded
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Dataset

from .columns import Col
from .dedup_id_title import dedup_id, dedup_title


COL_SE_NO = "SE No."
COL_TITLE = "Book Title"
COL_AUTHOR = "Author"
COL_LANGUAGE = "Original Language"
COL_DOWNLOAD = "Download (or Read Online)"
COL_WORDS = "No. of Words"
COL_GENRE = "Genre"
COL_YEAR = "Publication Year"


### LOAD ###

def find_books_csv(download_path: Path) -> Path:
    csv_files = sorted(download_path.glob("*.csv"))
    print(f"CSV files: {[f.name for f in csv_files]}")
    candidates = [f for f in csv_files if "book" in f.name.lower() and "about" not in f.name.lower()]
    if not candidates:
        candidates = csv_files
    chosen = candidates[0]
    print(f"Using: {chosen.name}")
    return chosen


def load_books(download_path: Path) -> tuple[pd.DataFrame, Dataset]:
    books_file = find_books_csv(download_path)
    df = pd.read_csv(books_file)
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    return df, Dataset.from_pandas(df, preserve_index=False)


### STATS ###

def print_raw_stats(df: pd.DataFrame):
    print(f"\n### RAW STATS ###")
    print(f"total books: {len(df):,}")

    if COL_LANGUAGE in df.columns:
        lang_counts = df[COL_LANGUAGE].value_counts()
        print(f"\noriginal language (top 10):")
        for lang, count in lang_counts.head(10).items():
            print(f"  {lang}: {count:,}")

    if COL_GENRE in df.columns:
        genre_counts = df[COL_GENRE].value_counts()
        print(f"\ngenre (top 15):")
        for genre, count in genre_counts.head(15).items():
            print(f"  {genre}: {count:,}")

    if COL_WORDS in df.columns:
        words = pd.to_numeric(
            df[COL_WORDS].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).dropna()
        if len(words):
            print(f"\nword count ({len(words):,} books with data):")
            print(f"  min={words.min():,.0f}  max={words.max():,.0f}  mean={words.mean():,.0f}  median={words.median():,.0f}")
            print(f"  p10={words.quantile(0.10):,.0f}  p25={words.quantile(0.25):,.0f}  p75={words.quantile(0.75):,.0f}  p90={words.quantile(0.90):,.0f}")

    if COL_YEAR in df.columns:
        years = pd.to_numeric(df[COL_YEAR], errors="coerce").dropna()
        if len(years):
            print(f"\npublication years: {int(years.min())} – {int(years.max())}")


def print_pipeline_stats(ds):
    counts = Counter(ds[Col.SKIP_REASON])
    kept = counts.pop(None, 0)
    total = kept + sum(counts.values())
    print(f"\n### PIPELINE STATS ###")
    print(f"total={total:,}  kept={kept:,}  {'  '.join(f'{k}={v:,}' for k, v in counts.items())}")


### INIT COLUMNS ###

def init_columns(book):
    book[Col.KEEP] = True
    book[Col.SKIP_REASON] = None

    # ID: Download URL is the canonical identifier on Standard Ebooks
    url = str(book.get(COL_DOWNLOAD) or "").strip()
    book[Col.BOOK_ID] = url if url else ""

    title = str(book.get(COL_TITLE) or "").strip()
    if not title:
        book[Col.BOOK_TITLE] = "Unknown Title"
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "no_title"
    else:
        book[Col.BOOK_TITLE] = title

    # "Original Language" is the source language; Standard Ebooks text is always English,
    # but we filter to keep only works originally written in English.
    if book[Col.KEEP]:
        lang = str(book.get(COL_LANGUAGE) or "").strip().lower()
        if lang and lang not in ("english", "en"):
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "non_english_original"

    return book


### PIPELINE ###

def pipeline(download_path: Path | None):
    if download_path is None:
        import kagglehub
        print("Downloading Standard Ebooks dataset from Kaggle...")
        download_path = Path(kagglehub.dataset_download("mohammadanas27/standard-ebooks-dataset"))
        print(f"Downloaded to: {download_path}")

    df, ds = load_books(download_path)
    print_raw_stats(df)

    ds = ds.map(init_columns, num_proc=1, desc="init columns")
    ds = dedup_id(ds)
    ds = dedup_title(ds)

    print_pipeline_stats(ds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-path", type=str, default=None,
                        help="path to already-downloaded dataset directory (skips kagglehub download)")
    args = parser.parse_args()
    pipeline(Path(args.download_path) if args.download_path else None)


if __name__ == "__main__":
    main()
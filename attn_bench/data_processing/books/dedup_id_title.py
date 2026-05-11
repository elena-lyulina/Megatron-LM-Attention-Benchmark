from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import ftfy

from .columns import Col


def dedup_id(ds):
    seen = set()
    def mark_duplicate(book):
        book_id = book[Col.BOOK_ID]
        if book_id in seen:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "duplicate_id"
        else:
            seen.add(book_id)
        return book
    return ds.map(mark_duplicate, num_proc=1, desc="dedup by id")


def normalize_title(title: str) -> str:
    # Fix mojibake, bad quotes, broken encodings
    title = ftfy.fix_text(title)
    # Unicode normalize + strip accents
    title = unicodedata.normalize("NFKD", title)
    title = "".join(c for c in title if unicodedata.category(c) != "Mn")
    # Keep only alphanumeric + whitespace
    title = re.sub(r"[^\w\s]", "", title)
    # Collapse whitespace
    title = re.sub(r"\s+", " ", title).strip()
    return title.lower()


def dedup_title(ds):
    seen = set()
    def mark_duplicate(book):
        if not book[Col.KEEP]:
            return book
        norm = normalize_title(book[Col.BOOK_TITLE] or "")
        if not norm:
            return book
        if norm in seen:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "duplicate_title"
        else:
            seen.add(norm)
        return book
    return ds.map(mark_duplicate, num_proc=1, desc="dedup by title")


def write_dedup_id_stats(ds, stats_dir: Path):
    dropped = [r for r in ds if r[Col.SKIP_REASON] == "duplicate_id"]
    if not dropped:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "duplicate_ids.txt"
    with open(path, "w") as f:
        f.write(f"total dropped: {len(dropped):,}\n\n")
        for row in dropped:
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
    print(f"Duplicate IDs ({len(dropped)}) -> {path}")


def write_dedup_title_stats(ds, stats_dir: Path):
    dropped = [r for r in ds if r[Col.SKIP_REASON] == "duplicate_title"]
    if not dropped:
        return
    # build norm_title -> kept book mapping to show pairs
    kept_by_norm = {normalize_title(r[Col.BOOK_TITLE] or ""): r
                    for r in ds if r[Col.SKIP_REASON] != "duplicate_title"}
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "duplicate_titles.txt"
    with open(path, "w") as f:
        f.write(f"total dropped: {len(dropped):,}\n\n")
        for row in dropped:
            kept = kept_by_norm.get(normalize_title(row[Col.BOOK_TITLE] or ""))
            kept_str = f"book_id={kept[Col.BOOK_ID]}  title={kept[Col.BOOK_TITLE]!r}" if kept else "unknown"
            f.write(f"dropped  book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
            f.write(f"kept     {kept_str}\n\n")
    print(f"Duplicate titles ({len(dropped)}) -> {path}")

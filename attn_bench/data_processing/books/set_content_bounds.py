from __future__ import annotations

import random
from pathlib import Path

from .columns import Col

MARGIN_FRAC_START = 0.10
MARGIN_FRAC_END = 0.20 # the end usually has additional info, so let's strip more
# with margin 20k and window 45k, a book must be at least 85k to survive => ~30 pages
# 20k cutoff -- minimum margin is 7 pages
# the percentage kicks in once the book exceeds 200k for the front (74 pages), and 100k in the back (37 pages)
MIN_MARGIN = 20_000 # skip the first 7 and the last 7-15 pages, don't take the books smaller than at least 30 pages.


def set_content_bounds(book):
    if not book[Col.KEEP]:
        return book
    text_len = len(book.get("text") or "")
    margin_start = max(MIN_MARGIN, int(text_len * MARGIN_FRAC_START))
    margin_end = max(MIN_MARGIN, int(text_len * MARGIN_FRAC_END))
    book[Col.CONTENT_SIZE] = text_len
    book[Col.CONTENT_START] = margin_start
    book[Col.CONTENT_END] = text_len - margin_end
    return book


WINDOW_CHARS = 45_000
BOUNDS_SAMPLE_N = 300
BOUNDS_SNIPPET_CHARS = 2_000


def write_content_bounds_samples(ds, stats_dir: Path):
    rng = random.Random(42)
    active = [row for row in ds if row[Col.KEEP] and row[Col.CONTENT_START] is not None]
    sample = rng.sample(active, min(BOUNDS_SAMPLE_N, len(active)))

    sep = "=" * 80
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "content_bounds_samples.txt"
    with open(path, "w") as f:
        for row in sample:
            text = row.get("text") or ""
            start = row[Col.CONTENT_START]
            end = row[Col.CONTENT_END]
            f.write(f"{sep}\n")
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
            f.write(f"text_len={row[Col.CONTENT_SIZE]:,}  content_start={start:,}  content_end={end:,}\n")
            f.write(f"{sep}\n\n")
            f.write("--- BEGINNING ---\n")
            f.write(text[start : start + BOUNDS_SNIPPET_CHARS])
            f.write("\n\n--- END ---\n")
            f.write(text[end - BOUNDS_SNIPPET_CHARS : end])
            f.write("\n\n")
    print(f"Content bounds samples ({len(sample)} books) -> {path}")


def mark_too_short(book):
    if not book[Col.KEEP]:
        return book
    if book[Col.CONTENT_END] - book[Col.CONTENT_START] < WINDOW_CHARS:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "too_short"
    return book


def write_too_short_stats(ds, stats_dir: Path):
    dropped = [r for r in ds if r[Col.SKIP_REASON] == "too_short"]
    if not dropped:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    text_lens = sorted(r[Col.CONTENT_SIZE] for r in dropped)
    n = len(text_lens)
    path = stats_dir / "too_short.txt"
    with open(path, "w") as f:
        f.write(f"total dropped: {n:,}\n")
        f.write(f"text length (chars): min={text_lens[0]:,}  p25={text_lens[n//4]:,}  median={text_lens[n//2]:,}  p75={text_lens[3*n//4]:,}  max={text_lens[-1]:,}\n\n")
        f.write(f"{'text_len':>12}  {'content_len':>12}  {'margin_cut':>12}  title\n")
        for row in sorted(dropped, key=lambda r: r[Col.CONTENT_SIZE], reverse=True):
            text_len = row[Col.CONTENT_SIZE]
            content_len = row[Col.CONTENT_END] - row[Col.CONTENT_START]
            margin_cut = row[Col.CONTENT_START] + (text_len - row[Col.CONTENT_END])
            f.write(f"{text_len:>12,}  {content_len:>12,}  {margin_cut:>12,}  {row[Col.BOOK_TITLE]!r}\n")
    print(f"Too-short ({n}) -> {path}")

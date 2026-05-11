"""
Measure Gutenberg header/footer sizes across a sample of books.

Finds ***START OF THE PROJECT GUTENBERG EBOOK*** and ***END*** markers and
reports the distribution of chars before/after them.

Usage:
    python explore_gutenberg.py
    python explore_gutenberg.py --n-books 100
"""

import argparse
import os
import re

import numpy as np
from datasets import load_dataset

START_RE = re.compile(r"\*+\s*START OF TH\w+ PROJECT GUTENBERG EBOOK.+?\*+", re.IGNORECASE | re.DOTALL)
END_RE   = re.compile(r"\*+\s*END OF TH\w+ PROJECT GUTENBERG EBOOK.+?\*+",   re.IGNORECASE | re.DOTALL)
# loose fallback just to check if any gutenberg text exists at all
GUTENBERG_RE = re.compile(r"project gutenberg", re.IGNORECASE)


# results:
# Sampled 1000 books  (missing START: 0, missing END: 0)
#
# Header chars (start of text → end of START marker):
#   n=1000  mean=674  std=138  min=493  p50=623  p95=907  max=1,338
# Footer chars (start of END marker → end of text):
#   n=1000  mean=18,868  std=929  min=13,210  p50=18,831  p95=20,207  max=37,357
def measure_gutenberg_margins(n_books: int) -> None:
    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True,
                      token=os.environ.get("HF_TOKEN"))

    header_sizes, footer_sizes = [], []
    missing_start, missing_end = 0, 0

    for book in ds.take(n_books):
        text = book.get("text", "")
        total = len(text)

        m_start = START_RE.search(text)
        m_end   = END_RE.search(text)

        if m_start:
            header_sizes.append(m_start.end())
        else:
            missing_start += 1
            if missing_start <= 3:
                has_gutenberg = bool(GUTENBERG_RE.search(text))
                print(f"[missing START #{missing_start}]  total_chars={total:,}  "
                      f"has 'project gutenberg': {has_gutenberg}")
                print(f"  first 2000 chars:\n{text[:2000]}\n")

        if m_end:
            footer_sizes.append(total - m_end.start())
        else:
            missing_end += 1

    def stats(label, sizes):
        a = np.array(sizes)
        print(f"{label}:")
        print(f"  n={len(a)}  mean={a.mean():,.0f}  std={a.std():,.0f}  "
              f"min={a.min():,}  p50={int(np.median(a)):,}  "
              f"p95={int(np.percentile(a, 95)):,}  max={a.max():,}")

    print(f"\nSampled {n_books} books  "
          f"(missing START: {missing_start}, missing END: {missing_end})\n")
    stats("Header chars (start of text → end of START marker)", header_sizes)
    stats("Footer chars (start of END marker → end of text)",   footer_sizes)


# ### DEDUP EXPLORATION ###

def base_id(full_id: str) -> str:
    """Strip encoding suffix: '41496-8' -> '41496'."""
    return full_id.rsplit("-", 1)[0] if "-" in full_id else full_id


def explore_dedup(n_books: int) -> None:
    ds = load_dataset("manu/project_gutenberg", split="en", streaming=True,
                      token=os.environ.get("HF_TOKEN"))

    books = list(ds.take(n_books))

    n_total = len(books)
    n_missing_id = sum(1 for b in books if not b.get("id"))
    print(f"\nSampled {n_total}  missing id: {n_missing_id}")

    full_ids = [b["id"] for b in books if b.get("id")]
    base_ids = [base_id(fid) for fid in full_ids]
    n_unique_full = len(set(full_ids))
    n_unique_base = len(set(base_ids))
    print(f"unique full IDs: {n_unique_full}  unique base IDs: {n_unique_base}")

    # find pairs with identical full id
    from collections import defaultdict
    by_full = defaultdict(list)
    for b in books:
        by_full[b["id"]].append(b)

    print("\n--- Same full ID (should be identical rows) ---")
    shown = 0
    for fid, group in by_full.items():
        if len(group) > 1 and shown < 3:
            texts_identical = all(g["text"] == group[0]["text"] for g in group)
            print(f"  id={fid!r}  count={len(group)}  text identical: {texts_identical}")
            shown += 1

    # find pairs with same base id but different encoding suffix
    by_base = defaultdict(list)
    for b in books:
        by_base[base_id(b["id"])].append(b)

    print("\n--- Same base ID, different encoding suffix ---")
    shown = 0
    for bid, group in by_base.items():
        suffixes = set(b["id"] for b in group)
        if len(suffixes) > 1 and shown < 3:
            print(f"  base={bid!r}  variants={sorted(suffixes)}")
            t0, t1 = group[0]["text"], group[1]["text"]
            overlap = sum(a == b for a, b in zip(t0[:2000], t1[:2000]))
            print(f"  first-2000-char overlap: {overlap}/2000")
            shown += 1

    if shown == 0:
        print("  (none found in this sample)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-books", type=int, default=50)
    parser.add_argument("--explore-dedup", action="store_true")
    args = parser.parse_args()
    if args.explore_dedup:
        explore_dedup(args.n_books)
    else:
        measure_gutenberg_margins(args.n_books)
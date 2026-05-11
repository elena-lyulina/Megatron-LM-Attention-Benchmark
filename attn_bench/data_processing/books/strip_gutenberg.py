from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

from .columns import Col

GUTENBERG_ZONE = 25000  # chars from start/end to search in; epub license chapters are ~15k chars
GUTENBERG_HEADER_PATTERNS = [
    re.compile(r"\*+\s*START OF TH\w+ PROJECT GUTENBERG EBOOK.+?\*+", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"Note:\s*Project Gutenberg also has an HTML version of this\s+"
        r"file which includes the original illustrations?\.\s+"
        r"See [^:\n]+:\s+\(https?://[^\)]+\)"
        r"(?:\s+or\s+\([^\)]+\))?",
        re.IGNORECASE,
    ),
    re.compile(
        r"Note:.*?Project\s+Gutenberg.*?See\s+https?://\S+",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"E-?text\s+prepared\s+b[yu]\s+.*?(?:the\s+)?(?:Project\s+Gutenberg|PG)(?:\s+Online)?\s+"
        r"Distributed\s+Proof\w+(?:\s+Team)?[^\n]*",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Produced\s+by\s+.*?(?:Project\s+Gutenberg|PG)\b[^\n]*",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"Credit\s+for\s+this\s+e-text:\s+.+?Project\s+Gutenberg[^\n]*",
        re.IGNORECASE | re.DOTALL,
    ),
]
GUTENBERG_FOOTER_PATTERNS = [
    re.compile(r"\*+\s*END OF TH\w+ PROJECT GUTENBERG EBOOK.+?\*+", re.IGNORECASE | re.DOTALL),
    re.compile(r"End of Project Gutenberg['’]s [^\n]+", re.IGNORECASE),
    re.compile(r"\*?\s*Now available online from Project Gutenberg\.", re.IGNORECASE),
]


def strip_gutenberg_markers(book):
    if not book[Col.KEEP]:
        return book
    text = book.get("text") or ""
    stripped = False
    for pattern in GUTENBERG_HEADER_PATTERNS:
        m = pattern.search(text[:GUTENBERG_ZONE])
        if m:
            text = text[m.end():]
            stripped = True
    for pattern in GUTENBERG_FOOTER_PATTERNS:
        m = pattern.search(text[-GUTENBERG_ZONE:])
        if m:
            text = text[:len(text) - GUTENBERG_ZONE + m.start()]
            stripped = True
    book["text"] = text.strip() if stripped else text
    book[Col.GUTENBERG_STRIPPED] = stripped
    book[Col.GUTENBERG_PRESENT] = bool(PROJECT_GUTENBERG_RE.search(text))
    return book


PROJECT_GUTENBERG_RE = re.compile(r"Project\s+Gutenberg", re.IGNORECASE)
GUTENBERG_CONTEXT_CHARS = 500
GUTENBERG_SAMPLE_N = 100
GUTENBERG_TOP_BOTTOM_LINES = 10


def write_gutenberg_strip_stats(ds, stats_dir: Path):
    total = len(ds)
    stripped = sum(1 for r in ds if r[Col.GUTENBERG_STRIPPED])
    present = sum(1 for r in ds if r[Col.GUTENBERG_PRESENT])
    stripped_and_present = sum(1 for r in ds if r[Col.GUTENBERG_STRIPPED] and r[Col.GUTENBERG_PRESENT])
    not_stripped_and_present = sum(1 for r in ds if not r[Col.GUTENBERG_STRIPPED] and r[Col.GUTENBERG_PRESENT])

    suspicious = [r for r in ds if r[Col.GUTENBERG_PRESENT]]
    rng = random.Random(42)
    sample = rng.sample(suspicious, min(GUTENBERG_SAMPLE_N, len(suspicious)))

    # collect positions first so max values are available for the summary
    from_start = []
    from_end = []
    for row in suspicious:
        text = row.get("text") or ""
        text_len = len(text)
        if not text_len:
            continue
        half = text_len // 2
        for m in re.finditer(r'gutenberg', text, re.IGNORECASE):
            pos = m.start()
            if pos < half:
                from_start.append(pos)
            else:
                from_end.append(text_len - pos)
    max_start = max(from_start) if from_start else 0
    max_end = max(from_end) if from_end else 0

    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "gutenberg_strip_stats.txt"
    sep = "=" * 80

    def clean_lines(text):
        text = re.sub(r'\n{2,}', '\n', text)
        return [l for l in text.split('\n') if l.strip()]

    with open(path, "w") as f:
        f.write(f"total books:              {total:,}\n")
        f.write(f"markers stripped:         {stripped:,}\n")
        f.write(f"'gutenberg' still present:{present:,}\n")
        f.write(f"  stripped + present:     {stripped_and_present:,}\n")
        f.write(f"  not stripped + present: {not_stripped_and_present:,}\n")
        f.write(f"'gutenberg' positions:    first-half max={max_start:,} chars from start  second-half max={max_end:,} chars from end\n")
        f.write(f"\nsampled {len(sample)} suspicious books (gutenberg_present=True)\n\n")

        for row in sample:
            text = row.get("text") or ""
            lines = clean_lines(text)
            text_len = len(text)

            f.write(f"{sep}\n")
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}  stripped={row[Col.GUTENBERG_STRIPPED]}\n")
            f.write(f"{sep}\n\n")

            f.write("--- TOP ---\n")
            for line in lines[:GUTENBERG_TOP_BOTTOM_LINES]:
                f.write(line + "\n")
            f.write("\n")

            f.write("--- GUTENBERG OCCURRENCES ---\n")
            for m in re.finditer(r'gutenberg', text, re.IGNORECASE):
                pos = m.start()
                pct = pos / text_len * 100 if text_len else 0
                ctx_start = max(0, pos - GUTENBERG_CONTEXT_CHARS)
                ctx_end = min(text_len, pos + len(m.group()) + GUTENBERG_CONTEXT_CHARS)
                ctx = re.sub(r'\n{2,}', '\n', text[ctx_start:ctx_end])
                f.write(f"[at {pct:.1f}% of book]\n")
                f.write(f"...{ctx}...\n\n")

            f.write("--- BOTTOM ---\n")
            for line in lines[-GUTENBERG_TOP_BOTTOM_LINES:]:
                f.write(line + "\n")
            f.write("\n\n")

    print(f"Gutenberg strip stats ({len(suspicious)} suspicious books) -> {path}")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in [
        (ax1, from_start, "chars from start of book"),
        (ax2, from_end,   "chars from end of book"),
    ]:
        if data:
            bins = np.logspace(np.log10(max(1, min(data))), np.log10(max(data)), 50)
            ax.hist(data, bins=bins)
        ax.set_xlabel(label)
        ax.set_ylabel("occurrences")
        ax.set_xscale("log")
    max_start = max(from_start) if from_start else 0
    max_end = max(from_end) if from_end else 0
    ax1.set_title(f"first-half occurrences (n={len(from_start)}, max={max_start:,} chars from start)")
    ax2.set_title(f"second-half occurrences (n={len(from_end)}, max={max_end:,} chars from end)")
    fig.suptitle("'gutenberg' occurrence positions")
    fig.tight_layout()
    plot_path = stats_dir / "gutenberg_positions.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Gutenberg position histogram -> {plot_path}")


def verify_no_project_gutenberg(book):
    if not book[Col.KEEP]:
        return book
    text = book.get("text") or ""
    start = book.get(Col.CONTENT_START) or 0
    end = book.get(Col.CONTENT_END) or len(text)
    # todo: remove after testing
    # start, end = 0, len(text)
    if PROJECT_GUTENBERG_RE.search(text[start:end]):
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "project_gutenberg"
    return book


def write_gutenberg_occurrences(ds, stats_dir: Path):
    books = [r for r in ds if r[Col.SKIP_REASON] == "project_gutenberg"]
    if not books:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "project_gutenberg_occurrences.txt"
    with open(path, "w") as f:
        f.write(f"total books: {len(books):,}\n\n")

        sep = "=" * 80

        for book in books:
            f.write(f"{sep}\n")
            f.write(f"book_id={book[Col.BOOK_ID]}  title={book[Col.BOOK_TITLE]!r}  stripped={book[Col.GUTENBERG_STRIPPED]}\n")
            f.write(f"{sep}\n\n")

            text = book.get("text") or ""
            text_len = len(text)
            for m in re.finditer(PROJECT_GUTENBERG_RE, text):
                pos = m.start()
                pct = pos / text_len * 100 if text_len else 0
                ctx_start = max(0, pos - GUTENBERG_CONTEXT_CHARS)
                ctx_end = min(text_len, pos + len(m.group()) + GUTENBERG_CONTEXT_CHARS)
                ctx = re.sub(r'\n{2,}', '\n', text[ctx_start:ctx_end])
                f.write(f"[at {pct:.1f}% of book]\n")
                f.write(f"...{ctx}...\n\n")

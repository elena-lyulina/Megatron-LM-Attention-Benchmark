from __future__ import annotations

from pathlib import Path

import nltk

from .columns import Col

SENTENCE_LOOKAHEAD = 3000


def load_punkt():
    nltk.download("punkt_tab", quiet=True)
    return nltk.data.load("tokenizers/punkt_tab/english.pickle")


def find_excerpt_start(book, punkt):
    if not book[Col.KEEP]:
        return book
    text = book.get("text") or ""
    pos = book[Col.WINDOW_START]
    spans = list(punkt.span_tokenize(text[pos: pos + SENTENCE_LOOKAHEAD]))
    if len(spans) < 2:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "no_excerpt_start"
        return book
    book[Col.EXCERPT_START] = pos + spans[1][0]
    return book


def write_no_excerpt_start_stats(ds, stats_dir: Path):
    examples = [row for row in ds if row[Col.SKIP_REASON] == "no_excerpt_start"]
    if not examples:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "no_excerpt_start.txt"
    sep = "=" * 80
    with open(path, "w") as f:
        for row in examples:
            pos = row[Col.WINDOW_START] or 0
            text = row.get("text") or ""
            f.write(f"{sep}\n")
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}  window_start={pos}\n\n")
            f.write(text[pos: pos + SENTENCE_LOOKAHEAD])
            f.write("\n\n")
    print(f"No-excerpt-start examples ({len(examples)}) -> {path}")

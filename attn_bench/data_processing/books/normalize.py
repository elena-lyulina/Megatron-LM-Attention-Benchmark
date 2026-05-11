from __future__ import annotations

import re
import unicodedata

import ftfy

from .columns import Col


def normalize_text(book):
    if not book[Col.KEEP]:
        return book
    text = book.get("text") or ""
    text = ftfy.fix_text(text) # fix encoding issues
    text = unicodedata.normalize("NFKC", text) # standardize unicode characters (e.g. quotes, etc)
    text = re.sub(r"[^\S\n]+$", "", text, flags=re.MULTILINE) # remove trailing whitespaces, whitespace-only lines will be empty abd be removed on the next step
    text = re.sub(r"\n{3,}", "\n\n", text) # collapse 3+ newlines to 2
    book["text"] = text
    return book

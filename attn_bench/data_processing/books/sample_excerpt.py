from __future__ import annotations

import random

from .columns import Col
from .set_content_bounds import WINDOW_CHARS

DETERMINISTIC_SAMPLING = True


def sample_window(book):
    if not book[Col.KEEP]:
        return book
    # we deduplicated book titles, so each book receives its own generator
    rng = random.Random(book[Col.BOOK_ID]) if DETERMINISTIC_SAMPLING else random
    book[Col.WINDOW_START] = rng.randint(book[Col.CONTENT_START], book[Col.CONTENT_END] - WINDOW_CHARS)
    return book

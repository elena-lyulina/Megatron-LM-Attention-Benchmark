from __future__ import annotations


class Col:
    BOOK_ID = "book_id"
    BOOK_TITLE = "book_title"
    KEEP = "keep"
    SKIP_REASON = "skip_reason"
    CONTENT_SIZE = "content_size"
    CONTENT_START = "content_start"
    CONTENT_END = "content_end"
    WINDOW_START = "window_start"
    EXCERPT_START = "excerpt_start"
    TOKEN_IDS = "token_ids"
    TEXT_EXCERPT = "text_excerpt"
    CLUSTER_ID = "cluster_id"
    CLUSTER_SIZE = "cluster_size"
    TITLE_EMBEDDING = "title_embedding"
    MINHASH_SIG = "minhash_sig"
    CHUNK_SIGS = "chunk_sigs"
    EXCERPT_CHUNK_SIGS = "excerpt_chunk_sigs"
    GUTENBERG_STRIPPED = "gutenberg_stripped"
    GUTENBERG_PRESENT = "gutenberg_present"
    PARA_LINE_COUNTS = "para_line_counts"
    PARA_MEAN_LINE_LEN = "para_mean_line_len"
    PARA_STD_LINE_LEN = "para_std_line_len"
    PARA_MAX_LINE_LEN = "para_max_line_len"
    PARA_NEAR_MAX_FRAC = "para_near_max_frac"
    PARA_SHORT_LINE_FRAC = "para_short_line_frac"
    PARA_TYPES = "para_types"
    PARA_UNWRAP = "para_unwrap"
    LINE_SNIPPET = "line_snippet"
    LINE_SNIPPET_OFFSET = "line_snippet_offset"
    EXCERPT_TOKEN_COUNT = "excerpt_token_count"
    PERPLEXITY = "perplexity"
    MIN_K_PP = "min_k_pp"


DEFAULTS = {
    Col.BOOK_ID: "unknown",
    Col.BOOK_TITLE: None,
    Col.KEEP: True,
    Col.SKIP_REASON: "",   # non-null so pyarrow always infers string, never null-typed shards
    Col.CONTENT_SIZE: 0,
    Col.CONTENT_START: 0,
    Col.CONTENT_END: 0,
    Col.WINDOW_START: None,
    Col.EXCERPT_START: None,
    Col.TOKEN_IDS: [],
    Col.TEXT_EXCERPT: "",
    Col.CLUSTER_ID: None,
    Col.CLUSTER_SIZE: 0,
    Col.MINHASH_SIG: [],
    Col.CHUNK_SIGS: [],
    Col.EXCERPT_CHUNK_SIGS: [],
    Col.GUTENBERG_STRIPPED: False,
    Col.GUTENBERG_PRESENT: False,
    Col.PARA_LINE_COUNTS: [],
    Col.PARA_MEAN_LINE_LEN: [],
    Col.PARA_STD_LINE_LEN: [],
    Col.PARA_MAX_LINE_LEN: [],
    Col.PARA_NEAR_MAX_FRAC: [],
    Col.PARA_SHORT_LINE_FRAC: [],
    Col.PARA_UNWRAP: [],
    Col.LINE_SNIPPET: "",
    Col.LINE_SNIPPET_OFFSET: None,
    Col.EXCERPT_TOKEN_COUNT: 0,
    Col.PERPLEXITY: None,
    Col.MIN_K_PP: None,
}


def init_columns(book):
    for col, default in DEFAULTS.items():
        book.setdefault(col, default)
    book[Col.BOOK_ID] = str(book.get("id"))

    meta = book.get("metadata") or {}

    if meta.get("language") != "en":
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "non_english"

    title = meta.get("title")
    if title is None or not str(title).strip():
        book[Col.BOOK_TITLE] = "Unknown Title"
        if book[Col.KEEP]:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "no_title"
    else:
        book[Col.BOOK_TITLE] = str(title).strip()

    return book

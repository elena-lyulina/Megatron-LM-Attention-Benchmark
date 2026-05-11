from __future__ import annotations

import os
from pathlib import Path

from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer

from .columns import Col
from .set_content_bounds import WINDOW_CHARS

TOKENIZER_ID = "meta-llama/Llama-3.2-1B"


def load_tokenizer(tokenizer_path: str):
    # mirror MegatronDocumentTokenizer.tokenizer property exactly

    # Determine if this is a local path or Hub ID
    # Local paths are absolute (start with /) or relative (contain / or .)
    is_local = os.path.isabs(tokenizer_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=is_local, use_fast=True)
    # Verify it's a fast tokenizer and extract the underlying tokenizers.Tokenizer
    if not hf_tokenizer.is_fast:
        raise ValueError(f"Tokenizer at {tokenizer_path} is not a fast tokenizer")

    tokenizer = hf_tokenizer._tokenizer  # raw tokenizers.Tokenizer, same object megatron uses
    bos_id = tokenizer.token_to_id(hf_tokenizer.bos_token)
    eos_id = tokenizer.token_to_id(hf_tokenizer.eos_token)
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
        pair=None,
    )
    print(f"Loaded tokenizer: {tokenizer_path}  vocab={tokenizer.get_vocab_size()}  bos={bos_id}  eos={eos_id}")
    return tokenizer, bos_id, eos_id


SEQ_LEN = 8192
# Megatron reads seq_length+1 tokens and splits into input[:-1] / labels[1:]
# (megatron/core/datasets/gpt_dataset.py:242-243, gpt_dataset.py:341-343), so we store SEQ_LEN+1 tokens
_TOKENS_PER_EXCERPT = SEQ_LEN + 1


def tokenize_excerpt(book, tokenizer, bos_id, eos_id):
    if not book[Col.KEEP]:
        return book
    text = book.get("text") or ""
    start = book[Col.EXCERPT_START]
    # encode gives [BOS, content..., EOS] via TemplateProcessing
    ids = tokenizer.encode(text[start: start + WINDOW_CHARS], add_special_tokens=True).ids
    book[Col.EXCERPT_TOKEN_COUNT] = len(ids)
    if len(ids) < _TOKENS_PER_EXCERPT:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "not_enough_tokens"
        return book
    # take [BOS, content[:_TOKENS_PER_EXCERPT-2]] then append EOS → exactly _TOKENS_PER_EXCERPT tokens
    book[Col.TOKEN_IDS] = ids[:_TOKENS_PER_EXCERPT - 1] + [eos_id]
    return book


def verify_tokenization(book, tokenizer, bos_id, eos_id):
    if not book[Col.KEEP]:
        return book
    token_ids = book[Col.TOKEN_IDS]
    if len(token_ids) != _TOKENS_PER_EXCERPT:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "wrong_token_count"
        return book
    if token_ids[0] != bos_id or token_ids.count(bos_id) != 1:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "bad_bos"
        return book
    if token_ids[-1] != eos_id or token_ids.count(eos_id) != 1:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "bad_eos"
        return book
    # mirror verify_tokenization.py: decode with special tokens, re-encode without
    excerpt_text = tokenizer.decode(token_ids, skip_special_tokens=False)
    re_token_ids = tokenizer.encode(excerpt_text, add_special_tokens=False).ids
    if len(re_token_ids) != _TOKENS_PER_EXCERPT or re_token_ids != token_ids:
        book[Col.KEEP] = False
        book[Col.SKIP_REASON] = "token_round_trip"
        return book
    book[Col.TEXT_EXCERPT] = excerpt_text
    return book


_SNIPPET_CHARS = 500


def write_tokenize_stats(ds, stats_dir: Path):
    dropped = [r for r in ds if r[Col.SKIP_REASON] == "not_enough_tokens"]
    if not dropped:
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    counts = sorted(r[Col.EXCERPT_TOKEN_COUNT] for r in dropped)
    n = len(counts)
    path = stats_dir / "not_enough_tokens.txt"
    with open(path, "w") as f:
        f.write(f"total dropped: {n:,}\n")
        f.write(f"token counts: min={counts[0]:,}  p25={counts[n//4]:,}  median={counts[n//2]:,}  p75={counts[3*n//4]:,}  max={counts[-1]:,}\n\n")
        sep = "=" * 80
        for row in sorted(dropped, key=lambda r: r[Col.EXCERPT_TOKEN_COUNT], reverse=True):
            start = row[Col.EXCERPT_START] or 0
            snippet = (row.get("text") or "")[start: start + WINDOW_CHARS]
            f.write(f"{sep}\n")
            f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}  tokens={row[Col.EXCERPT_TOKEN_COUNT]:,}\n\n")
            f.write(snippet)
            f.write("\n\n")
    print(f"Not-enough-tokens ({n}) -> {path}")


def write_verify_stats(ds, stats_dir: Path, tokenizer):
    reasons = ["wrong_token_count", "bad_bos", "bad_eos", "token_round_trip"]
    by_reason = {r: [] for r in reasons}
    for row in ds:
        if row[Col.SKIP_REASON] in by_reason:
            by_reason[row[Col.SKIP_REASON]].append(row)
    if not any(by_reason.values()):
        return
    stats_dir.mkdir(parents=True, exist_ok=True)
    path = stats_dir / "verify_failures.txt"
    sep = "=" * 80
    with open(path, "w") as f:
        for reason, rows in by_reason.items():
            f.write(f"{reason}: {len(rows):,}\n")
        f.write("\n")
        for reason, rows in by_reason.items():
            if not rows:
                continue
            f.write(f"{sep}\n{reason.upper()} ({len(rows)})\n{sep}\n\n")
            for row in rows:
                token_ids = row[Col.TOKEN_IDS]
                f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
                if reason == "wrong_token_count":
                    f.write(f"token_count={len(token_ids):,}  expected={_TOKENS_PER_EXCERPT}\n")
                elif reason == "bad_bos":
                    f.write(f"first 10 tokens: {token_ids[:10]}\n")
                elif reason == "bad_eos":
                    f.write(f"last 10 tokens: {token_ids[-10:]}\n")
                elif reason == "token_round_trip":
                    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
                    re_ids = tokenizer.encode(decoded, add_special_tokens=False).ids
                    f.write(f"re_encoded_count={len(re_ids):,}  expected={_TOKENS_PER_EXCERPT}\n")
                    f.write(decoded)
                f.write("\n\n")
    print(f"Verify failures -> {path}")

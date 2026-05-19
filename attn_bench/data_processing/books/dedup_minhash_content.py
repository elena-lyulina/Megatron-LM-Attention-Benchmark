from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from .columns import Col

CHUNK_NUM_PERM = 128 # an industry standard, provides good approximation without being computationally expensive
CHUNK_WORDS = 600 # big enough to be efficient, small enough to catch a duplicate
CHUNK_NGRAM_SIZE = 5 # allows matching noisy texts (OCR, typos)
CHUNK_SIM_THRESHOLD = 0.3  # per-chunk minhash Jaccard threshold -- low enough, drops a book if it has ~200 duplicate words (0.3 of 1 chunk), so we're aggressive with deduplication
MIN_CHUNK_MATCHES = 1  # drop a book if >= this many chunks match


def _word_chunks(text: str, n: int) -> list[list[str]]:
    words = text.split()
    chunks = [words[i:i + n] for i in range(0, len(words), n)]
    if chunks and len(chunks[-1]) < n // 2:
        chunks.pop()
    return chunks


def _chunk_minhash(words: list[str], num_perm: int = CHUNK_NUM_PERM, ngram_size: int = CHUNK_NGRAM_SIZE) -> MinHash:
    m = MinHash(num_perm=num_perm)
    norm = [re.sub(r"[^a-z]", "", w.lower()) for w in words]
    norm = [w for w in norm if w]
    for i in range(len(norm) - ngram_size + 1):
        m.update(" ".join(norm[i:i + ngram_size]).encode())
    return m


def compute_chunk_signatures(book, chunk_size: int = CHUNK_WORDS, num_perm: int = CHUNK_NUM_PERM):
    if not book[Col.KEEP]:
        book[Col.CHUNK_SIGS] = []
        return book
    text = (book.get("text") or "")[book[Col.CONTENT_START]:book[Col.CONTENT_END]]
    chunks = _word_chunks(text, chunk_size)
    book[Col.CHUNK_SIGS] = [_chunk_minhash(c, num_perm).hashvalues.tolist() for c in chunks]
    return book


def dedup_chunks_minhash(
    ds,
    sim_threshold: float = CHUNK_SIM_THRESHOLD,
    min_matches: int = MIN_CHUNK_MATCHES,
    num_perm: int = CHUNK_NUM_PERM,
    stats_dir: Path | None = None,
):
    lsh = MinHashLSH(threshold=sim_threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    active_indices = [i for i, k in enumerate(ds[Col.KEEP]) if k]
    all_chunk_sigs = ds[Col.CHUNK_SIGS]  # list of lists of ints, much smaller than text

    collect_stats = stats_dir is not None
    # book_idx -> matched chunk count / list of match pairs — only collected when writing stats
    matched_counts: dict[int, int] | None = {} if collect_stats else None
    match_examples: dict[int, list[tuple[int, int, int]]] | None = {} if collect_stats else None

    for global_idx in tqdm(active_indices, desc="chunk dedup LSH"):
        sigs = all_chunk_sigs[global_idx]
        if not sigs:
            if collect_stats:
                matched_counts[global_idx] = 0
            continue

        hashes = [] # stores minhashes that are deserialised from signatures
        for sig in sigs:
            m = MinHash(num_perm=num_perm)
            m.hashvalues = np.array(sig, dtype=np.uint64)
            hashes.append(m)

        matched = 0
        for chunk_i, m in enumerate(hashes):
            results = lsh.query(m) # matches with previously seen chunks
            if results:
                matched += 1
                if collect_stats:
                    key = results[0]  # {global_idx}_{chunk_i} e.g. "1234_5"
                    parts = key.rsplit("_", 1)
                    matched_book_idx, matched_chunk_idx = int(parts[0]), int(parts[1])
                    match_examples.setdefault(global_idx, []).append((chunk_i, matched_book_idx, matched_chunk_idx))

        if collect_stats:
            matched_counts[global_idx] = matched
        if matched >= min_matches:
            duplicate_indices.add(global_idx)
        else:
            # otherwise, insert all the chunks hashes into lsh index
            for chunk_i, m in enumerate(hashes):
                lsh.insert(f"{global_idx}_{chunk_i}", m)

    print(f"chunk_dedup: {len(duplicate_indices)} duplicates found (sim={sim_threshold}, min_matches={min_matches})")

    if collect_stats:
        _write_chunk_dedup_stats(ds, stats_dir, matched_counts, match_examples, sim_threshold, min_matches)

    def mark_chunk_duplicates(book, row_idx):
        if row_idx in duplicate_indices:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "chunk_duplicate"
        return book

    return ds.map(mark_chunk_duplicates, with_indices=True, num_proc=1, desc="chunk dedup mark")


def _write_chunk_dedup_stats(
    ds,
    stats_dir: Path,
    matched_counts: dict[int, int],
    match_examples: dict[int, list[tuple[int, int, int]]],
    sim_threshold: float,
    min_matches: int,
):
    from collections import Counter
    from datetime import datetime
    from attn_bench.utils.text_match import compare_texts

    stats_dir.mkdir(parents=True, exist_ok=True)

    dist = Counter(matched_counts.values())
    total_matched_chunks = sum(k * v for k, v in dist.items())
    max_matched = max(dist) if dist else 0
    total = len(matched_counts)

    path = stats_dir / "chunk_dedup_stats.txt"
    with open(path, "w") as f:
        f.write(f"total active books: {total:,}  chunk_size={CHUNK_WORDS}  ngram={CHUNK_NGRAM_SIZE}  sim_threshold={sim_threshold}  min_matches={min_matches}\n\n")
        f.write(f"total matched chunks: {total_matched_chunks:,}\n")

        f.write("### DISTRIBUTION ###\n\n")
        f.write(f"{'matched chunks':>15}  {'books':>8}  {'cumulative drop':>16}\n")
        cumulative = 0
        for n in range(1, max_matched + 1):
            count = dist.get(n, 0)
            if count == 0:
                continue
            cumulative += count
            f.write(f"{n:>15}  {count:>8,}  {cumulative:>16,} ({100*cumulative/total:.1f}%)\n")

    print(f"Chunk dedup stats -> {path}")

    titles = ds[Col.BOOK_TITLE]
    book_ids = ds[Col.BOOK_ID]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for book_idx, pairs in match_examples.items():
        n_matched = matched_counts[book_idx]
        folder = stats_dir / "matches" / f"{n_matched}_matched"

        my_row = ds[book_idx]
        my_text = (my_row.get("text") or "")[my_row[Col.CONTENT_START]:my_row[Col.CONTENT_END]]
        my_chunks = _word_chunks(my_text, CHUNK_WORDS)

        for my_ci, matched_book_idx, matched_ci in pairs:
            matched_row = ds[matched_book_idx]
            matched_text = (matched_row.get("text") or "")[matched_row[Col.CONTENT_START]:matched_row[Col.CONTENT_END]]
            matched_chunks = _word_chunks(matched_text, CHUNK_WORDS)

            if my_ci >= len(my_chunks) or matched_ci >= len(matched_chunks):
                continue

            chunk_a = " ".join(my_chunks[my_ci])
            chunk_b = " ".join(matched_chunks[matched_ci])

            desc = (
                f"book_a={book_ids[book_idx]}  title={titles[book_idx]!r}  chunk={my_ci}/{len(my_chunks)}  words={len(my_chunks[my_ci])}\n"
                f"book_b={book_ids[matched_book_idx]}  title={titles[matched_book_idx]!r}  chunk={matched_ci}/{len(matched_chunks)}  words={len(matched_chunks[matched_ci])}\n"
                f"params: chunk_size={CHUNK_WORDS}  ngram={CHUNK_NGRAM_SIZE}  sim_threshold={sim_threshold}  min_matches={min_matches}"
            )
            filename = f"{book_ids[book_idx]}_c{my_ci}_{book_ids[matched_book_idx]}_c{matched_ci}_{ts}"
            label_a = f"{book_ids[book_idx]} | {titles[book_idx][:40]} | chunk {my_ci}"
            label_b = f"{book_ids[matched_book_idx]} | {titles[matched_book_idx][:40]} | chunk {matched_ci}"

            try:
                compare_texts(
                    text1=chunk_a,
                    text2=chunk_b,
                    output_folder=folder,
                    filename=filename,
                    label1=label_a,
                    label2=label_b,
                    description=desc,
                )
            except Exception as e:
                print(f"compare_texts failed for {filename}: {e}")

    print(f"Chunk match HTML files -> {stats_dir / 'matches'}")
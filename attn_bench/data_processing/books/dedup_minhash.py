from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

from .columns import Col


# ── shared internals ──────────────────────────────────────────────────────────

def _normalize_words(text: str) -> list[str]:
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in text.split()]
    return [w for w in words if w]


def _word_chunks(words: list[str], n: int) -> list[list[str]]:
    chunks = [words[i:i + n] for i in range(0, len(words), n)]
    if chunks and len(chunks[-1]) < n // 2:
        chunks.pop()
    return chunks


def _chunk_ngrams(words: list[str], n: int) -> frozenset[str]:
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def _minhash_from_ngrams(ngrams: frozenset[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in ngrams:
        m.update(gram.encode())
    return m


def _exact_jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _sigs_for_text(text: str, chunk_size: int, num_perm: int, ngram_size: int) -> list[list[int]]:
    words = _normalize_words(text)
    return [
        _minhash_from_ngrams(_chunk_ngrams(c, ngram_size), num_perm).hashvalues.tolist()
        for c in _word_chunks(words, chunk_size)
        if len(c) >= ngram_size
    ]


def _write_match_distribution(f, matched_counts: dict[int, int], unit_label: str):
    dist = Counter(matched_counts.values())
    total = len(matched_counts)
    max_matched = max(dist) if dist else 0
    f.write("### DISTRIBUTION ###\n\n")
    f.write(f"{'matched chunks':>15}  {unit_label:>9}  {'cumulative drop':>16}\n")
    cumulative = 0
    for n in range(1, max_matched + 1):
        count = dist.get(n, 0)
        if count == 0:
            continue
        cumulative += count
        f.write(f"{n:>15}  {count:>9,}  {cumulative:>16,} ({100*cumulative/total:.1f}%)\n")
    return dist, total


def _write_compare_pair(chunk_a, chunk_b, idx_a, ci_a, idx_b, ci_b, desc, folder, book_ids, titles, ts, sim=None):
    from attn_bench.utils.text_match import compare_texts
    sim_prefix = f"{sim:.3f}_" if sim is not None else ""
    filename = f"{sim_prefix}{book_ids[idx_a]}_c{ci_a}_{book_ids[idx_b]}_c{ci_b}_{ts}"
    label_a = f"{book_ids[idx_a]} | {titles[idx_a][:40]} | chunk {ci_a}"
    label_b = f"{book_ids[idx_b]} | {titles[idx_b][:40]} | chunk {ci_b}"
    try:
        compare_texts(
            text1=chunk_a, text2=chunk_b,
            output_folder=folder, filename=filename,
            label1=label_a, label2=label_b,
            description=desc,
        )
    except Exception as e:
        print(f"compare_texts failed for {filename}: {e}")


# ── content dedup (LSH only, pre-computed chunk sigs) ─────────────────────────

CHUNK_NUM_PERM = 128
CHUNK_WORDS = 600
CHUNK_NGRAM_SIZE = 5
CHUNK_SIM_THRESHOLD = 0.3
MIN_CHUNK_MATCHES = 1


def compute_content_chunk_signatures(book, chunk_size: int = CHUNK_WORDS, num_perm: int = CHUNK_NUM_PERM, ngram_size: int = CHUNK_NGRAM_SIZE):
    if not book[Col.KEEP]:
        book[Col.CHUNK_SIGS] = []
        return book
    text = (book.get("text") or "")[book[Col.CONTENT_START]:book[Col.CONTENT_END]]
    book[Col.CHUNK_SIGS] = _sigs_for_text(text, chunk_size, num_perm, ngram_size)
    return book


def _load_chunk_data(ds, sigs_col: str, num_perm: int, desc: str = "loading chunk sigs") -> dict[int, list[MinHash]]:
    data: dict[int, list[MinHash]] = {}
    all_chunk_sigs = ds[sigs_col]
    active = [i for i, k in enumerate(ds[Col.KEEP]) if k]
    for idx in tqdm(active, desc=desc):
        sigs = all_chunk_sigs[idx]
        if not sigs:
            continue
        hashes = []
        for sig in sigs:
            m = MinHash(num_perm=num_perm)
            m.hashvalues = np.array(sig, dtype=np.uint64)
            hashes.append(m)
        data[idx] = hashes
    return data


def dedup_content_minhash(
    ds,
    sim_threshold: float = CHUNK_SIM_THRESHOLD,
    min_matches: int = MIN_CHUNK_MATCHES,
    num_perm: int = CHUNK_NUM_PERM,
    stats_dir: Path | None = None,
):
    chunk_data = _load_chunk_data(ds, Col.CHUNK_SIGS, num_perm, desc="loading content chunk sigs")
    active_indices = [i for i, k in enumerate(ds[Col.KEEP]) if k]
    lsh = MinHashLSH(threshold=sim_threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    collect_stats = stats_dir is not None
    # initialise all active books to 0 so total in stats covers books with empty sigs too
    matched_counts: dict[int, int] | None = {idx: 0 for idx in active_indices} if collect_stats else None
    match_examples: dict[int, list[tuple[int, int, int]]] | None = {} if collect_stats else None

    for idx, hashes in tqdm(chunk_data.items(), desc="content dedup LSH"):
        matched = 0
        for chunk_i, m in enumerate(hashes):
            results = lsh.query(m)
            if results:
                matched += 1
                if collect_stats:
                    src_idx, src_ci = (int(x) for x in results[0].rsplit("_", 1))
                    match_examples.setdefault(idx, []).append((chunk_i, src_idx, src_ci))

        if collect_stats:
            matched_counts[idx] = matched
        if matched >= min_matches:
            duplicate_indices.add(idx)
        else:
            for chunk_i, m in enumerate(hashes):
                lsh.insert(f"{idx}_{chunk_i}", m)

    print(f"content_dedup: {len(duplicate_indices)} duplicates found (sim={sim_threshold}, min_matches={min_matches})")

    if collect_stats:
        _write_content_dedup_stats(ds, stats_dir, matched_counts, match_examples, sim_threshold, min_matches)

    def mark_duplicates(book, row_idx):
        if row_idx in duplicate_indices:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "content_minhash_duplicate"
        return book

    return ds.map(mark_duplicates, with_indices=True, num_proc=1, desc="content dedup mark")


def _write_content_dedup_stats(
    ds,
    stats_dir: Path,
    matched_counts: dict[int, int],
    match_examples: dict[int, list[tuple[int, int, int]]],
    sim_threshold: float,
    min_matches: int,
):
    stats_dir.mkdir(parents=True, exist_ok=True)

    dist, total = Counter(matched_counts.values()), len(matched_counts)
    total_matched_chunks = sum(k * v for k, v in dist.items())

    path = stats_dir / "content_dedup_stats.txt"
    with open(path, "w") as f:
        f.write(f"total active books: {total:,}  chunk_size={CHUNK_WORDS}  ngram={CHUNK_NGRAM_SIZE}  sim_threshold={sim_threshold}  min_matches={min_matches}\n\n")
        f.write(f"total matched chunks: {total_matched_chunks:,}\n\n")
        _write_match_distribution(f, matched_counts, "books")
    print(f"Content dedup stats -> {path}")

    titles = ds[Col.BOOK_TITLE]
    book_ids = ds[Col.BOOK_ID]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for book_idx, pairs in match_examples.items():
        n_matched = matched_counts[book_idx]
        folder = stats_dir / "matches" / f"{n_matched}_matched"
        my_row = ds[book_idx]
        my_text = (my_row.get("text") or "")[my_row[Col.CONTENT_START]:my_row[Col.CONTENT_END]]
        my_chunks = _word_chunks(_normalize_words(my_text), CHUNK_WORDS)

        for my_ci, src_idx, src_ci in pairs:
            src_row = ds[src_idx]
            src_text = (src_row.get("text") or "")[src_row[Col.CONTENT_START]:src_row[Col.CONTENT_END]]
            src_chunks = _word_chunks(_normalize_words(src_text), CHUNK_WORDS)
            if my_ci >= len(my_chunks) or src_ci >= len(src_chunks):
                continue
            desc = (
                f"book_a={book_ids[book_idx]}  title={titles[book_idx]!r}  chunk={my_ci}/{len(my_chunks)}  words={len(my_chunks[my_ci])}\n"
                f"book_b={book_ids[src_idx]}  title={titles[src_idx]!r}  chunk={src_ci}/{len(src_chunks)}  words={len(src_chunks[src_ci])}\n"
                f"params: chunk_size={CHUNK_WORDS}  ngram={CHUNK_NGRAM_SIZE}  sim_threshold={sim_threshold}  min_matches={min_matches}"
            )
            _write_compare_pair(
                " ".join(my_chunks[my_ci]), " ".join(src_chunks[src_ci]),
                book_idx, my_ci, src_idx, src_ci,
                desc, folder, book_ids, titles, ts,
            )

    print(f"Content match HTML files -> {stats_dir / 'matches'}")


# ── excerpt dedup (LSH + exact Jaccard, pre-computed chunk sigs) ──────────────

# datasketch picks b=40, r=1 for EXCERPT_LSH_THRESHOLD=0.04 regardless of num_perm — extra perms are unused by LSH
EXCERPT_NUM_PERM = 40
EXCERPT_CHUNK_WORDS = 100
EXCERPT_NGRAM_SIZE = 5
EXCERPT_LSH_THRESHOLD = 0.04
EXCERPT_JACCARD_THRESHOLD = 0.05


def compute_excerpt_chunk_signatures(book, chunk_size: int = EXCERPT_CHUNK_WORDS, num_perm: int = EXCERPT_NUM_PERM, ngram_size: int = EXCERPT_NGRAM_SIZE):
    if not book[Col.KEEP] or not book[Col.TEXT_EXCERPT]:
        book[Col.EXCERPT_CHUNK_SIGS] = []
        return book
    book[Col.EXCERPT_CHUNK_SIGS] = _sigs_for_text(book[Col.TEXT_EXCERPT], chunk_size, num_perm, ngram_size)
    return book



def dedup_excerpts_minhash(
    ds,
    jaccard_threshold: float = EXCERPT_JACCARD_THRESHOLD,
    lsh_threshold: float = EXCERPT_LSH_THRESHOLD,
    num_perm: int = EXCERPT_NUM_PERM,
    chunk_size: int = EXCERPT_CHUNK_WORDS,
    ngram_size: int = EXCERPT_NGRAM_SIZE,
    stats_dir: Path | None = None,
):
    chunk_data = _load_chunk_data(ds, Col.EXCERPT_CHUNK_SIGS, num_perm, desc="loading excerpt chunk sigs")
    all_excerpts = ds[Col.TEXT_EXCERPT]
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    ngram_cache: dict[int, list[frozenset]] = {}

    def get_ngrams(idx: int) -> list[frozenset]:
        if idx not in ngram_cache:
            words = _normalize_words(all_excerpts[idx])
            chunks = [c for c in _word_chunks(words, chunk_size) if len(c) >= ngram_size]
            ngram_cache[idx] = [_chunk_ngrams(c, ngram_size) for c in chunks]
        return ngram_cache[idx]

    collect_stats = stats_dir is not None
    n_lsh_candidates = 0
    n_jaccard_matches = 0
    matched_counts: dict[int, int] | None = {} if collect_stats else None
    # excerpt_idx -> list of (chunk_i, src_idx, src_ci, jaccard)
    match_examples: dict[int, list[tuple[int, int, int, float]]] | None = {} if collect_stats else None
    # jaccard of every LSH candidate pair (for plotting the full distribution)
    all_candidate_jaccards: list[float] | None = [] if collect_stats else None

    for idx, hashes in tqdm(chunk_data.items(), desc="excerpt dedup LSH"):
        n_matched_chunks = 0
        for chunk_i, m in enumerate(hashes):
            candidates = lsh.query(m)
            n_lsh_candidates += len(candidates)
            chunk_matched = False
            for c in candidates:
                src_idx, src_ci = (int(x) for x in c.rsplit("_", 1))
                j = _exact_jaccard(get_ngrams(idx)[chunk_i], get_ngrams(src_idx)[src_ci])
                if collect_stats:
                    all_candidate_jaccards.append(j)
                if j >= jaccard_threshold and not chunk_matched:
                    n_jaccard_matches += 1
                    n_matched_chunks += 1
                    chunk_matched = True
                    if collect_stats:
                        match_examples.setdefault(idx, []).append((chunk_i, src_idx, src_ci, j))
                    else:
                        break  # fast path: no need to check remaining candidates

        if collect_stats:
            matched_counts[idx] = n_matched_chunks
        if n_matched_chunks >= 1:
            duplicate_indices.add(idx)
        else:
            for chunk_i, m in enumerate(hashes):
                lsh.insert(f"{idx}_{chunk_i}", m)

    ratio = n_jaccard_matches / n_lsh_candidates if n_lsh_candidates else 0.0
    print(
        f"excerpt_dedup: {len(duplicate_indices)} duplicates  "
        f"lsh_candidates={n_lsh_candidates:,}  jaccard_matches={n_jaccard_matches:,} ({100*ratio:.1f}%)"
        f"  (lsh={lsh_threshold}  jaccard={jaccard_threshold})"
    )

    if collect_stats:
        _write_excerpt_dedup_stats(
            ds, stats_dir, matched_counts, match_examples,
            n_lsh_candidates, n_jaccard_matches, all_candidate_jaccards,
            lsh_threshold, jaccard_threshold, chunk_size, ngram_size,
        )

    def mark_duplicates(book, row_idx):
        if row_idx in duplicate_indices:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "excerpt_minhash_duplicate"
        return book

    return ds.map(mark_duplicates, with_indices=True, num_proc=1, desc="excerpt dedup mark")


def _write_excerpt_dedup_stats(
    ds,
    stats_dir: Path,
    matched_counts: dict[int, int],
    match_examples: dict[int, list[tuple[int, int, int, float]]],
    n_lsh_candidates: int,
    n_jaccard_matches: int,
    all_candidate_jaccards: list[float],
    lsh_threshold: float,
    jaccard_threshold: float,
    chunk_size: int,
    ngram_size: int,
):
    import matplotlib.pyplot as plt

    stats_dir.mkdir(parents=True, exist_ok=True)
    total = len(matched_counts)
    ratio = n_jaccard_matches / n_lsh_candidates if n_lsh_candidates else 0.0

    path = stats_dir / "excerpt_dedup_stats.txt"
    with open(path, "w") as f:
        f.write(f"total active excerpts: {total:,}  chunk_size={chunk_size}  ngram={ngram_size}  lsh_threshold={lsh_threshold}  jaccard_threshold={jaccard_threshold}\n\n")
        f.write("### CANDIDATE FILTERING ###\n\n")
        f.write(f"{'LSH candidates:':25} {n_lsh_candidates:>10,}\n")
        f.write(f"{'Jaccard matches:':25} {n_jaccard_matches:>10,}  ({100*ratio:.1f}% of candidates)\n\n")
        _write_match_distribution(f, matched_counts, "excerpts")
    print(f"Excerpt dedup stats -> {path}")

    if all_candidate_jaccards:
        sims = np.array(sorted(all_candidate_jaccards))
        x_max = jaccard_threshold * 8
        thresholds = np.linspace(0.0, x_max, 200)
        ccdf = len(sims) - np.searchsorted(sims, thresholds)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(sims[sims <= x_max], bins=50, log=True)
        ax1.axvline(jaccard_threshold, color="red", linestyle="--", label=f"threshold={jaccard_threshold}")
        ax1.set_xlabel("Exact Jaccard (word 5-grams, per chunk)")
        ax1.set_ylabel("LSH candidate pairs (log)")
        ax1.set_title("Similarity distribution (all LSH candidates)")
        ax1.legend()
        ax2.plot(thresholds, np.maximum(ccdf, 0.5))  # avoid log(0)
        ax2.axvline(jaccard_threshold, color="red", linestyle="--", label=f"threshold={jaccard_threshold}")
        ax2.set_yscale("log")
        ax2.set_xlabel("Jaccard threshold")
        ax2.set_ylabel("Pairs above threshold (log)")
        ax2.set_title("Cumulative pairs vs threshold (all LSH candidates)")
        ax2.legend()
        fig.tight_layout()
        plot_path = stats_dir / "text_similarity.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Similarity plot -> {plot_path}")

    titles = ds[Col.BOOK_TITLE]
    book_ids = ds[Col.BOOK_ID]
    excerpts = ds[Col.TEXT_EXCERPT]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for excerpt_idx, examples in match_examples.items():
        n_matched = matched_counts[excerpt_idx]
        folder = stats_dir / "matches" / f"{n_matched}_matched"
        my_chunks = _word_chunks(_normalize_words(excerpts[excerpt_idx]), chunk_size)

        for my_ci, src_idx, src_ci, j in examples:
            src_chunks = _word_chunks(_normalize_words(excerpts[src_idx]), chunk_size)
            if my_ci >= len(my_chunks) or src_ci >= len(src_chunks):
                continue
            desc = (
                f"excerpt_a={book_ids[excerpt_idx]}  title={titles[excerpt_idx]!r}  chunk={my_ci}/{len(my_chunks)}\n"
                f"excerpt_b={book_ids[src_idx]}  title={titles[src_idx]!r}  chunk={src_ci}/{len(src_chunks)}\n"
                f"exact_jaccard={j:.3f}  chunk_size={chunk_size}  ngram={ngram_size}  jaccard_threshold={jaccard_threshold}"
            )
            _write_compare_pair(
                " ".join(my_chunks[my_ci]), " ".join(src_chunks[src_ci]),
                excerpt_idx, my_ci, src_idx, src_ci,
                desc, folder, book_ids, titles, ts, sim=j,
            )

    if match_examples:
        print(f"Excerpt match HTML files -> {stats_dir / 'matches'}")
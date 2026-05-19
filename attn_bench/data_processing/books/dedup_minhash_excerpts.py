from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

from .columns import Col

EXCERPT_NUM_PERM = 256
EXCERPT_NGRAM_SIZE = 5
EXCERPT_CHUNK_WORDS = 200
EXCERPT_LSH_THRESHOLD = 0.15   # wide net for candidate retrieval
EXCERPT_JACCARD_THRESHOLD = 0.2  # hard cutoff after exact Jaccard verification


def _normalize_words(text: str) -> list[str]:
    words = [re.sub(r"[^a-z]", "", w.lower()) for w in text.split()]
    return [w for w in words if w]


def _chunk_ngrams(words: list[str], ngram_size: int = EXCERPT_NGRAM_SIZE) -> frozenset[str]:
    return frozenset(" ".join(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1))


def _minhash(ngrams: frozenset[str], num_perm: int = EXCERPT_NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in ngrams:
        m.update(gram.encode())
    return m


def _exact_jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _build_chunk_data(ds, chunk_size: int, num_perm: int, ngram_size: int) -> dict[int, list[tuple[MinHash, frozenset]]]:
    data: dict[int, list[tuple[MinHash, frozenset]]] = {}
    for idx, row in enumerate(ds):
        if not row[Col.KEEP] or not row[Col.TEXT_EXCERPT]:
            continue
        words = _normalize_words(row[Col.TEXT_EXCERPT])
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        if chunks and len(chunks[-1]) < chunk_size // 2:
            chunks.pop()
        data[idx] = []
        for chunk_words in chunks:
            if len(chunk_words) < ngram_size:
                continue
            ngrams = _chunk_ngrams(chunk_words, ngram_size)
            data[idx].append((_minhash(ngrams, num_perm), ngrams))
    return data


def dedup_excerpts_minhash(
    ds,
    jaccard_threshold: float = EXCERPT_JACCARD_THRESHOLD,
    lsh_threshold: float = EXCERPT_LSH_THRESHOLD,
    num_perm: int = EXCERPT_NUM_PERM,
    chunk_size: int = EXCERPT_CHUNK_WORDS,
    ngram_size: int = EXCERPT_NGRAM_SIZE,
):
    chunk_data = _build_chunk_data(ds, chunk_size, num_perm, ngram_size)
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    duplicate_indices: set[int] = set()

    for idx, chunks in chunk_data.items():
        matched = False
        for chunk_i, (m, ngrams) in enumerate(chunks):
            for c in lsh.query(m):
                src_idx, src_ci = (int(x) for x in c.rsplit("_", 1))
                if _exact_jaccard(ngrams, chunk_data[src_idx][src_ci][1]) >= jaccard_threshold:
                    matched = True
                    break
            if matched:
                break
        if matched:
            duplicate_indices.add(idx)
        else:
            for chunk_i, (m, _) in enumerate(chunks):
                lsh.insert(f"{idx}_{chunk_i}", m)

    print(f"minhash_dedup: {len(duplicate_indices)} duplicates found (chunk_size={chunk_size}, jaccard={jaccard_threshold})")

    def mark_duplicates(book, row_idx):
        if row_idx in duplicate_indices:
            book[Col.KEEP] = False
            book[Col.SKIP_REASON] = "minhash_duplicate"
        return book

    return ds.map(mark_duplicates, with_indices=True, num_proc=1, desc="minhash dedup")


def write_similar_excerpts(
    ds,
    output_dir: Path,
    candidate_threshold: float = EXCERPT_LSH_THRESHOLD,
    num_perm: int = EXCERPT_NUM_PERM,
    chunk_size: int = EXCERPT_CHUNK_WORDS,
    ngram_size: int = EXCERPT_NGRAM_SIZE,
    top_n: int = 10,
):
    import matplotlib.pyplot as plt

    chunk_data = _build_chunk_data(ds, chunk_size, num_perm, ngram_size)
    lsh = MinHashLSH(threshold=candidate_threshold, num_perm=num_perm)
    # (exact_jaccard, excerpt_i, chunk_i, excerpt_j, chunk_j)
    pairs: list[tuple[float, int, int, int, int]] = []

    for idx, chunks in chunk_data.items():
        for chunk_i, (m, ngrams) in enumerate(chunks):
            for c in lsh.query(m):
                src_idx, src_ci = (int(x) for x in c.rsplit("_", 1))
                sim = _exact_jaccard(ngrams, chunk_data[src_idx][src_ci][1])
                pairs.append((sim, idx, chunk_i, src_idx, src_ci))
        for chunk_i, (m, _) in enumerate(chunks):
            lsh.insert(f"{idx}_{chunk_i}", m)

    print(f"write_similar_excerpts: {len(pairs)} candidate chunk pairs (lsh_threshold={candidate_threshold})")

    if not pairs:
        print("No candidate pairs found — texts are highly dissimilar.")
        return

    similarities = np.array(sorted(sim for sim, *_ in pairs))

    sweep = np.arange(0.0, 1.0, 0.1)
    sweep_counts = len(similarities) - np.searchsorted(similarities, sweep)
    print("Chunk pairs at Jaccard >=:  " + "  ".join(f"{t:.1f}: {c}" for t, c in zip(sweep, sweep_counts)))

    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(candidate_threshold, 1.0, 200)
    ccdf = len(similarities) - np.searchsorted(similarities, thresholds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(similarities, bins=50)
    ax1.axvline(EXCERPT_JACCARD_THRESHOLD, color="red", linestyle="--", label=f"dedup={EXCERPT_JACCARD_THRESHOLD}")
    ax1.set_xlabel("Exact Jaccard (word 5-grams, per chunk)")
    ax1.set_ylabel("Chunk pairs")
    ax1.set_title("Similarity distribution")
    ax1.legend()

    ax2.plot(thresholds, ccdf)
    ax2.axvline(EXCERPT_JACCARD_THRESHOLD, color="red", linestyle="--", label=f"dedup={EXCERPT_JACCARD_THRESHOLD}")
    ax2.set_xlabel("Jaccard threshold")
    ax2.set_ylabel("Chunk pairs above threshold")
    ax2.set_title("Cumulative pairs vs threshold")
    ax2.legend()

    fig.tight_layout()
    path = output_dir / "text_similarity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plots -> {path}")

    titles = ds[Col.BOOK_TITLE]
    excerpts = ds[Col.TEXT_EXCERPT]
    # deduplicate to best chunk pair per excerpt pair, then take top_n
    best: dict[tuple[int, int], tuple[float, int, int, int, int]] = {}
    for sim, i, ci, j, cj in pairs:
        key = (min(i, j), max(i, j))
        if key not in best or sim > best[key][0]:
            best[key] = (sim, i, ci, j, cj)
    top = sorted(best.values(), reverse=True)[:top_n]

    path = output_dir / "text_similarity_pairs.txt"
    sep_pair = "=" * 80
    sep_book = "-" * 80
    with open(path, "w") as f:
        for rank, (sim, i, ci, j, cj) in enumerate(top, 1):
            f.write(f"{sep_pair}\n")
            f.write(f"PAIR {rank}  sim={sim:.3f}  chunks: [{i}]c{ci} vs [{j}]c{cj}\n")
            f.write(f"{sep_pair}\n\n")
            f.write(f"{'#' * 10} BOOK 1: [{i}] {titles[i]} {'#' * 10}\n\n")
            f.write(excerpts[i])
            f.write(f"\n\n{sep_book}\n\n")
            f.write(f"{'#' * 10} BOOK 2: [{j}] {titles[j]} {'#' * 10}\n\n")
            f.write(excerpts[j])
            f.write(f"\n\n")
    print(f"Top {top_n} similar pairs -> {path}")

    import csv
    csv_path = output_dir / "lsh_minhash_stats.csv"
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "lsh_threshold": candidate_threshold,
        "jaccard_threshold": EXCERPT_JACCARD_THRESHOLD,
        "chunk_size": chunk_size,
        "n_excerpts_total": len(ds),
        "n_excerpts_with_chunks": len(chunk_data),
        **{f"pairs_ge_{t:.1f}": int(c) for t, c in zip(sweep, sweep_counts)},
    }
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"LSH/minhash stats -> {csv_path}")
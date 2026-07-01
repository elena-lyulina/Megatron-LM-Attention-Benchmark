"""
Build long-context versions of the Gutenberg memorization probes.

Each existing probe in `gutenberg_rep_jsonl/rep_{R}_token.jsonl` is one 8,192-token
excerpt (`[BOS, c0..c8189, EOS]`). This script re-derives the source text + excerpt
position for each probe and re-tokenizes a longer span around it — up to 3x seq len
of extra context on each side, clamped to the book's content zone — while keeping the
measured 8,192-token region byte-identical to what the model trained on.

The old `book_id <-> token_ids` map is gone, so we recover it from the tokens: the
per-book pipeline steps are deterministic, so we regenerate the canonical 8,192 for
every raw book and match each probe by its token sequence (strategy B in the plan).
Matching self-recovers `book_id`, `text`, and `excerpt_start`; nothing else is needed.

Output: `<output-dir>/rep_{R}_token.jsonl` (long records), `lengths.jsonl` (per-book
extra lengths for plotting), `unmatched_report.txt` (probes that failed to match).

Usage:
  python -m attn_bench.data_processing.books.extract_long_context \
      --jsonl-dir     /path/to/gutenberg_rep_jsonl \
      --raw-dir       /path/to/gutenberg-laion/raw-hf-full \
      --tokenizer-path /path/to/llama-3.2-1b \
      --output-dir    /path/to/gutenberg_rep_jsonl_long \
      [--stats-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from .columns import Col
from .find_excerpt_start import find_excerpt_start, load_punkt
from .main_gutenberg_laion import extract_text, init_columns_laion
from .normalize import normalize_text
from .sample_excerpt import sample_window
from .set_content_bounds import mark_too_short, set_content_bounds
from .strip_gutenberg import strip_gutenberg_markers
from .tokenize_excerpts import SEQ_LEN, TOKENIZER_ID, load_tokenizer, tokenize_excerpt

HF_CACHE = False

CONTENT_TOKENS = SEQ_LEN - 2          # sample content tokens: c0..c8189 (BOS/EOS excluded)
MAX_EXTRA = 3 * SEQ_LEN               # cap on extra tokens per side (24,576)
# Char bounds for the two extra tokenizations. We only keep the last/first MAX_EXTRA
# tokens, so a bound just avoids tokenizing far more than can ever be used. Sized to
# guarantee the token caps even for dense text: forward needs CONTENT_TOKENS + MAX_EXTRA
# = 32,766 tokens (220k chars = 6.7 chars/tok headroom vs ~4.3 avg); backward needs
# MAX_EXTRA = 24,576 tokens (170k chars = 6.9 headroom).
FORWARD_CHARS = 220_000
BACKWARD_CHARS = 170_000
SNIPPET_TOKENS = 60                   # decoded preview length in the unmatched report


### PROBES ###

def _digest(token_ids) -> str:
    return hashlib.blake2b(np.asarray(token_ids, dtype=np.int32).tobytes(), digest_size=16).hexdigest()


def load_probes(jsonl_dir: Path):
    """rep_{R}_token.jsonl -> {rep: [input_ids, ...]} plus digest -> [(rep, idx), ...]."""
    probes: dict[int, list[list[int]]] = {}
    by_digest: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path in sorted(jsonl_dir.glob("rep_*_token.jsonl")):
        rep = int(re.search(r"rep_(\d+)_token", path.name).group(1))
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                ids = json.loads(line)["input_ids"]
                by_digest[_digest(ids)].append((rep, len(rows)))
                rows.append(ids)
        probes[rep] = rows
        print(f"  {path.name}: {len(rows):,} probes")
    total = sum(len(v) for v in probes.values())
    print(f"Loaded {total:,} probes across {len(probes)} buckets")
    return probes, by_digest


### CANONICAL REGENERATION ###

def add_canonical_digest(book):
    ids = book[Col.TOKEN_IDS]
    book["canon_digest"] = _digest(ids) if book[Col.KEEP] and len(ids) == SEQ_LEN else ""
    return book


def regenerate_canonicals(raw_dir: Path, tokenizer, bos_id, eos_id, punkt, num_proc):
    """Run the per-book pipeline half on the full raw corpus -> canonical 8,192 + text.

    Skips all cross-book steps (dedup, ppl, containment, bucketing); keeps only the
    steps that produce `text`, content bounds, `excerpt_start`, and the canonical
    tokens. `mark_too_short` is required so `sample_window` never gets an empty range.
    """
    ds = load_from_disk(str(raw_dir))
    print(f"Loaded raw corpus: {len(ds):,} books")

    def m(fn, desc):
        return ds.map(fn, num_proc=num_proc, load_from_cache_file=HF_CACHE, desc=desc)

    ds = m(init_columns_laion, "init columns")
    ds = m(extract_text, "extract text")
    # drop the heavy raw epub bytes once text is extracted (saves memory + speeds row access)
    ds = ds.remove_columns([c for c in ("epub",) if c in ds.column_names])
    ds = m(strip_gutenberg_markers, "strip gutenberg")
    ds = m(normalize_text, "normalize text")
    ds = m(set_content_bounds, "set content bounds")
    ds = m(mark_too_short, "mark too short")
    ds = m(sample_window, "sample window")
    ds = m(partial(find_excerpt_start, punkt=punkt), "find excerpt start")
    ds = m(partial(tokenize_excerpt, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id), "tokenize canonical")
    ds = m(add_canonical_digest, "digest canonical")

    ds = ds.filter(lambda b: b["canon_digest"] != "", num_proc=num_proc, desc="keep tokenized")
    print(f"Regenerated {len(ds):,} canonical excerpts")
    return ds


### LONG BUILD ###

def build_long_row(task, tokenizer, bos_id, eos_id):
    """Build the long sequence for one (book, bucket) task. Runs in parallel via `.map`.

    `task` carries text + positions + `original_ids` (the matched probe). Sets `ok=False`
    on a sample-reproduction miss (text/determinism drift) instead of raising, so the map
    doesn't die on one book — the caller filters and logs those.
    """
    text = task["text"] or ""
    start = task["excerpt_start"]
    content = task["original_ids"][1:1 + CONTENT_TOKENS]   # c0..c8189, used verbatim from the probe

    # Tokenize forward from excerpt_start ONLY to obtain the suffix tokens (which exist
    # nowhere yet). Its first CONTENT_TOKENS reproduce the sample region and are used purely
    # as a check — we keep the probe's `content`, not these. The check proves the regenerated
    # text is byte-identical to the original run, so the suffix it continues into is genuinely
    # contiguous with the real sample.
    fwd = tokenizer.encode(text[start:min(len(text), start + FORWARD_CHARS)], add_special_tokens=False)
    if fwd.ids[:CONTENT_TOKENS] != content:
        task["ok"] = False
        task["input_ids"] = []
        task["sample_offset"] = 0
        task["extra_prefix_len"] = 0
        task["extra_suffix_len"] = 0
        return task

    # extra suffix: tokens past the sample, clamped to the content zone, capped at MAX_EXTRA
    zone_len = task["content_end"] - start           # chars from excerpt_start to content_end
    extra_suffix = []
    for i in range(CONTENT_TOKENS, min(len(fwd.ids), CONTENT_TOKENS + MAX_EXTRA)):
        if fwd.offsets[i][0] >= zone_len:
            break
        extra_suffix.append(fwd.ids[i])

    # extra prefix: separate tokenization of the content zone before the sample, last MAX_EXTRA.
    # bounded to BACKWARD_CHARS so we don't tokenize the whole front of a long book.
    bwd_start = max(task["content_start"], start - BACKWARD_CHARS)
    bwd = tokenizer.encode(text[bwd_start:start], add_special_tokens=False)
    extra_prefix = bwd.ids[-MAX_EXTRA:] if MAX_EXTRA else []

    input_ids = [bos_id] + extra_prefix + content + extra_suffix + [eos_id]
    sample_offset = 1 + len(extra_prefix)
    # guarantee the measured region in the assembled output is byte-identical to the probe
    assert input_ids[sample_offset:sample_offset + CONTENT_TOKENS] == content

    task["ok"] = True
    task["input_ids"] = input_ids
    task["sample_offset"] = sample_offset
    task["extra_prefix_len"] = len(extra_prefix)
    task["extra_suffix_len"] = len(extra_suffix)
    return task


### MAIN ###

def run(jsonl_dir, raw_dir, tokenizer_path, output_dir, regen_workers, build_workers, stats_only):
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    punkt = load_punkt()
    tokenizer, bos_id, eos_id = load_tokenizer(tokenizer_path)

    print("\n### PROBES ###")
    probes, by_digest = load_probes(jsonl_dir)

    print("\n### REGENERATE CANONICALS ###")
    ds = regenerate_canonicals(raw_dir, tokenizer, bos_id, eos_id, punkt, regen_workers)

    print("\n### MATCH ###")
    # digest -> ds index (verify full equality on hit to exclude collisions)
    digest_to_idx: dict[str, int] = {}
    collisions = 0
    for i, d in enumerate(ds["canon_digest"]):
        if d in digest_to_idx:
            collisions += 1
        else:
            digest_to_idx[d] = i
    if collisions:
        print(f"  WARNING: {collisions} duplicate canonical digests among regenerated books")

    matched: dict[int, list[tuple[int, int]]] = {}   # ds_idx -> [(rep, probe_idx), ...]
    unmatched: list[tuple[int, int]] = []            # (rep, probe_idx)
    for digest, refs in by_digest.items():
        idx = digest_to_idx.get(digest)
        if idx is not None and ds[idx][Col.TOKEN_IDS] == probes[refs[0][0]][refs[0][1]]:
            matched[idx] = refs
        else:
            unmatched.extend(refs)

    total = sum(len(v) for v in probes.values())
    n_matched = sum(len(v) for v in matched.values())
    print(f"  matched {n_matched:,} / {total:,} probes ({100 * n_matched / total:.1f}%)")

    print("\n### BUILD LONG ###")
    # one row per (book, bucket). Build on the memory-mapped dataset via select+map
    # (the path that actually shards across num_proc — from_list runs ~serially here).
    # ds.select accepts repeated indices, so a book matched by >1 probe is duplicated.
    flat_idx, flat_rep, flat_probe = [], [], []
    for idx, refs in matched.items():
        for rep, probe_idx in refs:
            flat_idx.append(idx)
            flat_rep.append(rep)
            flat_probe.append(probes[rep][probe_idx])

    ds_m = ds.select(flat_idx).flatten_indices()
    ds_m = ds_m.add_column("bucket_rep", flat_rep)
    ds_m = ds_m.add_column("original_ids", flat_probe)
    # keep only what the writer needs; drop text + intermediates so Arrow isn't hauling
    # ~200k chars/row through the map output.
    keep_cols = {"book_id", "bucket_rep", "original_ids"}
    built = ds_m.map(
        partial(build_long_row, tokenizer=tokenizer, bos_id=bos_id, eos_id=eos_id),
        num_proc=build_workers, desc="build long",
        remove_columns=[c for c in ds_m.column_names if c not in keep_cols],
    )

    # write per-bucket files (streaming), collect length stats, log build failures
    writers = {} if not stats_only else None
    counts = defaultdict(int)
    lengths_rows = []
    build_failures = []                              # (rep, book_id)
    for rec in built:
        rep = rec["bucket_rep"]
        if not rec["ok"]:
            build_failures.append((rep, rec["book_id"]))
            continue
        lengths_rows.append({
            "book_id": rec["book_id"],
            "bucket_rep": rep,
            "extra_prefix_len": rec["extra_prefix_len"],
            "extra_suffix_len": rec["extra_suffix_len"],
        })
        counts[rep] += 1
        if writers is not None:
            if rep not in writers:
                writers[rep] = open(output_dir / f"rep_{rep}_token.jsonl", "w")
            out = {
                "book_id": rec["book_id"],
                "bucket_rep": rep,
                "input_ids": rec["input_ids"],
                "original_ids": rec["original_ids"],
                "sample_offset": rec["sample_offset"],
                "sample_len": CONTENT_TOKENS,
                "extra_prefix_len": rec["extra_prefix_len"],
                "extra_suffix_len": rec["extra_suffix_len"],
            }
            writers[rep].write(json.dumps(out) + "\n")
    if writers is not None:
        for rep in sorted(writers):
            writers[rep].close()
            print(f"  rep_{rep}_token.jsonl: {counts[rep]:,} records")
    else:
        print("  stats-only: token files not written")
    if build_failures:
        print(f"  WARNING: {len(build_failures)} matched books failed sample reproduction")

    print("\n### WRITE ###")
    write_lengths(output_dir, lengths_rows)
    write_unmatched(output_dir, unmatched, build_failures, probes, tokenizer)
    print_length_summary(lengths_rows)
    print(f"\nDone in {(time.time() - t0) / 60:.1f}min  ->  {output_dir}")


def write_lengths(output_dir: Path, rows: list[dict]):
    path = output_dir / "lengths.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  {path.name}: {len(rows):,} rows")


def write_unmatched(output_dir: Path, join_misses, build_failures, probes, tokenizer):
    """join_misses: (rep, probe_idx) probes with no matching book.
    build_failures: (rep, book_id) matched books whose sample didn't reproduce."""
    path = output_dir / "unmatched_report.txt"
    per_bucket = defaultdict(int)
    for rep, _ in join_misses:
        per_bucket[rep] += 1
    for rep, _ in build_failures:
        per_bucket[rep] += 1
    with open(path, "w") as f:
        total = sum(len(v) for v in probes.values())
        dropped = len(join_misses) + len(build_failures)
        f.write(f"dropped probes: {dropped:,} / {total:,}  "
                f"(join misses: {len(join_misses):,}, build failures: {len(build_failures):,})\n\n")
        f.write("per-bucket match rate:\n")
        for rep in sorted(probes):
            n = len(probes[rep])
            miss = per_bucket.get(rep, 0)
            f.write(f"  rep_{rep:<4} {n - miss:>6,} / {n:>6,}  ({100 * (n - miss) / n:.1f}%)\n")
        if join_misses:
            f.write("\n" + "=" * 80 + "\nJOIN MISSES\n" + "=" * 80 + "\n")
            for rep, idx in join_misses:
                ids = probes[rep][idx]
                snippet = tokenizer.decode(ids[:SNIPPET_TOKENS], skip_special_tokens=True).replace("\n", " ")
                f.write(f"rep_{rep} idx={idx}  first20={ids[:20]}\n  {snippet!r}\n")
        if build_failures:
            f.write("\n" + "=" * 80 + "\nBUILD FAILURES (sample not reproduced)\n" + "=" * 80 + "\n")
            for rep, book_id in build_failures:
                f.write(f"rep_{rep} book_id={book_id}\n")
    print(f"  {path.name}: {len(join_misses):,} join misses, {len(build_failures):,} build failures")


def print_length_summary(rows: list[dict]):
    if not rows:
        return
    pre = np.array([r["extra_prefix_len"] for r in rows])
    suf = np.array([r["extra_suffix_len"] for r in rows])
    print(f"\nExtra-length summary ({len(rows):,} books):")
    for name, arr in (("prefix", pre), ("suffix", suf)):
        pct = 100 * (arr >= MAX_EXTRA).mean()
        print(f"  {name}: mean={arr.mean():,.0f}  median={np.median(arr):,.0f}  "
              f"max={arr.max():,}  at-cap({MAX_EXTRA:,})={pct:.1f}%")


def main():
    p = argparse.ArgumentParser(description="Build long-context Gutenberg probes")
    p.add_argument("--jsonl-dir", required=True, help="existing gutenberg_rep_jsonl dir")
    p.add_argument("--raw-dir", required=True, help="raw-hf-full HF dataset dir")
    p.add_argument("--tokenizer-path", default=TOKENIZER_ID)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-workers", type=int, default=None,
                   help="map workers for the canonical-regen phase (default: all cpus)")
    p.add_argument("--build-workers", type=int, default=32,
                   help="map workers for the build-long phase; fewer is faster here since each "
                        "worker only gets a handful of rows and spawn overhead dominates")
    p.add_argument("--stats-only", action="store_true",
                   help="compute length distribution + reports without writing token files")
    args = p.parse_args()
    run(
        jsonl_dir=Path(args.jsonl_dir),
        raw_dir=Path(args.raw_dir),
        tokenizer_path=args.tokenizer_path,
        output_dir=Path(args.output_dir),
        regen_workers=args.num_workers or os.cpu_count(),
        build_workers=args.build_workers,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

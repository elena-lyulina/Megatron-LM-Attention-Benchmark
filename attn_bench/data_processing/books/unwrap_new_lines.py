from __future__ import annotations

import random
from collections import Counter, defaultdict

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .columns import Col
from .main_gutenberg import get_encode_devices
from .set_content_bounds import WINDOW_CHARS

NEAR_MAX_THRESH = 10    # chars from max line len — high frac signals hard-wrapped prose
SHORT_LINE_THRESH = 45  # chars — high frac signals poetry
CLASSIFY_BATCH_SIZE = 16
CLASSIFY_CONFIDENCE_THRESH = 0.3
CLASSIFY_MODEL_ID = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
# the model scores the hypothesis for each label, e.g. "This is an example of prose — continuous narrative or descriptive text."
CLASSIFY_HYPOTHESIS_TEMPLATE = "This is an example of {}."
# short label (stored in PARA_TYPES) -> description passed to the NLI classifier
PARA_TYPE_DESCRIPTIONS = {
    "prose":   "prose — continuous narrative or descriptive text",
    "poetry":  "poetry — verse with intentional line breaks",
    "drama":   "drama or script — dialogue with speaker names",
    "list":    "a list — enumerated or bulleted items",
    "data":    "a table or data — structured numbers or aligned columns",
    "matter":  "front or back matter — table of contents, index, preface, or appendix",
    "title":   "a title or heading — short chapter or section heading",
    "mixed":   "mixed content — combination of multiple text types",
}
_CLASSIFY_LABELS = list(PARA_TYPE_DESCRIPTIONS.values())
_CLASSIFY_LABEL_TO_TYPE = {v: k for k, v in PARA_TYPE_DESCRIPTIONS.items()}


def _extract_paragraphs(text: str, window_start: int) -> list[list[str]]:
    # returns each paragraph as a list of non-empty lines
    window = text[window_start : window_start + WINDOW_CHARS]
    result = []
    for para in window.split("\n\n"):
        lines = [ln for ln in para.split("\n") if ln.strip()]
        if lines:
            result.append(lines)
    return result


def split_and_stat_paragraphs(book):
    if not book[Col.KEEP] or book[Col.WINDOW_START] is None:
        return book
    text = book.get("text") or ""

    line_counts, mean_lens, std_lens, max_lens, near_max_fracs, short_fracs = [], [], [], [], [], []

    for lines in _extract_paragraphs(text, book[Col.WINDOW_START]):
        lens = [len(ln.strip()) for ln in lines]
        line_counts.append(len(lines))
        mean_lens.append(float(np.mean(lens)))
        std_lens.append(float(np.std(lens)))
        max_len = max(lens)
        max_lens.append(max_len)
        near_max_fracs.append(sum(1 for l in lens if max_len - l <= NEAR_MAX_THRESH) / len(lens))
        short_fracs.append(sum(1 for l in lens if l < SHORT_LINE_THRESH) / len(lens))

    book[Col.PARA_LINE_COUNTS] = line_counts
    book[Col.PARA_MEAN_LINE_LEN] = mean_lens
    book[Col.PARA_STD_LINE_LEN] = std_lens
    book[Col.PARA_MAX_LINE_LEN] = max_lens
    book[Col.PARA_NEAR_MAX_FRAC] = near_max_fracs
    book[Col.PARA_SHORT_LINE_FRAC] = short_fracs
    return book


def classify_paragraphs(ds):
    from transformers import pipeline as hf_pipeline

    device = get_encode_devices()
    if isinstance(device, list):
        device = device[0]

    is_cuda = isinstance(device, str) and device.startswith("cuda")
    classifier = hf_pipeline(
        "zero-shot-classification", # for hugging face to select the right pipeline
        model=CLASSIFY_MODEL_ID,
        device=device,
        batch_size=CLASSIFY_BATCH_SIZE,
        model_kwargs={"torch_dtype": torch.bfloat16} if is_cuda else {},
    )

    keep_mask = ds[Col.KEEP]
    active_indices = [i for i, k in enumerate(keep_mask) if k]

    all_para_types: list[list] = [[] for _ in range(len(ds))]
    # store a batch of paragraphs
    buffer_texts: list[str] = []
    # store books ids these paragraphs belong to, len(buffer_book_ids) = len(buffer_texts)
    buffer_book_ids: list[int] = []

    def classify_and_clean_buffer():
        if not buffer_texts:
            return
        results = classifier(
            buffer_texts,
            candidate_labels=_CLASSIFY_LABELS,
            hypothesis_template=CLASSIFY_HYPOTHESIS_TEMPLATE,
        )
        # could happen for a batch size of 1, then it just returns a single dict instead of a list
        if isinstance(results, dict):
            results = [results]
        # iterated over each paragraph's result
        for book_idx, result in zip(buffer_book_ids, results):
            #  pipeline always returns labels and scores sorted by score descending
            top_score = result["scores"][0]
            label = _CLASSIFY_LABEL_TO_TYPE[result["labels"][0]] if top_score >= CLASSIFY_CONFIDENCE_THRESH else None
            all_para_types[book_idx].append(label)
        buffer_texts.clear()
        buffer_book_ids.clear()

    # could also try to sort by lengths so each batch is more balanced
    for book_idx in tqdm(active_indices, desc="classifying paragraphs"):
        row = ds[book_idx]
        text = row.get("text") or ""
        window_start = row[Col.WINDOW_START]
        if window_start is None:
            continue
        for lines in _extract_paragraphs(text, window_start):
            buffer_texts.append("\n".join(lines))
            buffer_book_ids.append(book_idx)
            # fill the buffer until we reach the batch size -- in this case, apply classifier and clean buffer
            if len(buffer_texts) == CLASSIFY_BATCH_SIZE:
                classify_and_clean_buffer()
    classify_and_clean_buffer()

    return ds.add_column(Col.PARA_TYPES, all_para_types)


PARA_STATS_SAMPLE_N = 100


def write_paragraph_line_stats(ds, stats_dir: Path):
    import matplotlib.pyplot as plt

    line_counts, mean_lens, near_max_fracs, short_fracs = [], [], [], []
    for row in ds:
        if not row[Col.KEEP]:
            continue
        line_counts.extend(row[Col.PARA_LINE_COUNTS])
        mean_lens.extend(row[Col.PARA_MEAN_LINE_LEN])
        near_max_fracs.extend(row[Col.PARA_NEAR_MAX_FRAC])
        short_fracs.extend(row[Col.PARA_SHORT_LINE_FRAC])

    n = len(line_counts)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, data, label in [
        (axes[0, 0], line_counts,    "line count per paragraph"),
        (axes[0, 1], mean_lens,      "mean line length (chars)"),
        (axes[1, 0], near_max_fracs, f"near-max frac (±{NEAR_MAX_THRESH} chars)"),
        (axes[1, 1], short_fracs,    f"short line frac (<{SHORT_LINE_THRESH} chars)"),
    ]:
        ax.hist(data, bins=50)
        ax.set_xlabel(label)
        ax.set_ylabel("paragraphs")
    fig.suptitle(f"Paragraph line stats  (n={n:,} paragraphs)")
    fig.tight_layout()
    path = stats_dir / "paragraph_line_stats.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Paragraph line stats -> {path}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(near_max_fracs, short_fracs, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel(f"near-max frac (±{NEAR_MAX_THRESH} chars)  — hard-wrap signal")
    ax.set_ylabel(f"short line frac (<{SHORT_LINE_THRESH} chars)  — poetry signal")
    ax.set_title(f"Paragraph line stats scatter  (n={n:,})")
    fig.tight_layout()
    path = stats_dir / "paragraph_line_stats_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Paragraph line stats scatter -> {path}")


def write_classification_stats(ds, stats_dir: Path):
    import matplotlib.pyplot as plt

    type_counts: Counter = Counter()
    type_examples: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for book_idx, row in enumerate(ds):
        if not row[Col.KEEP]:
            continue
        for para_idx, label in enumerate(row[Col.PARA_TYPES]):
            key = label if label is not None else "none"
            type_counts[key] += 1
            type_examples[key].append((book_idx, para_idx))

    labels_sorted = sorted(type_counts, key=type_counts.get, reverse=True)
    counts = [type_counts[lbl] for lbl in labels_sorted]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels_sorted, counts)
    ax.set_xlabel("paragraph type")
    ax.set_ylabel("count")
    ax.set_title(f"Paragraph type distribution  (n={sum(counts):,} paragraphs)")
    for i, (lbl, cnt) in enumerate(zip(labels_sorted, counts)):
        ax.text(i, cnt + max(counts) * 0.005, f"{cnt:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = stats_dir / "paragraph_type_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Paragraph type distribution -> {path}")

    rng = random.Random(42)
    sep = "=" * 80
    path = stats_dir / "paragraph_type_samples.txt"
    with open(path, "w") as f:
        for label in labels_sorted:
            examples = type_examples[label]
            sample = rng.sample(examples, min(PARA_STATS_SAMPLE_N, len(examples)))
            f.write(f"{sep}\n")
            f.write(f"TYPE: {label}  (total={type_counts[label]:,}, sampled={len(sample)})\n")
            f.write(f"{sep}\n\n")
            for book_idx, para_idx in sample:
                row = ds[book_idx]
                paras = _extract_paragraphs(row["text"], row[Col.WINDOW_START])
                if para_idx >= len(paras):
                    continue
                f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
                f.write("\n".join(paras[para_idx]))
                f.write("\n\n")
    print(f"Paragraph type samples ({PARA_STATS_SAMPLE_N} per type) -> {path}")


SENTENCE_END_CHARS = frozenset('.!?:;')
UNWRAP_STATS_SAMPLE_N = 100


def _vote_newline(prev_line: str, next_line: str) -> bool:
    """True = hard-wrap (remove \n), False = real break (keep \n)."""
    if not prev_line or not next_line:
        return False
    last_char = prev_line[-1]
    first_char = next_line[0]
    if last_char in SENTENCE_END_CHARS or not first_char.islower():
        return False
    return True


def vote_paragraph_unwrap(book):
    if not book[Col.KEEP] or book[Col.WINDOW_START] is None:
        return book
    text = book.get("text") or ""
    decisions = []
    for lines in _extract_paragraphs(text, book[Col.WINDOW_START]):
        if len(lines) <= 1:
            decisions.append(False)
            continue
        votes = [_vote_newline(lines[i], lines[i + 1]) for i in range(len(lines) - 1)]
        decisions.append(sum(votes) > len(votes) - sum(votes))
    book[Col.PARA_UNWRAP] = decisions
    return book


def write_unwrap_stats(ds, stats_dir: Path):
    unwrap_examples: list[tuple[int, int]] = []
    keep_examples: list[tuple[int, int]] = []

    for book_idx, row in enumerate(ds):
        if not row[Col.KEEP]:
            continue
        for para_idx, decision in enumerate(row[Col.PARA_UNWRAP]):
            (unwrap_examples if decision else keep_examples).append((book_idx, para_idx))

    total = len(unwrap_examples) + len(keep_examples)
    print(f"Unwrap decisions: unwrap={len(unwrap_examples):,} ({100*len(unwrap_examples)/total:.1f}%)  keep={len(keep_examples):,} ({100*len(keep_examples)/total:.1f}%)")

    rng = random.Random(42)
    sep = "=" * 80
    path = stats_dir / "unwrap_samples.txt"
    with open(path, "w") as f:
        f.write(f"total={total:,}  unwrap={len(unwrap_examples):,}  keep={len(keep_examples):,}\n\n")
        for label, examples in [("UNWRAP", unwrap_examples), ("KEEP", keep_examples)]:
            sample = rng.sample(examples, min(UNWRAP_STATS_SAMPLE_N, len(examples)))
            f.write(f"{sep}\n")
            f.write(f"DECISION: {label}  (total={len(examples):,}, sampled={len(sample)})\n")
            f.write(f"{sep}\n\n")
            for book_idx, para_idx in sample:
                row = ds[book_idx]
                paras = _extract_paragraphs(row["text"], row[Col.WINDOW_START])
                if para_idx >= len(paras):
                    continue
                f.write(f"book_id={row[Col.BOOK_ID]}  title={row[Col.BOOK_TITLE]!r}\n")
                f.write("\n".join(paras[para_idx]))
                f.write("\n\n")
    print(f"Unwrap samples -> {path}")

"""
Explore document length distributions of PANORAMA and Nemotron-PII datasets.

Streams a fraction of each dataset, tokenizes with the LLaMA 3.2 1B tokenizer,
and plots length distributions. Also shows per-profile concatenation lengths
for PANORAMA.

Usage:
    python explore_pii_datasets.py               # 1% of each dataset
    python explore_pii_datasets.py --fraction 0.05
"""

import argparse
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

SEQ_LEN = 8192  # target training sequence length
TOKENIZER_ID = "meta-llama/Llama-3.2-1B"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, token=os.environ["HF_TOKEN"])
    print(f"Loaded tokenizer: {TOKENIZER_ID}  (vocab size {tok.vocab_size})")
    return tok


def token_len(tokenizer, text):
    return len(tokenizer.encode(text))


def load_all_splits(repo_id, token):
    """Concatenate all available splits (train/test/validation) into one streaming dataset."""
    from datasets import concatenate_datasets, get_dataset_split_names
    splits = get_dataset_split_names(repo_id, token=token)
    parts = [load_dataset(repo_id, split=s, streaming=True, token=token) for s in splits]
    return concatenate_datasets(parts), splits


def n_to_sample(ds, fraction):
    try:
        total = sum(s.num_examples for s in ds.info.splits.values())
        return max(1, round(total * fraction))
    except Exception:
        return None


def print_stats(label, lengths):
    arr = np.array(lengths)
    frac = np.mean(arr >= SEQ_LEN)
    print(f"  {label}: n={len(arr)}  median={int(np.median(arr))}  "
          f"p95={int(np.percentile(arr, 95))}  max={int(arr.max())}  "
          f"(>={SEQ_LEN}: {frac:.1%})")


# ### PANORAMA ###

def explore_panorama(tokenizer, fraction):
    ds, splits = load_all_splits("srirxml/PANORAMA", token=os.environ["HF_TOKEN"])
    n = n_to_sample(ds, fraction)
    print(f"\n=== PANORAMA  splits={splits}  [{fraction:.0%} = {n} docs] ===")

    doc_lengths = []
    content_type_lengths = defaultdict(list)
    profile_lengths = defaultdict(int)  # profile_id -> total tokens

    first_row = next(iter(ds))
    print(f"  available fields: {list(first_row.keys())}")
    # detect field names from first row
    text_field = next((k for k in ["text", "content"] if k in first_row), None)
    type_field = next((k for k in ["content-type", "content_type", "type", "category", "doc_type", "source"] if k in first_row), None)
    id_field = next((k for k in ["profile_id", "persona_id", "id"] if k in first_row), None)
    print(f"  using: text={text_field!r}  type={type_field!r}  profile_id={id_field!r}")

    for row in tqdm(ds.take(n), total=n,
                    unit="doc", desc="PANORAMA [docs processed / elapsed < remaining]"):
        text = row.get(text_field) or "" if text_field else ""
        tlen = token_len(tokenizer, text)
        doc_lengths.append(tlen)
        content_type_lengths[row.get(type_field) or "unknown" if type_field else "unknown"].append(tlen)
        profile_id = str(row.get(id_field) or "") if id_field else ""
        if profile_id:
            profile_lengths[profile_id] += tlen

    print(f"  {len(doc_lengths)} documents, {len(profile_lengths)} profiles")
    print_stats("per-document", doc_lengths)

    print("  per content type:")
    for ctype, lengths in sorted(content_type_lengths.items()):
        arr = np.array(lengths)
        print(f"    {ctype:30s}  n={len(arr):5d}  "
              f"median={int(np.median(arr)):5d}  p95={int(np.percentile(arr, 95)):5d}")

    profile_concat = list(profile_lengths.values())
    if profile_concat:
        print_stats("per-profile concat", profile_concat)

    return doc_lengths, content_type_lengths, profile_concat


# ### NEMOTRON-PII ###

def explore_nemotron(tokenizer, fraction):
    ds, splits = load_all_splits("nvidia/Nemotron-PII", token=os.environ["HF_TOKEN"])
    n = n_to_sample(ds, fraction)
    print(f"\n=== Nemotron-PII  splits={splits}  [{fraction:.0%} = {n} docs] ===")

    doc_lengths = []
    domain_lengths = defaultdict(list)

    for row in tqdm(ds.take(n), total=n,
                    unit="doc", desc="Nemotron-PII [docs processed / elapsed < remaining]"):
        text = row.get("text") or ""
        tlen = token_len(tokenizer, text)
        doc_lengths.append(tlen)
        domain_lengths[row.get("domain") or "unknown"].append(tlen)

    print(f"  {len(doc_lengths)} documents")
    print_stats("per-document", doc_lengths)

    print("  top domains by median length:")
    by_median = sorted(domain_lengths.items(), key=lambda kv: -np.median(kv[1]))
    for domain, lengths in by_median[:10]:
        print(f"    {domain:40s}  n={len(lengths):4d}  median={int(np.median(lengths)):5d}")

    return doc_lengths, domain_lengths


# ### PLOTTING ###

def plot_results(panorama_doc, panorama_by_type, panorama_profile, nemotron_doc, nemotron_by_domain, fraction, out_path):
    # 2 rows x 3 cols: [per-doc | by-type | per-profile-concat] per dataset
    n_groups = max(len(panorama_by_type), len(nemotron_by_domain))
    width = max(20, n_groups * 0.6)
    fig, axes = plt.subplots(2, 3, figsize=(width, 14))
    pct = max(1, round(fraction * 100))
    fig.suptitle(f"PII Dataset Token Length Distributions  ({pct}% sample)", fontsize=14)

    def plot_hist(ax, data, color, title):
        arr = np.array(data)
        ax.hist(arr, bins=60, color=color, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("tokens")
        ax.set_ylabel("count")
        stats = (f"n={len(arr):,}\n"
                 f"min={int(arr.min())}  mean={int(arr.mean())}  max={int(arr.max())}")
        ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=7.5,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    def plot_by_group(ax, groups, title, xlabel):
        labels = sorted(groups.keys(), key=lambda l: np.median(groups[l]), reverse=True)
        medians = [np.median(groups[l]) for l in labels]
        means = [np.mean(groups[l]) for l in labels]
        x = np.arange(len(labels))
        ax.bar(x, medians, label="median", color="steelblue")
        ax.bar(x, [m - med for m, med in zip(means, medians)], bottom=medians,
               label="mean", color="darkorange", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("tokens")
        ax.legend(loc="upper right")

    # row 0: PANORAMA
    plot_hist(axes[0, 0], panorama_doc, "steelblue", "PANORAMA: per-document lengths")
    plot_by_group(axes[0, 1], panorama_by_type, "PANORAMA: length by content type", "content type")
    if panorama_profile:
        plot_hist(axes[0, 2], panorama_profile, "darkorange", "PANORAMA: per-profile concatenated lengths")
    else:
        axes[0, 2].set_visible(False)

    # row 1: Nemotron-PII
    plot_hist(axes[1, 0], nemotron_doc, "seagreen", "Nemotron-PII: per-document lengths")
    plot_by_group(axes[1, 1], nemotron_by_domain, "Nemotron-PII: length by domain", "domain")
    axes[1, 2].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ### MAIN ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.01,
                        help="Fraction of each dataset to sample (default: 0.01 = 1%%)")
    args = parser.parse_args()

    tokenizer = load_tokenizer()

    panorama_doc, panorama_by_type, panorama_profile = explore_panorama(tokenizer, args.fraction)
    nemotron_doc, nemotron_by_domain = explore_nemotron(tokenizer, args.fraction)

    pct = max(1, round(args.fraction * 100))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(__file__).parents[1] / "results" / "plots" / f"pii-tokens-panorama-nemotron-{pct}pct-{timestamp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_results(panorama_doc, panorama_by_type, panorama_profile, nemotron_doc, nemotron_by_domain, args.fraction, out_path)


if __name__ == "__main__":
    main()
"""
Plot FineWeb-Edu contamination diagnostics from sampled_containment.jsonl.

Outputs:
  containment_thresholds.png    — perplexity vs min_k_pp at multiple contamination_fraction cutoffs
  contamination_distributions.png — coverage histogram, mean-hits histogram, and scatter plots

Reads sampled_containment.jsonl (output of check_fineweb_containment.py).
Requires perplexity and min_k_pp fields — run after scoring step 20.

Usage:
    python plot_fineweb_containment_thresholds.py \\
        --input    /path/to/sampled_containment.jsonl \\
        --output-dir /path/to/stats/fineweb_containment/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Percentiles of non-zero contamination fractions (contaminated books only).
# First entry is always 0 (= any match).
PERCENTILES = [0, 10, 25, 40, 50, 60, 75, 90, 95, 99]

N_COLS = 5


def write_threshold_scatter(records: list[dict], out_dir: Path):
    """Perplexity vs min_k_pp colored by contamination_fraction at multiple cutoffs."""
    scored = [r for r in records if r.get('perplexity') is not None and r.get('min_k_pp') is not None]
    if not scored:
        print("No perplexity/min_k_pp scores — skipping threshold scatter.")
        return

    log_ppl = np.log10([r['perplexity'] for r in scored])
    min_k_pp = np.array([r['min_k_pp'] for r in scored])
    fractions = np.array([r.get('contamination_fraction', 0.0) for r in scored])

    nonzero = fractions[fractions > 0]
    thresholds = []
    for p in PERCENTILES:
        if p == 0:
            thresholds.append((p, 0.0))
        else:
            thresholds.append((p, float(np.percentile(nonzero, p))))

    print("Thresholds (percentile -> contamination_fraction):")
    for p, t in thresholds:
        print(f"  p{p:>2d} -> frac > {t:.4f}")

    n_rows = (len(thresholds) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 4, n_rows * 3.5), squeeze=False)

    for i, (p, threshold) in enumerate(thresholds):
        ax = axes[i // N_COLS][i % N_COLS]
        contaminated = fractions > threshold
        n_cont = int(contaminated.sum())
        pct = n_cont / len(scored) * 100

        label = "any match" if p == 0 else f"p{p} of contaminated"
        ax.set_title(f"{label}  (frac > {threshold:.4f})\n{n_cont:,} contaminated ({pct:.1f}%)", fontsize=8)

        ax.scatter(log_ppl[~contaminated], min_k_pp[~contaminated],
                   c='steelblue', marker='o', s=5, alpha=0.35, linewidths=0)
        ax.scatter(log_ppl[contaminated], min_k_pp[contaminated],
                   c='red', marker='x', s=12, alpha=0.7, linewidths=0.8)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
        if i % N_COLS == 0:
            ax.set_ylabel("Min-K%++ z-score", fontsize=7)
        if i // N_COLS == n_rows - 1:
            ax.set_xlabel("Perplexity", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(thresholds), n_rows * N_COLS):
        axes[j // N_COLS][j % N_COLS].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=7, label='not in FineWeb'),
        plt.Line2D([0], [0], marker='x', color='red', markersize=7,
                   markeredgewidth=1, label='in FineWeb (frac > threshold)'),
    ]
    fig.legend(handles=legend_handles, loc='lower right', fontsize=9)
    fig.suptitle(
        f"Perplexity vs Min-K%++  n={len(scored):,}  colored by FineWeb-Edu contamination fraction threshold",
        fontsize=11,
    )
    fig.tight_layout()

    path = out_dir / 'containment_thresholds.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {path}")


def write_contamination_distributions(records: list[dict], out_dir: Path):
    """Coverage histogram, mean-hits histogram, and scatter of ppl/min_k_pp vs fraction."""
    contaminated = [r for r in records if r.get('in_fineweb') and r.get('fineweb_ngram_hits')]
    if not contaminated:
        print("No contaminated records with ngram hits — skipping distributions plot.")
        return

    fractions_cont = np.array([r['contamination_fraction'] for r in contaminated])
    mean_hits = np.array([np.mean(r['fineweb_ngram_hits']) for r in contaminated])
    median_hits = np.array([np.median(r['fineweb_ngram_hits']) for r in contaminated])

    scored = [r for r in records if r.get('perplexity') is not None and r.get('min_k_pp') is not None]
    log_ppl_all = np.log10([r['perplexity'] for r in scored])
    min_k_pp_all = np.array([r['min_k_pp'] for r in scored])
    fractions_all = np.array([r.get('contamination_fraction', 0.0) for r in scored])
    in_fw_all = np.array([bool(r.get('in_fineweb')) for r in scored])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Top-left: histogram of contamination_fraction (contaminated books only)
    ax = axes[0, 0]
    ax.hist(fractions_cont, bins=60, color='salmon', edgecolor='none')
    ax.set_xlabel("contamination_fraction")
    ax.set_ylabel("count")
    ax.set_title(f"Coverage per contaminated book  (n={len(contaminated):,})", fontsize=10)

    # Top-right: histogram of mean and median hits per ngram (log x-scale)
    ax = axes[0, 1]
    bins = np.logspace(np.log10(max(mean_hits.min(), 0.5)), np.log10(mean_hits.max() + 1), 50)
    ax.hist(mean_hits, bins=bins, color='salmon', alpha=0.7, edgecolor='none', label='mean')
    ax.hist(median_hits, bins=bins, color='steelblue', alpha=0.6, edgecolor='none', label='median')
    ax.set_xscale('log')
    ax.set_xlabel("FineWeb hits per matched ngram (log scale)")
    ax.set_ylabel("count")
    ax.set_title(f"Ngram popularity distribution  (n={len(contaminated):,})", fontsize=10)
    ax.legend(fontsize=8)

    # Bottom-left: scatter perplexity vs contamination_fraction
    ax = axes[1, 0]
    if scored:
        vmax = fractions_all[in_fw_all].max() if in_fw_all.any() else 1.0
        ax.scatter(log_ppl_all[~in_fw_all], fractions_all[~in_fw_all],
                   c='lightgrey', s=4, alpha=0.4, linewidths=0)
        sc = ax.scatter(log_ppl_all[in_fw_all], fractions_all[in_fw_all],
                        c=fractions_all[in_fw_all], cmap='Reds', vmin=0, vmax=vmax,
                        s=6, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='contamination_fraction')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("contamination_fraction")
        ax.set_title("Perplexity vs coverage", fontsize=10)

    # Bottom-right: scatter min_k_pp vs contamination_fraction
    ax = axes[1, 1]
    if scored:
        ax.scatter(min_k_pp_all[~in_fw_all], fractions_all[~in_fw_all],
                   c='lightgrey', s=4, alpha=0.4, linewidths=0)
        sc = ax.scatter(min_k_pp_all[in_fw_all], fractions_all[in_fw_all],
                        c=fractions_all[in_fw_all], cmap='Reds', vmin=0, vmax=vmax,
                        s=6, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='contamination_fraction')
        ax.set_xlabel("Min-K%++ z-score")
        ax.set_ylabel("contamination_fraction")
        ax.set_title("Min-K%++ vs coverage", fontsize=10)

    fig.suptitle(
        f"FineWeb-Edu contamination distributions  n={len(records):,}  contaminated={len(contaminated):,}",
        fontsize=11,
    )
    fig.tight_layout()

    path = out_dir / 'contamination_distributions.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='sampled_containment.jsonl from check_fineweb_containment.py')
    parser.add_argument('--output-dir', required=True,
                        help='directory to write plots')
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records found.")
        return

    print(f"Loaded {len(records):,} records")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_threshold_scatter(records, out_dir)
    write_contamination_distributions(records, out_dir)


if __name__ == '__main__':
    main()
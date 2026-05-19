"""
Plot FineWeb-Edu contamination diagnostics from sampled_containment.jsonl.

Outputs:
  containment_thresholds.png    — perplexity vs min_k_pp at fixed coverage cutoffs
  contamination_distributions.png — coverage histogram, ngram hit histogram, and scatter plots

Reads sampled_containment.jsonl (output of check_fineweb_containment.py).
Also reads hash_to_ngram.json from --output-dir for the ngram hit histogram.
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

FIXED_THRESHOLDS = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50, 0.70, 0.90]

N_COLS = 5


def write_threshold_scatter(records: list[dict], out_dir: Path):
    """Perplexity vs min_k_pp colored by contamination_fraction at fixed coverage cutoffs."""
    scored = [r for r in records if r.get('perplexity') is not None and r.get('min_k_pp') is not None]
    if not scored:
        print("No perplexity/min_k_pp scores — skipping threshold scatter.")
        return

    log_ppl = np.log10([r['perplexity'] for r in scored])
    min_k_pp = np.array([r['min_k_pp'] for r in scored])
    fractions = np.array([r.get('contamination_fraction', 0.0) for r in scored])

    n_rows = (len(FIXED_THRESHOLDS) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 4, n_rows * 3.5), squeeze=False)

    for i, threshold in enumerate(FIXED_THRESHOLDS):
        ax = axes[i // N_COLS][i % N_COLS]
        contaminated = fractions > threshold
        n_cont = int(contaminated.sum())
        pct = n_cont / len(scored) * 100

        label = "any match" if threshold == 0.0 else f"coverage > {threshold:.0%}"
        ax.set_title(f"{label}\n{n_cont:,} books ({pct:.1f}%)", fontsize=8)

        ax.scatter(log_ppl[~contaminated], min_k_pp[~contaminated],
                   c='steelblue', marker='o', s=8, alpha=0.35, linewidths=0)
        ax.scatter(log_ppl[contaminated], min_k_pp[contaminated],
                   c='red', marker='o', s=8, alpha=0.7, linewidths=0)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
        if i % N_COLS == 0:
            ax.set_ylabel("Min-K%++ z-score", fontsize=7)
        if i // N_COLS == n_rows - 1:
            ax.set_xlabel("Perplexity", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(len(FIXED_THRESHOLDS), n_rows * N_COLS):
        axes[j // N_COLS][j % N_COLS].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
                   markersize=7, label='not contaminated'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=7, label='contaminated (frac > threshold)'),
    ]
    fig.legend(handles=legend_handles, loc='lower right', fontsize=9)
    fig.suptitle(
        f"Perplexity vs Min-K%++  n={len(scored):,}  colored by FineWeb-Edu contamination fraction threshold",
        fontsize=11,
    )
    fig.tight_layout()

    path = out_dir / 'containment_thresholds.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {path}")


def write_contamination_distributions(records: list[dict], out_dir: Path):
    """Coverage histogram, ngram hit histogram, and scatter of ppl/min_k_pp vs fraction and max hits."""
    from matplotlib.colors import LogNorm

    contaminated = [r for r in records if r.get('in_fineweb')]
    if not contaminated:
        print("No contaminated records — skipping distributions plot.")
        return

    fractions_cont = np.array([r['contamination_fraction'] for r in contaminated])

    scored = [r for r in records if r.get('perplexity') is not None and r.get('min_k_pp') is not None]
    log_ppl_all = np.log10([r['perplexity'] for r in scored])
    min_k_pp_all = np.array([r['min_k_pp'] for r in scored])
    fractions_all = np.array([r.get('contamination_fraction', 0.0) for r in scored])
    in_fw_all = np.array([bool(r.get('in_fineweb')) for r in scored])
    max_hits_all = np.array([r.get('fineweb_max_ngram_hits', 0) for r in scored])

    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    # Top-left: histogram of contamination_fraction (contaminated books only)
    ax = axes[0, 0]
    ax.hist(fractions_cont, bins=60, color='salmon', edgecolor='none')
    ax.set_xlabel("contamination_fraction")
    ax.set_ylabel("count")
    ax.set_yscale('log')
    ax.set_title(f"Coverage per contaminated book  (n={len(contaminated):,})", fontsize=10)

    # Top-right: flat histogram of total_hits per unique matched ngram from hash_to_ngram.json
    ax = axes[0, 1]
    hash_to_ngram_path = out_dir / 'hash_to_ngram.json'
    if hash_to_ngram_path.exists():
        with open(hash_to_ngram_path) as f:
            h2n = json.load(f)
        all_hits = np.array([v['total_hits'] for v in h2n.values()])
        bins = np.logspace(np.log10(max(all_hits.min(), 0.5)), np.log10(all_hits.max() + 1), 60)
        ax.hist(all_hits, bins=bins, color='salmon', edgecolor='none')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("FineWeb hits per matched ngram (log scale)")
        ax.set_ylabel("count")
        ax.set_title(f"Ngram hit count distribution  ({len(all_hits):,} unique matched ngrams)", fontsize=10)
    else:
        ax.text(0.5, 0.5, 'hash_to_ngram.json not found\nrun check_fineweb_containment.py first',
                transform=ax.transAxes, ha='center', va='center', fontsize=9)
        ax.set_title("Ngram hit count distribution", fontsize=10)

    # Middle-left: scatter perplexity vs contamination_fraction
    ax = axes[1, 0]
    if scored:
        vmax_frac = fractions_all[in_fw_all].max() if in_fw_all.any() else 1.0
        ax.scatter(log_ppl_all[~in_fw_all], fractions_all[~in_fw_all],
                   c='lightgrey', s=8, alpha=0.4, linewidths=0)
        sc = ax.scatter(log_ppl_all[in_fw_all], fractions_all[in_fw_all],
                        c=fractions_all[in_fw_all], cmap='coolwarm', vmin=0, vmax=vmax_frac,
                        s=8, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='contamination_fraction')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("contamination_fraction")
        ax.set_title("Perplexity vs coverage", fontsize=10)

    # Middle-right: scatter min_k_pp vs contamination_fraction
    ax = axes[1, 1]
    if scored:
        ax.scatter(min_k_pp_all[~in_fw_all], fractions_all[~in_fw_all],
                   c='lightgrey', s=8, alpha=0.4, linewidths=0)
        sc = ax.scatter(min_k_pp_all[in_fw_all], fractions_all[in_fw_all],
                        c=fractions_all[in_fw_all], cmap='coolwarm', vmin=0, vmax=vmax_frac,
                        s=8, alpha=0.8, linewidths=0)
        plt.colorbar(sc, ax=ax, label='contamination_fraction')
        ax.set_xlabel("Min-K%++ z-score")
        ax.set_ylabel("contamination_fraction")
        ax.set_title("Min-K%++ vs coverage", fontsize=10)

    # Bottom-left: scatter perplexity vs max_ngram_hits (log y-axis, log colormap)
    ax = axes[2, 0]
    if scored:
        vmax_hits = max(max_hits_all.max(), 1)
        log_norm = LogNorm(vmin=0.5, vmax=vmax_hits)
        sc = ax.scatter(log_ppl_all, max_hits_all,
                        c=np.maximum(max_hits_all, 0.5), cmap='coolwarm', norm=log_norm,
                        s=8, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label='max_ngram_hits (log scale)')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
        ax.set_xlabel("Perplexity")
        ax.set_ylabel("max_ngram_hits")
        ax.set_yscale('symlog', linthresh=1)
        ax.set_title("Perplexity vs max ngram hits", fontsize=10)

    # Bottom-right: scatter min_k_pp vs max_ngram_hits (log y-axis, log colormap)
    ax = axes[2, 1]
    if scored:
        sc = ax.scatter(min_k_pp_all, max_hits_all,
                        c=np.maximum(max_hits_all, 0.5), cmap='coolwarm', norm=log_norm,
                        s=8, alpha=0.6, linewidths=0)
        plt.colorbar(sc, ax=ax, label='max_ngram_hits (log scale)')
        ax.set_xlabel("Min-K%++ z-score")
        ax.set_ylabel("max_ngram_hits")
        ax.set_yscale('symlog', linthresh=1)
        ax.set_title("Min-K%++ vs max ngram hits", fontsize=10)

    fig.suptitle(
        f"FineWeb-Edu contamination distributions  n={len(records):,}  contaminated={len(contaminated):,}",
        fontsize=11,
    )
    fig.tight_layout()

    path = out_dir / 'contamination_distributions.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved -> {path}")


def write_contamination_stats(records: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(records)
    contaminated = [r for r in records if r.get('in_fineweb')]
    n_contaminated = len(contaminated)

    path = out_dir / 'fineweb_containment_stats.txt'
    with open(path, 'w') as f:
        f.write(f"n={n_total}\n")
        f.write(f"contaminated={n_contaminated} ({n_contaminated / n_total * 100:.1f}%)\n")
        f.write(f"clean={n_total - n_contaminated} ({(n_total - n_contaminated) / n_total * 100:.1f}%)\n")
        if contaminated:
            matched = np.array([r['fineweb_matched_ngrams'] for r in contaminated])
            fractions = np.array([r['contamination_fraction'] for r in contaminated])
            max_hits = np.array([r['fineweb_max_ngram_hits'] for r in contaminated])
            f.write(
                f"matched_ngrams (contaminated only): "
                f"min={matched.min()}  median={int(np.median(matched))}  max={matched.max()}\n"
            )
            f.write(
                f"contamination_fraction (contaminated only): "
                f"min={fractions.min():.4f}  median={np.median(fractions):.4f}  max={fractions.max():.4f}\n"
            )
            f.write(
                f"max_ngram_hits (contaminated only): "
                f"min={max_hits.min()}  median={int(np.median(max_hits))}  max={max_hits.max()}\n"
            )
    print(f"Containment stats -> {path}")


def write_containment_scatter(records: list[dict], out_dir: Path):
    """Perplexity vs Min-K%++ colored by FineWeb containment. Skipped if scores absent."""
    scored = [
        r for r in records
        if r.get('perplexity') is not None and r.get('min_k_pp') is not None
    ]
    if not scored:
        print("No perplexity/min_k_pp scores found — skipping containment scatter.")
        return

    log_ppl = np.log10([r['perplexity'] for r in scored])
    min_k_pp_vals = np.array([r['min_k_pp'] for r in scored])
    in_fw = np.array([r['in_fineweb'] for r in scored])

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, mask, color in [
        ('not in FineWeb', ~in_fw, 'steelblue'),
        ('in FineWeb', in_fw, 'red'),
    ]:
        ax.scatter(
            log_ppl[mask], min_k_pp_vals[mask],
            c=color, marker='o', s=8, alpha=0.5, linewidths=0,
            label=f"{label} (n={mask.sum():,})",
        )

    ax.set_xlabel("Perplexity (log₁₀ scale)")
    ax.set_ylabel("Min-K%++ z-score")
    ax.set_title(f"Perplexity vs Min-K%++  n={len(scored):,}  colored by FineWeb-Edu containment")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10 ** x:.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = out_dir / 'containment_ppl_min_k_pp.png'
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Containment scatter -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='sampled_containment.jsonl from check_fineweb_containment.py')
    parser.add_argument('--output-dir', required=True,
                        help='directory to write plots (also read hash_to_ngram.json from here)')
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

    write_contamination_stats(records, out_dir)
    write_containment_scatter(records, out_dir)
    write_threshold_scatter(records, out_dir)
    write_contamination_distributions(records, out_dir)


if __name__ == '__main__':
    main()

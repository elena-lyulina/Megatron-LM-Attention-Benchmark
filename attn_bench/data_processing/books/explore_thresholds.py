"""
Explore post-contamination filtering thresholds for the Gutenberg dataset.

Reads sampled_containment.jsonl (must have perplexity, contamination_fraction,
fineweb_max_ngram_hits fields). Produces a single PNG with 5 annotated heatmaps —
one per symmetric perplexity cutoff pair (p5/p95 … p25/p75) — arranged in a 2×3
grid (last cell hidden). Each subplot shows exact book counts surviving all three
filters across a grid of coverage_max × max_hit_max values, with the number of
books surviving the perplexity cut alone in the title.

Usage:
    python -m attn_bench.data_processing.books.explore_thresholds \\
        --input  /path/to/sampled_containment.jsonl \\
        --output-dir /path/to/stats/threshold_exploration/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PPL_PAIRS = [(0, 100), (5, 95), (10, 90), (15, 85), (20, 80), (25, 75)]
COVERAGE_MAX_VALUES = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
MAX_HIT_MAX_VALUES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

N_COLS = 3


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records")
    return records


def fill_subplot(
    ax,
    grid: np.ndarray,
    n_after_ppl: int,
    n_total: int,
    ppl_lo_q: int,
    ppl_hi_q: int,
    ppl_lo_val: float,
    ppl_hi_val: float,
    coverage_vals: list,
    max_hit_vals: list,
):
    im = ax.imshow(grid, aspect='auto', cmap='Blues')
    plt.colorbar(im, ax=ax, label='books surviving')

    vmax = grid.max()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            color = 'white' if val > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:,}', ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(range(len(coverage_vals)))
    ax.set_xticklabels([f'{v:.0%}' for v in coverage_vals], fontsize=8)
    ax.set_yticks(range(len(max_hit_vals)))
    ax.set_yticklabels([str(v) for v in max_hit_vals], fontsize=8)
    ax.set_xlabel('coverage_max  (contamination_fraction ≤ x)', fontsize=8)
    ax.set_ylabel('max_hit_max  (fineweb_max_ngram_hits ≤ x)', fontsize=8)
    ax.set_title(
        f'PPL cut  p{ppl_lo_q}/p{ppl_hi_q}  ({ppl_lo_val:.1f} – {ppl_hi_val:.1f})\n'
        f'after PPL cut: {n_after_ppl:,} / {n_total:,}  ({n_after_ppl / n_total * 100:.1f}%)',
        fontsize=9,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                        help='sampled_containment.jsonl with perplexity, contamination_fraction, fineweb_max_ngram_hits')
    parser.add_argument('--output-dir', required=True,
                        help='directory to write the combined heatmap PNG')
    args = parser.parse_args()

    records = load_records(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppls = np.array([r.get('perplexity') or float('nan') for r in records])
    coverages = np.array([r.get('contamination_fraction', 0.0) for r in records])
    max_hits = np.array([r.get('fineweb_max_ngram_hits', 0) for r in records])
    n_total = len(records)

    valid_ppl = ~np.isnan(ppls)

    n_rows = (len(PPL_PAIRS) + N_COLS - 1) // N_COLS
    cell_w = len(COVERAGE_MAX_VALUES) * 1.5 + 2.0
    cell_h = len(MAX_HIT_MAX_VALUES) * 1.2 + 2.0
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * cell_w, n_rows * cell_h))

    for idx, (ppl_lo_q, ppl_hi_q) in enumerate(PPL_PAIRS):
        ax = axes[idx // N_COLS][idx % N_COLS]

        ppl_lo_val = float(np.nanpercentile(ppls, ppl_lo_q))
        ppl_hi_val = float(np.nanpercentile(ppls, ppl_hi_q))

        ppl_mask = valid_ppl & (ppls >= ppl_lo_val) & (ppls <= ppl_hi_val)
        n_after_ppl = int(ppl_mask.sum())
        print(f'p{ppl_lo_q}/p{ppl_hi_q}  ({ppl_lo_val:.1f}–{ppl_hi_val:.1f}):  {n_after_ppl:,} books after PPL cut')

        grid = np.zeros((len(MAX_HIT_MAX_VALUES), len(COVERAGE_MAX_VALUES)), dtype=int)
        for i, max_hit_max in enumerate(MAX_HIT_MAX_VALUES):
            for j, coverage_max in enumerate(COVERAGE_MAX_VALUES):
                mask = ppl_mask & (coverages <= coverage_max) & (max_hits <= max_hit_max)
                grid[i, j] = int(mask.sum())

        fill_subplot(
            ax=ax,
            grid=grid,
            n_after_ppl=n_after_ppl,
            n_total=n_total,
            ppl_lo_q=ppl_lo_q,
            ppl_hi_q=ppl_hi_q,
            ppl_lo_val=ppl_lo_val,
            ppl_hi_val=ppl_hi_val,
            coverage_vals=COVERAGE_MAX_VALUES,
            max_hit_vals=MAX_HIT_MAX_VALUES,
        )

    for idx in range(len(PPL_PAIRS), n_rows * N_COLS):
        axes[idx // N_COLS][idx % N_COLS].set_visible(False)

    fig.suptitle('Gutenberg filtering threshold exploration', fontsize=12)
    fig.tight_layout()

    out_path = out_dir / 'threshold_exploration.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
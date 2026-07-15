"""Plots for individual (per-sequence) long-sequence position-wise loss, from the
*_individual.jsonl files written by --store-individual (attn_bench/evaluation/long_inference.py).

Unlike plot_long_gutenberg.py's plot_loss_grid (mean/std/count over a bucket), this reads raw
per-position, per-sequence records -- one line per sequence -- so a single sequence can be
inspected in detail: NLL, how far off the model's guess was (true-token rank), and, when a
tokenizer is given and the window is short enough to stay legible, the actual token text.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory

from attn_bench.plotting.long_inference_util import SAMPLE_LEN, VOCAB_SIZE, smooth

_COLORS = plt.get_cmap("tab10").colors


### DATA ###

def load_individual(path: Path) -> dict:
    """Read one *_individual.jsonl into {seq_id: record}."""
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            records[r["seq_id"]] = r
    return records


### PLOT ###

def plot_individual_grid(results_by_label, seq_ids, x_range=None, tokenizer=None, ncols=3,
                         colors=None, smooth_window=0, smooth_window_true_rank=0,
                         show_unsmoothed=False,
                         sample_end=SAMPLE_LEN, show_rank=True, sharey=True, metric="nll",
                         show_random_baseline=False,
                         max_tokens_for_labels=100, suptitle=None):
    """One cell per seq_id; every model overlaid as a coloured NLL line (+ optional
    true-token-rank line underneath).

    results_by_label: {label: {seq_id: record}} -- load each model's file with
    load_individual() first (random access by seq_id, cheap to hold in memory at this scale).
    seq_ids: which sequences to plot, one grid cell each (a cell is skipped if no model has it,
    e.g. when comparing models run on different rep buckets).
    x_range: (lo, hi) position window shared by every cell, or None for the full sequence.
    show_rank: include the true-token-rank panel underneath NLL (default True). Set False
    for an NLL-only cell.
    sharey: one common y-scale across every cell's NLL panel, and separately across every
    cell's rank panel (default True).
    metric: "nll" (default) plots NLL as-is; "ppl" plots exp(NLL) (perplexity), computed
    after smoothing -- same convention as plot_loss_grid.
    show_random_baseline: draw a horizontal line at the NLL/perplexity of a uniform random
    guess over the tokenizer vocab (same as plot_loss_grid).
    tokenizer: an already-loaded HF tokenizer (needs .decode()). When given and the window is
    at most max_tokens_for_labels wide, a token-label strip is added under each cell: the true
    token (black) and, per model, its argmax prediction (green if it matches the true token,
    red otherwise). Left out (with a note) if the window is too wide to stay legible.
    smooth_window: rolling-mean window applied to NLL.
    smooth_window_true_rank: rolling-mean window applied to true-token rank. Defaults to 0
    (raw) since rank is already an integer and smoothing blurs the "how wrong" signal it's
    there for, but a noisy rank curve can be worth smoothing too -- set independently of
    smooth_window since the two series have very different scales/noise.
    show_unsmoothed: also draw the raw (pre-smoothing) line very faintly behind the smoothed
    one, on whichever panel(s) have smoothing on (smooth_window / smooth_window_true_rank).
    No effect on a panel whose window is 0 -- there'd be nothing to distinguish it from.
    """
    labels = list(results_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, _COLORS[i % len(_COLORS)]) for i, lab in enumerate(labels)}

    cells = [sid for sid in seq_ids if any(sid in results_by_label[l] for l in labels)]
    if not cells:
        raise ValueError(f"None of seq_ids={seq_ids} found under any label in results_by_label")

    window = None if x_range is None else x_range[1] - x_range[0]
    show_tokens = tokenizer is not None and window is not None and window <= max_tokens_for_labels
    if tokenizer is not None and not show_tokens:
        print(f"Window too wide for token labels ({window} > {max_tokens_for_labels} positions) "
              "-- plotting without the token strip.")

    ncols = min(ncols, len(cells))
    nrows = int(np.ceil(len(cells) / ncols))

    panels = ["nll"] + (["rank"] if show_rank else []) + (["tokens"] if show_tokens else [])
    rows_per_cell = len(panels)
    # Scale with the actual row span (true + gaps + one row per model), not just the model
    # count -- _MODEL_ROW_GAP makes that span grow faster than len(labels) once there's more
    # than one model, and the panel needs to grow with it or the text overlaps.
    token_span = _token_row_y(len(labels))[-1] + 1 if show_tokens else 0  # +1: top/bottom pad
    token_ratio = 0.22 * token_span
    ratio_by_panel = {"nll": 5.0, "rank": 1.3, "tokens": token_ratio}
    height_ratios = [ratio_by_panel[p] for p in panels]

    fig_h = nrows * (sum(height_ratios) / 3.0) * 2.6
    fig = plt.figure(figsize=(6.5 * ncols, fig_h), layout="constrained")
    gs = fig.add_gridspec(nrows * rows_per_cell, ncols, height_ratios=height_ratios * nrows)

    handles = [plt.Line2D([], [], color=colors[l], linewidth=1.8) for l in labels]
    leg_labels = list(labels)
    # seq length boundary: same style/label convention as plot_loss_grid, so the two read
    # consistently if you're looking at both.
    boundary_style = dict(color="#999999", linestyle=(0, (2, 2)), linewidth=1.0)
    if sample_end is not None:
        handles = handles + [plt.Line2D([], [], **boundary_style)]
        leg_labels = leg_labels + [f"seq length ({sample_end})"]
    # Random-guess baseline: ln(V) in NLL space, V itself in perplexity space -- same as
    # plot_loss_grid.
    baseline_style = dict(color="#cc3333", linestyle=(0, (1, 1)), linewidth=1.2)
    baseline_value = np.log(VOCAB_SIZE) if metric == "nll" else VOCAB_SIZE
    if show_random_baseline:
        handles = handles + [plt.Line2D([], [], **baseline_style)]
        leg_labels = leg_labels + [f"random guess (V={VOCAB_SIZE:,})"]
    # loc="outside upper center" (no bbox_to_anchor) lets constrained_layout reserve exactly
    # the space this needs and no more -- a fixed bbox_to_anchor fraction (the old approach)
    # is a fraction of the *whole* figure height, so it looks fine on a short grid and leaves
    # a huge gap on a tall one. suptitle is folded into the legend's own title= instead of a
    # separate fig.suptitle(): two independently-positioned artists both trying to sit "above
    # everything" is what caused them to overlap.
    fig.legend(
        handles,
        leg_labels,
        title=suptitle,
        loc="outside upper center",
        ncols=len(leg_labels),
        frameon=False,
        borderpad=0,
        borderaxespad=0.2,
        handletextpad=0.2,
    )

    nll_axes = []
    rank_axes = []
    for i, seq_id in enumerate(cells):
        r, c = divmod(i, ncols)
        nll_ax = fig.add_subplot(gs[rows_per_cell * r + panels.index("nll"), c],
                                 sharey=(nll_axes[0] if sharey and nll_axes else None))
        nll_axes.append(nll_ax)
        rank_ax = None
        if show_rank:
            rank_ax = fig.add_subplot(gs[rows_per_cell * r + panels.index("rank"), c], sharex=nll_ax,
                                      sharey=(rank_axes[0] if sharey and rank_axes else None))
            rank_axes.append(rank_ax)
        tok_ax = (fig.add_subplot(gs[rows_per_cell * r + panels.index("tokens"), c], sharex=nll_ax)
                 if show_tokens else None)

        title_len = None
        for lab in labels:
            rec = results_by_label[lab].get(seq_id)
            if rec is None:
                continue
            title_len = rec["length"]
            lo = 0 if x_range is None else max(0, x_range[0])
            hi = rec["length"] if x_range is None else min(rec["length"], x_range[1])
            pos = np.arange(lo, hi)
            raw_nll = np.array(rec["nll"][lo:hi])
            if show_unsmoothed and smooth_window > 1:
                faint_nll = np.exp(raw_nll) if metric == "ppl" else raw_nll
                nll_ax.plot(pos, faint_nll, color=colors[lab], linewidth=0.6, alpha=0.25)
            nll = smooth(raw_nll, smooth_window)
            if metric == "ppl":
                nll = np.exp(nll)
            nll_ax.plot(pos, nll, color=colors[lab], linewidth=1.0)
            if rank_ax is not None:
                raw_rank = np.array(rec["true_token_rank"][lo:hi])
                if show_unsmoothed and smooth_window_true_rank > 1:
                    rank_ax.plot(pos, raw_rank, color=colors[lab], linewidth=0.6, alpha=0.25)
                rank_ax.plot(pos, smooth(raw_rank, smooth_window_true_rank), color=colors[lab], linewidth=0.9)

        # One vertical gridline per position when tokens are shown, so the curve above lines
        # up exactly with the token below it -- the default locator can space major ticks
        # every 2/5/10 positions, which drifts out of alignment with per-token text.
        if show_tokens:
            nll_ax.xaxis.set_major_locator(MultipleLocator(1))
        if sample_end is not None and (x_range is None or x_range[0] <= sample_end <= x_range[1]):
            nll_ax.axvline(sample_end, **boundary_style)
        if show_random_baseline:
            nll_ax.axhline(baseline_value, **baseline_style)
        nll_ax.set_title(f"seq_id: {seq_id}  (len={title_len})" if title_len else str(seq_id))
        nll_ax.spines[["top", "right"]].set_visible(False)
        # More y-ticks than matplotlib's default -- with sharey, one outlier cell can stretch
        # the axis wide enough that the default locator collapses to just 2-3 ticks, making
        # every other (normal-range) cell look flat.
        nll_ax.yaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=6))
        nll_ax.grid(True, color="#ececec", linewidth=0.8)
        if c == 0:
            nll_ax.set_ylabel("Perplexity" if metric == "ppl" else "NLL")

        stacked = [nll_ax] + ([rank_ax] if rank_ax is not None else [])
        if rank_ax is not None:
            if show_tokens:
                rank_ax.xaxis.set_major_locator(MultipleLocator(1))
            rank_ax.set_yscale("symlog", linthresh=1)
            rank_ax.spines[["top", "right"]].set_visible(False)
            # grid() called after set_yscale -- changing scale resets the axis's tick
            # locators, which would otherwise wipe out a grid set up beforehand.
            rank_ax.grid(True, color="#ececec", linewidth=0.8)
            if c == 0:
                rank_ax.set_ylabel("true token rank")
        for ax in stacked[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        if tok_ax is not None:
            plt.setp(stacked[-1].get_xticklabels(), visible=False)
            _plot_token_strip(tok_ax, results_by_label, labels, seq_id, x_range, tokenizer)
            lo = 0 if x_range is None else max(0, x_range[0])
            hi = title_len if x_range is None else min(title_len, x_range[1])
            _connect_panels(fig, stacked + [tok_ax], lo, hi)
        else:
            stacked[-1].set_xlabel("position (from sample start)")

    return fig


def _connect_panels(fig, axes_top_to_bottom, lo, hi):
    """Draw a light vertical connector, at every position, from the bottom of each panel to
    the top of the one below it -- axvline alone stops at each panel's own border, so without
    this a spike in the NLL panel can't be traced visually down through rank and into the
    exact token that caused it. Uses axes-fraction for the y endpoint (0/1 = bottom/top of
    that panel) so it doesn't care about each panel's own data range, scale, or inversion."""
    for upper, lower in zip(axes_top_to_bottom[:-1], axes_top_to_bottom[1:]):
        trans_upper = blended_transform_factory(upper.transData, upper.transAxes)
        trans_lower = blended_transform_factory(lower.transData, lower.transAxes)
        for pos in range(lo, hi):
            fig.add_artist(ConnectionPatch(
                xyA=(pos, 0), coordsA=trans_upper, xyB=(pos, 1), coordsB=trans_lower,
                color="#ececec", linewidth=0.8, zorder=0,
            ))


_TOKEN_FONT_SIZE = 8
_TRUE_ROW_GAP = 0.6    # extra vertical space between the true-token row and the first model row
_MODEL_ROW_GAP = 2.5   # extra vertical space between consecutive model rows


def _token_row_y(n_models):
    """Row y-position for "true" (row 0) + each of n_models model rows: a small gap after
    true, then a much bigger gap between models -- models need more separation since each
    one's rotated text can be several characters tall and would otherwise run into the next
    model's row, whereas true-vs-first-model rarely collide as badly in practice."""
    row_y = [0.0]
    for i in range(n_models):
        gap = _TRUE_ROW_GAP if i == 0 else _MODEL_ROW_GAP
        row_y.append(row_y[-1] + 1 + gap)
    return row_y


def _plot_token_strip(ax, results_by_label, labels, seq_id, x_range, tokenizer):
    """Under a cell: true token (top row, black) + each model's argmax prediction (green if
    it matches the true token, red otherwise), one row per model, vertical text. A gap
    separates the true row from the model rows so they don't visually run together."""
    present = [lab for lab in labels if seq_id in results_by_label[lab]]
    any_rec = results_by_label[present[0]][seq_id]
    lo = 0 if x_range is None else max(0, x_range[0])
    hi = any_rec["length"] if x_range is None else min(any_rec["length"], x_range[1])

    # rotation_mode="anchor" makes ha exact: the true row grows right from its position
    # (ha="left"), the prediction rows grow left from theirs (ha="right") -- so at any given
    # position the two token halves meet at the shared tick instead of both drifting the same
    # way and colliding with whichever neighbor happens to also be long.
    common_kwargs = dict(rotation=90, va="center", rotation_mode="anchor", fontsize=_TOKEN_FONT_SIZE)
    rows = ["true"] + present
    row_y = _token_row_y(len(present))

    # A vertical guide line per position, continuing the NLL/rank panels' gridlines down
    # through the token strip so a spike above can be traced to the exact token below it.
    for pos in range(lo, hi):
        ax.axvline(pos, color="#ececec", linewidth=0.8, zorder=0)
    for pos in range(lo, hi):
        true_id = any_rec["true_token"][pos]
        ax.text(pos, row_y[0], tokenizer.decode([true_id]), color="black", ha="left", **common_kwargs)
        for ri, lab in zip(row_y[1:], present):
            pred_id = results_by_label[lab][seq_id]["argmax_token"][pos]
            color = "#2ca02c" if pred_id == true_id else "#d62728"
            ax.text(pos, ri, tokenizer.decode([pred_id]), color=color, ha="right", **common_kwargs)

    ax.set_yticks(row_y)
    ax.set_yticklabels(rows, fontsize=8)
    ax.set_ylim(min(row_y) - 0.5, max(row_y) + 0.5)
    ax.invert_yaxis()  # "true" row on top
    ax.set_xlim(lo - 0.5, hi - 0.5)
    # tick_params, not set_xticks([]) -- this axes shares its x-axis with the panels above
    # (sharex=nll_ax), and set_xticks([]) installs an empty FixedLocator that propagates to
    # the whole shared group, silently wiping out their MultipleLocator(1) grid ticks too.
    ax.tick_params(axis="x", length=0, labelbottom=False, labeltop=False)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    # No xlabel here -- the tokens themselves make the x-axis meaning obvious, and the label
    # would just repeat what the NLL/rank panels above already say when tokens aren't shown.

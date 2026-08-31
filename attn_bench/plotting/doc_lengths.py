"""Document-length and packed-chunk-length distributions for the training datasets.

Loaders live in data_loading.py (load_doc_lengths, load_packed_chunk_length_hist).
This file is plotting only.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from attn_bench.plotting.data_loading import select_models

PERCENTILES = [1, 5, 25, 50, 75, 90, 95, 99]

# Pretty names for the legend / titles; falls back to the raw dataset name if not listed.
DISPLAY_NAMES = {
    "fineweb-edu-dedup-160B-datatrove": "fineweb-edu-full",
    "fineweb-edu-dedup-160B-datatrove_0.25": "fineweb-edu-0.25",
    "gutenberg_rep_1_256": "gutenberg",
}

_THOUSANDS = FuncFormatter(lambda x, _: f"{int(x):,}")


def _display_name(name):
    return DISPLAY_NAMES.get(name, name)


def _log_bins(max_value, n_bins=80):
    # +1 so the last edge doesn't land just below max_value from log10/pow roundoff,
    # which would silently drop every value at exactly max_value into no bin.
    return np.logspace(0, np.log10(max_value + 1), n_bins)


def _darken(color, factor=0.6):
    """RGB(A) -> a darker, fully opaque version -- for a mean line drawn in the same hue as
    its (opaque) histogram bars, so it reads as that series' own accent instead of blending
    into solid bars of the identical color."""
    r, g, b = color[:3]
    return (r * factor, g * factor, b * factor)


def _style_length_axis(ax, xlabel, ylabel, title, thousands_yaxis=True, log_x=True):
    """Shared chrome for every length-distribution plot in this file: log-x by default, a
    denser and more visible grid, labels/title/legend. thousands_yaxis: comma-format the
    y-axis ticks -- off for plots whose y-axis is a fraction, not a raw count."""
    if log_x:
        ax.set_xscale("log")
    if thousands_yaxis:
        ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="#dadada", linewidth=0.8)
    ax.grid(which="minor", color="#ececec", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def print_doc_length_stats(name, lengths):
    pct = np.percentile(lengths, PERCENTILES)
    print(f"### {_display_name(name)} ###")
    print(f"  documents:    {len(lengths):,}")
    print(f"  total tokens: {int(lengths.sum()):,}")
    print(f"  mean:         {lengths.mean():,.1f}")
    print(f"  std:          {lengths.std():,.1f}")
    print(f"  min / max:    {lengths.min():,} / {lengths.max():,}")
    print("  percentiles:  " + "  ".join(f"p{p}={int(v):,}" for p, v in zip(PERCENTILES, pct)))


def _plot_length_histogram(lengths, weights, mean, xlabel, ylabel, title, figsize):
    """Shared by plot_doc_length_histogram and plot_chunk_length_histogram -- both are a
    log-x histogram + mean line + grid, differing only in what one data point represents
    (a document vs. a chunk) and whether it's a flat array or a pre-aggregated histogram
    (weights=None vs. weights=hist)."""
    fig, ax = plt.subplots(figsize=figsize)
    _, _, patches = ax.hist(lengths, bins=_log_bins(lengths.max()), weights=weights)
    ax.axvline(mean, color=_darken(patches[0].get_facecolor()), linestyle="--", linewidth=1.5,
              label=f"mean = {mean:,.0f}")
    _style_length_axis(ax, xlabel, ylabel, title)
    plt.tight_layout()
    return fig


def plot_doc_length_histogram(name, lengths, figsize=(8, 4)):
    """Single dataset, log-x histogram of per-document token length."""
    return _plot_length_histogram(
        lengths, None, lengths.mean(),
        xlabel="document length (tokens)", ylabel="number of documents",
        title=f"{_display_name(name)}  (n={len(lengths):,})", figsize=figsize)


def plot_doc_length_overlay(datasets, models=None, figsize=(9, 5)):
    """Datasets overlaid on one log-x histogram, each with its own mean line. Title shows
    the pooled mean over every plotted document (dominated by whichever dataset has the
    most documents).

    models: list of dataset names to plot, in order (default: every dataset in `datasets`)."""
    plotted = select_models(datasets, models)
    global_max = max(lengths.max() for lengths in plotted.values())
    bins = _log_bins(global_max)

    fig, ax = plt.subplots(figsize=figsize)
    for name, lengths in plotted.items():
        _, _, patches = ax.hist(lengths, bins=bins, alpha=0.55,
                                label=f"{_display_name(name)} (n={len(lengths):,})")
        color = patches[0].get_facecolor()
        ax.axvline(lengths.mean(), color=color, linestyle="--", linewidth=1.5,
                  label=f"{_display_name(name)} mean = {lengths.mean():,.0f}")

    overall_mean = np.concatenate(list(plotted.values())).mean()
    _style_length_axis(ax, "document length (tokens)", "number of documents",
                       f"Document length distribution  (overall mean = {overall_mean:,.0f} tokens)")
    plt.tight_layout()
    return fig


def range_budget(lengths, lo=0, hi=None, verbose=True):
    """Tokens (and docs) in documents with length in [lo, hi). hi=None means no upper bound,
    i.e. the ">= lo" selection budget. This is the primitive for length-informed data
    selection: how many training tokens a given length range gives you."""
    mask = lengths >= lo
    if hi is not None:
        mask = mask & (lengths < hi)
    tokens = lengths[mask].astype(np.float64).sum()
    total = lengths.astype(np.float64).sum()
    docs = int(mask.sum())
    if verbose:
        hi_str = "inf" if hi is None else f"{hi:,}"
        print(f"[{lo:,}, {hi_str}): {docs:,} docs, {tokens/1e9:.2f}B tokens "
             f"({100*tokens/total:.1f}% of {total/1e9:.1f}B)")
    return {"docs": docs, "tokens": float(tokens), "pct": 100 * tokens / total}


def plot_token_budget(name, lengths, max_tok=40000, bin_size=1000, figsize=(10, 4.5)):
    """Token budget by document-length range: how many *tokens* (not documents) sit in each
    length bin. Bars show tokens per bin, capped at `max_tok` with one overflow bar pooling
    everything longer. The line is the reverse-cumulative "tokens in docs >= x" curve
    (includes the overflow, so it starts at the total token count and drops as the length
    threshold rises) -- read exact budgets off it, since on the log y-axis bar heights are
    not visually proportional."""
    max_k = max_tok // 1000
    edges = np.arange(0, max_tok + bin_size, bin_size)
    left_k = edges[:-1] / 1000

    weights = lengths.astype(np.float64)
    total = weights.sum()
    tokens_per_bin = np.histogram(lengths, bins=edges, weights=weights)[0]
    overflow = weights[lengths >= max_tok].sum()
    reverse_cumulative = (total - np.concatenate([[0], np.cumsum(tokens_per_bin)])) / 1e9

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    ax.grid(color="#dadada", linewidth=0.8)
    ax.grid(which="minor", color="#ececec", linewidth=0.6)
    ax.bar(left_k, tokens_per_bin / 1e9, width=0.9, align="edge", color="#4292C6",
          label=f"tokens per {bin_size//1000}k bin")
    if overflow > 0:
        ax.bar(max_tok / 1000, overflow / 1e9, width=0.9, align="edge",
              color="#c6dbef", hatch="//", label=f">={max_k}k (overflow)")
    ax.plot(edges / 1000, reverse_cumulative, color="#333333", linewidth=1,
           label="budget for docs >= x (cumulative)")
    ax.set_xlabel("document length (k tokens)")
    ax.set_ylabel("tokens (billions)")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.1)  # floor below the smallest capped bin so every bar is visible

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.tick_params(axis="y", which="both", right=True, labelright=True)
    ax.tick_params(axis="y", which="minor", labelsize=5)
    ax.tick_params(axis="y", which="major", pad=9)

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_minor_formatter(
        FuncFormatter(lambda v, _: f"{int(round(v)) % 10}" if 0 <= v <= max_k else ""))
    ax.tick_params(axis="x", which="minor", labelsize=5)

    ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"{_display_name(name)}: token budget by length  ({total/1e9:,.1f}B tokens total)")
    plt.tight_layout()
    return fig


def chunk_hist_stats(hist):
    """Summary stats over a packed chunk-length histogram (hist[i] = number of chunks with
    length i+1)."""
    lengths = np.arange(1, len(hist) + 1)
    total = int(hist.sum())
    mean = float((lengths * hist).sum() / total)
    std = float(np.sqrt((((lengths - mean) ** 2) * hist).sum() / total))
    cumulative = np.cumsum(hist)
    nonzero = np.flatnonzero(hist)
    percentiles = {p: int(lengths[np.searchsorted(cumulative, p / 100 * total)]) for p in PERCENTILES}
    return {"chunks": total, "mean": mean, "std": std,
           "min": int(lengths[nonzero[0]]), "max": int(lengths[nonzero[-1]]), "percentiles": percentiles}


def print_chunk_stats(name, stats, job_id=None):
    print(f"### {_display_name(name)} ###" + (f"  (job {job_id})" if job_id else ""))
    print(f"  chunks:       {stats['chunks']:,}")
    print(f"  mean:         {stats['mean']:,.1f}")
    print(f"  std:          {stats['std']:,.1f}")
    print(f"  min / max:    {stats['min']:,} / {stats['max']:,}")
    print("  percentiles:  " + "  ".join(f"p{p}={v:,}" for p, v in stats["percentiles"].items()))


def plot_chunk_length_histogram(name, hist, figsize=(8, 4)):
    """Single dataset, log-x histogram of packed chunk length (one full training epoch)."""
    seq_len = len(hist)
    stats = chunk_hist_stats(hist)
    return _plot_length_histogram(
        np.arange(1, seq_len + 1), hist, stats["mean"],
        xlabel="chunk length (tokens)", ylabel="number of chunks",
        title=f"{_display_name(name)}: packed chunk length  (n={stats['chunks']:,}, seq_len={seq_len:,})",
        figsize=figsize)


def plot_chunk_length_overlay(chunk_hists, models=None, colors=None, seq_len=None,
                              log_x=True, log_y=False, figsize=(9, 5)):
    """Packed chunk-length datasets overlaid on one histogram -- the packed-chunk counterpart
    of plot_doc_length_overlay. Weighted by *tokens* (chunks x length), not raw chunk count:
    otherwise a dataset packed into shorter chunks would dominate the chart just by having
    more of them for the same token budget.

    chunk_hists: {dataset_name: hist}, e.g. from load_packed_chunk_length_hist() (unwrap the
    (hist, job_id) tuples first). seq_len: optional reference line, matches plot_doc_vs_chunk_length.
    log_x: bins log-spaced when True (default), else linear.
    log_y: off by default -- a narrow-chunk dataset can concentrate into one bin and dwarf a
    smoother one on a linear axis; turn on to compare datasets of very different shape."""
    plotted = select_models(chunk_hists, models)
    palette = colors or {}
    max_len = max(len(hist) for hist in plotted.values())
    bins = _log_bins(max_len) if log_x else np.linspace(1, max_len, 80)

    fig, ax = plt.subplots(figsize=figsize)
    for name, hist in plotted.items():
        lengths = np.arange(1, len(hist) + 1)
        tokens = hist * lengths
        stats = chunk_hist_stats(hist)
        _, _, patches = ax.hist(lengths, bins=bins, weights=tokens, alpha=0.55,
                                color=palette.get(name),
                                label=f"{_display_name(name)} (n={int(tokens.sum()):,} tokens)")
        color = patches[0].get_facecolor()
        ax.axvline(stats["mean"], color=color, linestyle="--", linewidth=1.5,
                  label=f"{_display_name(name)} mean = {stats['mean']:,.0f}")

    if seq_len is not None:
        ax.axvline(seq_len, color="black", linestyle=":", linewidth=1.2, label=f"seq_len = {seq_len:,}")
    if log_y:
        ax.set_yscale("log")

    _style_length_axis(ax, "chunk length (tokens)", "number of tokens",
                       "Packed chunk length distribution", log_x=log_x)
    plt.tight_layout()
    return fig


def plot_doc_vs_chunk_length(name, doc_lengths, chunk_hist, figsize=(9, 4.5)):
    """Overlay raw document lengths against packed chunk lengths for the same dataset: docs
    longer than seq_len get cut into several seq_len-sized chunks (mass appears at/near
    seq_len that wasn't there before); a doc straddling a sample boundary splits into two
    shorter chunks. Most short docs are unaffected (whole doc = one chunk).

    Fraction of total per bin, not density=True: with log-spaced bins, density=True divides
    by the (linear) bin width, and the narrow first bin near x=1 would blow a negligible
    sliver of short chunks into the tallest bar. Weighting by 1/N avoids that."""
    seq_len = len(chunk_hist)
    chunk_lengths = np.arange(1, seq_len + 1)
    max_len = max(doc_lengths.max(), seq_len)
    bins = _log_bins(max_len)

    doc_weights = np.full(len(doc_lengths), 1 / len(doc_lengths))
    chunk_weights = chunk_hist / chunk_hist.sum()
    doc_mean = doc_lengths.mean()
    chunk_mean = chunk_hist_stats(chunk_hist)["mean"]

    fig, ax = plt.subplots(figsize=figsize)
    _, _, doc_patches = ax.hist(doc_lengths, bins=bins, weights=doc_weights, alpha=0.55,
                                label=f"raw docs (n={len(doc_lengths):,}, mean={doc_mean:,.0f})")
    _, _, chunk_patches = ax.hist(chunk_lengths, bins=bins, weights=chunk_weights, alpha=0.55,
                                  label=f"packed chunks (n={int(chunk_hist.sum()):,}, mean={chunk_mean:,.0f})")
    # Same-color-as-its-own-bars mean line as the overlay chart, not darkened -- these bars
    # are already semi-transparent, so the exact color still stands out fine against them.
    ax.axvline(doc_mean, color=doc_patches[0].get_facecolor(), linestyle="--", linewidth=1.5)
    ax.axvline(chunk_mean, color=chunk_patches[0].get_facecolor(), linestyle="--", linewidth=1.5)
    ax.axvline(seq_len, color="black", linestyle=":", linewidth=1.2, label=f"seq_len = {seq_len:,}")
    _style_length_axis(ax, "length (tokens)", "fraction of total",
                       f"{_display_name(name)}: doc length vs packed chunk length",
                       thousands_yaxis=False)
    plt.tight_layout()
    return fig
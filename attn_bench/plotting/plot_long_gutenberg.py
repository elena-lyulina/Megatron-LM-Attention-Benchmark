"""Plots for the long-Gutenberg position-wise loss (and later, state norms).

load_nll reads one rep_{R}.npz (mean + std across samples, per position). plot_loss_grid
draws one cell per repetition bucket: a tall loss panel (all models, one colour each), with
an optional short panel underneath (show_count=True) showing how many sequences still reach
each position, so a noisy tail can be checked against a thinning sample count.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from attn_bench.plotting.long_inference_util import SAMPLE_LEN, SEQ_LEN, VOCAB_SIZE
from attn_bench.plotting.long_inference_util import denser_grid as _denser_grid
from attn_bench.plotting.long_inference_util import smooth as _smooth

COVERAGE_HUE = "#4292C6"  # single sequential hue for the overall coverage curve

# Fixed categorical order (tab10); one colour per model, assigned in the order given.
_COLORS = plt.get_cmap("tab10").colors

# Shared style skin. Applied via plt.rc_context so it scopes to these figures only and does
# not leak into the notebook's global rcParams. Helvetica falls back to Arial then the
# matplotlib default, so it still renders where Helvetica isn't installed (e.g. the cluster).
_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "regular",
    "axes.labelsize": 12,
    "axes.labelcolor": "#333333",
    "axes.edgecolor": "#bbbbbb",
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,       # grid sits behind the data lines
    "axes.grid": True,
    "axes.grid.axis": "both",     # horizontal + vertical major gridlines
    "grid.color": "#dadada",
    "grid.linewidth": 0.9,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
}


def load_nll(path: Path) -> dict:
    d = np.load(path)
    cnt = d["count"]
    mean = d["nll_sum"] / cnt
    var = np.maximum(d["nll_sqsum"] / cnt - mean ** 2, 0.0)  # clamp fp noise below 0
    return {
        "position": d["position"],
        "mean": mean,
        "std": np.sqrt(var),
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def load_state_norm(path: Path) -> dict:
    # A state file holds per-layer, per-boundary, per-head Frobenius norms summed over the
    # bucket's sequences (norm_sum / norm_sqsum), plus count[b] = sequences reaching boundary b.
    # For the overview we collapse both the layer and head axes: mean over the 16*8 per-(layer,
    # head) sequence-means at each boundary, with std taken across those same values so the band
    # shows how much the single line hides across layers/heads (not across sequences).
    d = np.load(path)
    cnt = d["count"]                                   # [num_boundaries]
    per_lh = d["norm_sum"] / cnt[None, :, None]        # [layer, boundary, head] mean over seqs
    flat = per_lh.transpose(1, 0, 2).reshape(per_lh.shape[1], -1)  # [boundary, layer*head]
    return {
        "position": d["boundary"],
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def load_state_norm_by_layer(path: Path) -> dict:
    # Same file as load_state_norm, but collapse only the head axis: mean over heads of the
    # per-sequence-mean norm, kept per layer. Returns mean[layer, boundary] so each layer is
    # one line. Layers are ordered as stored (ascending layer number in `layer`).
    d = np.load(path)
    cnt = d["count"]                                    # [num_boundaries]
    num_heads = d["norm_sum"].shape[-1]
    mean = d["norm_sum"].sum(axis=-1) / (num_heads * cnt[None, :])  # [layer, boundary]
    return {
        "position": d["boundary"],
        "layer": d["layer"],
        "mean": mean,
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def _rep_path(config_dir: Path, rep: int) -> Path:
    return Path(config_dir) / f"rep_{rep}.npz"


def _state_path(config_dir: Path, rep: int) -> Path:
    return Path(config_dir) / f"rep_{rep}_state.npz"


def plot_loss_grid(results_by_label, reps, ncols=3, sample_end=SAMPLE_LEN,
                   show_std=False, smooth=0, xmax=None, ymax=None, sharey=True,
                   linestyles=None, suptitle=None, colors=None,
                   path_fn=_rep_path, bucket_title_fn=lambda b: f"repetition {b}",
                   show_count=False, metric="nll", show_random_baseline=False, log_y=False):
    """One cell per bucket; every model overlaid as a coloured mean line.

    results_by_label: {label: entry} -- entry is passed to path_fn(entry, bucket) to resolve
    each npz. Default path_fn=_rep_path expects entry to be a config_dir holding rep_{R}.npz
    (Gutenberg's layout: one dir per model, bucket = repetition number). Pass a custom path_fn
    when the bucket changes more than the filename -- e.g. fineweb's seen/unseen partitions
    each live in their own directory, so entry would be {bucket: dir} and
    path_fn=lambda dirs, b: dirs[b] / "some_fixed_name.npz".
    reps: buckets to draw (a cell is made only for buckets that have at least one file).
    show_std: shade +/- one std across samples around each mean.
    smooth: rolling-mean window over position to tame per-token spikes (0 = raw).
    xmax / ymax: optional axis caps (loss blows up in the suffix, so a cap can help).
    linestyles: optional {label: linestyle} to give a line a secondary encoding on top of
    its colour. Useful when two lines nearly coincide (e.g. tp=1 vs tp=4): draw one dashed
    so both stay readable where they overlap instead of one just hiding the other.
    suptitle: optional figure title, placed in the top strip above the legend (kept here so
    the title/legend/plot spacing stays consistent instead of floating on a fixed fraction).
    colors: optional {label: colour} override; labels not listed fall back to the tab10 cycle.
    Lets related models share a hue family (e.g. the GDN variants as shades of blue).
    bucket_title_fn: how to render each cell's title from its bucket value (default
    "repetition {b}"; e.g. fineweb passes `str.capitalize` for "Seen"/"Unseen").
    show_count: add a short panel under each loss cell (sharing its x-axis) plotting each
    label's raw sequence count per position -- the same field every model already writes,
    so a wobbly/noisy tail can be checked against a thinning sample count for both Gutenberg
    and FineWeb without any new data. Off by default so existing calls render unchanged.
    metric: "nll" (default) plots the stored per-position mean NLL as-is. "ppl" plots
    exp(mean NLL) instead -- the standard perplexity definition, applied to the already
    -aggregated cross-entropy (not an average of per-sample exp(NLL)). show_std bands are
    exponentiated at their mean-std/mean+std bounds, not symmetrically around the mean,
    since perplexity is log-normal-shaped, not symmetric.
    show_random_baseline: draw a horizontal line at the NLL/perplexity of a uniform random
    guess over the tokenizer vocab (ln(VOCAB_SIZE) / VOCAB_SIZE) -- lines above it are
    doing worse than guessing, which does happen in the extreme repetition bumps here.
    log_y: log-scale the y-axis. Independent of metric -- on a log axis, "ppl" is just an
    affine rescaling of "nll" (same curve shape, different labels), so leave this off to
    actually see perplexity's magnitude (e.g. how enormous the repetition-bump spikes are
    in raw terms). Consider it for "nll" with a large xmax/many reps instead, where NLL
    itself spans a wide range and a linear axis compresses the low end.
    """
    labels = list(results_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, _COLORS[i % len(_COLORS)]) for i, lab in enumerate(labels)}
    linestyles = linestyles or {}

    # Keep only buckets that actually have data on disk.
    buckets = [r for r in reps if any(path_fn(results_by_label[l], r).exists() for l in labels)]
    if not buckets:
        dirs = "\n  ".join(str(d) for d in results_by_label.values())
        raise ValueError(f"No files found for reps={reps} under:\n  {dirs}")
    nb = len(buckets)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    # constrained_layout handles the title/legend/cell spacing without hand-tuned fractions
    # (which used to make the suptitle collide with the cell titles). A thin top grid row
    # holds one shared legend above the plots; the suptitle sits above that. Per-cell size is
    # kept small on purpose so the notebook renders the cells large instead of scaling a huge
    # figure down.
    with plt.rc_context(_STYLE):
        # Each plot row gets its own thin legend strip directly above it (repeated identically),
        # so on a tall grid the legend is always next to the row you are reading. constrained
        # layout keeps the strips, cells and suptitle from colliding without hand-tuned fractions.
        # leg_ratio sized for 2 legend rows (models can add a boundary + random-baseline entry,
        # which easily overflows one row and made constrained_layout shrink the plots to fit).
        cell_w, cell_h, leg_ratio, count_ratio = 8.0, 4.3, 0.24, 0.3
        rows_per_group = 3 if show_count else 2
        extra_h = count_ratio if show_count else 0.0
        fig_h = nrows * cell_h * (1 + leg_ratio + extra_h) + (0.25 if suptitle else 0.0)
        fig = plt.figure(figsize=(cell_w * ncols, fig_h), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.0, hspace=0.04)
        # Interleaved rows: [legend, plot, (count,) legend, plot, (count,) ...].
        # A narrow empty column on the right gives a bit of far-right margin without widening
        # the gap between the plot columns (which is what shrinking the whole layout would do).
        right_margin = 0.08   # width of the spacer relative to a plot column
        row_ratios = [leg_ratio, 1.0, count_ratio] if show_count else [leg_ratio, 1.0]
        gs = fig.add_gridspec(rows_per_group * nrows, ncols + 1, height_ratios=row_ratios * nrows,
                              width_ratios=[1.0] * ncols + [right_margin])

        handles = [plt.Line2D([], [], color=colors[l], linewidth=1.8, linestyle=linestyles.get(l, "-"))
                   for l in labels]
        # The dashed vertical boundary (drawn per cell below) also gets one legend entry. Reuse a
        # single style dict for the line and its swatch so they can never drift apart.
        boundary_style = dict(color="#999999", linestyle=(0, (2, 2)), linewidth=1.0)
        leg_handles = handles + [plt.Line2D([], [], **boundary_style)]
        leg_labels = labels + ["seq length (8192)"]

        # Random-guess baseline: ln(V) in NLL space, V itself in perplexity space.
        baseline_style = dict(color="#cc3333", linestyle=(0, (1, 1)), linewidth=1.2)
        baseline_value = np.log(VOCAB_SIZE) if metric == "nll" else VOCAB_SIZE
        if show_random_baseline:
            leg_handles = leg_handles + [plt.Line2D([], [], **baseline_style)]
            leg_labels = leg_labels + [f"random guess (V={VOCAB_SIZE:,})"]

        loss_axes = []
        for i, rep in enumerate(buckets):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[rows_per_group * r + 1, c],
                                 sharey=(loss_axes[0] if sharey and loss_axes else None))
            loss_axes.append(ax)
            if log_y:
                ax.set_yscale("log")
            count_ax = None
            if show_count:
                count_ax = fig.add_subplot(gs[rows_per_group * r + 2, c], sharex=ax)
            for lab in labels:
                p = path_fn(results_by_label[lab], rep)
                if not p.exists():
                    continue
                data = load_nll(p)
                pos = data["position"]
                mean = _smooth(data["mean"], smooth)
                if show_std:
                    std = _smooth(data["std"], smooth)
                    lo, hi = mean - std, mean + std
                if metric == "ppl":
                    mean = np.exp(mean)
                    if show_std:
                        lo, hi = np.exp(lo), np.exp(hi)
                ax.plot(pos, mean, color=colors[lab], linewidth=1.0, label=lab,
                        linestyle=linestyles.get(lab, "-"))
                if show_std:
                    ax.fill_between(pos, lo, hi, color=colors[lab], alpha=0.15, linewidth=0)
                if count_ax is not None:
                    count_ax.plot(pos, data["count"], color=colors[lab], linewidth=0.9,
                                  linestyle=linestyles.get(lab, "-"))

            _denser_grid(ax)

            # Sample-end / extrapolation boundary.
            ax.axvline(sample_end, **boundary_style)
            if show_random_baseline:
                ax.axhline(baseline_value, **baseline_style)
            ax.set_title(bucket_title_fn(rep))
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("Perplexity" if metric == "ppl" else "NLL")
            if xmax is not None:
                ax.set_xlim(0, xmax)

            if count_ax is not None:
                # Shared x-axis with the loss panel above: the loss panel's own tick labels
                # would just repeat, so only the count panel (bottom of the cell) carries them.
                ax.tick_params(labelbottom=False)
                count_ax.axvline(sample_end, **boundary_style)
                count_ax.set_ylim(bottom=0)
                # More major ticks than the default sparse auto-locator, plus the same denser
                # minor grid as the loss panel, so the count is easy to read off precisely.
                count_ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
                _denser_grid(count_ax)
                # n samples specifically: 2 evenly-spaced minor ticks between majors (3 parts),
                # denser than the auto-picked spacing everywhere else.
                count_ax.yaxis.set_minor_locator(AutoMinorLocator(3))
                count_ax.spines[["top", "right"]].set_visible(False)
                count_ax.set_xlabel("position (from sample start)")
                if c == 0:
                    count_ax.set_ylabel("n samples")
            else:
                ax.set_xlabel("position (from sample start)")
        if ymax is not None:
            loss_axes[0].set_ylim(top=ymax)

        # One legend strip per plot row (same legend), spanning the columns above that row.
        for r in range(nrows):
            if not any(r * ncols + c < nb for c in range(ncols)):
                continue
            legax = fig.add_subplot(gs[rows_per_group * r, 0:ncols])   # span plot columns, not the spacer
            legax.axis("off")
            # Wrap into 2 rows instead of one -- one row can overflow the figure width once the
            # boundary + random-baseline entries are added, which made constrained_layout shrink
            # the plots to make room.
            legax.legend(leg_handles, leg_labels, loc="center", ncol=int(np.ceil(len(leg_labels) / 2)),
                         frameon=False, fontsize=11, handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
        if suptitle:
            fig.suptitle(suptitle, fontweight="semibold", fontsize=16)
    return fig


def plot_state_norm_grid(results_by_label, reps, ncols=3, seq_len=SEQ_LEN,
                         show_std=False, smooth=0, xmax=SEQ_LEN, ymax=None, sharey=True,
                         linestyles=None, suptitle=None, colors=None):
    """One cell per repetition bucket; each GDN model's mean state norm overlaid as a line.

    Same scaffolding as plot_loss_grid, but reads rep_{R}_state.npz and plots the state
    norm averaged over all layers and heads against the boundary token position.

    results_by_label: {label: config_dir} -- config_dir holds rep_{R}_state.npz.
    reps: buckets to draw (a cell is made only for buckets that have a state file).
    show_std: shade +/- one std of the per-(layer, head) means around each line, i.e. the
    spread the layer/head average hides. Off by default: that spread is large enough to swamp
    the differences between models, so it belongs in the per-layer breakdown, not this overview.
    smooth: rolling-mean window over the boundary axis (0 = raw).
    xmax: cap the boundary axis (defaults to seq_len; pass None to show the full tail, which
    runs far past training length but thins out as sequences drop off).
    linestyles / suptitle / colors: as in plot_loss_grid.
    """
    labels = list(results_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, _COLORS[i % len(_COLORS)]) for i, lab in enumerate(labels)}
    linestyles = linestyles or {}

    buckets = [r for r in reps if any(_state_path(results_by_label[l], r).exists() for l in labels)]
    if not buckets:
        dirs = "\n  ".join(str(d) for d in results_by_label.values())
        raise ValueError(f"No rep_{{R}}_state.npz found for reps={reps} under:\n  {dirs}")
    nb = len(buckets)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    with plt.rc_context(_STYLE):
        cell_w, cell_h, leg_ratio = 8.0, 4.3, 0.14
        fig_h = nrows * cell_h * (1 + leg_ratio) + (0.25 if suptitle else 0.0)
        fig = plt.figure(figsize=(cell_w * ncols, fig_h), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.02, wspace=0.0, hspace=0.04)
        right_margin = 0.08
        gs = fig.add_gridspec(2 * nrows, ncols + 1, height_ratios=[leg_ratio, 1.0] * nrows,
                              width_ratios=[1.0] * ncols + [right_margin])

        handles = [plt.Line2D([], [], color=colors[l], linewidth=1.8, linestyle=linestyles.get(l, "-"))
                   for l in labels]
        boundary_style = dict(color="#999999", linestyle=(0, (2, 2)), linewidth=1.0)
        leg_handles = handles + [plt.Line2D([], [], **boundary_style)]
        leg_labels = labels + [f"seq length ({seq_len})"]

        norm_axes = []
        for i, rep in enumerate(buckets):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[2 * r + 1, c], sharey=(norm_axes[0] if sharey and norm_axes else None))
            norm_axes.append(ax)
            for lab in labels:
                p = _state_path(results_by_label[lab], rep)
                if not p.exists():
                    continue
                data = load_state_norm(p)
                pos = data["position"]
                mean = _smooth(data["mean"], smooth)
                ax.plot(pos, mean, color=colors[lab], linewidth=1.0, label=lab,
                        linestyle=linestyles.get(lab, "-"))
                if show_std:
                    std = _smooth(data["std"], smooth)
                    ax.fill_between(pos, mean - std, mean + std,
                                    color=colors[lab], alpha=0.15, linewidth=0)

            _denser_grid(ax)

            # Training sequence length -- everything to the right is extrapolation past it.
            ax.axvline(seq_len, **boundary_style)
            ax.set_title(f"repetition {rep}")
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("state norm (Frobenius)")
            ax.set_xlabel("position (token)")
            if xmax is not None:
                ax.set_xlim(0, xmax)
        if ymax is not None:
            norm_axes[0].set_ylim(top=ymax)

        for r in range(nrows):
            if not any(r * ncols + c < nb for c in range(ncols)):
                continue
            legax = fig.add_subplot(gs[2 * r, 0:ncols])
            legax.axis("off")
            legax.legend(leg_handles, leg_labels, loc="center", ncol=len(leg_labels), frameon=False,
                         fontsize=11, handlelength=1.4, columnspacing=1.2, handletextpad=0.5)
        if suptitle:
            fig.suptitle(suptitle, fontweight="semibold", fontsize=16)
    return fig


def plot_state_norm_by_layer(results_by_label, rep, ncols=2, seq_len=SEQ_LEN,
                             smooth=0, xmax=SEQ_LEN, ymax=None, sharey=True,
                             cmap="viridis", suptitle=None):
    """One subplot per model for a single rep bucket; 16 head-averaged layer lines each.

    Breaks down the cross-layer spread that plot_state_norm_grid collapses into one line.
    Colour runs along a sequential gradient from the first layer (dark) to the last (bright),
    with a shared colourbar instead of a 16-entry legend.

    results_by_label: {label: config_dir} holding rep_{rep}_state.npz.
    rep: the single repetition bucket to draw (required -- one figure is one rep).
    smooth: rolling-mean window over the boundary axis (0 = raw).
    xmax: cap the boundary axis (defaults to seq_len; pass None for the full tail).
    """
    labels = [l for l in results_by_label if _state_path(results_by_label[l], rep).exists()]
    if not labels:
        dirs = "\n  ".join(str(d) for d in results_by_label.values())
        raise ValueError(f"No rep_{rep}_state.npz found under:\n  {dirs}")
    nb = len(labels)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    cmap = plt.get_cmap(cmap)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.3 * nrows),
                                 sharey=sharey, squeeze=False, layout="constrained")
        norm = layers = None
        for i, lab in enumerate(labels):
            ax = axes[i // ncols][i % ncols]
            data = load_state_norm_by_layer(_state_path(results_by_label[lab], rep))
            pos, layers, mean = data["position"], data["layer"], data["mean"]
            norm = plt.Normalize(vmin=layers.min(), vmax=layers.max())
            for li, lid in enumerate(layers):
                ax.plot(pos, _smooth(mean[li], smooth), color=cmap(norm(lid)), linewidth=1.0)

            _denser_grid(ax)

            # Training sequence length -- everything to the right is extrapolation past it.
            ax.axvline(seq_len, color="#999999", linestyle=(0, (2, 2)), linewidth=1.0)
            ax.set_title(lab)
            ax.spines[["top", "right"]].set_visible(False)
            if i % ncols == 0:
                ax.set_ylabel("state norm (Frobenius)")
            if i // ncols == nrows - 1:
                ax.set_xlabel("position (token)")
            if xmax is not None:
                # Give a little headroom past the seq_len marker so it does not hide under the
                # right spine when xmax == seq_len (the default). Data runs past it either way.
                ax.set_xlim(0, max(xmax, seq_len * 1.02))
        if ymax is not None:
            axes[0][0].set_ylim(top=ymax)

        # Hide any unused cells (e.g. 3 models in a 2x2 grid).
        for j in range(nb, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), label="layer", pad=0.02, fraction=0.05)
        cbar.set_ticks(layers)   # integer layer numbers, not the default continuous ticks
        if suptitle:
            fig.suptitle(suptitle, fontweight="semibold", fontsize=16)
    return fig


def plot_coverage(df, buckets, seq_len=SEQ_LEN, sample_len=SAMPLE_LEN, hue=COVERAGE_HUE):
    """Fraction of sequences that still have a real token at each position.

    df: one row per book per bucket, with extra_prefix_len / extra_suffix_len (from
    lengths.jsonl). The sample band [0, sample_len) is always full; the prefix extends left
    (negative positions), the suffix right. Grey = per bucket, coloured = overall.
    """
    xs_pre = np.arange(int(-1.5 * seq_len), 0)   # 1.5L of warmup before the sample start
    xs_suf = np.arange(0, int(4.5 * seq_len))    # 3.5L past the sample start

    def coverage_frac(sub):
        n = len(sub)
        p = np.sort(sub["extra_prefix_len"].to_numpy())
        s = np.sort(sub["extra_suffix_len"].to_numpy())
        pre = (n - np.searchsorted(p, -xs_pre, "left")) / n
        d_suf = np.maximum(xs_suf - sample_len, 0)         # tokens needed past the sample
        suf = (n - np.searchsorted(s, d_suf, "left")) / n
        suf[d_suf == 0] = 1.0                              # inside the sample: always present
        return np.concatenate([xs_pre, xs_suf]), np.concatenate([pre, suf])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for rep in buckets:
            x, y = coverage_frac(df[df["bucket_rep"] == rep])
            ax.plot(x, y, color="0.85", linewidth=0.8)         # per-bucket, recessive
        x, y = coverage_frac(df)
        ax.plot(x, y, color=hue, linewidth=2, label="overall")
        ax.axvspan(0, sample_len, color="0.92", label="sample (always present)")
        for pos in (-seq_len, seq_len, 2 * seq_len, 3 * seq_len, 4 * seq_len):
            ax.axvline(pos, color="#999999", linestyle=(0, (2, 2)), linewidth=0.8)
        ax.set_xlim(xs_pre[0], xs_suf[-1])
        ax.set_xlabel("token position relative to sample start")
        ax.set_ylabel("fraction of sequences present")
        ax.set_title("Coverage: context reach (grey = per bucket, blue = overall)")
        ax.legend(loc="upper right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
    return fig

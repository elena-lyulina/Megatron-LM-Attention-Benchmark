"""Behavioral: cross-model position-wise loss under length extrapolation, on long-context
Gutenberg (repetition buckets) and FineWeb-Edu (seen/unseen partitions)."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from attn_bench.plotting.utils import (LONG_INFERENCE_STYLE, SAMPLE_LEN,
                                       SEQ_LEN, TAB10_COLORS, VOCAB_SIZE,
                                       denser_grid, smooth)

COVERAGE_HUE = "#4292C6"  # single sequential hue for the overall coverage curve


def compare_nll(nll_a, nll_b, label_a="a", label_b="b", rel_tol=0.01):
    """Per-rep NLL gap between two already-loaded {rep: load_long_inference_nll(...)} dicts,
    over the positions present in both.

    Prints a table of max/mean absolute difference (and as a % of the region's average NLL,
    so the size is readable -- 0.05 nats means nothing without the loss scale). Returns the
    worst mean-relative-diff across reps, for a quick pass/fail check.
    """
    print(f"{label_a} vs {label_b}")
    print(f"  {'rep':>4}  {'shared_pos':>10}  {'max_abs':>8}  {'max_%':>7}  {'mean_abs':>9}  {'mean_%':>7}")
    worst_rel = 0.0
    for rep in sorted(set(nll_a) & set(nll_b)):
        a, b = nll_a[rep], nll_b[rep]
        # intersect on shared positions; both position arrays are sorted ascending
        common, ia, ib = np.intersect1d(a["position"], b["position"], return_indices=True)
        ma, mb = a["mean"][ia], b["mean"][ib]
        diff = np.abs(ma - mb)
        ref = 0.5 * (ma + mb).mean()   # region-average NLL: the scale to judge the diff against
        max_rel, mean_rel = diff.max() / ref, diff.mean() / ref
        worst_rel = max(worst_rel, mean_rel)
        print(f"  {rep:>4}  {len(common):>10}  {diff.max():>8.2e}  {100*max_rel:>6.2f}%  {diff.mean():>9.2e}  {100*mean_rel:>6.2f}%")
    flag = "OK" if worst_rel < rel_tol else f"CHECK (> {100*rel_tol:g}%)"
    print(f"  worst mean diff over all reps: {100*worst_rel:.2f}% of typical loss  [{flag}]\n")
    return worst_rel


def plot_loss_panel(nll_by_label, ncols=3,
                    show_std=False, smooth_window=0, xmin=0, xmax=None, ymax=None, sharey=True,
                    linestyles=None, suptitle=None, colors=None,
                    bucket_title_fn=lambda b: f"repetition {b}", bucket_order=None,
                    show_count=False, metric="nll", show_random_baseline=False, log_y=False,
                    vlines=None, vline_colors=None, cell_w=8.0, cell_h=4.3):
    """One cell per bucket; every model overlaid as a coloured mean line.

    nll_by_label: {label: {bucket: load_long_inference_nll(...)}}, from
    load_long_inference_nll_grid (bucket=repetition) or load_long_fineweb_inference_nll_grid.
    cell_w / cell_h: widen cell_w for few panels with many legend entries (legend width
    scales with the figure, not panel count).
    metric: "ppl" plots exp(mean NLL) of the aggregated cross-entropy, not per-sample exp(NLL).
    log_y: independent of metric -- "ppl" on a log axis is just a rescaling of "nll".
    show_count: adds a count-per-position panel under each loss cell.
    vlines / vline_colors: {name: position} reference lines, drawn on both panels.
    """
    labels = list(nll_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, TAB10_COLORS[i % len(TAB10_COLORS)]) for i, lab in enumerate(labels)}
    linestyles = linestyles or {}

    buckets = bucket_order if bucket_order is not None else sorted(
        {b for per_bucket in nll_by_label.values() for b in per_bucket})
    if not buckets:
        raise ValueError("nll_by_label has no buckets loaded")
    nb = len(buckets)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    # constrained_layout handles the title/legend/cell spacing without hand-tuned fractions
    # (which used to make the suptitle collide with the cell titles). A thin top grid row
    # holds one shared legend above the plots; the suptitle sits above that. Per-cell size is
    # kept small on purpose so the notebook renders the cells large instead of scaling a huge
    # figure down.
    with plt.rc_context(LONG_INFERENCE_STYLE):
        # Each plot row gets its own thin legend strip directly above it (repeated identically),
        # so on a tall grid the legend is always next to the row you are reading. constrained
        # layout keeps the strips, cells and suptitle from colliding without hand-tuned fractions.
        # leg_ratio sized for 2 legend rows (models can add a boundary + random-baseline entry,
        # which easily overflows one row and made constrained_layout shrink the plots to fit).
        leg_ratio, count_ratio = 0.24, 0.3
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
        leg_handles = list(handles)
        leg_labels = list(labels)

        # Random-guess baseline: ln(V) in NLL space, V itself in perplexity space.
        baseline_style = dict(color="#cc3333", linestyle=(0, (1, 1)), linewidth=1.2)
        baseline_value = np.log(VOCAB_SIZE) if metric == "nll" else VOCAB_SIZE
        if show_random_baseline:
            leg_handles = leg_handles + [plt.Line2D([], [], **baseline_style)]
            leg_labels = leg_labels + [f"random guess (V={VOCAB_SIZE:,})"]

        # Named vertical lines (e.g. the seq-length boundary, mean/median chunk lengths) --
        # one colour each, cycled if more are given than the palette has. Avoids teal/navy
        # (model line colours in the long-vs-split1024 charts) and red (random-guess
        # baseline) so these never blend into an existing line.
        vlines = vlines or {}
        vline_palette = vline_colors or {}
        _VLINE_COLORS = ["#7570b3", "#e6ab02", "#e7298a", "#66a61e", "#a6761d"]
        vline_styles = {name: dict(color=vline_palette.get(name, _VLINE_COLORS[i % len(_VLINE_COLORS)]),
                                   linestyle=(0, (4, 2)), linewidth=1.2)
                        for i, name in enumerate(vlines)}
        if vlines:
            leg_handles = leg_handles + [plt.Line2D([], [], **vline_styles[name]) for name in vlines]
            leg_labels = leg_labels + [f"{name} ({pos:g})" for name, pos in vlines.items()]

        loss_axes = []

        def _draw_bucket(bucket, r, c):
            ax = fig.add_subplot(gs[rows_per_group * r + 1, c],
                                 sharey=(loss_axes[0] if sharey and loss_axes else None))
            loss_axes.append(ax)
            if log_y:
                ax.set_yscale("log")
            count_ax = None
            if show_count:
                count_ax = fig.add_subplot(gs[rows_per_group * r + 2, c], sharex=ax)
            for lab in labels:
                data = nll_by_label[lab].get(bucket)
                if data is None:
                    continue
                pos = data["position"]
                mean = smooth(data["mean"], smooth_window)
                count = data["count"]
                std = smooth(data["std"], smooth_window) if show_std else None
                # Slice to the visible window *before* plotting -- ax.plot() autoscales the
                # y-axis to everything it's given, and set_xlim() below only restricts the
                # view afterward, so a spike past xmax (e.g. the noisy small-sample tail)
                # would otherwise stretch the axis even though it's scrolled off-screen.
                if xmax is not None:
                    mask = (pos >= xmin) & (pos <= xmax)
                    pos, mean, count = pos[mask], mean[mask], count[mask]
                    if std is not None:
                        std = std[mask]
                if show_std:
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
                    count_ax.plot(pos, count, color=colors[lab], linewidth=0.9,
                                  linestyle=linestyles.get(lab, "-"))

            denser_grid(ax)

            if show_random_baseline:
                ax.axhline(baseline_value, **baseline_style)
            for name, pos_v in vlines.items():
                ax.axvline(pos_v, **vline_styles[name])
            ax.set_title(bucket_title_fn(bucket))
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel("Perplexity" if metric == "ppl" else "NLL")
            if xmax is not None:
                ax.set_xlim(xmin, xmax)

            if count_ax is not None:
                # Shared x-axis with the loss panel above: the loss panel's own tick labels
                # would just repeat, so only the count panel (bottom of the cell) carries them.
                ax.tick_params(labelbottom=False)
                for name, pos_v in vlines.items():
                    count_ax.axvline(pos_v, **vline_styles[name])
                count_ax.set_ylim(bottom=0)
                # More major ticks than the default sparse auto-locator, plus the same denser
                # minor grid as the loss panel, so the count is easy to read off precisely.
                count_ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
                denser_grid(count_ax)
                # n samples specifically: 2 evenly-spaced minor ticks between majors (3 parts),
                # denser than the auto-picked spacing everywhere else.
                count_ax.yaxis.set_minor_locator(AutoMinorLocator(3))
                count_ax.spines[["top", "right"]].set_visible(False)
                count_ax.set_xlabel("position (from sample start)")
                if c == 0:
                    count_ax.set_ylabel("n samples")
            else:
                ax.set_xlabel("position (from sample start)")

        for i, bucket in enumerate(buckets):
            _draw_bucket(bucket, *divmod(i, ncols))
        if ymax is not None:
            loss_axes[0].set_ylim(top=ymax)

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
            fig.suptitle(suptitle, fontweight="bold", fontsize=16)
    return fig


def plot_coverage(df, buckets, seq_len=SEQ_LEN, sample_len=SAMPLE_LEN, hue=COVERAGE_HUE):
    """Fraction of sequences that still have a real token at each position.

    df: one row per book per bucket, with extra_prefix_len / extra_suffix_len (from
    lengths.jsonl). The sample band [0, sample_len) is always full; the prefix extends left
    (negative positions), the suffix right. Grey = per bucket, coloured = overall.
    """
    xs_pre = np.arange(int(-1.5 * seq_len), 0)   # 1.5L of warmup before the sample start
    xs_suf = np.arange(0, int(4.5 * seq_len))    # 4.5L past the sample start

    def coverage_frac(sub):
        n = len(sub)
        p = np.sort(sub["extra_prefix_len"].to_numpy())
        s = np.sort(sub["extra_suffix_len"].to_numpy())
        pre = (n - np.searchsorted(p, -xs_pre, "left")) / n
        d_suf = np.maximum(xs_suf - sample_len, 0)         # tokens needed past the sample
        suf = (n - np.searchsorted(s, d_suf, "left")) / n
        suf[d_suf == 0] = 1.0                              # inside the sample: always present
        return np.concatenate([xs_pre, xs_suf]), np.concatenate([pre, suf])

    with plt.rc_context(LONG_INFERENCE_STYLE):
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
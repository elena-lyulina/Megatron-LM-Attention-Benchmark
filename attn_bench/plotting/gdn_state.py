"""GDN recurrent state norm plots (mechanistic -- does the state stay bounded past training
length, or does it keep growing?). Only the GDN variants carry a state."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from attn_bench.plotting.utils import (LONG_INFERENCE_STYLE, SEQ_LEN,
                                       TAB10_COLORS, denser_grid, smooth)


def plot_state_norm_panel(state_by_label, ncols=3, seq_len=SEQ_LEN,
                          show_std=False, smooth_window=0, xmax=SEQ_LEN, ymax=None, sharey=True,
                          linestyles=None, suptitle=None, colors=None):
    """One cell per repetition bucket; each GDN model's mean state norm overlaid as a line.

    state_by_label: {label: {rep: load_long_inference_state_norm(...)}}, already loaded.
    show_std: shade +/- one std of the per-(layer, head) spread the mean line hides. Off by
    default -- that spread is large enough to swamp the model differences this view is for.
    xmax: cap the boundary axis (default seq_len; None for the full tail, which thins out).
    """
    labels = list(state_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, TAB10_COLORS[i % len(TAB10_COLORS)]) for i, lab in enumerate(labels)}
    linestyles = linestyles or {}

    buckets = sorted({rep for per_rep in state_by_label.values() for rep in per_rep})
    if not buckets:
        raise ValueError("state_by_label has no reps loaded")
    nb = len(buckets)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    with plt.rc_context(LONG_INFERENCE_STYLE):
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
                data = state_by_label[lab].get(rep)
                if data is None:
                    continue
                pos = data["position"]
                mean = smooth(data["mean"], smooth_window)
                ax.plot(pos, mean, color=colors[lab], linewidth=1.0, label=lab,
                        linestyle=linestyles.get(lab, "-"))
                if show_std:
                    std = smooth(data["std"], smooth_window)
                    ax.fill_between(pos, mean - std, mean + std,
                                    color=colors[lab], alpha=0.15, linewidth=0)

            # More major x-ticks than the default auto-locator picks over this wide a range,
            # plus a denser and more visible grid (both major and minor) -- with many colored
            # lines already filling the axes, the default light gray blends in too easily.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=6))
            denser_grid(ax)
            ax.grid(which="major", color="#bbbbbb", linewidth=0.9)
            ax.grid(which="minor", color="#c8c8c8", linewidth=0.6)

            # Training sequence length -- past it, the model is running on lengths it never
            # trained on.
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
            fig.suptitle(suptitle, fontweight="bold", fontsize=16)
    return fig


def plot_state_norm_by_layer(state_by_label, ncols=2, seq_len=SEQ_LEN,
                             smooth_window=0, xmax=SEQ_LEN, ymax=None, sharey=True,
                             cmap="viridis", suptitle=None):
    """One subplot per model, for a single repetition bucket; 16 head-averaged layer lines
    each, breaking down the cross-layer spread plot_state_norm_panel collapses into one line.
    Colour runs first-layer (dark) to last (bright), with a shared colourbar instead of a
    16-entry legend.

    state_by_label: {label: load_long_inference_state_norm_by_layer(...)} for one chosen rep.
    xmax: cap the boundary axis (default seq_len; None for the full tail).
    """
    labels = list(state_by_label)
    if not labels:
        raise ValueError("state_by_label is empty")
    nb = len(labels)
    ncols = min(ncols, nb)
    nrows = int(np.ceil(nb / ncols))

    cmap = plt.get_cmap(cmap)
    with plt.rc_context(LONG_INFERENCE_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.3 * nrows),
                                 sharey=sharey, squeeze=False, layout="constrained")
        norm = layers = None
        for i, lab in enumerate(labels):
            ax = axes[i // ncols][i % ncols]
            data = state_by_label[lab]
            pos, layers, mean = data["position"], data["layer"], data["mean"]
            norm = plt.Normalize(vmin=layers.min(), vmax=layers.max())
            for li, lid in enumerate(layers):
                ax.plot(pos, smooth(mean[li], smooth_window), color=cmap(norm(lid)), linewidth=1.0)

            # More major x-ticks than the default auto-locator picks over this wide a range,
            # plus a denser and more visible grid (both major and minor) -- with 16 colored
            # lines already filling the axes, the default light gray blends in too easily.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=6))
            denser_grid(ax)
            ax.grid(which="major", color="#bbbbbb", linewidth=0.9)
            ax.grid(which="minor", color="#c8c8c8", linewidth=0.6)

            # Training sequence length -- past it, the model is running on lengths it never
            # trained on.
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
            fig.suptitle(suptitle, fontweight="bold", fontsize=16)
    return fig
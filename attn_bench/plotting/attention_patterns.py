"""Mechanistic: attention weights and gating scores for the pretrained attention variants.

Loaders live in data_loading.py (load_attention_patterns, load_attention_gating,
load_weighted_avg_attention_patterns). This file is plotting only.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from attn_bench.plotting.data_loading import (
    ALL_MEM_BUCKETS, load_weighted_avg_attention_patterns, mem_bucket_label)
from attn_bench.plotting.model_registry import MODEL_COLORS


def _as_label(bucket):
    return bucket if isinstance(bucket, str) else mem_bucket_label(bucket)


def _save_fig(fig, save_path, dpi=150):
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")


def _first_token_attention_per_layer(mean_map, query_slice):
    """mean_map: [L, H, S, S] -> [L], attention to key position 0 averaged over heads and
    (optionally sliced) query positions."""
    col0 = mean_map[:, :, :, 0]           # [L, H, S]  attention to key position 0
    if query_slice is not None:
        col0 = col0[:, :, query_slice]
    return col0.mean(axis=(1, 2))          # [L]


def _draw_first_token_lines(ax, items, color_for, query_slice, markersize=4):
    """Draws one attention-to-key-0-per-layer line per item on ax; shared by the pooled
    (one line per model) and by-bucket (one line per bucket, per model panel) views below.

    items: [(label, mean_map, count)]. color_for(label) -> matplotlib color.
    Returns [(mean_over_layers, line)] so the caller can build/sort its own legend.
    """
    entries = []
    for label, mean_map, count in items:
        per_layer = _first_token_attention_per_layer(mean_map, query_slice)
        overall = float(per_layer.mean())
        line, = ax.plot(range(len(per_layer)), per_layer, marker="o", markersize=markersize,
                        color=color_for(label), label=f"{label} (mean={overall:.3f}, n={count})")
        entries.append((overall, line))
    return entries


def plot_first_token_attention_avg(attn_by_model, *, query_slice=None, figsize=(8, 5), save_path=None):
    """Mean attention to the first token (key position 0) per layer -- the attention sink.
    Rouge-L mem-buckets are pooled per model, then averaged over heads and query positions.

    attn_by_model: {model_name: load_attention_patterns(..., 'attn_scores', ...)}
    query_slice: optional slice over query positions before averaging (default: all) -- e.g.
    slice(prompt_len, None) to skip early rows, which see few keys and inflate attention to
    key 0.
    """
    fig, ax = plt.subplots(figsize=figsize)
    items = []
    for name, buckets in attn_by_model.items():
        if not buckets:
            print(f"skip {name}: no attn_scores files found")
            continue
        averaged = load_weighted_avg_attention_patterns(buckets)
        items.append((name, averaged["mean"], averaged["count"]))
    if not items:
        print("no attn_scores data for any model")
        return None
    entries = _draw_first_token_lines(ax, items, lambda label: MODEL_COLORS.get(label, "black"), query_slice)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention to first token (key 0)")
    ax.set_title("Attention sink: first-token attention per layer\n"
                 "(avg over heads, query positions, all samples)")
    # legend sorted by overall mean (highest sink first)
    entries.sort(key=lambda e: e[0], reverse=True)
    handles = [e[1] for e in entries]
    ax.legend(handles=handles, labels=[h.get_label() for h in handles])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_first_token_attention_by_bucket(attn_by_model, buckets=None, query_slice=None,
                                         ncols=2, cmap="viridis", figsize=None, save_path=None):
    """Same first-token-attention-per-layer view as plot_first_token_attention, but split by
    Rouge-L mem-bucket instead of pooled -- does the sink change with how memorized the
    sample is? One panel per model, one line per bucket (dark=low Rouge-L, bright=high).

    attn_by_model: {model_name: load_attention_patterns(..., 'attn_scores', ...)}
    buckets: subset of ALL_MEM_BUCKETS to plot (default: all 10). Pass fewer (e.g. the two
    extremes) to compare "barely memorized" vs "memorized" without a crowded legend.
    query_slice: see plot_first_token_attention.
    """
    buckets = list(buckets) if buckets is not None else list(ALL_MEM_BUCKETS)
    bucket_pos = {label: i for i, label in enumerate(buckets)}
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=0, vmax=max(len(buckets) - 1, 1))
    color_for = lambda label: cmap(norm(bucket_pos[label]))

    names = [n for n, b in attn_by_model.items() if b]
    n_models = len(names)
    ncols = min(ncols, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (6 * ncols, 4.5 * nrows),
                             sharey=True, squeeze=False)
    for i, name in enumerate(names):
        ax = axes[i // ncols][i % ncols]
        model_buckets = attn_by_model[name]
        items = [(label, model_buckets[label]["mean"], int(model_buckets[label]["count"]))
                 for label in buckets
                 if label in model_buckets and int(model_buckets[label]["count"]) > 0]
        _draw_first_token_lines(ax, items, color_for, query_slice, markersize=3)

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer")
        if i % ncols == 0:
            ax.set_ylabel("Mean attention to first token (key 0)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, title="Rouge-L")

    for j in range(n_models, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Attention sink by memorization bucket\n(avg over heads, query positions)",
                fontweight="bold", fontsize=14)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_map(maps, bucket, layer, head, *, figsize=None, vmin=1e-4, vmax=None, save_path=None):
    """Multi-panel heatmap of the (query x key) attention/norm map, one panel per model.

    maps: {model_name: load_attention_patterns(...)[bucket_label]} -- all for the same
    kind and bucket.
    bucket: bucket index/label (used only for the title)
    layer, head: 0-based indices
    """
    model_names = list(maps.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        m = maps[name]
        mat = m["mean"][layer, head].astype(np.float32)   # [S, S]
        prompt_len = int(m["prompt_len"])

        pos = mat[mat > 0]
        vm = vmax if vmax is not None else (float(np.quantile(pos, 0.995)) if pos.size else 1.0)
        im = ax.imshow(mat, aspect="auto", origin="upper",
                       norm=LogNorm(vmin=vmin, vmax=max(vm, vmin * 10)), cmap="viridis")
        # prefix/suffix divider on both axes
        ax.axvline(prompt_len - 0.5, color="white", linewidth=1.0, linestyle="--")
        ax.axhline(prompt_len - 0.5, color="white", linewidth=1.0, linestyle="--")
        ax.set_title(f"{name}  (n={m['count']})")
        ax.set_xlabel("Key position")
        if ax is axes[0]:
            ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Rouge-L {_as_label(bucket)}  layer={layer}  head={head}", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_bucket_maps(maps_by_model, bucket, layer, head, **kwargs):
    """Convenience over plot_map: pick one Rouge-L mem-bucket from the full all-buckets dict
    (load_attention_patterns output per model) and draw one heatmap panel per model. Run
    twice with different buckets to compare e.g. high (09-10) vs low (00-01) memorization.

    maps_by_model: {model_name: load_attention_patterns(...)}
    bucket: bucket index 0..9 or label like '09-10'
    """
    label = _as_label(bucket)
    selected = {name: b[label] for name, b in maps_by_model.items() if label in b}
    if not selected:
        print(f"no maps for bucket {label} in any model")
        return None
    return plot_map(selected, bucket=label, layer=layer, head=head, **kwargs)


def plot_full_attn_maps_panel(maps, bucket, *, models=(), heads=(), layers=(),
                              cell_w=None, cell_h=None, target_width=14.0,
                              vmin=1e-4, save_path=None):
    """Full panel: rows=layers x models, cols=heads (+ always-included head-avg). Each cell
    is a (query x key) heatmap with LogNorm, color scale shared within each (layer, head)
    block so models are directly comparable.

    maps: {model_name: load_attention_patterns(...)[bucket_label]}, same kind/bucket for all.
    models/heads/layers: optional subsets (default: all); heads always keeps the avg column.
    cell_w/cell_h: default None scales up (never down) so a filtered, few-column call still
    fills target_width instead of staying pinned to the full 16-layer grid's tiny size.
    """
    model_names = [m for m in models if m in maps] if models else list(maps.keys())
    n_models = len(model_names)

    matrices = {name: maps[name]["mean"].astype(np.float32) for name in model_names}
    prompt_len = int(maps[model_names[0]]["prompt_len"])

    ref = next(iter(matrices.values()))
    n_layers_total, n_heads_total, _, _ = ref.shape
    layer_ids = list(layers) if layers else list(range(n_layers_total))
    head_ids = list(heads) if heads else list(range(n_heads_total))

    head_avg = {name: matrices[name].mean(axis=1) for name in model_names}  # (L, S, S), all heads

    # cols are plain (no model multiplication) -- one col per head, plus a final "avg" col.
    col_specs = [(f"H{h}", h) for h in head_ids] + [("avg", None)]

    n_cols = len(col_specs)
    n_row_groups = len(layer_ids)
    n_rows = n_row_groups * n_models

    base_w, base_h = 1.5, 0.8
    scale = min(max(target_width / (base_w * n_cols), 1.0), 3.5)
    cell_w = cell_w if cell_w is not None else base_w * scale
    cell_h = cell_h if cell_h is not None else base_h * scale

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False)
    # Default top margin is a fixed *fraction* of figure height, which becomes a huge gap in
    # absolute inches once the figure is this tall -- pin it to a small constant gap instead.
    total_height = cell_h * n_rows
    top = max(1 - 0.8 / total_height, 0.85)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=top)

    for gi, lid in enumerate(layer_ids):
        for ci, (col_label, hid) in enumerate(col_specs):
            cell_data = ({name: head_avg[name][lid] for name in model_names} if hid is None
                        else {name: matrices[name][lid, hid] for name in model_names})

            pos_vals = np.concatenate([cell_data[k].ravel() for k in model_names])
            pos_vals = pos_vals[pos_vals > 0]
            vmax = float(np.quantile(pos_vals, 0.995)) if len(pos_vals) > 0 else 1.0
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))

            for mi, name in enumerate(model_names):
                row = gi * n_models + mi
                ax = axes[row, ci]
                ax.imshow(cell_data[name], aspect="auto", origin="upper",
                          norm=norm, cmap="viridis", interpolation="nearest")
                ax.axvline(prompt_len - 0.5, color="white", linewidth=0.4, linestyle="--")
                ax.set_xticks([])
                ax.set_yticks([])

                if ci == 0:
                    ax.set_ylabel(f"L{lid}\n{name}", fontsize=7,
                                  rotation=0, ha="right", va="center", labelpad=4)
                if gi == 0 and mi == 0:
                    ax.set_title(col_label, fontsize=9, pad=3)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                if mi == 0:
                    ax.spines["top"].set_linewidth(1.5)
                    ax.spines["top"].set_color("white")

    fig.suptitle(
        f"Attention maps  Rouge-L {_as_label(bucket)}  |  rows: layer x model  |  cols: head",
        fontsize=12, y=top + (1 - top) * 0.8,
    )
    _save_fig(fig, save_path, dpi=None)
    return fig


def plot_gating_distribution(gating, *, layer=None, head=None, buckets=None, density=False,
                             merge_bins=1, figsize=(8, 5), save_path=None):
    """Gate-score distribution (Gated Attention paper convention). Gated model only.
    Aggregates the per-(bucket, layer, head) histogram to the requested granularity; one
    curve per Rouge-L bucket if `buckets` given, else pooled.

    gating: load_attention_gating(...) result. layer/head: restrict to one (default: pooled).
    density: True = probability density; False (default) = fraction per bin (paper convention).
    merge_bins: combine this many adjacent bins first (must divide the captured count, 100).
    """
    hist = gating["hist"].astype(np.float64)   # [n_buckets, L, H, n_bins]
    edges = gating["bin_edges"]

    if merge_bins > 1:
        n_bins = hist.shape[-1]
        if n_bins % merge_bins != 0:
            raise ValueError(f"merge_bins ({merge_bins}) must divide the captured bin count ({n_bins})")
        hist = hist.reshape(*hist.shape[:-1], n_bins // merge_bins, merge_bins).sum(axis=-1)
        edges = edges[::merge_bins]

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    def _density(sel_hist):
        # sum over all axes except the bin axis
        counts = sel_hist.reshape(-1, sel_hist.shape[-1]).sum(axis=0)
        total = counts.sum()
        if total == 0:
            return None
        frac = counts / total                    # fraction per bin (sums to 1)
        return frac / widths if density else frac

    fig, ax = plt.subplots(figsize=figsize)

    def _slice(bucket_idx):
        h = hist[bucket_idx] if bucket_idx is not None else hist
        # h: [..., L, H, n_bins]
        if layer is not None:
            h = h[..., layer, :, :]   # index the L axis (3rd from last)
        if head is not None:
            h = h[..., head, :]       # index the H axis (2nd from last)
        return h

    if buckets is None:
        dens = _density(_slice(None))
        if dens is not None:
            ax.plot(centers, dens, color="black", lw=2)
        title_extra = "all buckets pooled"
    else:
        counts = gating.get("count")
        cmap = plt.cm.viridis(np.linspace(0, 1, len(buckets)))
        for c, b in zip(cmap, buckets):
            bi = b if isinstance(b, int) else ALL_MEM_BUCKETS.index(b)
            dens = _density(_slice(bi))
            if dens is not None:
                n = f", n={int(counts[bi])}" if counts is not None else ""
                ax.plot(centers, dens, color=c, lw=1.8, label=f"Rouge-L {_as_label(b)}{n}")
        ax.legend(fontsize=8)
        title_extra = "by Rouge-L bucket"

    loc = []
    if layer is not None:
        loc.append(f"layer={layer}")
    if head is not None:
        loc.append(f"head={head}")
    loc_str = ", ".join(loc) if loc else "all layers/heads"

    ax.set_xlabel("gating score  sigmoid(gate)")
    ax.set_ylabel("normalized density" if density else "fraction of gate values")
    ax.set_xlim(0, 1)
    ax.set_title(f"Gate-score distribution ({loc_str}; {title_extra})", fontsize=11)
    plt.tight_layout()
    _save_fig(fig, save_path)
    return fig
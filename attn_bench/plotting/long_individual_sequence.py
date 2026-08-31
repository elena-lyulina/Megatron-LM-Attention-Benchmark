"""Behavioral: per-sequence position-wise loss, from the *_individual.jsonl files written
by --store-individual (attn_bench/evaluation/long_inference.py).

Unlike long_inference.py's plot_loss_panel (mean/std/count over a bucket), this reads raw
per-position, per-sequence records -- one line per sequence -- so a single sequence can be
inspected in detail: NLL, how far off the model's guess was (true-token rank), and, when a
tokenizer is given and the window is short enough to stay legible, the actual token text.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory

from attn_bench.plotting.data_loading import load_long_individual_sequence
from attn_bench.plotting.utils import (SAMPLE_LEN, TAB10_COLORS, VOCAB_SIZE,
                                       smooth)

### Sequence selection (filtering/ranking which sequences are worth plotting) ###

def _iter_aligned_records(paths):
    """Stream *_individual.jsonl files (paths: {label: path}, one entry works too) in
    lockstep, yielding (seq_id, {label: record}). Assumes every file lists the same
    sequences in the same order; raises on a mismatch rather than silently misaligning.
    Never holds more than one seq_id's records across all files at once."""
    labels = list(paths)
    files = [open(p) for p in paths.values()]
    try:
        for lines in zip(*files):
            recs = [json.loads(l) for l in lines]
            seq_id = recs[0]["seq_id"]
            for r in recs[1:]:
                if r["seq_id"] != seq_id:
                    raise ValueError(f"seq_id mismatch: {seq_id!r} vs {r['seq_id']!r} -- "
                                      "files are not aligned")
            yield seq_id, dict(zip(labels, recs))
    finally:
        for f in files:
            f.close()


def filter_sequences(paths, filter_fns, n_top=None, sort_key=None, reverse=True):
    """Stream *_individual.jsonl files (paths: {label: path}) once, keeping seq_ids where
    every filter_fn(seq_id, records) is truthy. records is {label: record} -- a filter can
    compare across labels or look at just one. Only survivors get reloaded in full, so this
    never holds more than one sequence's records in memory while streaming.

    n_top + sort_key: rank survivors by sort_key(seq_id, records), keep only the top n_top.

    Returns (seq_ids, records, scores) -- records: {label: {seq_id: record}}, reloaded;
    scores: {seq_id: sort_key(...)} if sort_key given, else None.
    """
    kept = []
    scores = {} if sort_key is not None else None
    for seq_id, records in _iter_aligned_records(paths):
        if all(fn(seq_id, records) for fn in filter_fns):
            kept.append(seq_id)
            if sort_key is not None:
                scores[seq_id] = sort_key(seq_id, records)

    if n_top is not None:
        if sort_key is None:
            raise ValueError("n_top requires sort_key")
        kept = sorted(kept, key=lambda sid: scores[sid], reverse=reverse)[:n_top]

    out_records = {lab: load_long_individual_sequence(p, seq_ids=set(kept)) for lab, p in paths.items()}
    return kept, out_records, scores


def score_sequences(records_by_label, filter_fns, sort_key):
    """In-memory sibling of filter_sequences, for records already loaded instead of streamed
    from disk. Same filter_fns/sort_key contract; missing labels come through as None. No
    n_top/reload -- every survivor's score is returned.

    records_by_label: {label: {seq_id: record}}, same shape filter_sequences returns.
    Returns {seq_id: sort_key(...)}.
    """
    label0 = next(iter(records_by_label))
    scores = {}
    for seq_id in records_by_label[label0]:
        records = {lab: records_by_label[lab].get(seq_id) for lab in records_by_label}
        if all(fn(seq_id, records) for fn in filter_fns):
            scores[seq_id] = sort_key(seq_id, records)
    return scores


### Plot ###

def plot_individual_panel(results_by_label, seq_ids, x_range=None, tokenizer=None, ncols=3,
                          colors=None, smooth_window=0, smooth_window_true_rank=0,
                          show_unsmoothed=False,
                          sample_end=SAMPLE_LEN, show_rank=True, sharey=True, metric="nll",
                          show_random_baseline=False,
                          max_tokens_for_labels=100, suptitle=None,
                          zoom_range=None):
    """One cell per seq_id; every model overlaid as a coloured NLL line (+ optional
    true-token-rank line underneath).

    results_by_label: {label: {seq_id: record}}, from data_loading.load_long_individual_sequence.
    seq_ids: one grid cell each; a cell is skipped if no model has it.
    sharey: main and zoom blocks (see zoom_range) share y-scales independently.
    tokenizer: token-label strip (true black, pred green/red) when window <= max_tokens_for_labels.
    show_unsmoothed: faint raw line behind the smoothed one, only where smoothing is on.
    zoom_range: optional (lo, hi) -- a second, narrower stacked block per seq_id.
    """
    labels = list(results_by_label)
    palette = colors or {}
    colors = {lab: palette.get(lab, TAB10_COLORS[i % len(TAB10_COLORS)]) for i, lab in enumerate(labels)}

    cells = [sid for sid in seq_ids if any(sid in results_by_label[l] for l in labels)]
    if not cells:
        raise ValueError(f"None of seq_ids={seq_ids} found under any label in results_by_label")

    def _panels_for(rng, note):
        window = None if rng is None else rng[1] - rng[0]
        show_tok = tokenizer is not None and window is not None and window <= max_tokens_for_labels
        if tokenizer is not None and not show_tok:
            print(f"{note}window too wide for token labels ({window} > {max_tokens_for_labels} "
                  "positions) -- plotting without the token strip.")
        panels = ["nll"] + (["rank"] if show_rank else []) + (["tokens"] if show_tok else [])
        return panels, show_tok

    def _height_ratios_for(panels, show_tok):
        # Scale with the actual row span (true + gaps + one row per model), not just the
        # model count -- _MODEL_ROW_GAP makes that span grow faster than len(labels) once
        # there's more than one model, and the panel needs to grow with it or text overlaps.
        token_span = _token_row_y(len(labels))[-1] + 1 if show_tok else 0  # +1: top/bottom pad
        token_ratio = 0.22 * token_span
        ratio_by_panel = {"nll": 5.0, "rank": 1.3, "tokens": token_ratio}
        return [ratio_by_panel[p] for p in panels]

    main_panels, main_show_tokens = _panels_for(x_range, "")
    blocks = [("main", x_range, main_panels, main_show_tokens)]
    if zoom_range is not None:
        zoom_panels, zoom_show_tokens = _panels_for(zoom_range, "zoom range: ")
        blocks.append(("zoom", zoom_range, zoom_panels, zoom_show_tokens))
    block_height_ratios = [_height_ratios_for(panels, show_tok) for _, _, panels, show_tok in blocks]
    rows_per_cell = sum(len(hr) for hr in block_height_ratios)

    ncols = min(ncols, len(cells))
    nrows = int(np.ceil(len(cells) / ncols))

    height_ratios = [h for hr in block_height_ratios for h in hr] * nrows
    fig_h = nrows * (sum(sum(hr) for hr in block_height_ratios) / 3.0) * 2.6
    fig = plt.figure(figsize=(6.5 * ncols, fig_h), layout="constrained")
    gs = fig.add_gridspec(nrows * rows_per_cell, ncols, height_ratios=height_ratios)

    handles = [plt.Line2D([], [], color=colors[l], linewidth=1.8) for l in labels]
    leg_labels = list(labels)
    # seq length boundary: same style/label convention as plot_loss_panel, so the two read
    # consistently if you're looking at both.
    boundary_style = dict(color="#999999", linestyle=(0, (2, 2)), linewidth=1.0)
    if sample_end is not None:
        handles = handles + [plt.Line2D([], [], **boundary_style)]
        leg_labels = leg_labels + [f"seq length ({sample_end})"]
    # Random-guess baseline: ln(V) in NLL space, V itself in perplexity space -- same as
    # plot_loss_panel.
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

    # Separate axis-sharing groups per block kind, so "main" cells share a y-scale among
    # themselves and "zoom" cells share a (likely very different) y-scale among themselves.
    nll_axes = {name: [] for name, *_ in blocks}
    rank_axes = {name: [] for name, *_ in blocks}

    def _draw_block(seq_id, r, c, row, name, rng, panels, show_tok):
        nll_ax = fig.add_subplot(gs[row + panels.index("nll"), c],
                                 sharey=(nll_axes[name][0] if sharey and nll_axes[name] else None))
        nll_axes[name].append(nll_ax)
        rank_ax = None
        if show_rank:
            rank_ax = fig.add_subplot(gs[row + panels.index("rank"), c], sharex=nll_ax,
                                      sharey=(rank_axes[name][0] if sharey and rank_axes[name] else None))
            rank_axes[name].append(rank_ax)
        tok_ax = (fig.add_subplot(gs[row + panels.index("tokens"), c], sharex=nll_ax)
                 if show_tok else None)

        title_len = None
        for lab in labels:
            rec = results_by_label[lab].get(seq_id)
            if rec is None:
                continue
            title_len = rec["length"]
            lo = 0 if rng is None else max(0, rng[0])
            hi = rec["length"] if rng is None else min(rec["length"], rng[1])
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

        # One vertical gridline per position when tokens are shown, so the curve above
        # lines up exactly with the token below it -- the default locator can space major
        # ticks every 2/5/10 positions, which drifts out of alignment with per-token text.
        if show_tok:
            nll_ax.xaxis.set_major_locator(MultipleLocator(1))
        if sample_end is not None and (rng is None or rng[0] <= sample_end <= rng[1]):
            nll_ax.axvline(sample_end, **boundary_style)
        if show_random_baseline:
            nll_ax.axhline(baseline_value, **baseline_style)
        if title_len is None:
            title = str(seq_id)
        elif name == "main":
            title = f"seq_id: {seq_id}  (len={title_len})"
        else:
            title = f"seq_id: {seq_id}  (range {rng[0]}-{rng[1]})"
        nll_ax.set_title(title)
        nll_ax.spines[["top", "right"]].set_visible(False)
        # More y-ticks than matplotlib's default -- with sharey, one outlier cell can
        # stretch the axis wide enough that the default locator collapses to just 2-3
        # ticks, making every other (normal-range) cell look flat.
        nll_ax.yaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=6))
        nll_ax.grid(True, color="#ececec", linewidth=0.8)
        if c == 0:
            nll_ax.set_ylabel("Perplexity" if metric == "ppl" else "NLL")

        stacked = [nll_ax] + ([rank_ax] if rank_ax is not None else [])
        if rank_ax is not None:
            if show_tok:
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
            _plot_token_strip(tok_ax, results_by_label, labels, seq_id, rng, tokenizer)
            lo = 0 if rng is None else max(0, rng[0])
            hi = title_len if rng is None else min(title_len, rng[1])
            _connect_panels(fig, stacked + [tok_ax], lo, hi)
        else:
            stacked[-1].set_xlabel("position (from sample start)")

        return row + len(panels)

    for i, seq_id in enumerate(cells):
        r, c = divmod(i, ncols)
        row = rows_per_cell * r
        for name, rng, panels, show_tok in blocks:
            row = _draw_block(seq_id, r, c, row, name, rng, panels, show_tok)

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
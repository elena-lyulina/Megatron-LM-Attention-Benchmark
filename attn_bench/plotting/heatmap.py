"""Every heatmap-drawing function, metric-agnostic (the metric to plot is always a
parameter, never hardcoded). Genuinely Rouge-L-specific plots (histogram, Rouge-L-bin
distribution heatmap) stay in rouge_l.py instead."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

from attn_bench.plotting.data_loading import select_models
from attn_bench.plotting.utils import (_draw_panel_grid, _panel_key,
                                       _resolve_metric, suptitle_centered)


def format_val(v):
    if math.isnan(v):
        return ''
    s = f'{v:.3f}'
    return s[1:] if 0 < v < 1 else s


def _metric_label(metric, n=None, p=None):
    return f'hayes extraction (n={n}, p={p})' if metric == 'hayes' else metric


def _normalize_means(means_data, reps, mode):
    """Normalize means_data in-place relative to rep=0.

    mode : None       — no normalization
           'subtract' — score(rep_k) - score(rep_0)
           'divide'   — score(rep_k) / score(rep_0)
    """
    if mode is None or 0 not in reps:
        return
    for name in means_data:
        baseline = means_data[name][0]
        for r in reps:
            if mode == 'subtract':
                means_data[name][r] -= baseline
            elif mode == 'divide':
                means_data[name][r] = (means_data[name][r] / baseline) if baseline != 0 else float('nan')


def _cell_value(results, expr, r, offset, prefix, suffix, metric, n=None, p=None):
    """(mean, per-sample scores) for one cell. metric='hayes' computes the Hayes et al. 2025
    (n, p)-discoverable extraction rate on the fly from p_z scores instead of reading a metric
    already stored in Results: mean = fraction of samples with 1-(1-p_z)^n >= p; scores =
    per-sample p_hat, reused as the Wilcoxon comparison array everywhere a stored metric's
    per-sample scores would otherwise go."""
    if metric == 'hayes':
        if n is None or p is None:
            raise ValueError("metric='hayes' requires n and p")
        p_z = np.array(results.get_stats(expr, r, offset, prefix, suffix, 'p_z').scores)
        p_hat = 1 - (1 - p_z) ** n
        return float((p_hat >= p).mean()), p_hat
    md = _resolve_metric(results.get_all_metrics(expr, r, offset, prefix, suffix), metric)
    return md.mean, np.array(md.scores)


def _read_cols(cols, metric, n, p):
    """label -> {r: (mean, scores)}, reading each column's Results at its own (offset,
    prefix, suffix). A None Results (point never captured for this column) yields no entries,
    same downstream effect as a captured Results missing a given repetition."""
    data = {}
    for label, results, offset, prefix, suffix in cols:
        data[label] = {}
        if results is None:
            continue
        expr = results.expr[0]
        for r in results.repetitions:
            try:
                data[label][r] = _cell_value(results, expr, r, offset, prefix, suffix, metric, n, p)
            except KeyError:
                pass
    return data


def _draw_heatmap_ax(ax, cols, metric, xlabel, vmin=None, vmax=None, reference=None,
                     ref_cols=None, reference_name=None, normalize=None, annot_fontsize=12,
                     ytick_fontsize=10, xtick_fontsize=10, text_scale=1.0, show_pvalues=True,
                     title=None, n=None, p=None):
    """Draw a single heatmap into ax. Rows = repetitions, columns = `cols` -- one engine
    shared by both heatmap families (columns=models vs swept-values is just how `cols` is built).

    cols      : [(label, results_or_None, offset, prefix, suffix), ...], each at its own point.
    reference : a label already in `cols` -- other columns compare against its own scores.
    ref_cols  : parallel to `cols`, a *different* Results object per column, for a reference
                not itself in `cols`. Give at most one of reference/ref_cols.
    reference_name : display name for the p-value note when using ref_cols.
    normalize : None | 'subtract' | 'divide' -- each column vs its own rep=0.
    title     : ax title, p-value note auto-appended -- omit for the caller to title it.

    Returns has_pvalues, for the caller's own suptitle p-value note.
    """
    reference_name = reference if reference_name is None else reference_name

    cell_data = _read_cols(cols, metric, n, p)
    means_data = {label: {r: mp[0] for r, mp in d.items()} for label, d in cell_data.items()}
    scores_data = {label: {r: mp[1] for r, mp in d.items()} for label, d in cell_data.items()}

    reps = sorted({r for _, results, *_ in cols if results is not None for r in results.repetitions})
    _normalize_means(means_data, reps, normalize)

    df = pd.DataFrame({label: {r: means_data[label].get(r, float('nan')) for r in reps} for label, *_ in cols})
    df.index.name = 'repetition'

    if normalize == 'subtract':
        _vmin = vmin if vmin is not None else -1
        _vmax = vmax if vmax is not None else  1
        cmap = 'RdYlGn'
    elif normalize == 'divide':
        _vmin = vmin if vmin is not None else 0
        _vmax = vmax if vmax is not None else  5
        cmap = 'RdYlGn'
    else:
        _vmin = vmin if vmin is not None else 0
        _vmax = vmax if vmax is not None else  1
        cmap = 'YlOrRd'

    if reference is not None:
        ref_scores_by_label = {label: scores_data.get(reference, {}) for label, *_ in cols}
    elif ref_cols is not None:
        ref_scores_by_label = {label: {r: sc for r, (_, sc) in d.items()}
                               for label, d in _read_cols(ref_cols, metric, n, p).items()}
    else:
        ref_scores_by_label = None

    annot, has_pvalues = [], False
    for r in reps:
        row = []
        for label, *_ in cols:
            val_str = format_val(means_data[label].get(r, float('nan')))
            ref_scores = ref_scores_by_label.get(label) if ref_scores_by_label else None
            if (not val_str or label == reference or ref_scores is None
                    or r not in ref_scores or r not in scores_data[label]):
                row.append(val_str)
            else:
                s_ref, s_cmp = ref_scores[r], scores_data[label][r]
                if s_ref.size < 2 or s_cmp.size < 2 or not np.isfinite(s_cmp).any() or not np.isfinite(s_ref).any():
                    row.append(val_str)
                else:
                    try:
                        with np.errstate(invalid='ignore'):
                            _, pval = wilcoxon(s_ref, s_cmp)
                        star = '*' if pval < 0.05 else ''
                        row.append(f'{val_str}{star} / {pval:.3f}' if show_pvalues else f'{val_str}{star}')
                        has_pvalues = True
                    except ValueError:
                        row.append(val_str)
        annot.append(row)

    sns.heatmap(
        df.astype(float), annot=annot, fmt='', cmap=cmap,
        vmin=_vmin, vmax=_vmax, ax=ax, annot_kws={'size': annot_fontsize},
        linewidths=0.5, linecolor='white',
    )
    for text in ax.texts:
        if '*' in text.get_text():
            text.set_fontweight('bold')
    ax.tick_params(axis='y', labelsize=ytick_fontsize)  # shrinks with vscale -- see plot_models_heatmap
    # The rotated model-name labels are the biggest single chunk of vertical chrome (a
    # rotated label's height footprint is its rendered length, i.e. its font size) -- also
    # tying this to vscale is what actually frees up room for the rows above, not just
    # shrinking the rows' own tick font.
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    # Colorbar ticks are a separate axes seaborn creates internally -- tick_params above
    # only reaches the main ax, so this one has to be sized explicitly too or it's left
    # looking oversized exactly like the other fixed-size text was.
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=10 * text_scale)
    ax.set_xlabel(xlabel, fontsize=10 * text_scale)
    ax.set_ylabel('repetitions', fontsize=10 * text_scale)

    if title is not None:
        if not has_pvalues:
            pval_note = ''
        elif show_pvalues:
            pval_note = f'\n/ p-values vs {reference_name} (* p < 0.05)'
        else:
            pval_note = f'\n(* p < 0.05 vs {reference_name})'
        ax.set_title(f'{title}{pval_note}', fontsize=11 * text_scale, weight='bold')

    return has_pvalues


def _column_key(columns, v, fixed_offset, fixed_prefix, fixed_suffix):
    if columns == 'offset': return (v, fixed_prefix, fixed_suffix)
    if columns == 'prefix': return (fixed_offset, v, fixed_suffix)
    if columns == 'suffix': return (fixed_offset, fixed_prefix, v)


def plot_models_heatmap(results_dict, metric, offset=0, prefix=500, suffix=500,
                       vmin=None, vmax=None, reference='full', normalize=None, models=None,
                       fig_hscale=1.0, fig_vscale=1.0, title=None, show_pvalues=True,
                       n=None, p=None):
    """models : list of model names to plot, in order (default: every model in results_dict).
    metric: 'hayes' computes the Hayes et al. 2025 (n, p)-discoverable extraction rate on the
    fly instead of reading a stored metric -- pass n, p in that case (see _cell_value).
    show_pvalues: False drops the ' / p-value' text, keeping just the significance star --
    use for wide model lists where the full annotation overflows into neighboring cells.
    fig_hscale / fig_vscale: shrink the per-model width / per-repetition height (e.g. 0.7),
    for a wide model list or a long repetition list. The `+1.5`/`+1.2` fixed-chrome allowance
    isn't scaled -- shrinking it too would let chrome eat a growing share of an ever-smaller
    canvas. All text (annotations, ticks, title, colorbar) follows min(fig_hscale, fig_vscale)
    so it shrinks with whichever axis is actually getting tighter."""
    results_dict = select_models(results_dict, models)
    reps = next(iter(results_dict.values())).repetitions
    text_scale = min(fig_hscale, fig_vscale)
    fig, ax = plt.subplots(figsize=(len(results_dict) * 1.5 * fig_hscale + 1.5,
                                    len(reps) * 0.55 * fig_vscale + 1.2))
    cols = [(name, results, offset, prefix, suffix) for name, results in results_dict.items()]
    _draw_heatmap_ax(ax, cols, metric, xlabel='model', vmin=vmin, vmax=vmax, reference=reference,
                     normalize=normalize, annot_fontsize=12 * text_scale, ytick_fontsize=10 * fig_vscale,
                     xtick_fontsize=10 * fig_vscale, text_scale=text_scale, show_pvalues=show_pvalues,
                     title=f'offset={offset}, prefix={prefix}, suffix={suffix}', n=n, p=p)
    # See utils.py's _draw_panel_grid comment for why the margin is forced explicitly.
    suptitle_centered(fig, [ax], title if title is not None else _metric_label(metric, n, p),
                      fontsize=14 * text_scale, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_models_heatmap_panel(results_grid, metric, panels, values,
                             fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                             ncols=4, vmin=None, vmax=None, reference='full', normalize=None,
                             models=None, show_pvalues=True, n=None, p=None):
    """Panel of heatmaps, one per value of `panels`, columns always models -- see
    plot_heatmap_per_model_panel for the transpose (columns=swept values).

    panels    : 'offset' | 'prefix' | 'suffix' | 'diagonal' | 'models' -- 'models' panels a
                model subset per fixed point (values=[(label, model_list),...], `models=`
                unused); 'diagonal' sweeps suffix_start=offset+prefix (values=[(offset,
                prefix),...], fixed_offset/prefix unused).
    show_pvalues: False drops the p-value text, keeping just the significance star.
    n, p      : only used when metric='hayes'.
    """
    norm_note = {'subtract': ' − rep=0', 'divide': ' / rep=0'}.get(normalize, '')

    # Size off the panels this call will actually draw, not an arbitrary first entry in the
    # whole grid -- results_grid can span many (offset, prefix, suffix) points with wildly
    # different model/repetition coverage (e.g. a sparse point with 2 of 5 models, 1 of 10
    # reps), and sizing off whichever key happens to be first crushes every other panel into
    # a figure that was never tall/wide enough for it (same failure mode fixed for
    # plot_rouge_heatmap's max_n_reps).
    panel_entries = [results_grid[k] for v in values
                     if (k := _panel_key(panels, v, fixed_offset, fixed_prefix, fixed_suffix)) in results_grid]
    if panels == 'models':
        n_models = max(len(m) for _, m in values)
        reps_pools = [e.values() for e in panel_entries]
    else:
        sub_entries = [select_models(e, models) for e in panel_entries]
        n_models = max((len(sub) for sub in sub_entries), default=1)
        reps_pools = [sub.values() for sub in sub_entries]
    n_reps = max((len(r.repetitions) for pool in reps_pools for r in pool), default=1)
    cell_w, cell_h = n_models * 1.5 + 1.5, n_reps * 0.55 + 1.2

    # _draw_panel_grid calls suptitle_fn(fixed_desc) only after every panel is drawn, so
    # whether any panel actually got a p-value (has_pvalues) has to be threaded out through
    # this mutable holder rather than computed up front -- same pattern
    # plot_heatmap_per_model_panel already uses in its own (non-_draw_panel_grid) loop.
    has_pvalues_holder = []

    def draw(ax, entry, offset, prefix, suffix, v):
        row_models = v[1] if panels == 'models' else models
        sub = select_models(entry, row_models)
        cols = [(name, results, offset, prefix, suffix) for name, results in sub.items()]
        has_pvalues_holder.append(_draw_heatmap_ax(ax, cols, metric, xlabel='model', vmin=vmin, vmax=vmax,
                         reference=reference, normalize=normalize, show_pvalues=show_pvalues, n=n, p=p))

    def suptitle_fn(fixed_desc):
        pval_note = ''
        if any(has_pvalues_holder):
            pval_note = (f'\n/ p-values vs {reference} (* p < 0.05)' if show_pvalues
                        else f'\n(* p < 0.05 vs {reference})')
        return f'{_metric_label(metric, n, p)}{norm_note}: {fixed_desc}{pval_note}'

    _draw_panel_grid(results_grid, panels, values, fixed_offset, fixed_prefix, fixed_suffix,
                     ncols, cell_w, cell_h, draw, suptitle_fn, suptitle_fontsize=16)


def plot_heatmap_per_model(results_grid, metric, model, columns, values,
                           fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                           vmin=None, vmax=None, reference=None, show_pvalues=True,
                           n=None, p=None):
    """One heatmap for a single model: rows = repetitions, columns = swept `columns` values --
    the single-heatmap sibling of plot_heatmap_per_model_panel (one panel per model in a grid).

    columns   : 'offset' | 'prefix' | 'suffix' -- which axis is on the columns.
    reference : optional model name (if different from `model`) -- every cell gets a
                Wilcoxon star/p-value vs its scores in the same column.
    show_pvalues: False drops the p-value text, keeping just the significance star.
    n, p      : only used when metric='hayes'.
    """
    keys = [(v, key) for v in values
           if (key := _column_key(columns, v, fixed_offset, fixed_prefix, fixed_suffix)) in results_grid]
    if not keys:
        print(f'No results found for columns={columns}, values={values}')
        return

    columns_label = {'offset': 'offset', 'prefix': 'prefix', 'suffix': 'suffix'}[columns]
    n_reps = len({r for _, key in keys if model in results_grid[key]
                 for r in results_grid[key][model].repetitions})
    cell_w, cell_h = len(keys) * 1.5 + 1.5, n_reps * 0.55 + 1.2
    fig, ax = plt.subplots(figsize=(cell_w, cell_h))

    cols = [(v, results_grid[key].get(model), *key) for v, key in keys]
    ref_cols = ([(v, results_grid[key].get(reference), *key) for v, key in keys]
               if reference is not None and reference != model else None)
    has_pvalues = _draw_heatmap_ax(ax, cols, metric, xlabel=columns_label, vmin=vmin, vmax=vmax,
                                   ref_cols=ref_cols, reference_name=reference,
                                   show_pvalues=show_pvalues, n=n, p=p)

    fixed_desc = ', '.join(f'{k}={v}' for k, v in
                           [('offset', fixed_offset), ('prefix', fixed_prefix), ('suffix', fixed_suffix)]
                           if k != columns)
    if not has_pvalues:
        pval_note = ''
    elif show_pvalues:
        pval_note = f' / p-values vs {reference} (* p < 0.05)'
    else:
        pval_note = f' (* p < 0.05 vs {reference})'
    # See utils.py's _draw_panel_grid comment for why the margin is forced explicitly.
    suptitle_centered(fig, [ax], f'{model}: {_metric_label(metric, n, p)}  ({fixed_desc}){pval_note}',
                      fontsize=14, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_heatmap_per_model_panel(results_grid, metric, columns, values, models,
                             fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                             ncols=4, vmin=None, vmax=None, reference=None, show_pvalues=True,
                             n=None, p=None):
    """One heatmap per model, in a grid; rows = repetitions, columns = swept `columns`
    values -- the transpose of plot_models_heatmap_panel (columns=models instead).

    columns   : 'offset' | 'prefix' | 'suffix' -- which axis is on the columns.
    models    : list of model names, one heatmap each, in order.
    reference : optional model name -- every other model's cells get a Wilcoxon star/p-value
                vs it in the same column; its own heatmap is drawn plain.
    show_pvalues: False drops the p-value text, keeping just the significance star.
    n, p      : only used when metric='hayes'.
    """
    keys = [(v, key) for v in values
           if (key := _column_key(columns, v, fixed_offset, fixed_prefix, fixed_suffix)) in results_grid]
    if not keys:
        print(f'No results found for columns={columns}, values={values}')
        return

    rows = [name for name in models if any(name in results_grid[key] for _, key in keys)]
    if not rows:
        print(f'No results found for models={models}')
        return

    columns_label = {'offset': 'offset', 'prefix': 'prefix', 'suffix': 'suffix'}[columns]
    n_reps = len({r for _, key in keys for name in rows if name in results_grid[key]
                  for r in results_grid[key][name].repetitions})
    cell_w, cell_h = len(keys) * 1.5 + 1.5, n_reps * 0.55 + 1.2
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows), squeeze=False)
    has_pvalues = False
    for idx, name in enumerate(rows):
        ax = axes[idx // ncols][idx % ncols]
        cols = [(v, results_grid[key].get(name), *key) for v, key in keys]
        ref_cols = ([(v, results_grid[key].get(reference), *key) for v, key in keys]
                   if reference is not None and name != reference else None)
        has_pvalues |= _draw_heatmap_ax(ax, cols, metric, xlabel=columns_label, vmin=vmin, vmax=vmax,
                                        ref_cols=ref_cols, reference_name=reference,
                                        show_pvalues=show_pvalues, n=n, p=p)
        ax.set_title(name, fontsize=11, weight='bold')
    for idx in range(len(rows), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fixed_desc = ', '.join(f'{k}={v}' for k, v in
                           [('offset', fixed_offset), ('prefix', fixed_prefix), ('suffix', fixed_suffix)]
                           if k != columns)
    if not has_pvalues:
        pval_note = ''
    elif show_pvalues:
        pval_note = f' / p-values vs {reference} (* p < 0.05)'
    else:
        pval_note = f' (* p < 0.05 vs {reference})'
    # See utils.py's _draw_panel_grid comment for why the margin is forced explicitly.
    suptitle_centered(fig, list(fig.axes), f'{_metric_label(metric, n, p)}: {columns_label}s={values}, {fixed_desc}{pval_note}',
                      fontsize=16, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()

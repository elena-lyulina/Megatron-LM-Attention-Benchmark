"""Shared plotting and data-loading utilities for memorization metric notebooks."""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from scipy.stats import wilcoxon

from verbatim_eval.controlled_expr import Results


# --- Data loading ---

def load_results(exp_name, results_base, prefix=500, suffix=500, offset=0, policy='greedy'):
    path = f'{results_base}/{exp_name}/offset_{offset}_prefix_{prefix}_suffix_{suffix}_{policy}.pkl'
    return Results.load(path)


def load_results_grid(offsets, prefixes, suffixes, exp_names, results_base, policy='greedy'):
    """Load all available (offset, prefix, suffix) combinations into a flat dict."""
    grid = {}
    for offset in offsets:
        for prefix in prefixes:
            for suffix in suffixes:
                key = (offset, prefix, suffix)
                per_model = {}
                for name, exp in exp_names.items():
                    try:
                        per_model[name] = load_results(exp, results_base, offset=offset,
                                                        prefix=prefix, suffix=suffix, policy=policy)
                    except FileNotFoundError:
                        pass
                if per_model:
                    grid[key] = per_model
    return grid


# --- Heatmap helpers ---

MODEL_COLORS = {
    'full':       '#6BAED6',
    'gated':      '#F4A0B5',
    'learn-sink': '#FD9E4B',
    'fix-sink':   '#74C476',
}


def _fmt_val(v):
    if math.isnan(v):
        return ''
    s = f'{v:.3f}'
    return s[1:] if 0 < v < 1 else s


def _suptitle_centered(fig, visible_axes, title, **kwargs):
    """Center suptitle over the actual axes area (not full figure width)."""
    x0 = min(ax.get_position().x0 for ax in visible_axes)
    x1 = max(ax.get_position().x1 for ax in visible_axes)
    fig.suptitle(title, x=(x0 + x1) / 2, ha='center', **kwargs)


def _panel_titles(dim, values, fixed_offset, fixed_prefix, fixed_suffix):
    dim_label = {'offset': 'offset', 'prefix': 'prefix', 'suffix': 'suffix'}[dim]
    fixed = [(k, v) for k, v in [('offset', fixed_offset), ('prefix', fixed_prefix), ('suffix', fixed_suffix)]
             if k != dim]
    fixed_desc = ', '.join(f'{k}={v}' for k, v in fixed)
    varying_desc = f'{dim_label}s={values}'
    return (lambda v: f'{dim_label}={v}'), f'{varying_desc}, {fixed_desc}'


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


def _draw_heatmap_ax(ax, results_dict, metric, offset=0, prefix=500, suffix=500,
                     vmin=None, vmax=None, reference='full', normalize=None):
    """Draw a single heatmap into ax. Rows = repetitions, cols = attention variants.
    Non-reference cells: 'value* / p-value' (Wilcoxon vs reference; * = p < 0.05).
    """
    scores_data, means_data = {}, {}
    for model_name, results in results_dict.items():
        expr = results.expr[0]
        scores_data[model_name], means_data[model_name] = {}, {}
        for r in results.repetitions:
            try:
                md = results.get_all_metrics(expr, r, offset, prefix, suffix)[metric]
                scores_data[model_name][r] = np.array(md.scores)
                means_data[model_name][r] = md.mean
            except KeyError:
                scores_data[model_name][r] = np.array([float('nan')])
                means_data[model_name][r] = float('nan')

    reps = next(iter(results_dict.values())).repetitions
    _normalize_means(means_data, reps, normalize)

    df = pd.DataFrame({name: {r: means_data[name][r] for r in reps} for name in results_dict})
    df.index.name = 'repetition'

    if normalize == 'subtract':
        _vmin = vmin if vmin is not None else -1
        _vmax = vmax if vmax is not None else  1
        cmap = 'RdYlGn'
    elif normalize == 'divide':
        _vmin = vmin if vmin is not None else 0
        _vmax = vmax if vmax is not None else 5
        cmap = 'RdYlGn'
    else:
        _vmin = vmin if vmin is not None else 0
        _vmax = vmax if vmax is not None else 1
        cmap = 'YlOrRd'

    ref_scores = scores_data.get(reference)
    annot, has_pvalues = [], False
    for r in reps:
        row = []
        for name in df.columns:
            val_str = _fmt_val(means_data[name][r])
            if not val_str or name == reference or ref_scores is None:
                row.append(val_str)
            else:
                s_ref, s_cmp = ref_scores[r], scores_data[name][r]
                if s_ref.size < 2 or s_cmp.size < 2 or not np.isfinite(s_cmp).any():
                    row.append(val_str)
                else:
                    try:
                        _, p = wilcoxon(s_ref, s_cmp)
                        star = '*' if p < 0.05 else ''
                        row.append(f'{val_str}{star} / {p:.3f}')
                        has_pvalues = True
                    except ValueError:
                        row.append(val_str)
        annot.append(row)

    sns.heatmap(
        df.astype(float), annot=annot, fmt='', cmap=cmap,
        vmin=_vmin, vmax=_vmax, ax=ax, annot_kws={'size': 12},
        linewidths=0.5, linecolor='white',
    )
    for text in ax.texts:
        if '*' in text.get_text():
            text.set_fontweight('bold')

    pval_note = f'\n/ p-values vs {reference} (* p < 0.05)' if has_pvalues else ''
    ax.set_title(f'offset={offset}, prefix={prefix}, suffix={suffix}{pval_note}', fontsize=11, weight='bold')
    ax.set_xlabel('attention variant', fontsize=10)
    ax.set_ylabel('repetitions', fontsize=10)


def plot_attn_heatmap(results_dict, metric, offset=0, prefix=500, suffix=500,
                      vmin=None, vmax=None, reference='full', normalize=None):
    reps = next(iter(results_dict.values())).repetitions
    fig, ax = plt.subplots(figsize=(len(results_dict) * 1.5 + 1.5, len(reps) * 0.55 + 1.2))
    _draw_heatmap_ax(ax, results_dict, metric, offset, prefix, suffix, vmin, vmax, reference, normalize)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _suptitle_centered(fig, [ax], metric, fontsize=14, weight='bold')
    plt.show()


def plot_attn_heatmaps_panel(results_grid, metric, dim, values,
                              fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                              ncols=4, vmin=None, vmax=None, reference='full', normalize=None):
    """Panel of heatmaps varying one dimension.

    dim       : 'offset' | 'prefix' | 'suffix'
    values    : list of values for the varying dimension
    normalize : None | 'subtract' | 'divide'  (relative to rep=0 per model)
    """
    def _key(v):
        if dim == 'offset': return (v, fixed_prefix, fixed_suffix)
        if dim == 'prefix': return (fixed_offset, v, fixed_suffix)
        if dim == 'suffix': return (fixed_offset, fixed_prefix, v)

    panels = [(v, _key(v)) for v in values if _key(v) in results_grid]
    if not panels:
        print(f'No results found for dim={dim}, values={values}')
        return

    panel_title, fixed_desc = _panel_titles(dim, values, fixed_offset, fixed_prefix, fixed_suffix)
    pval_note = f'\n/ p-values vs {reference} (* p < 0.05)'
    norm_note = {'subtract': ' − rep=0', 'divide': ' / rep=0'}.get(normalize, '')

    n = len(panels)
    nrows = math.ceil(n / ncols)
    first_results = next(iter(results_grid.values()))
    n_models = len(first_results)
    n_reps = len(next(iter(first_results.values())).repetitions)
    cell_w, cell_h = n_models * 1.5 + 1.5, n_reps * 0.55 + 1.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows), squeeze=False)
    visible = []
    for idx, (v, key) in enumerate(panels):
        offset, prefix, suffix = key
        ax = axes[idx // ncols][idx % ncols]
        _draw_heatmap_ax(ax, results_grid[key], metric, offset, prefix, suffix,
                         vmin, vmax, reference, normalize)
        ax.set_title(panel_title(v), fontsize=11, weight='bold')
        visible.append(ax)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _suptitle_centered(fig, visible, f'{metric}{norm_note}: {fixed_desc}{pval_note}',
                       fontsize=16, weight='bold')
    plt.show()


# --- Lineplot helpers ---

def _symlog(x, linthresh=1, base=2):
    safe = np.maximum(np.abs(x), linthresh)
    return np.sign(x) * np.where(
        np.abs(x) <= linthresh,
        np.abs(x),
        linthresh * (1 + np.log(safe / linthresh) / np.log(base))
    )


def _inv_symlog(y, linthresh=1, base=2):
    safe = np.maximum(np.abs(y), linthresh)
    return np.sign(y) * np.where(
        np.abs(y) <= linthresh,
        np.abs(y),
        linthresh * (base ** ((safe - linthresh) / linthresh))
    )


def _display(metric):
    if metric.startswith('Ref_') or metric.endswith('_ref') or metric.endswith('_gen'):
        return metric
    return f'Gen_{metric}'


def _metrics_list(metric_pair):
    """Normalise metric_pair to a list of (metric, linestyle, label_suffix) triples."""
    if isinstance(metric_pair, str):
        return [(metric_pair, '-', '')]
    metric_ref, metric_gen = metric_pair
    return [
        (metric_ref, '--', f' {_display(metric_ref)}'),
        (metric_gen, '-',  f' {_display(metric_gen)}'),
    ]


def _metric_pair_title(metric_pair):
    if isinstance(metric_pair, str):
        return _display(metric_pair)
    metric_ref, metric_gen = metric_pair
    return f'{_display(metric_ref)} vs {_display(metric_gen)}'


def plot_lineplot(ax, results_dict, metric_pair, prefix=500, suffix=500, offset=0, show_std=True):
    metrics = _metrics_list(metric_pair)
    n_models = len(results_dict)
    fallback_colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))
    dodge_offsets = np.linspace(-0.05, 0.05, n_models)

    for i, (model_name, results) in enumerate(results_dict.items()):
        color = MODEL_COLORS.get(model_name, fallback_colors[i])
        expr = results.expr[0]
        reps = np.array(results.repetitions)
        reps_dodged = _inv_symlog(_symlog(reps) + dodge_offsets[i])

        for metric, ls, lbl_suffix in metrics:
            means = np.array([results.get_stats(expr, r, offset, prefix, suffix, metric).mean for r in reps])
            stds  = np.array([results.get_stats(expr, r, offset, prefix, suffix, metric).std  for r in reps])
            ax.plot(reps_dodged, means, linestyle=ls, color=color, marker='o', markersize=4,
                    label=f'{model_name}{lbl_suffix}', alpha=0.85)
            if show_std:
                ax.fill_between(reps_dodged, means - stds, means + stds, alpha=0.25, color=color)

    ax.set_xscale('symlog', base=2, linthresh=1)
    ax.set_xticks(next(iter(results_dict.values())).repetitions)
    ax.set_xticklabels(next(iter(results_dict.values())).repetitions)
    ax.set_xlabel('repetitions', fontsize=10)
    ax.set_ylabel(_metric_pair_title(metric_pair), fontsize=10)
    ax.grid(True, which='both', alpha=0.2)
    ax.legend(fontsize=8, ncol=2)


def plot_lineplots_panel(results_grid, metric_pair, dim, values,
                         fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                         ncols=4, show_std=True):
    """Panel of line plots varying one dimension.

    metric_pair : single metric string  e.g. 'Rouge-L'
                  or pair of strings    e.g. ('Ref_PPL', 'PPL')
    dim         : 'offset' | 'prefix' | 'suffix'
    values      : list of values for the varying dimension
    """
    def _key(v):
        if dim == 'offset': return (v, fixed_prefix, fixed_suffix)
        if dim == 'prefix': return (fixed_offset, v, fixed_suffix)
        if dim == 'suffix': return (fixed_offset, fixed_prefix, v)

    panels = [(v, _key(v)) for v in values if _key(v) in results_grid]
    if not panels:
        print(f'No results found for dim={dim}, values={values}')
        return

    panel_title, fixed_desc = _panel_titles(dim, values, fixed_offset, fixed_prefix, fixed_suffix)

    n = len(panels)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows), squeeze=False)
    visible = []
    for idx, (v, key) in enumerate(panels):
        offset, prefix, suffix = key
        ax = axes[idx // ncols][idx % ncols]
        plot_lineplot(ax, results_grid[key], metric_pair, prefix=prefix, suffix=suffix,
                      offset=offset, show_std=show_std)
        ax.set_title(panel_title(v), fontsize=11)
        visible.append(ax)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _suptitle_centered(fig, visible,
                       f'{_metric_pair_title(metric_pair)}: {fixed_desc}',
                       fontsize=14, weight='bold')
    plt.show()


# --- Qualitative example display ---

_CELL_STYLE  = 'padding:6px 8px;vertical-align:top;max-width:280px;min-width:200px;text-align:left;'
_SCROLL_STYLE = (
    'max-height:160px;overflow-y:auto;white-space:pre-wrap;overflow-wrap:break-word;'
    'font-size:11px;font-family:monospace;color:#111;line-height:1.4;text-align:left;'
)
_METRIC_STYLE = 'font-size:10px;font-family:monospace;color:#111;text-align:left;line-height:1.8;'
_SEP_STYLE    = 'border:none;border-top:1px solid #999;margin:4px 0;'
_TH_STYLE     = 'padding:6px 10px;text-align:left;background:#ddd;color:#111;font-size:12px;'
_N_METRIC_LINES = 5


def _td_text(text, bg):
    escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    return (f'<td style="background:{bg};{_CELL_STYLE}">'
            f'<div style="{_SCROLL_STYLE}">{escaped}</div></td>')


def _td_metrics(lines, bg):
    content = '<br>'.join(lines) + f'<hr style="{_SEP_STYLE}">'
    return f'<td style="background:{bg};{_CELL_STYLE}"><div style="{_METRIC_STYLE}">{content}</div></td>'


def _fmt(v):
    return f'{v:.3f}'


def _load_jsonl_reordered(exp_name, rep, results_base, prefix, suffix, offset, policy):
    folder = Path(f'{results_base}/{exp_name}/inference'
                  f'/offset_{offset}_prefix_{prefix}_suffix_{suffix}'
                  f'/rep_{rep}_{policy}')
    rank_files = sorted(folder.glob('rank*.jsonl'))
    world_size = len(rank_files)
    per_rank = [[] for _ in rank_files]
    for r, f in enumerate(rank_files):
        with open(f) as fh:
            for line in fh:
                per_rank[r].append(json.loads(line))
    n = sum(len(x) for x in per_rank)
    data = [None] * n
    for r in range(world_size):
        for pos, item in enumerate(per_rank[r]):
            data[pos * world_size + r] = item
    return [x for x in data if x is not None]


def _load_examples(tok, exp_name, rep, indices, results_base, prefix, suffix, offset, policy):
    data = _load_jsonl_reordered(exp_name, rep, results_base, prefix, suffix, offset, policy)
    return [{
        'prefix':  tok.decode(data[i]['prefix'],           skip_special_tokens=True),
        'ref':     tok.decode(data[i]['true_suffix'],      skip_special_tokens=True),
        'gen':     tok.decode(data[i]['generated_suffix'], skip_special_tokens=True),
        'Rouge-L': data[i]['Rouge-L'],
        'LCS':     data[i]['lcs_norm'],
        'PPL':     data[i]['perplexity'],
        'NLL':     data[i]['nll_mean'],
        'TTR_gen': data[i]['TTR_gen'],
        'Ref_PPL': data[i]['ref_perplexity'],
        'Ref_NLL': data[i]['ref_nll_mean'],
        'TTR_ref': data[i]['TTR_ref'],
    } for i in indices]


# --- Rouge-L distribution ---

def load_reps_from_jsonl(exp_name, results_base, prefix, suffix, offset=0, policy='greedy'):
    """Return sorted rep values found in the inference jsonl folders (no pkl needed)."""
    folder = Path(f'{results_base}/{exp_name}/inference/offset_{offset}_prefix_{prefix}_suffix_{suffix}')
    if not folder.exists():
        return []
    reps = []
    for d in folder.iterdir():
        if d.name.startswith('rep_') and d.name.endswith(f'_{policy}'):
            try:
                reps.append(int(d.name[4:-(len(policy) + 1)]))
            except ValueError:
                pass
    return sorted(reps)


def _load_rouge_per_rep(exp_names, results_base, reps, prefix, suffix, offset=0, policy='greedy'):
    data = {}
    for name, exp in exp_names.items():
        per_rep = {}
        for rep in reps:
            try:
                records = _load_jsonl_reordered(exp, rep, results_base, prefix, suffix, offset, policy)
                per_rep[rep] = np.array([r['Rouge-L'] for r in records])
            except (FileNotFoundError, OSError):
                pass
        data[name] = per_rep
    return data


def plot_rouge_hist(exp_names, results_base, reps, prefix, suffix, offset=0, policy='greedy', n_bins=30):
    """Rouge-L histogram per model, all reps pooled."""
    rouge_data = _load_rouge_per_rep(exp_names, results_base, reps, prefix, suffix, offset, policy)
    n_models = len(exp_names)
    bins = np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, per_rep) in zip(axes, rouge_data.items()):
        color = MODEL_COLORS.get(name, '#888888')
        all_vals = np.concatenate(list(per_rep.values())) if per_rep else np.array([])
        ax.hist(all_vals, bins=bins, density=True, color=color, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.set_title(name, fontsize=11, weight='bold')
        ax.set_xlabel('Rouge-L', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('density', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _suptitle_centered(fig, axes,
                       f'Rouge-L distribution (all reps)  prefix={prefix}, suffix={suffix}, offset={offset}',
                       fontsize=14, weight='bold')
    plt.show()


def plot_rouge_heatmap(exp_names, results_base, reps, prefix, suffix, offset=0, policy='greedy', n_bins=10):
    """Rep × Rouge-L-bin heatmap per model. Cells = row-normalised fraction."""
    rouge_data = _load_rouge_per_rep(exp_names, results_base, reps, prefix, suffix, offset, policy)
    n_models = len(exp_names)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(n_bins)]

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, len(reps) * 0.55 + 1.5))
    if n_models == 1:
        axes = [axes]

    for i, (ax, (name, per_rep)) in enumerate(zip(axes, rouge_data.items())):
        available = [r for r in reps if r in per_rep]
        mat = np.zeros((len(available), n_bins))
        for j, rep in enumerate(available):
            counts, _ = np.histogram(per_rep[rep], bins=bins)
            total = counts.sum()
            mat[j] = counts / total if total > 0 else counts

        df = pd.DataFrame(mat, index=available, columns=bin_labels)
        df.index.name = 'rep'

        sns.heatmap(df, ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                    annot=True, fmt='.2f', annot_kws={'size': 7},
                    linewidths=0.3, linecolor='white', cbar=(i == n_models - 1))
        ax.set_title(name, fontsize=11, weight='bold')
        ax.set_xlabel('Rouge-L bin', fontsize=10)
        ax.set_ylabel('rep' if i == 0 else '', fontsize=10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _suptitle_centered(fig, axes,
                       f'Rouge-L by rep  prefix={prefix}, suffix={suffix}, offset={offset}',
                       fontsize=14, weight='bold')
    plt.show()


def show_examples(tok, rep, exp_map, col_bg, sample_indices,
                  results_base, prefix=500, suffix=500, offset=0, policy='greedy'):
    data_by_attn = {
        name: _load_examples(tok, exp, rep, sample_indices, results_base, prefix, suffix, offset, policy)
        for name, exp in exp_map.items()
    }

    cols = ['prefix', 'ref suffix'] + [f'gen suffix: {name}' for name in exp_map]
    header = ''.join(f'<th style="{_TH_STYLE}">{c}</th>' for c in cols)

    rows = [f'<tr>{header}</tr>']
    for idx in range(len(sample_indices)):
        ref_ex = next(iter(data_by_attn.values()))[idx]

        metric_cells = (
            _td_metrics(['&nbsp;'] * _N_METRIC_LINES, col_bg.get('prefix', '#f0f0f0'))
            + _td_metrics([
                '&nbsp;',
                '&nbsp;',
                f'Ref_PPL: {_fmt(ref_ex["Ref_PPL"])}',
                f'Ref_NLL: {_fmt(ref_ex["Ref_NLL"])}',
                f'TTR_ref: {_fmt(ref_ex["TTR_ref"])}',
            ], col_bg.get('ref', '#c8e6c9'))
        )
        for name, examples in data_by_attn.items():
            e = examples[idx]
            metric_cells += _td_metrics([
                f'Rouge-L: {_fmt(e["Rouge-L"])}',
                f'LCS:     {_fmt(e["LCS"])}',
                f'PPL:     {_fmt(e["PPL"])}',
                f'NLL:     {_fmt(e["NLL"])}',
                f'TTR_gen: {_fmt(e["TTR_gen"])}',
            ], col_bg.get(name, '#f5f5f5'))

        text_cells = (
            _td_text(ref_ex['prefix'], col_bg.get('prefix', '#f0f0f0')) +
            _td_text(ref_ex['ref'],    col_bg.get('ref', '#c8e6c9'))
        )
        for name, examples in data_by_attn.items():
            text_cells += _td_text(examples[idx]['gen'], col_bg.get(name, '#f5f5f5'))

        rows.append(f'<tr style="border-top:2px solid #aaa">{metric_cells}</tr>')
        rows.append(f'<tr>{text_cells}</tr>')

    table = (
        f'<details style="margin-top:24px;width:100%">'
        f'<summary style="cursor:pointer;font-size:16px;font-weight:bold;padding:4px 0">'
        f'repetition = {rep}</summary>'
        f'<div style="width:100%;overflow-x:auto">'
        f'<table style="border-collapse:collapse;border:1px solid #ccc">'
        + ''.join(rows) +
        '</table></div>'
        f'</details>'
    )
    display(HTML(table))
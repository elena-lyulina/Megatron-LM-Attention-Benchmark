"""Generic plotting scaffolding shared across the memorization-metrics notebooks: the
panel-grid layout engine (used by both heatmap.py and this file's own lineplot functions),
metric-key resolution across capture-pipeline generations, and a few constants/helpers
shared with the long-inference plots. Heatmap-drawing functions themselves live in
heatmap.py."""

import math

import matplotlib.pyplot as plt
import numpy as np

from attn_bench.plotting.data_loading import select_models
from attn_bench.plotting.model_registry import MODEL_COLORS

SEQ_LEN = 8192     # training sequence length
SAMPLE_LEN = 8190  # sample content tokens; predicted at positions 0..8189, so 8190 = first suffix token

# Llama-3 tokenizer vocab, padded (confirmed identical across every model's pretraining log:
# "padded vocab (size: 128256) with 0 dummy tokens"). NLL of a uniform random guess over the
# vocab is ln(VOCAB_SIZE); perplexity of that guess is VOCAB_SIZE itself.
VOCAB_SIZE = 128256

# Fallback color cycle for a label not in MODEL_COLORS.
TAB10_COLORS = plt.get_cmap("tab10").colors

# Shared style skin for the long-inference plots (loss panel, state norm). Applied via
# plt.rc_context so it scopes to those figures only and does not leak into the notebook's
# global rcParams. Helvetica falls back to Arial then the matplotlib default, so it still
# renders where Helvetica isn't installed (e.g. the cluster).
LONG_INFERENCE_STYLE = {
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


def denser_grid(ax):
    """Lighter minor grid between the major gridlines, on both axes (matplotlib's default
    auto-picked minor-tick spacing). Minor tick marks themselves are hidden so only the
    gridlines get denser, not the axis edges."""
    ax.minorticks_on()
    ax.grid(which="minor", color="#ececec", linewidth=0.6)
    ax.tick_params(which="minor", length=0)


def smooth(y, w):
    # Centered rolling mean with correct edge normalization (window shrinks at the ends).
    if not w or w <= 1:
        return y
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


# --- Panel-grid scaffolding (shared by heatmap.py and this file's lineplot functions) ---

def suptitle_centered(fig, visible_axes, title, **kwargs):
    """Center suptitle over the actual axes area (not full figure width)."""
    x0 = min(ax.get_position().x0 for ax in visible_axes)
    x1 = max(ax.get_position().x1 for ax in visible_axes)
    fig.suptitle(title, x=(x0 + x1) / 2, ha='center', **kwargs)


def _panel_titles(panels, values, fixed_offset, fixed_prefix, fixed_suffix):
    if panels == 'models':
        # values: [(panel_label, model_list), ...] -- offset/prefix/suffix don't vary here.
        fixed_desc = f'offset={fixed_offset}, prefix={fixed_prefix}, suffix={fixed_suffix}'
        return (lambda v: v[0]), fixed_desc
    if panels == 'diagonal':
        # values: [(offset, prefix), ...] -- anti-diagonal sweep at fixed suffix_start =
        # offset+prefix (disentangles anchor length from discarded run-up depth, see
        # project_mem_offset_prefix_disentangle memory). suffix is the only fixed axis.
        fixed_desc = f'suffix={fixed_suffix}'
        return (lambda v: f'suffix_start={v[0] + v[1]} (offset={v[0]}, prefix={v[1]}, suffix={fixed_suffix})'), fixed_desc
    panels_label = {'offset': 'offset', 'prefix': 'prefix', 'suffix': 'suffix'}[panels]
    fixed = [(k, v) for k, v in [('offset', fixed_offset), ('prefix', fixed_prefix), ('suffix', fixed_suffix)]
             if k != panels]
    fixed_desc = ', '.join(f'{k}={v}' for k, v in fixed)
    varying_desc = f'{panels_label}s={values}'
    return (lambda v: f'{panels_label}={v}'), f'{varying_desc}, {fixed_desc}'


def _panel_key(panels, v, fixed_offset, fixed_prefix, fixed_suffix):
    if panels == 'offset': return (v, fixed_prefix, fixed_suffix)
    if panels == 'prefix': return (fixed_offset, v, fixed_suffix)
    if panels == 'suffix': return (fixed_offset, fixed_prefix, v)
    if panels == 'models': return (fixed_offset, fixed_prefix, fixed_suffix)
    if panels == 'diagonal': return (v[0], v[1], fixed_suffix)  # v = (offset, prefix)


def _draw_panel_grid(results_grid, panels, values, fixed_offset, fixed_prefix, fixed_suffix,
                     ncols, cell_w, cell_h, draw_fn, suptitle_fn, suptitle_fontsize=14,
                     bold_titles=True):
    """Shared scaffolding for a grid of panels, one per value of `panels` -- offset/prefix/
    suffix, or 'models' (each panel gets its own model subset of the same fixed (offset,
    prefix, suffix) point; values is [(panel_label, model_list), ...]): resolve which panels
    have data, lay out the figure, draw each one via draw_fn(ax, entry, offset, prefix,
    suffix, v), hide unused cells, center a suptitle built by suptitle_fn(fixed_desc). Used
    by both the heatmap and lineplot panels."""
    def _key(v):
        return _panel_key(panels, v, fixed_offset, fixed_prefix, fixed_suffix)

    panel_axes = [(v, _key(v)) for v in values if _key(v) in results_grid]
    if not panel_axes:
        print(f'No results found for panels={panels}, values={values}')
        return

    panel_title, fixed_desc = _panel_titles(panels, values, fixed_offset, fixed_prefix, fixed_suffix)
    n = len(panel_axes)
    nrows = math.ceil(n / ncols)
    # Cap total figure height, shrinking cell_h (not cell_w) proportionally if the plain
    # cell_h*nrows would exceed it. Some notebook capture pipelines silently mis-render
    # (crush to a small, distorted image) once a figure gets tall enough -- confirmed empty
    # margin: a 13.4in-tall figure captured fine, 20.1in didn't, independent of width or
    # axis count. 16in stays comfortably under that with real headroom either side.
    MAX_FIG_HEIGHT_IN = 16
    if cell_h * nrows > MAX_FIG_HEIGHT_IN:
        cell_h = MAX_FIG_HEIGHT_IN / nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows), squeeze=False)
    title_kwargs = {'fontsize': 11, 'weight': 'bold'} if bold_titles else {'fontsize': 11}
    visible = []
    for idx, (v, key) in enumerate(panel_axes):
        offset, prefix, suffix = key
        ax = axes[idx // ncols][idx % ncols]
        draw_fn(ax, results_grid[key], offset, prefix, suffix, v)
        ax.set_title(panel_title(v), **title_kwargs)
        visible.append(ax)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    # tight_layout doesn't reliably leave room for the 2-line suptitle added after it (on
    # some subplot-count/ncols combinations it silently fails to honor rect= at all --
    # "Tight layout not applied" -- and adding the suptitle before tight_layout instead
    # doesn't get auto-detected either, at least on this matplotlib version). Forcing the
    # top margin explicitly after both is the one approach that actually held up across
    # every nrows tried; costs a bit of unused headroom on short (1-2 row) panels.
    suptitle_centered(fig, visible, suptitle_fn(fixed_desc), fontsize=suptitle_fontsize, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()


# Older captures store these under friendly keys directly; newer ones use different raw
# names instead, and don't store PPL/Ref_PPL at all -- only derivable as exp(NLL), same
# convention dashboard/export_data.py already uses.
_METRIC_ALIASES = {'LCS': 'lcs_norm', 'NLL': 'gen_nll_mean', 'Ref_NLL': 'ref_nll_mean'}
_METRIC_EXP_OF = {'PPL': 'gen_nll_mean', 'Ref_PPL': 'ref_nll_mean'}


class _DerivedMetric:
    __slots__ = ('scores', 'mean', 'std')

    def __init__(self, scores, mean, std):
        self.scores, self.mean, self.std = scores, mean, std


def _resolve_metric(all_metrics, metric):
    """Same metric value regardless of which capture-pipeline generation produced this
    point. Raises KeyError (same as a plain dict lookup) if the metric truly isn't there
    under any name."""
    if metric in all_metrics:
        return all_metrics[metric]
    if metric in _METRIC_ALIASES:
        return all_metrics[_METRIC_ALIASES[metric]]
    if metric in _METRIC_EXP_OF:
        nll = all_metrics[_METRIC_EXP_OF[metric]]
        scores = np.exp(np.asarray(nll.scores))
        return _DerivedMetric(scores=scores, mean=float(scores.mean()), std=float(scores.std()))
    raise KeyError(metric)


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


def plot_lineplot(ax, results_dict, metric_pair, prefix=500, suffix=500, offset=0, show_std=True, models=None):
    results_dict = select_models(results_dict, models)
    metrics = _metrics_list(metric_pair)
    is_paired = len(metrics) > 1
    n_models = len(results_dict)
    fallback_colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))
    dodge_offsets = np.linspace(-0.05, 0.05, n_models)

    model_handles = []
    for i, (model_name, results) in enumerate(results_dict.items()):
        color = MODEL_COLORS.get(model_name, fallback_colors[i])
        expr = results.expr[0]
        reps = np.array(results.repetitions)
        reps_dodged = _inv_symlog(_symlog(reps) + dodge_offsets[i])

        for metric, ls, lbl_suffix in metrics:
            # _resolve_metric, not a raw dict lookup -- some metrics (LCS, match_25/50/75,
            # exact_match) are computed on the fly and only reachable via get_all_metrics,
            # not stored under their own raw key; different capture generations of the
            # same model can also store a metric under different key names, or derive it
            # (PPL/Ref_PPL from NLL/Ref_NLL). KeyError -> NaN, not a crash, same as the
            # heatmap functions degrade per-cell.
            def _stat(r):
                try:
                    return _resolve_metric(results.get_all_metrics(expr, r, offset, prefix, suffix), metric)
                except KeyError:
                    return None
            stats = [_stat(r) for r in reps]
            means = np.array([s.mean if s is not None else float('nan') for s in stats])
            stds  = np.array([s.std  if s is not None else float('nan') for s in stats])
            ax.plot(reps_dodged, means, linestyle=ls, color=color, marker='o', markersize=4,
                    label=f'{model_name}{lbl_suffix}', alpha=0.85)
            if show_std:
                ax.fill_between(reps_dodged, means - stds, means + stds, alpha=0.25, color=color)
        model_handles.append(plt.Line2D([], [], color=color, marker='o', markersize=4, label=model_name))

    ax.set_xscale('symlog', base=2, linthresh=1)
    ax.set_xticks(next(iter(results_dict.values())).repetitions)
    ax.set_xticklabels(next(iter(results_dict.values())).repetitions)
    ax.set_xlabel('repetitions', fontsize=10)
    ax.set_ylabel(_metric_pair_title(metric_pair), fontsize=10)
    ax.grid(True, which='both', alpha=0.2)

    if is_paired:
        # Two legends instead of one combined model x {ref,gen} list: at legend-swatch
        # size, dashed vs solid is nearly invisible packed in among many same-length
        # entries. The style key is fixed at 2 entries with a long handle, so the dash
        # pattern actually reads, and it doesn't grow as models are added/removed.
        model_legend = ax.legend(handles=model_handles, fontsize=8, ncol=2, loc='upper left', framealpha=0.9)
        ax.add_artist(model_legend)
        style_handles = [plt.Line2D([], [], color='0.3', linestyle=ls, linewidth=2) for _, ls, _ in metrics]
        style_labels = [lbl_suffix.strip() for _, _, lbl_suffix in metrics]
        ax.legend(handles=style_handles, labels=style_labels, fontsize=8, loc='lower right',
                 handlelength=3, framealpha=0.9)
    else:
        ax.legend(fontsize=8, ncol=2)


def plot_lineplot_panel(results_grid, metric_pair, panels, values,
                        fixed_offset=0, fixed_prefix=500, fixed_suffix=500,
                        ncols=4, show_std=True, models=None):
    """Panel of line plots, one per value of `panels`.

    metric_pair : single metric string  e.g. 'Rouge-L'
                  or pair of strings    e.g. ('Ref_PPL', 'PPL')
    panels      : 'offset' | 'prefix' | 'suffix' -- which axis varies panel-to-panel
    values      : list of values for `panels`
    """
    def draw(ax, entry, offset, prefix, suffix, v):
        plot_lineplot(ax, entry, metric_pair, prefix=prefix, suffix=suffix,
                      offset=offset, show_std=show_std, models=models)

    _draw_panel_grid(results_grid, panels, values, fixed_offset, fixed_prefix, fixed_suffix,
                     ncols, 8, 5, draw,
                     lambda fixed_desc: f'{_metric_pair_title(metric_pair)}: {fixed_desc}',
                     suptitle_fontsize=14, bold_titles=False)
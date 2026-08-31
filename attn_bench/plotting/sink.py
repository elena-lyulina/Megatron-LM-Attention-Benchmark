"""Plots specific to the inference-time sink-value ablation (off-by-one / learnable sink):
trained softmax_offset heatmaps, paired PPL/TTR panels, generation-quality comparisons,
inference-config correctness checks (long-context capture)."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from attn_bench.plotting import model_registry
from attn_bench.plotting.data_loading import load_long_inference_nll_by_config
from attn_bench.plotting.long_inference import compare_nll, plot_loss_panel
from attn_bench.plotting.utils import SEQ_LEN, plot_lineplot

_COMPARISON_LINESTYLES = ('-', '--', ':', '-.')


def plot_trained_sink_offsets(offsets, source_label=None):
    """Two heatmaps side by side: the trained softmax_offset (log scale) and its exp() --
    the effective sink weight relative to a real token -- both per layer x head.
    offsets: from data_loading.load_trained_sink_offsets."""
    n_layers, n_heads = offsets.shape
    scales = np.exp(offsets)

    fig, axes = plt.subplots(1, 2, figsize=(max(n_heads * 0.9 + 3, 8), n_layers * 0.45 + 2))
    kw = dict(annot=True, fmt='.2f', annot_kws={'size': 8}, linewidths=0.3, linecolor='white',
              xticklabels=[f'h{i}' for i in range(n_heads)],
              yticklabels=[f'L{i}' for i in range(n_layers)])

    sns.heatmap(offsets, ax=axes[0], cmap='RdBu_r', center=0, **kw)
    axes[0].set_title('softmax_offset (log scale)', fontsize=11, weight='bold')
    axes[0].set_xlabel('head')
    axes[0].set_ylabel('layer')

    sns.heatmap(scales, ax=axes[1], cmap='YlOrRd', **kw)
    axes[1].set_title('exp(softmax_offset) (effective sink scale)', fontsize=11, weight='bold')
    axes[1].set_xlabel('head')
    axes[1].set_ylabel('layer')

    title = 'Trained softmax_offset (before any sscale patching)'
    plt.suptitle(title, fontsize=12, weight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def _resolve_sink_models(results, base_key, sscales):
    if sscales is None:
        return None
    return (['full'] if 'full' in results else []) + [model_registry.sscale_label(base_key, s) for s in sscales]


def plot_sink_ppl(results, base_key, sscales=None, show_std=True, prefix=500, suffix=500, offset=0):
    """PPL line plot (x = repetitions, one line per sink value) for one sink family.
    base_key: model_registry key ('off-by-one'/'learn-sink'), used for the panel title and
    to resolve sscales into result-dict keys via sscale_label. sscales: optional subset of
    scale values to plot (plus 'full', if present) -- default None plots every label already
    in results. show_std: shade +/- one std across samples."""
    models = _resolve_sink_models(results, base_key, sscales)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lineplot(ax, results, ('Ref_PPL', 'PPL'), prefix=prefix, suffix=suffix, offset=offset,
                 show_std=show_std, models=models)
    ax.set_title(f'PPL: {base_key} attn — sink value ablation', fontsize=11, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_sink_ttr(results, base_key, sscales=None, show_std=True, prefix=500, suffix=500, offset=0):
    """TTR line plot (x = repetitions, one line per sink value) for one sink family. Same
    parameters as plot_sink_ppl."""
    models = _resolve_sink_models(results, base_key, sscales)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_lineplot(ax, results, ('TTR_ref', 'TTR_gen'), prefix=prefix, suffix=suffix, offset=offset,
                 show_std=show_std, models=models)
    ax.set_title(f'TTR: {base_key} attn — sink value ablation', fontsize=11, weight='bold')
    plt.tight_layout()
    plt.show()


def _plot_quality_metric_lines(ax, quality_data, metric, n=None, title=''):
    reps = None
    for label, entry in quality_data.items():
        if metric not in entry:
            continue
        scores = entry[metric]
        reps = sorted(scores.keys())
        ys = [scores[r][str(n)] for r in reps] if metric == 'distinct_n' else [scores[r] for r in reps]
        ax.plot(range(len(reps)), ys, marker='o', markersize=4, label=label)
    if reps is not None:
        ax.set_xticks(range(len(reps)))
        ax.set_xticklabels([str(r) for r in reps], rotation=45)
    ax.set_xlabel('repetitions')
    ax.set_title(title, fontsize=10, weight='bold')
    ax.legend(fontsize=7, loc='best')


def plot_generation_quality(quality_sets, distinct_ns=(1, 2, 3)):
    """quality_sets: [(title, data_loading.load_generation_quality(...) result), ...] --
    one column per set. Draws perplexity (one row) then distinct-n (one row per set, one
    column per n) as two separate figures."""
    fig, axes = plt.subplots(1, len(quality_sets), figsize=(8 * len(quality_sets), 5))
    axes = np.atleast_1d(axes)
    for ax, (title, qdata) in zip(axes, quality_sets):
        _plot_quality_metric_lines(ax, qdata, 'perplexity', title=f'Perplexity (Qwen2.5-1.5B): {title}')
        ax.set_ylabel('perplexity')
    plt.suptitle('Generation quality — Perplexity under Qwen2.5-1.5B', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(len(quality_sets), len(distinct_ns),
                             figsize=(6 * len(distinct_ns), 4.5 * len(quality_sets)))
    axes = np.atleast_2d(axes)
    for row, (title, qdata) in enumerate(quality_sets):
        for col, n in enumerate(distinct_ns):
            ax = axes[row, col]
            _plot_quality_metric_lines(ax, qdata, 'distinct_n', n=n, title=f'Distinct-{n}: {title}')
            ax.set_ylabel(f'distinct-{n}')
    plt.suptitle('Generation quality — Distinct-n (token level)', fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_sink_nll_comparison(model_key, configs, reps, xmax=None, smooth_window=100):
    """For one sink model, plot position-wise NLL from two or more named long-context
    inference configs overlaid -- e.g. a chain like tp=1 (unfused) -> tp=4 (unfused) -> tp=1
    (fused), each isolating one variable from the last -- plus a compare_nll table per
    *adjacent* pair (non-adjacent pairs would conflate several changes at once). Each label
    gets its own linestyle, cycled: with configs this close, same-style solid lines would
    just draw over each other and hide all but the last.

    configs: {label: config_dir_name}, in comparison order.
    """
    labels = list(configs)
    linestyles = dict(zip(labels, itertools.cycle(_COMPARISON_LINESTYLES)))
    nll_by_label = load_long_inference_nll_by_config(model_key, configs, reps)
    plot_loss_panel(nll_by_label, smooth_window=smooth_window, xmax=xmax, show_std=False,
                    linestyles=linestyles, suptitle=model_key, vlines={'seq length': SEQ_LEN})
    plt.show()
    for label_a, label_b in zip(labels[:-1], labels[1:]):
        compare_nll(nll_by_label[label_a], nll_by_label[label_b], label_a=label_a, label_b=label_b)

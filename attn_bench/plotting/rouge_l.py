"""Rouge-L-specific plots. Reads scores straight from the already-loaded `Results` pickle
(`.get_stats(...).scores`)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from attn_bench.plotting.data_loading import ALL_MEM_BUCKETS, select_models
from attn_bench.plotting.heatmap import format_val
from attn_bench.plotting.model_registry import MODEL_COLORS
from attn_bench.plotting.utils import suptitle_centered


def plot_mem_bucket_counts(attn_by_model, models=None, figsize=None, keys=None,
                           x_label='Rouge-L bucket',
                           title='Rouge-L memorization bucket sizes (pooled across all reps)'):
    """Bar chart of sample counts per series entry, one panel per model.

    attn_by_model: {model_name: load_attention_patterns(...)} keyed by Rouge-L label, or
    load_attention_patterns(..., split_by="rep") keyed by repetition -- same {mean, count, prompt_len}
    values. The Rouge-L cut shows the outcome axis; the by-rep cut is a flat sanity check
    (every rep sees every doc).
    keys: series entries to bar, in order (default: all 10 Rouge-L labels). Pass the rep list
    for a by-rep dict.
    """
    keys = list(keys) if keys is not None else list(ALL_MEM_BUCKETS)
    xs = [str(k) for k in keys]
    attn_by_model = select_models(attn_by_model, models)
    n_models = len(attn_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=figsize or (4 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, buckets) in zip(axes, attn_by_model.items()):
        counts = [int(buckets[k]['count']) if k in buckets else 0 for k in keys]
        color = MODEL_COLORS.get(name, '#888888')
        ax.bar(xs, counts, color=color, edgecolor='white', linewidth=0.4)
        ax.set_title(f'{name} (n={sum(counts)})', fontsize=11, weight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    axes[0].set_ylabel('documents', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    suptitle_centered(fig, axes, title, fontsize=14, weight='bold')
    plt.show()


def plot_rouge_hist(results_dict, offset=0, prefix=500, suffix=500, models=None, n_bins=30):
    """Rouge-L histogram per model, every repetition in each model's Results pooled together."""
    results_dict = select_models(results_dict, models)
    n_models = len(results_dict)
    bins = np.linspace(0, 1, n_bins + 1)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, results) in zip(axes, results_dict.items()):
        color = MODEL_COLORS.get(name, '#888888')
        expr = results.expr[0]
        all_vals = np.concatenate([
            np.array(results.get_stats(expr, rep, offset, prefix, suffix, 'Rouge-L').scores)
            for rep in results.repetitions
        ]) if results.repetitions else np.array([])
        ax.hist(all_vals, bins=bins, density=True, color=color, alpha=0.75, edgecolor='white', linewidth=0.4)
        ax.set_title(name, fontsize=11, weight='bold')
        ax.set_xlabel('Rouge-L', fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel('density', fontsize=10)
    # See utils.py's _draw_panel_grid comment -- tight_layout's rect= doesn't reliably
    # reserve the suptitle margin, subplots_adjust after the fact does.
    suptitle_centered(fig, axes,
                      f'Rouge-L distribution (all reps)  prefix={prefix}, suffix={suffix}, offset={offset}',
                      fontsize=14, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_rouge_heatmap(results_dict, offset=0, prefix=500, suffix=500, models=None, n_bins=10):
    """Repetition x Rouge-L-bin heatmap per model. Cells = row-normalised fraction."""
    results_dict = select_models(results_dict, models)
    n_models = len(results_dict)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(n_bins)]
    # Max across models, not just the first one in the dict -- different models can have
    # genuinely different numbers of captured repetitions at this (offset, prefix, suffix)
    # point; sizing off whichever model happens to be first crushes every other model's
    # rows into a figure that was never tall enough for them.
    max_n_reps = max(len(results.repetitions) for results in results_dict.values())

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, max_n_reps * 0.55 + 1.5))
    if n_models == 1:
        axes = [axes]

    for i, (ax, (name, results)) in enumerate(zip(axes, results_dict.items())):
        expr = results.expr[0]
        model_reps = results.repetitions
        mat = np.zeros((len(model_reps), n_bins))
        for j, rep in enumerate(model_reps):
            scores = np.array(results.get_stats(expr, rep, offset, prefix, suffix, 'Rouge-L').scores)
            counts, _ = np.histogram(scores, bins=bins)
            total = counts.sum()
            mat[j] = counts / total if total > 0 else counts

        df = pd.DataFrame(mat, index=model_reps, columns=bin_labels)
        df.index.name = 'rep'

        annot = [[format_val(v) for v in row] for row in mat]
        sns.heatmap(df, ax=ax, cmap='YlOrRd', vmin=0, vmax=1,
                    annot=annot, fmt='', annot_kws={'size': 7},
                    linewidths=0.3, linecolor='white', cbar=(i == n_models - 1))
        ax.set_title(name, fontsize=11, weight='bold')
        ax.set_xlabel('Rouge-L bin', fontsize=10)
        ax.set_ylabel('rep' if i == 0 else '', fontsize=10)
        ax.tick_params(axis='x', labelrotation=45, labelsize=8)

    suptitle_centered(fig, axes,
                      f'Rouge-L by rep  prefix={prefix}, suffix={suffix}, offset={offset}',
                      fontsize=14, weight='bold')
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()


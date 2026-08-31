"""Cross-model sequence-level memorization overlap: do different attention variants
memorize the same Gutenberg excerpts, or different ones?

Every model reads the same unshuffled `rep_R_token.jsonl` (DistributedSampler(shuffle=False)
in evaluation/megatron_inference_sparse.py), so sample i is the same excerpt everywhere --
no book_id join needed, just matching sample order. PDM's `load_inference_data` (building
the Results pkl) restores that same order from the round-robin rank files, so
`Results.get_stats(...).scores` is already aligned -- no need to touch the raw jsonls.
"""

import matplotlib.pyplot as plt
import numpy as np

from attn_bench.plotting import model_registry
from attn_bench.plotting.data_loading import load_mem_results_scores
from attn_bench.plotting.utils import suptitle_centered


def load_sequence_matrix(rep, models=None, metric='Rouge-L', offset=0, prefix=500, suffix=500,
                         policy='greedy', results_base=model_registry.MEM_RESULTS_DIR):
    """{model_name: np.ndarray[n_sequences]} of `metric`, one repetition bucket -- sample
    order is shared across every model (see module docstring), so index i is the same
    excerpt everywhere.

    metric: any key from Results.get_all_metrics, e.g. 'Rouge-L', 'LCS', 'PPL', 'NLL',
            'Ref_PPL', 'Ref_NLL', 'TTR_ref', 'TTR_gen', 'token_acc', 'divergence_point'.
    Models missing this rep, or with a mismatched sequence count (breaks the shared-order
    assumption), are skipped with a note. Raises if fewer than 2 models end up with data.
    """
    models = list(models) if models else list(model_registry.MODELS)
    matrix = {}
    n_expected = None
    for key in models:
        exp_name = model_registry.MODELS[key]
        try:
            results = load_mem_results_scores(exp_name, results_base, prefix=prefix, suffix=suffix,
                                              offset=offset, policy=policy)
        except FileNotFoundError:
            print(f'Missing (not yet run): {exp_name}')
            continue
        if rep not in results.repetitions:
            print(f'No rep={rep} for {exp_name} (has {results.repetitions})')
            continue
        scores = np.asarray(results.get_stats(results.expr[0], rep, offset, prefix, suffix, metric).scores)
        if n_expected is None:
            n_expected = len(scores)
        elif len(scores) != n_expected:
            print(f'Skipping {exp_name}: {len(scores)} sequences, expected {n_expected} '
                  f'(rep={rep}) -- sample order would not line up with the other models')
            continue
        matrix[key] = scores

    if len(matrix) < 2:
        raise ValueError(f'Fewer than 2 models have rep={rep} data for offset={offset}, '
                         f'prefix={prefix}, suffix={suffix} -- nothing to compare')
    return matrix


def plot_sequence_overlap_heatmap(rep, models=None, metric='Rouge-L',
                                  offset=0, prefix=500, suffix=500, policy='greedy',
                                  results_base=model_registry.MEM_RESULTS_DIR,
                                  sort_by='mean', cmap='YlOrRd', vmin=0, vmax=1,
                                  figsize=None, dpi=150):
    """Raster heatmap: rows = models, columns = this rep bucket's sequences (sorted by
    `sort_by`), cell = `metric`. Reads as a pattern instead of n_models x n_sequences
    separate numbers -- a solid column (dark for every model) is a sequence every mechanism
    memorizes; an isolated dark cell is model-specific memorization.

    sort_by: 'mean' (consensus across models, descending -- default) | a model key (sort by
             that one model's score) | None (keep the underlying file's original order).
    Returns (mat [n_models, n_sequences], row_labels) for further inspection.
    """
    matrix = load_sequence_matrix(rep, models, metric, offset, prefix, suffix, policy, results_base)
    labels = list(matrix.keys())
    mat = np.stack([matrix[m] for m in labels])  # [n_models, n_sequences]

    if sort_by == 'mean':
        order = np.argsort(-mat.mean(axis=0))
    elif sort_by is None:
        order = np.arange(mat.shape[1])
    elif sort_by in labels:
        order = np.argsort(-matrix[sort_by])
    else:
        raise ValueError(f"sort_by must be 'mean', None, or one of {labels}, got {sort_by!r}")
    mat = mat[:, order]

    n_models, n_seqs = mat.shape
    fig, ax = plt.subplots(figsize=figsize or (14, n_models * 0.45 + 1.0), dpi=dpi)
    # aspect='auto' + interpolation='nearest': stretch to the figure box without blending
    # adjacent sequences into each other -- there's no meaningful value *between* columns.
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_yticks(range(n_models))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks([])  # 660 column ticks would be unreadable and meaningless -- read the pattern, not a column
    ax.set_xlabel(f'sequences (n={n_seqs}, sorted by {sort_by})', fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(metric, fontsize=10)

    suptitle_centered(fig, [ax], f'{metric} per sequence, rep={rep}  '
                                 f'(offset={offset}, prefix={prefix}, suffix={suffix})',
                      fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()
    return mat, labels

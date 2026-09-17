"""Precompute offset x prefix grid data (raw points + griddata-interpolated surface) for the
memorization dashboard, one JSON per (model, suffix), one entry per (repetition, metric). GitHub
Pages is static -- no scipy in the browser -- so this runs the same interpolation the notebook
plots use (load_offset_prefix_grid_data) once, ahead of time, instead of shipping the computation
client-side. The dashboard's suffix slider swaps in a different {model}__s{suffix}.json.

Nucleus results (`--policy nucleus-p0.95-n10`) go to {model}__s{suffix}__nucleus.json with the
per-cell values of every draw reduction (mean / single draw / max or min of the N draws by a
chosen metric) and NO grid: 12 reductions x 10 suffixes of 80x80 grids would be ~400 MB per
model, so the dashboard interpolates the (few) nucleus cells itself. Cells are whatever pkls
exist on disk.

Run from anywhere: `python3 attn_bench/dashboard/export_data.py`, optionally with model
names to re-export only those.
"""

import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)  # deterministic relative paths (PDM_DIR, sys.path)
                                            # regardless of where this script is invoked from

PDM_DIR = os.environ.get('PDM_DIR', '../../../PDM')
sys.path.append(f'{PDM_DIR}/src')
import verbatim_eval.controlled_expr as _ce
import verbatim_eval.rouge_ttr as _rt
import verbatim_eval.utils as _ut

sys.modules['controlled_expr'] = _ce  # needed for pickle deserialization
sys.modules['rouge_ttr'] = _rt
sys.modules['utils'] = _ut

sys.path.insert(0, '../..')
import numpy as np

from attn_bench.plotting import model_registry
from attn_bench.plotting.data_loading import (discover_offset_prefix_points,
                                              grid_axis,
                                              load_mem_results_scores_grid,
                                              load_offset_prefix_grid_data,
                                              start_finish_flags,
                                              start_finish_rates)

MODELS = ['full-scf1',  'swa-w4096-scf1', 'swa-w1024-scf1', 'swa-w256-scf1',
          'sink-scf1', 'gated-scf1', 'gdn', 'gdn-upd', 'gdn-xdl', 'gdn-xdl-xsl-0.5', 'gdn-xdl-xsl',
          'qwen', 'mla', 'kda', 'gemma']
REPS = [0, 1, 16, 32, 64, 128, 256]
# One JSON per suffix per model -- inference feasibility was defined at suffix=250, so the
# populated candidate points are the same set at every smaller suffix (a shorter suffix only
# opens up an unsampled sliver in the far corner). The 250 file preserves the old behaviour.
# 249 is the re-run of the same grid with every repetition bucket (0,1,2,4,...,256) in one
# job; the 25..150 boundaries here still come from the original suffix=250 run. A model
# that has no pkl at a suffix yet exports as an empty payload (all-NaN grid, 0 points), so
# the dashboard -- which fetches every model per suffix -- still loads.
SUFFIXES = [25, 50, 75, 100, 150, 249, 250]
MAX_DOC_LENGTH = 8192
GRID_RES = 80
# grid_scale='log' swaps ~6 of the linear grid nodes for explicit low nodes (down to grid_min,
# plus 50), so the dashboard's log-axis view has resolution in the low corner and the fill
# reaches exactly to the offset=0 / prefix=50 sample rows, without growing the mesh.

# (key, kwargs passed to load_offset_prefix_grid_data, post-transform on zs/GZ) -- one row per
# dashboard dropdown entry. post-transform is None or a numpy ufunc-like callable.
METRICS = [
    ('rouge_l', dict(metric='Rouge-L'), None),
    ('rouge_l_var', dict(metric='Rouge-L', stat='var'), None),
    ('lcs', dict(metric='lcs_norm'), None),
    ('ttr_gen', dict(metric='TTR_gen'), None),
    ('ttr_ref', dict(metric='TTR_ref'), None),
    ('exact_match', dict(metric='exact_match'), None),
    ('divergence_point', dict(metric='divergence_point'), None),
    # Passages reproducing >= 5 leading suffix tokens, and of those the fraction reproducing
    # the whole suffix (undefined -> null with no starters). The unconditional finish rate
    # is exact_match (lcs_norm == 1 <=> identical suffix).
    ('start_rate', dict(metric='start_rate'), None),
    ('conditional_finish_rate', dict(metric='conditional_finish_rate'), None),
    # Perplexity isn't stored directly -- exp() of the stored mean NLL.
    ('ppl_gen', dict(metric='gen_nll_mean'), 'exp'),
    ('ppl_ref', dict(metric='ref_nll_mean'), 'exp'),
    ('hayes_n10_p99', dict(metric='hayes', n=10, p=0.99), None),
    ('hayes_n10_p75', dict(metric='hayes', n=10, p=0.75), None),
    ('hayes_n10_p50', dict(metric='hayes', n=10, p=0.5), None),
    ('hayes_n10_p25', dict(metric='hayes', n=10, p=0.25), None),
]

# Same candidate grid the notebook validated for this exact model/rep/suffix combo
# (mem_plotting_style.ipynb, cells "6b59507c"/"882ed00b") -- kept in sync by hand since the
# notebook doesn't expose it as an importable constant.
CANDIDATE_OFFSETS = [0, 50, 150, 250, 500, 1000, 2000, 3971, 5942, 7892]
CANDIDATE_PREFIXES = [50, 250, 500, 1000, 2000, 3971, 5942, 7892]


def points_for_suffix(suffix):
    bound = MAX_DOC_LENGTH - suffix
    return sorted({(o, p) for o in CANDIDATE_OFFSETS for p in CANDIDATE_PREFIXES if o + p <= bound})


OUT_DIR = Path('data')

# Nucleus draw reductions: how the N draws of one document collapse to one value, before the
# mean over documents. 'single' is draw 0 (each draw is an iid sample, so one draw is "a random
# one"). max_/min_<key> pick, per document, the draw extremal in that metric and then report
# every metric of that draw. Keys are dashboard metric keys; the mapping to pkl metric names is
# METRICS above (only draw-level metrics can select).
SELECTION_METRICS = ['rouge_l', 'lcs', 'ttr_gen', 'ppl_gen', 'divergence_point']
REDUCTIONS = ['mean', 'single'] + [f'{how}_{key}' for how in ('max', 'min') for key in SELECTION_METRICS]
NUCLEUS_SUFFIXES = [25, 50, 75, 100, 150, 249, 250, 500, 750, 1000]


def _round(v):
    return None if v != v else round(float(v), 4)  # v != v -- NaN check, avoids importing numpy/math here


def _apply(transform, values):
    if transform is None:
        return values
    if transform == 'exp':
        import math
        return [v if v != v else math.exp(v) for v in values]  # math.exp(nan) is legal (-> nan)
    raise ValueError(f'unknown transform {transform!r}')


def export_model(model, suffix):
    points = points_for_suffix(suffix)
    # Read each pickle once, rather than once per metric and repetition.
    grid_data = load_mem_results_scores_grid(
        sorted({o for o, _ in points}), sorted({p for _, p in points}), [suffix], models=[model],
        results_base=model_registry.MEM_RESULTS_DIR, backend=('hf', 'megatron'))
    feasible_bound = MAX_DOC_LENGTH - suffix
    reps_out = {}
    for rep in REPS:
        # Coordinates are identical across every metric for a given (model, rep) -- computed
        # once per rep, not once per metric, to avoid repeating 80*80 + 80 + 80 floats 12x.
        coords_set = False
        rep_entry = {'metrics': {}}
        for metric_key, kwargs, transform in METRICS:
            offset_vals, prefix_vals, zs, G_OFFSET, G_PREFIX, GZ, _ = load_offset_prefix_grid_data(
                model, rep, suffix, grid_res=GRID_RES, max_doc_length=MAX_DOC_LENGTH, points=points,
                grid_scale='log', grid_min=20.0, interp='bilinear', grid_data=grid_data, **kwargs)
            if not coords_set:
                rep_entry['points_offset'] = [int(v) for v in offset_vals]
                rep_entry['points_prefix'] = [int(v) for v in prefix_vals]
                rep_entry['grid_offset'] = [_round(v) for v in G_OFFSET[0]]
                rep_entry['grid_prefix'] = [_round(v) for v in G_PREFIX[:, 0]]
                coords_set = True
            zs_t = _apply(transform, zs)
            gz_t = [_apply(transform, row) for row in GZ]
            rep_entry['metrics'][metric_key] = {
                'points_z': [_round(v) for v in zs_t],
                'z_grid': [[_round(v) for v in row] for row in gz_t],
                'n_points': int(np.isfinite(zs).sum()),
            }
        reps_out[rep] = rep_entry
    payload = {
        'model': model,
        'suffix': suffix,
        'max_doc_length': MAX_DOC_LENGTH,
        'feasible_bound': feasible_bound,
        'reps': reps_out,
    }
    out_path = OUT_DIR / f'{model}__s{suffix}.json'
    out_path.write_text(json.dumps(payload))
    print(f'{model} s{suffix}: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)')


def _draw_index(all_metrics, reduction):
    """Per-document draw index [docs] for a max_/min_ reduction, from the selecting metric's
    [docs, N] scores. None for mean/single (handled by the caller)."""
    how, key = reduction.split('_', 1)
    metric_name = dict((k, kw['metric']) for k, kw, _ in METRICS)[key]
    scores = np.asarray(all_metrics[metric_name].scores)
    # ppl_gen is stored as NLL (exp applied at display time): monotone, so argmax/argmin carry over.
    return scores.argmax(axis=1) if how == 'max' else scores.argmin(axis=1)


def _reduce(scores, reduction, idx):
    """[docs, N] -> [docs] under one reduction; [docs] (document-level) passes through."""
    scores = np.asarray(scores)
    if scores.ndim == 1:
        return scores
    if reduction == 'mean':
        return scores.mean(axis=1)
    if reduction == 'single':
        return scores[:, 0]
    return scores[np.arange(scores.shape[0]), idx]


def export_model_nucleus(model, suffix, policy):
    points = discover_offset_prefix_points(model, suffix, policy=policy)
    reps_out = {}
    if points:
        grid_data = load_mem_results_scores_grid(
            sorted({o for o, _ in points}), sorted({p for _, p in points}), [suffix], models=[model],
            results_base=model_registry.MEM_RESULTS_DIR, policy=policy, backend=('hf', 'megatron'))
    for rep in REPS:
        cells = []
        for offset, prefix in points:
            r = grid_data.get((offset, prefix, suffix), {}).get(model)
            if r is None or rep not in r.repetitions:
                continue
            all_metrics = r.get_all_metrics(r.expr[0], rep, offset, prefix, suffix)
            values = {}
            for reduction in REDUCTIONS:
                idx = _draw_index(all_metrics, reduction) if reduction not in ('mean', 'single') else None
                out = {}
                for metric_key, kwargs, transform in METRICS:
                    if kwargs['metric'] == 'hayes':
                        # Analytic, from the true suffix only -- the same for every draw.
                        p_z = np.asarray(all_metrics['p_z'].scores)
                        out[metric_key] = float((1 - (1 - p_z) ** kwargs['n'] >= kwargs['p']).mean())
                    elif kwargs['metric'] in ('start_rate', 'conditional_finish_rate'):
                        # Per-draw indicators, reduced like any score: mean -> per-document
                        # fraction of the N draws, single/max/min -> the chosen draw's indicator.
                        flags = start_finish_flags(all_metrics['divergence_point'].scores, suffix)
                        out[metric_key] = start_finish_rates(*(_reduce(f, reduction, idx) for f in flags))[kwargs['metric']]
                    elif kwargs['metric'] in all_metrics:
                        per_doc = _reduce(all_metrics[kwargs['metric']].scores, reduction, idx)
                        z = float(per_doc.var() if kwargs.get('stat') == 'var' else per_doc.mean())
                        out[metric_key] = _apply(transform, [z])[0]
                values[reduction] = out
            cells.append((offset, prefix, values))
        reps_out[rep] = {
            'points_offset': [c[0] for c in cells],
            'points_prefix': [c[1] for c in cells],
            'reductions': {red: {key: [_round(c[2][red].get(key, float('nan'))) for c in cells]
                                 for key, _, _ in METRICS}
                           for red in REDUCTIONS},
        }
    axis = grid_axis(GRID_RES, MAX_DOC_LENGTH, 'log', 20.0)
    payload = {
        'model': model,
        'suffix': suffix,
        'policy': policy,
        'max_doc_length': MAX_DOC_LENGTH,
        'feasible_bound': MAX_DOC_LENGTH - suffix,
        'grid_axis': [_round(v) for v in axis],
        'reductions': REDUCTIONS,
        'reps': reps_out,
    }
    out_path = OUT_DIR / f'{model}__s{suffix}__nucleus.json'
    out_path.write_text(json.dumps(payload))
    print(f'{model} s{suffix} {policy}: {out_path} ({out_path.stat().st_size / 1024:.0f} KB, '
          f'{len(points)} cells)')


if __name__ == '__main__':
    # Optional model filter -- a full re-export is ~98 files x ~4 MB, so pass the models you
    # actually changed when adding one: `python3 export_data.py gemma`. `--suffixes 249`
    # (comma-separated) likewise restricts the suffixes, e.g. after adding a new one.
    args = sys.argv[1:]
    policy = 'greedy'
    if '--policy' in args:
        i = args.index('--policy')
        policy = args[i + 1]
        del args[i:i + 2]
    suffixes = SUFFIXES if policy == 'greedy' else NUCLEUS_SUFFIXES
    if '--suffixes' in args:
        i = args.index('--suffixes')
        suffixes = [int(v) for v in args[i + 1].split(',')]
        del args[i:i + 2]
        known = SUFFIXES if policy == 'greedy' else NUCLEUS_SUFFIXES
        unknown_s = [v for v in suffixes if v not in known]
        if unknown_s:
            raise SystemExit(f'unknown suffix(es) {unknown_s} -- known: {known}')
    selected = args or MODELS
    unknown = [m for m in selected if m not in MODELS]
    if unknown:
        raise SystemExit(f'unknown model(s) {unknown} -- known: {MODELS}')
    OUT_DIR.mkdir(exist_ok=True)
    for suffix in suffixes:
        for model in selected:
            if policy == 'greedy':
                export_model(model, suffix)
            else:
                export_model_nucleus(model, suffix, policy)

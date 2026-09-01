"""Precompute offset x prefix grid data (raw points + griddata-interpolated surface) for the
memorization dashboard, one JSON per (model, suffix), one entry per (repetition, metric). GitHub
Pages is static -- no scipy in the browser -- so this runs the same interpolation the notebook
plots use (load_offset_prefix_grid_data) once, ahead of time, instead of shipping the computation
client-side. The dashboard's suffix slider swaps in a different {model}__s{suffix}.json.

Run from anywhere: `python3 attn_bench/dashboard/export_data.py`.
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
from attn_bench.plotting.data_loading import load_offset_prefix_grid_data

MODELS = ['full-scf1', 'swa-w4096-scf1', 'swa-w1024-scf1', 'swa-w256-scf1',
          'sink-scf1', 'gated-scf1', 'gdn', 'gdn-xdl', 'gdn-xdl-xsl-0.5', 'gdn-xdl-xsl']
REPS = [0, 1, 16, 32, 64, 128, 256]
# One JSON per suffix per model -- inference feasibility was defined at suffix=250, so the
# populated candidate points are the same set at every smaller suffix (a shorter suffix only
# opens up an unsampled sliver in the far corner). The 250 file preserves the old behaviour.
SUFFIXES = [25, 50, 75, 100, 150, 250]
MAX_DOC_LENGTH = 8192
GRID_RES = 80
# grid_scale='log' swaps ~6 of the linear grid nodes for explicit low nodes (down to grid_min,
# plus 50), so the dashboard's log-axis view has resolution in the low corner and the fill
# reaches exactly to the offset=0 / prefix=50 sample rows, without growing the mesh.

# (key, kwargs passed to load_offset_prefix_grid_data, post-transform on zs/GZ) -- one row per
# dashboard dropdown entry. post-transform is None or a numpy ufunc-like callable.
METRICS = [
    ('rouge_l', dict(metric='Rouge-L'), None),
    ('lcs', dict(metric='lcs_norm'), None),
    ('ttr_gen', dict(metric='TTR_gen'), None),
    ('ttr_ref', dict(metric='TTR_ref'), None),
    ('exact_match', dict(metric='exact_match'), None),
    ('divergence_point', dict(metric='divergence_point'), None),
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
                grid_scale='log', grid_min=20.0, interp='bilinear', **kwargs)
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
                'n_points': len(offset_vals),
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


if __name__ == '__main__':
    OUT_DIR.mkdir(exist_ok=True)
    for suffix in SUFFIXES:
        for model in MODELS:
            export_model(model, suffix)

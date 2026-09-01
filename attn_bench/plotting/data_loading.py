"""Loaders for every results family"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, griddata

from attn_bench.plotting import model_registry


def _resolve_models(models):
    """None or empty -> every model in model_registry.MODELS, in registry order."""
    return list(models) if models else list(model_registry.MODELS)


def select_models(loaded, models=None):
    """Filter an already-loaded {label: ...} dict down to `models`, in that order (default:
    every label as loaded). Every plot function takes its data already loaded, plus this
    same `models=` filter -- picking a subset never means reloading."""
    if models is None:
        return loaded
    return {m: loaded[m] for m in models if m in loaded}


# --- mem-results (Rouge-L / LCS / PPL / NLL / TTR sweeps) ---

def load_mem_results_scores(exp_name, results_base, prefix=500, suffix=500, offset=0, policy='greedy'):
    """Load one (offset, prefix, suffix) pickle for one model. Returns a `Results` object:
        r.repetitions                       # [0, 1, 2, 4, 8, ..., 256] -- reps this pickle covers
        stats = r.get_stats(expr=r.expr[0], rep=32, offset=0, prefix=500, suffix=500, metric='Rouge-L')
        stats.scores                        # np.ndarray, one score per sample
        stats.mean, stats.std               # floats, across all samples in this (rep, offset, prefix, suffix)

    Metrics: Rouge-L, LCS, match_25/50/75, exact_match, divergence_point, token_acc, PPL/NLL/TTR (plain name = generation, Ref_ prefix = reference).
    """
    # Imported here, not at module level: this is the only function in the file that needs
    # PDM/verbatim_eval, so everything else (doc lengths, long-inference, attention patterns)
    # stays usable without it on the path.
    from verbatim_eval.controlled_expr import Results
    path = f'{results_base}/{exp_name}/metrics/offset_{offset}_prefix_{prefix}_suffix_{suffix}_{policy}.pkl'
    return Results.load(path)


def load_mem_results_scores_grid(offsets, prefixes, suffixes, models=None,
                                 results_base=model_registry.MEM_RESULTS_DIR, policy='greedy',
                                 backend='megatron'):
    """Load every (offset, prefix, suffix) combo for every model in `models` (default: every
    model in the registry).

    backend: a single string pins one backend; a list/tuple (e.g. ('hf', 'megatron')) tries
    each in order per cell, independently -- so a model with partial backend coverage can end
    up sourced from different backends across cells in the same grid.

    Returns {(offset, prefix, suffix): {model_name: Results}}, only combos with a file found.
    """
    models = _resolve_models(models)
    backends = (backend,) if isinstance(backend, str) else tuple(backend)
    grid = {}
    for offset in offsets:
        for prefix in prefixes:
            for suffix in suffixes:
                key = (offset, prefix, suffix)
                per_model = {}
                for name in models:
                    for b in backends:
                        try:
                            per_model[name] = load_mem_results_scores(
                                model_registry.model_folder(name, b), results_base,
                                offset=offset, prefix=prefix, suffix=suffix, policy=policy)
                            break
                        except FileNotFoundError:
                            continue
                if per_model:
                    grid[key] = per_model
    return grid


def load_mem_results_scores_grid_by_name(offsets, prefixes, suffixes, exp_names,
                                         results_base=model_registry.MEM_RESULTS_DIR, policy='greedy'):
    """Like load_mem_results_scores_grid, but for arbitrary {label: exp_name} pairs (e.g.
    explicit per-backend labels built with model_registry.model_folder) instead of models
    resolved through model_registry.MODELS + a backend priority list -- for plots that want
    two backends of the same checkpoint shown side by side rather than one falling back to
    the other. Missing experiments are skipped, same as load_mem_results_by_name.

    Returns {(offset, prefix, suffix): {label: Results}} -- only combos with at least one
    label's file on disk are included.
    """
    grid = {}
    for offset in offsets:
        for prefix in prefixes:
            for suffix in suffixes:
                key = (offset, prefix, suffix)
                per_model = {}
                for label, exp_name in exp_names.items():
                    try:
                        per_model[label] = load_mem_results_scores(
                            exp_name, results_base, offset=offset, prefix=prefix, suffix=suffix, policy=policy)
                    except FileNotFoundError:
                        pass
                if per_model:
                    grid[key] = per_model
    return grid


def discover_offset_prefix_points(models, suffix, results_base=model_registry.MEM_RESULTS_DIR,
                                  backend=('hf', 'megatron'), policy='greedy'):
    """Every (offset, prefix) with an existing suffix=`suffix` pkl for any of `models`,
    scanning each model's metrics/ dir across every backend folder (model_registry.model_folder)
    and taking the union. Feed the result's unique offsets/prefixes into
    load_mem_results_scores_grid, or the pairs themselves into a dim='diagonal' panel plot."""
    if isinstance(models, str):
        models = [models]
    backends = (backend,) if isinstance(backend, str) else tuple(backend)
    pat = re.compile(rf'offset_(\d+)_prefix_(\d+)_suffix_{suffix}_{policy}\.pkl$')
    points = set()
    for model in models:
        for b in backends:
            metrics_dir = Path(results_base) / model_registry.model_folder(model, b) / 'metrics'
            if not metrics_dir.exists():
                continue
            for f in metrics_dir.iterdir():
                m = pat.match(f.name)
                if m:
                    points.add((int(m.group(1)), int(m.group(2))))
    return sorted(points)


def load_offset_prefix_grid_data(model, rep, suffix, backends=('hf', 'megatron'), metric='Rouge-L',
                                 interp='linear', grid_res=80, max_doc_length=8192, points=None,
                                 n=None, p=None, grid_scale='linear', grid_min=30.0, smooth_sigma=0.0):
    """(offset, prefix) -> metric points for one (model, rep, suffix), interpolated onto a
    regular grid over the full document-length domain, masked to the feasible region
    (offset+prefix+suffix<=max_doc_length). Shared by every offset x prefix 2D/3D panel.

    points: explicit [(offset, prefix), ...] to load (default: discovered on disk).
    interp: 'linear' / 'cubic' go through scipy.griddata (Delaunay); 'bilinear' interpolates on
    the candidate (offset x prefix) lattice directly -- smoother than 'linear' (no arbitrary
    per-quad diagonal crease) while still exact at every sample, and no overshoot like 'cubic'.
    smooth_sigma>0 additionally runs a NaN-aware Gaussian over the grid (sigma in grid cells);
    it softens edges but pulls the surface off the sample values, so keep it small or 0.
    metric='hayes': Hayes et al. 2025 (n, p)-discoverable extraction rate, computed on the fly.
    grid_scale='log' keeps the full linear grid (so a linear-axis view still gets a clean
    diagonal along the feasible boundary) and adds log-spaced nodes from grid_min up through
    the low corner -- a plain linear grid's ~100-token first cell hides every sample below it
    on a log axis.

    Returns (offset_vals, prefix_vals, zs, G_OFFSET, G_PREFIX, GZ, feasible_bound).
    """
    if metric == 'hayes' and (n is None or p is None):
        raise ValueError("metric='hayes' requires n and p")
    if points is None:
        points = discover_offset_prefix_points(model, suffix, backend=backends)
    offsets = sorted({o for o, prefix in points})
    prefixes = sorted({prefix for o, prefix in points})
    grid_data = load_mem_results_scores_grid(offsets, prefixes, [suffix], models=[model],
                                             results_base=model_registry.MEM_RESULTS_DIR, backend=list(backends))
    offset_vals, prefix_vals, zs = [], [], []
    for offset, prefix in points:
        key = (offset, prefix, suffix)
        if key not in grid_data or model not in grid_data[key]:
            continue
        r = grid_data[key][model]
        expr = r.expr[0]
        try:
            if metric == 'hayes':
                p_z = np.array(r.get_stats(expr, rep, offset, prefix, suffix, 'p_z').scores)
                z = float((1 - (1 - p_z) ** n >= p).mean())
            else:
                z = r.get_all_metrics(expr, rep, offset, prefix, suffix)[metric].mean
        except KeyError:
            continue
        offset_vals.append(offset); prefix_vals.append(prefix); zs.append(z)
    offset_vals, prefix_vals, zs = np.array(offset_vals), np.array(prefix_vals), np.array(zs)

    feasible_bound = max_doc_length - suffix
    if grid_scale == 'log':
        # Geometric spacing in the low decades so the contour fill tracks the (log-dense)
        # sample points there instead of stretching one wide cell across them, then linear
        # spacing out to the edge (keeps the boundary diagonal clean in a linear-axis view).
        # 50 is forced in so the fill reaches exactly to the prefix=50 sample row; the grid
        # also starts at grid_min (= the dashboard's offset=0 park position). Total ~= grid_res.
        low = np.geomspace(grid_min, 550.0, 13)
        high = np.linspace(550.0, max_doc_length, grid_res - low.size)
        # 0 kept so the fill carries the true offset=0 values (the dashboard parks the offset=0
        # dots on that column); 50 forced so the fill reaches exactly to the prefix=50 row.
        axis = np.union1d(np.union1d(low, high), [0.0, 50.0])
    else:
        axis = np.linspace(0, max_doc_length, grid_res)
    g_prefix = axis
    g_offset = axis
    G_OFFSET, G_PREFIX = np.meshgrid(g_offset, g_prefix)

    if interp == 'bilinear':
        # Bilinear on the candidate (offset x prefix) lattice itself. Unlike griddata's
        # Delaunay 'linear' -- which splits every lattice quad along an arbitrary diagonal and
        # leaves a visible crease -- this warps each quad as one smooth patch, while still
        # passing exactly through every sample (so a dot's colour matches the fill under it).
        ox, py = np.unique(offset_vals), np.unique(prefix_vals)
        lattice = np.full((ox.size, py.size), np.nan)
        ix = {v: i for i, v in enumerate(ox)}
        iy = {v: i for i, v in enumerate(py)}
        for o, p, zv in zip(offset_vals, prefix_vals, zs):
            lattice[ix[o], iy[p]] = zv
        holes = np.isnan(lattice)          # infeasible / never-run (offset, prefix) combos
        if holes.any():
            mo, mp = np.meshgrid(ox, py, indexing='ij')
            lattice[holes] = griddata((offset_vals, prefix_vals), zs, (mo[holes], mp[holes]), method='linear')
            edge = np.isnan(lattice)
            if edge.any():
                lattice[edge] = griddata((offset_vals, prefix_vals), zs, (mo[edge], mp[edge]), method='nearest')
        rgi = RegularGridInterpolator((ox, py), lattice, method='linear', bounds_error=False, fill_value=None)
        GZ = rgi(np.stack([G_OFFSET.ravel(), G_PREFIX.ravel()], axis=-1)).reshape(G_OFFSET.shape)
        # RGI has no convex hull -- restrict to the same region griddata('linear') would cover.
        GZ[np.isnan(griddata((offset_vals, prefix_vals), zs, (G_OFFSET, G_PREFIX), method='linear'))] = np.nan
    else:
        GZ = griddata((offset_vals, prefix_vals), zs, (G_OFFSET, G_PREFIX), method=interp)
    GZ[G_OFFSET + G_PREFIX > feasible_bound] = np.nan

    if smooth_sigma:
        from scipy.ndimage import gaussian_filter
        valid = ~np.isnan(GZ)
        num = gaussian_filter(np.where(valid, GZ, 0.0), smooth_sigma)
        den = gaussian_filter(valid.astype(float), smooth_sigma)
        GZ = np.where(valid, num / np.maximum(den, 1e-9), np.nan)

    return offset_vals, prefix_vals, zs, G_OFFSET, G_PREFIX, GZ, feasible_bound


def list_reps_with_mem_results_text(exp_name, results_base, prefix, suffix, offset=0, policy='greedy'):
    """Repetition values that have raw-text records on disk (load_mem_results_text_records)."""
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


def load_mem_results_text_records(exp_name, rep, results_base, prefix, suffix, offset, policy):
    """One repetition's raw per-sample records: prefix/true_suffix/generated_suffix token ids
    plus per-sample metric values. The only source for the actual text -- for metric numbers
    alone, use load_mem_results_scores(...).get_stats(...).scores instead.

    Samples are written by parallel ranks in round-robin order (sample i -> rank i %
    world_size), so records are reassembled into the original sample order here."""
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


def load_mem_results_by_name(exp_names, results_base, prefix=500, suffix=500, offset=0, policy='greedy'):
    """Load one (offset, prefix, suffix) pickle per {label: exp_name} pair -- for arbitrary
    labels that aren't model_registry keys, e.g. inference-time sink-scale variants that
    don't have their own MODELS entry. Missing experiments are skipped with a printed note.

    Returns {label: Results}.
    """
    out = {}
    for label, exp_name in exp_names.items():
        try:
            out[label] = load_mem_results_scores(exp_name, results_base, prefix=prefix, suffix=suffix,
                                                 offset=offset, policy=policy)
        except FileNotFoundError:
            print(f'Missing (not yet run): {exp_name} with prefix={prefix}, suffix={suffix}, offset={offset}')
    return out


def load_trained_sink_offsets(exp_names, results_base):
    """Trained per-head softmax_offset (pre-patch value, same across every sscale variant
    of a model -- any one file works), from the first sink_scale_metadata.json found in
    exp_names. Only informative for learnable sink -- off-by-one's offset is fixed at 0.
    Returns (offsets [n_layers, n_heads], label), or (None, None) if nothing's found yet."""
    for label, exp_name in exp_names.items():
        p = Path(results_base) / exp_name / 'sink_scale_metadata.json'
        if p.exists():
            offsets = np.array(json.loads(p.read_text())['original_softmax_offset'])
            if offsets.ndim == 1:
                offsets = offsets[:, None]
            return offsets, label
    return None, None


def load_generation_quality(exp_names, results_base, offset, prefix, suffix, policy='greedy'):
    """{label: {'perplexity': {rep: value}, 'distinct_n': {rep: {n: value}}}} from the JSON
    files compute_generation_quality.py writes alongside a mem-results experiment. Labels
    with neither file are skipped."""
    stem = f'offset_{offset}_prefix_{prefix}_suffix_{suffix}_{policy}'
    out = {}
    for label, exp_name in exp_names.items():
        d = Path(results_base) / exp_name
        ppl_f = d / f'{stem}_perplexity.json'
        dn_f = d / f'{stem}_distinct_n.json'
        entry = {}
        if ppl_f.exists():
            entry['perplexity'] = {int(k): v for k, v in json.loads(ppl_f.read_text()).items()}
        if dn_f.exists():
            entry['distinct_n'] = {int(k): v for k, v in json.loads(dn_f.read_text()).items()}
        if entry:
            out[label] = entry
    return out


# --- long-inference (position-wise NLL, GDN state norm) ---

def load_gutenberg_long_lengths(path=model_registry.GUTENBERG_LONG_LENGTHS_PATH):
    """One row per book per repetition bucket (extra_prefix_len/extra_suffix_len etc.), for
    long_inference.plot_coverage. Returns (df, buckets) -- buckets is the sorted list of
    bucket_rep values present."""
    df = pd.DataFrame([json.loads(l) for l in Path(path).open()])
    buckets = sorted(int(b) for b in df["bucket_rep"].unique())
    return df, buckets


def _long_inference_rep_path(config_dir, rep):
    return Path(config_dir) / f"rep_{rep}.npz"


def _long_inference_state_path(config_dir, rep):
    return Path(config_dir) / f"rep_{rep}_state.npz"


def load_long_inference_nll(path):
    """One rep's position-wise NLL. Returns {position, mean, std, count, seq_len}, each an
    array over token position except seq_len (int)."""
    d = np.load(path)
    cnt = d["count"]
    mean = d["nll_sum"] / cnt
    var = np.maximum(d["nll_sqsum"] / cnt - mean ** 2, 0.0)  # clamp fp noise below 0
    return {
        "position": d["position"],
        "mean": mean,
        "std": np.sqrt(var),
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def _load_grid(results_by_label, reps, path_fn, loader_fn):
    grid = {}
    for label, entry in results_by_label.items():
        per_rep = {}
        for rep in reps:
            p = path_fn(entry, rep)
            if Path(p).exists():
                per_rep[rep] = loader_fn(p)
        if per_rep:
            grid[label] = per_rep
    return grid


def load_long_inference_nll_grid(reps, models=None, config=model_registry.LONG_INFERENCE_CONFIG,
                                 results_base=model_registry.LONG_GUTENBERG_RESULTS_DIR):
    """Load `load_long_inference_nll` for every (model, rep) found on disk, for
    `models` (a list of names from model_registry.MODELS; default/empty -> every model in the
    registry) under one `config` (capture directory name).

    Returns {label: {rep: load_long_inference_nll(...)}} -- only combos found on disk.
    For FineWeb (no rep buckets, just seen/unseen), use `load_long_fineweb_inference_nll_grid` instead.
    """
    models = _resolve_models(models)
    results_by_label = {m: Path(results_base) / model_registry.MODELS[m] / config for m in models}
    return _load_grid(results_by_label, reps, _long_inference_rep_path, load_long_inference_nll)


def load_long_inference_nll_by_config(model_key, configs, reps, results_base=model_registry.LONG_GUTENBERG_RESULTS_DIR):
    """load_long_inference_nll for one model under several named configs -- the inverse
    axis from load_long_inference_nll_grid (which varies model at one fixed config). For
    comparing two capture configs of the same model (e.g. different TP degree, or an old
    vs redone run). configs: {label: config_dir_name}.

    Returns {label: {rep: load_long_inference_nll(...)}}.
    """
    base_folder = Path(results_base) / model_registry.MODELS[model_key]
    results_by_label = {label: base_folder / config for label, config in configs.items()}
    return _load_grid(results_by_label, reps, _long_inference_rep_path, load_long_inference_nll)


def load_long_inference_state_norm(path):
    """One rep's GDN state norm, collapsed over layers and heads. Returns {position, mean,
    std, count, seq_len} -- mean/std are the per-boundary mean/std across the per-(layer,
    head) sequence-mean norms (the spread across layers/heads, not across sequences)."""
    d = np.load(path)
    cnt = d["count"]                                   # [num_boundaries]
    per_lh = d["norm_sum"] / cnt[None, :, None]        # [layer, boundary, head] mean over seqs
    flat = per_lh.transpose(1, 0, 2).reshape(per_lh.shape[1], -1)  # [boundary, layer*head]
    return {
        "position": d["boundary"],
        "mean": flat.mean(axis=1),
        "std": flat.std(axis=1),
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def load_long_inference_state_norm_by_layer(path):
    """One rep's GDN state norm, collapsed over heads only. Returns {position, layer, mean,
    count, seq_len} -- mean has shape [layer, boundary]."""
    d = np.load(path)
    cnt = d["count"]                                    # [num_boundaries]
    num_heads = d["norm_sum"].shape[-1]
    mean = d["norm_sum"].sum(axis=-1) / (num_heads * cnt[None, :])  # [layer, boundary]
    return {
        "position": d["boundary"],
        "layer": d["layer"],
        "mean": mean,
        "count": cnt,
        "seq_len": int(d["seq_len"]),
    }


def load_long_inference_state_norm_grid(reps, models=None, config=model_registry.LONG_INFERENCE_CONFIG,
                                        results_base=model_registry.LONG_GUTENBERG_RESULTS_DIR):
    """Same `reps`/`models`/`config`/`results_base` convenience as `load_long_inference_nll_grid`,
    for state norm."""
    models = _resolve_models(models)
    results_by_label = {m: Path(results_base) / model_registry.MODELS[m] / config for m in models}
    return _load_grid(results_by_label, reps, _long_inference_state_path, load_long_inference_state_norm)


def load_long_inference_state_norm_by_layer_grid(rep, models=None, config=model_registry.LONG_INFERENCE_CONFIG,
                                                 results_base=model_registry.LONG_GUTENBERG_RESULTS_DIR):
    """Load `load_long_inference_state_norm_by_layer` for one `rep`, for `models` (a list of
    names from model_registry.MODELS; default/empty -> every model in the registry).

    Returns {label: load_long_inference_state_norm_by_layer(...)} -- only models with a
    state file for this rep are included.
    """
    models = _resolve_models(models)
    out = {}
    for m in models:
        p = _long_inference_state_path(Path(results_base) / model_registry.MODELS[m] / config, rep)
        if p.exists():
            out[m] = load_long_inference_state_norm_by_layer(p)
    return out


def load_long_fineweb_inference_nll_grid(partitions=('seen', 'unseen'), models=None,
                                         config=model_registry.LONG_FINEWEB_CONFIG,
                                         key=model_registry.LONG_FINEWEB_KEY,
                                         results_base=model_registry.LONG_FINEWEB_RESULTS_DIR):
    """Load `load_long_inference_nll` for every (model, partition) found on disk, for
    `models` (a list of names from model_registry.MODELS; default/empty -> every model in the
    registry). FineWeb has no repetition buckets -- `partitions` ('seen'/'unseen') plays the
    same per-cell role `reps` plays for Gutenberg.

    Returns {label: {partition: load_long_inference_nll(...)}}.
    """
    models = _resolve_models(models)
    results_by_label = {m: model_registry.MODELS[m] for m in models}

    def path_fn(model_folder, partition):
        return (Path(results_base) / model_folder / model_registry.LONG_FINEWEB_PARTITION_DIRS[partition]
                / config / f"{key}.npz")

    return _load_grid(results_by_label, partitions, path_fn, load_long_inference_nll)


# --- long individual sequence (per-sequence NLL / true-token-rank) ---

_SEQ_ID_RE = re.compile(r'"seq_id":\s*"([^"]*)"')


def list_long_individual_sequence_ids(path):
    """Every seq_id in a *_individual.jsonl, in file order -- a cheap regex pass over just
    the seq_id field, without parsing each line's full per-position arrays."""
    ids = []
    with open(path) as f:
        for line in f:
            m = _SEQ_ID_RE.search(line)
            if m:
                ids.append(m.group(1))
    return ids


def load_long_individual_sequence(path, limit=None, seq_ids=None):
    """Read one *_individual.jsonl into {seq_id: record}. Record fields:
        `idx`, `seq_id`, `length` (int),
        `nll`, `true_token`, `true_token_rank`, `argmax_token` (per-position lists, one entry per token).

    limit: stop after this many sequences.
    seq_ids: only keep sequences with these ids.
    """
    records = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            if seq_ids is not None and not any(f'"seq_id": "{sid}"' in line for sid in seq_ids):
                continue
            r = json.loads(line)
            records[r["seq_id"]] = r
            if seq_ids is not None and len(records) == len(seq_ids):
                break
    return records


def load_long_individual_sequence_grid(rep, models=None, config=model_registry.LONG_INFERENCE_CONFIG,
                                       results_base=model_registry.LONG_GUTENBERG_RESULTS_DIR, **kwargs):
    """Load `load_long_individual_sequence` for one `rep`, for `models` (a list of
    names from model_registry.MODELS; default/empty -> every model in the registry) under one
    `config` (capture directory name). `kwargs` (limit, seq_ids) are passed through to
    every load.

    Returns {label: {seq_id: record}}. When results_base/MODELS[m]/config doesn't fit the
    layout, build a {label: path} dict by hand and call `load_long_individual_sequence`
    yourself in a loop instead of this convenience wrapper."""
    models = _resolve_models(models)
    paths = {m: Path(results_base) / model_registry.MODELS[m] / config / f"rep_{rep}_individual.jsonl"
             for m in models}
    return {label: load_long_individual_sequence(path, **kwargs) for label, path in paths.items()}


# --- attention patterns (query x key maps, gating histograms) ---

N_MEM_BUCKETS = 10  # Rouge-L score is split into 10 buckets: '00-01', '01-02', ..., '09-10'


def mem_bucket_label(bi):
    return f"{bi:02d}-{bi + 1:02d}"


ALL_MEM_BUCKETS = [mem_bucket_label(bi) for bi in range(N_MEM_BUCKETS)]


def _load_attention_pattern_mem_bucket(exp_base, model_name, kind, mem_bucket, offset=0, prefix_len=500, suffix_len=50):
    """One mem-bucket's (Rouge-L score range, not repetition count) average attention map.
    Each GPU rank only saw some samples during inference. This combines all ranks into one true average.

    `kind`: 'attn_scores' or 'norm_attn'.
    Returns {mean: [L,H,S,S] float32, count: int, prompt_len: int}."""
    if kind not in ("attn_scores", "norm_attn"):
        raise ValueError(f"kind must be 'attn_scores' or 'norm_attn', got {kind!r}")
    label = mem_bucket if isinstance(mem_bucket, str) else mem_bucket_label(mem_bucket)
    d = (Path(exp_base) / model_name / "inference"
         / f"offset_{offset}_prefix_{prefix_len}_suffix_{suffix_len}")
    rank_files = sorted(d.glob(f"{kind}_rouge_l_{label}_rank*.npz"))
    if not rank_files:
        raise FileNotFoundError(f"No {kind}_rouge_l_{label}_rank*.npz in {d}")

    weighted_sum = None
    total = 0
    prompt_len = None
    for f in rank_files:
        npz = np.load(f)
        c = int(npz["count"])
        prompt_len = int(npz["prompt_len"])
        if c == 0:
            continue
        contrib = npz["mean"].astype(np.float32) * c
        weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib
        total += c

    if weighted_sum is None or total == 0:
        ref = np.load(rank_files[0])
        mean = np.zeros_like(ref["mean"], dtype=np.float32)
    else:
        mean = weighted_sum / total
    return {"mean": mean, "count": total, "prompt_len": prompt_len}


def load_attention_patterns(exp_base, model_name, kind, offset=0, prefix_len=500, suffix_len=50):
    """{label: _load_attention_pattern_mem_bucket(...)} for every one of the ALL_MEM_BUCKETS
    (Rouge-L score ranges) that has files."""
    out = {}
    for label in ALL_MEM_BUCKETS:
        try:
            out[label] = _load_attention_pattern_mem_bucket(exp_base, model_name, kind, label,
                                                             offset, prefix_len, suffix_len)
        except FileNotFoundError:
            pass
    return out


def load_attention_patterns_grid(kind, models=None, exp_base=model_registry.MEM_RESULTS_DIR_OLD,
                                 offset=0, prefix_len=500, suffix_len=50):
    """Load `load_attention_patterns` for every model in `models` (a list of names
    from model_registry.MODELS; default/empty -> every model in the registry). Returns
    {model_name: load_attention_patterns(...)}.

    Attention-pattern captures currently only exist for the 4 base models (full/gated/
    learn-sink/off-by-one) against the older MEM_RESULTS_BASE_OLD data version -- any other
    model in the default full-registry list just comes back with an empty dict.
    """
    models = _resolve_models(models)
    return {m: load_attention_patterns(exp_base, model_registry.MODELS[m], kind, offset, prefix_len, suffix_len)
            for m in models}


def load_attention_gating(exp_base, model_name, offset=0, prefix_len=500, suffix_len=50):
    """Sum-merged per-rank gating histograms (gated model only).
    Returns {hist: [n_buckets,L,H,n_bins] int64, bin_edges: [n_bins+1], count: [n_buckets] int64}."""
    d = (Path(exp_base) / model_name / "inference"
         / f"offset_{offset}_prefix_{prefix_len}_suffix_{suffix_len}")
    rank_files = sorted(d.glob("gating_scores_rank*.npz"))
    if not rank_files:
        raise FileNotFoundError(f"No gating_scores_rank*.npz in {d}")

    hist = None
    count = None
    edges = None
    for f in rank_files:
        npz = np.load(f)
        edges = npz["bin_edges"]
        h = npz["hist"].astype(np.int64)
        c = npz["count"].astype(np.int64)
        hist = h if hist is None else hist + h
        count = c if count is None else count + c
    return {"hist": hist, "bin_edges": edges, "count": count}


def load_weighted_avg_attention_patterns(mem_buckets):
    """`load_attention_patterns` gives one average attention map per mem-bucket (10 Rouge-L score ranges).
    This combines all of them into a single average over every sample, weighting each bucket's average by how many samples it had.

    Returns {mean: [L,H,S,S] float32, count: int, prompt_len: int}."""
    acc = None
    total = 0
    prompt_len = None
    for b in mem_buckets.values():
        c = int(b["count"])
        prompt_len = int(b["prompt_len"])
        if c == 0:
            continue
        contrib = b["mean"].astype(np.float64) * c
        acc = contrib if acc is None else acc + contrib
        total += c
    if acc is None or total == 0:
        ref = next(iter(mem_buckets.values()))["mean"]
        return {"mean": np.zeros_like(ref, dtype=np.float32), "count": 0, "prompt_len": prompt_len}
    return {"mean": (acc / total).astype(np.float32), "count": total, "prompt_len": prompt_len}


# --- doc lengths (document / packed-chunk length distributions) ---

def load_doc_lengths(data_dir=model_registry.DOC_LENGTHS_DIR):
    """{dataset_name: np.ndarray of per-document token lengths} for every *.npy in data_dir."""
    return {p.stem: np.load(p) for p in sorted(Path(data_dir).glob("*.npy"))}


def load_packed_chunk_length_hist(chunk_data_dir=model_registry.PACKED_CHUNK_LENGTHS_DIR):
    """{dataset_name: (hist, job_id)} for every *_chunk_len_hist.npy in chunk_data_dir.
    hist[i] = number of packed chunks with length i+1 (1-indexed lengths). job_id is None
    if no matching *_run_metadata.json exists."""
    out = {}
    chunk_data_dir = Path(chunk_data_dir)
    for p in sorted(chunk_data_dir.glob("*_chunk_len_hist.npy")):
        name = p.stem.removesuffix("_chunk_len_hist")
        hist = np.load(p)
        meta_path = chunk_data_dir / f"{name}_run_metadata.json"
        job_id = json.loads(meta_path.read_text())["job_id"] if meta_path.exists() else None
        out[name] = (hist, job_id)
    return out
"""
Attention pattern visualization for the four pretrained models.

Works on the Rouge-L-bucketed capture written by attn_capture.AttentionCapture:
  attn_scores_rouge_l_{NN-MM}_rank{N}.npz   – mean attention map [L, H, S, S] per bucket
  norm_attn_rouge_l_{NN-MM}_rank{N}.npz     – mean norm-based map [L, H, S, S] per bucket
  gating_scores_rank{N}.npz                 – per-(bucket, layer, head) gate histogram (gated only)

S = prompt_len + suffix_length - 1; row = query position, col = key position. Maps are
NOT renormalized: for sink/off-by-one models the row-sum deficit is the virtual-sink mass,
and BOS attention is simply column 0 — compute those yourself at whatever granularity.

Functions:
  load_maps(exp_base, model, kind, bucket)  – load+merge per-rank maps for one bucket
  load_all_maps(exp_base, model, kind)      – {bucket_label: merged map} for all buckets
  load_gating(exp_base, model)              – load+merge gating histogram
  plot_map(maps, bucket, layer, head)       – multi-panel attention/norm heatmap
  plot_full_grid(maps, bucket)              – full grid: cols=layers, rows=heads×models (+avg)
  plot_gating_distribution(gating, ...)     – normalized gate-score density per layer

Usage example:
  from attn_bench.evaluation.plot_attention_patterns import *

  EXP_BASE = "/users/elyulina/store/mem-results/SparseGutenberg"
  MODELS = {
      "full":  "llama3-1b-full-attn-fineweb40B-gutenberg3B",
      "gated": "llama3-1b-gated-attn-fineweb40B-gutenberg3B",
      "obo":   "llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215",
      "sink":  "llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215",
  }
  PREFIX_LEN, SUFFIX_LEN, OFFSET = 500, 50, 0

  attn = {k: load_all_maps(EXP_BASE, n, "attn_scores", offset=OFFSET,
                           prefix_len=PREFIX_LEN, suffix_len=SUFFIX_LEN)
          for k, n in MODELS.items()}
  plot_map({k: v["09-10"] for k, v in attn.items()}, bucket="09-10", layer=8, head=0)
  gating = load_gating(EXP_BASE, MODELS["gated"], offset=OFFSET,
                       prefix_len=PREFIX_LEN, suffix_len=SUFFIX_LEN)
  plot_gating_distribution(gating, layer=8)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

N_BUCKETS = 10


def bucket_label(bi: int) -> str:
    """'00-01', ..., '09-10' for bucket index 0..9."""
    return f"{bi:02d}-{bi + 1:02d}"


ALL_BUCKETS = [bucket_label(bi) for bi in range(N_BUCKETS)]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _run_dir(
    exp_base: str | Path,
    model_name: str,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 50,
) -> Path:
    return (Path(exp_base) / model_name / "inference"
            / f"offset_{offset}_prefix_{prefix_len}_suffix_{suffix_len}")


def _as_label(bucket: int | str) -> str:
    return bucket if isinstance(bucket, str) else bucket_label(bucket)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_maps(
    exp_base: str | Path,
    model_name: str,
    kind: str,
    bucket: int | str,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 50,
) -> dict[str, Any]:
    """Load and count-weighted-merge per-rank maps for one Rouge-L bucket.

    kind   : 'attn_scores' | 'norm_attn'
    bucket : bucket index 0..9 or label like '09-10'
    Returns {'mean': [L,H,S,S] float32, 'count': int, 'prompt_len': int}.
    """
    if kind not in ("attn_scores", "norm_attn"):
        raise ValueError(f"kind must be 'attn_scores' or 'norm_attn', got {kind!r}")
    label = _as_label(bucket)
    d = _run_dir(exp_base, model_name, offset, prefix_len, suffix_len)
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
        # No samples landed in this bucket on any rank
        ref = np.load(rank_files[0])
        mean = np.zeros_like(ref["mean"], dtype=np.float32)
    else:
        mean = weighted_sum / total
    return {"mean": mean, "count": total, "prompt_len": prompt_len}


def load_all_maps(
    exp_base: str | Path,
    model_name: str,
    kind: str,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 50,
) -> dict[str, dict[str, Any]]:
    """{bucket_label: load_maps(...)} for every bucket that has files."""
    out = {}
    for label in ALL_BUCKETS:
        try:
            out[label] = load_maps(exp_base, model_name, kind, label,
                                   offset, prefix_len, suffix_len)
        except FileNotFoundError:
            pass
    return out


def load_gating(
    exp_base: str | Path,
    model_name: str,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 50,
) -> dict[str, Any]:
    """Load and sum-merge the per-rank gating histograms (gated model only).

    Returns {'hist': [n_buckets,L,H,n_bins] int64, 'bin_edges': [n_bins+1],
             'count': [n_buckets] int64}.
    """
    d = _run_dir(exp_base, model_name, offset, prefix_len, suffix_len)
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _model_label(key: str) -> str:
    return {"full": "Full", "gated": "Gated", "obo": "Off-by-one", "sink": "Sink"}.get(key, key)


def _model_color(key: str) -> str:
    return {"full": "steelblue", "gated": "darkorange",
            "obo": "forestgreen", "sink": "crimson"}.get(key, "black")


# ---------------------------------------------------------------------------
# plot_map
# ---------------------------------------------------------------------------

def plot_map(
    maps: dict[str, dict],
    bucket: int | str,
    layer: int,
    head: int,
    *,
    figsize: tuple[float, float] | None = None,
    vmin: float = 1e-4,
    vmax: float | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Multi-panel heatmap of the (query × key) attention/norm map, one panel per model.

    Parameters
    ----------
    maps   : {model_key: load_maps(...)}  — all for the same kind and bucket
    bucket : bucket index/label (used only for the title)
    layer  : 0-based layer index
    head   : 0-based head index
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    model_keys = list(maps.keys())
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, model_keys):
        m = maps[key]
        mat = m["mean"][layer, head].astype(np.float32)   # [S, S]
        prompt_len = int(m["prompt_len"])

        pos = mat[mat > 0]
        vm = vmax if vmax is not None else (float(np.quantile(pos, 0.995)) if pos.size else 1.0)
        im = ax.imshow(mat, aspect="auto", origin="upper",
                       norm=LogNorm(vmin=vmin, vmax=max(vm, vmin * 10)), cmap="viridis")
        # prefix/suffix divider on both axes
        ax.axvline(prompt_len - 0.5, color="white", linewidth=1.0, linestyle="--")
        ax.axhline(prompt_len - 0.5, color="white", linewidth=1.0, linestyle="--")
        ax.set_title(f"{_model_label(key)}  (n={m['count']})")
        ax.set_xlabel("Key position")
        if ax is axes[0]:
            ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Rouge-L {_as_label(bucket)}  layer={layer}  head={head}", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_full_grid
# ---------------------------------------------------------------------------

def plot_full_grid(
    maps: dict[str, dict],
    bucket: int | str,
    *,
    cell_w: float = 1.5,
    cell_h: float = 0.8,
    vmin: float = 1e-4,
    save_path: str | Path | None = None,
) -> Any:
    """Full grid: cols=layers, rows=(heads + head-avg) × models.

    Each cell is a (query × key) heatmap with LogNorm. Color scale is shared within each
    (head-group, layer) block so the models are directly comparable.

    Parameters
    ----------
    maps   : {model_key: load_maps(...)}  — same kind and bucket for all models
    bucket : bucket index/label (used only for the title)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    model_keys = list(maps.keys())
    n_models = len(model_keys)

    matrices = {key: maps[key]["mean"].astype(np.float32) for key in model_keys}
    prompt_len = int(maps[model_keys[0]]["prompt_len"])

    ref = next(iter(matrices.values()))
    n_layers, n_heads, _, _ = ref.shape

    head_avg = {key: matrices[key].mean(axis=1) for key in model_keys}  # (L, S, S)

    n_groups = n_heads + 1
    n_rows = n_groups * n_models
    n_cols = n_layers

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_w * n_cols, cell_h * n_rows), squeeze=False)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    def _fill_group(group_idx: int, group_label: str, mats: dict[str, np.ndarray]) -> None:
        for col in range(n_layers):
            pos_vals = np.concatenate([mats[k][col].ravel() for k in model_keys])
            pos_vals = pos_vals[pos_vals > 0]
            vmax = float(np.quantile(pos_vals, 0.995)) if len(pos_vals) > 0 else 1.0
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))

            for mi, key in enumerate(model_keys):
                row = group_idx * n_models + mi
                ax = axes[row, col]
                ax.imshow(mats[key][col], aspect="auto", origin="upper",
                          norm=norm, cmap="viridis", interpolation="nearest")
                ax.axvline(prompt_len - 0.5, color="white", linewidth=0.4, linestyle="--")
                ax.set_xticks([])
                ax.set_yticks([])

                if col == 0:
                    ax.set_ylabel(f"{group_label}\n{_model_label(key)}", fontsize=4,
                                  rotation=0, ha="right", va="center", labelpad=2)
                if group_idx == 0 and mi == 0:
                    ax.set_title(f"L{col}", fontsize=6, pad=2)

                for spine in ax.spines.values():
                    spine.set_linewidth(0.3)
                if mi == 0:
                    ax.spines["top"].set_linewidth(1.5)
                    ax.spines["top"].set_color("white")

    for h in range(n_heads):
        _fill_group(h, f"H{h}", {key: matrices[key][:, h, :, :] for key in model_keys})
    _fill_group(n_heads, "avg", head_avg)

    fig.suptitle(
        f"Attention maps  Rouge-L {_as_label(bucket)}  |  rows: head × model  |  cols: layer",
        fontsize=8, y=1.001,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_gating_distribution
# ---------------------------------------------------------------------------

def plot_gating_distribution(
    gating: dict[str, Any],
    *,
    layer: int | None = None,
    head: int | None = None,
    buckets: list[int | str] | None = None,
    figsize: tuple[float, float] = (8, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Normalized gate-score density (as in the Gated Attention paper).

    Aggregates the per-(bucket, layer, head) histogram down to the requested granularity
    and plots normalized density vs gating score. If `buckets` is given, one curve per
    Rouge-L bucket (to compare memorized vs non-memorized); otherwise a single pooled curve.

    Parameters
    ----------
    gating : load_gating(...) result
    layer  : restrict to one layer (default: all layers pooled)
    head   : restrict to one head (default: all heads pooled)
    buckets: list of bucket indices/labels to overlay (default: all pooled into one curve)
    """
    import matplotlib.pyplot as plt

    hist = gating["hist"].astype(np.float64)   # [n_buckets, L, H, n_bins]
    edges = gating["bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    def _density(sel_hist: np.ndarray) -> np.ndarray | None:
        # sum over all axes except the bin axis
        counts = sel_hist.reshape(-1, sel_hist.shape[-1]).sum(axis=0)
        total = counts.sum()
        if total == 0:
            return None
        return counts / (total * widths)

    fig, ax = plt.subplots(figsize=figsize)

    def _slice(bucket_idx):
        h = hist[bucket_idx] if bucket_idx is not None else hist
        # h: [..., L, H, n_bins]
        if layer is not None:
            h = h[..., layer, :, :]   # index the L axis (3rd from last)
        if head is not None:
            h = h[..., head, :]       # index the H axis (2nd from last)
        return h

    if buckets is None:
        dens = _density(_slice(None))
        if dens is not None:
            ax.plot(centers, dens, color="black", lw=2)
        title_extra = "all buckets pooled"
    else:
        cmap = plt.cm.viridis(np.linspace(0, 1, len(buckets)))
        for c, b in zip(cmap, buckets):
            bi = b if isinstance(b, int) else ALL_BUCKETS.index(b)
            dens = _density(_slice(bi))
            if dens is not None:
                ax.plot(centers, dens, color=c, lw=1.8, label=f"Rouge-L {_as_label(b)}")
        ax.legend(fontsize=8)
        title_extra = "by Rouge-L bucket"

    loc = []
    if layer is not None:
        loc.append(f"layer={layer}")
    if head is not None:
        loc.append(f"head={head}")
    loc_str = ", ".join(loc) if loc else "all layers/heads"

    ax.set_xlabel("gating score  sigmoid(gate)")
    ax.set_ylabel("normalized density")
    ax.set_xlim(0, 1)
    ax.set_title(f"Gate-score distribution ({loc_str}; {title_extra})", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
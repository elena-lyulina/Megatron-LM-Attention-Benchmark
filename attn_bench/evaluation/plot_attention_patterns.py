"""
Attention pattern visualization for the four pretrained models.

Functions:
  load_stats(exp_base, model_name, rep)     – load and merge per-rank stats npz
  load_example(exp_base, model_name, rep)   – load rank-0 example matrix npz
  plot_heatmap(examples, rep, layer, head)  – 4-panel attention heatmap
  plot_sink_strength(stats_by_rep, models)  – BOS/sink mass vs. repetition
  plot_entropy(stats_by_rep, models)        – mean entropy vs. repetition
  plot_gate(example, rep, layer)            – gate scalar heatmap (gated model)

Usage example:
  from attn_bench.evaluation.plot_attention_patterns import *

  EXP_BASE = "/users/elyulina/store/mem-results/SparseGutenberg"
  MODELS = {
      "full":    "llama3-1b-full-attn-fineweb40B-gutenberg3B",
      "gated":   "llama3-1b-gated-attn-fineweb40B-gutenberg3B",
      "obo":     "llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215",
      "sink":    "llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215",
  }
  REPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
  PREFIX_LEN, SUFFIX_LEN, OFFSET = 500, 500, 0

  stats = {
      model_key: {rep: load_stats(EXP_BASE, name, rep, OFFSET, PREFIX_LEN, SUFFIX_LEN)
                  for rep in REPS}
      for model_key, name in MODELS.items()
  }
  examples = {
      model_key: {rep: load_example(EXP_BASE, name, rep, OFFSET, PREFIX_LEN, SUFFIX_LEN)
                  for rep in REPS}
      for model_key, name in MODELS.items()
  }
  plot_heatmap(examples, rep=16, layer=8, head=0)
  plot_sink_strength(stats, MODELS)
  plot_entropy(stats, MODELS)
  plot_gate(examples["gated"][16], rep=16, layer=8)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _inference_dir(
    exp_base: str | Path,
    model_name: str,
    rep: int,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 500,
) -> Path:
    base = Path(exp_base) / model_name / "inference"
    run = f"offset_{offset}_prefix_{prefix_len}_suffix_{suffix_len}"
    return base / run / f"rep_{rep}_greedy"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stats(
    exp_base: str | Path,
    model_name: str,
    rep: int,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 500,
) -> dict[str, np.ndarray]:
    """Load and concatenate per-rank attn_stats_rank*.npz files.

    Returns a dict with the same keys as the per-rank files, but with
    arrays concatenated along axis 0 (samples dimension).
    """
    d = _inference_dir(exp_base, model_name, rep, offset, prefix_len, suffix_len)
    rank_files = sorted(d.glob("attn_stats_rank*.npz"))
    if not rank_files:
        raise FileNotFoundError(f"No attn_stats_rank*.npz in {d}")

    parts: dict[str, list[np.ndarray]] = {}
    for f in rank_files:
        npz = np.load(f)
        for k in npz.files:
            parts.setdefault(k, []).append(npz[k].astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


def load_example(
    exp_base: str | Path,
    model_name: str,
    rep: int,
    offset: int = 0,
    prefix_len: int = 500,
    suffix_len: int = 500,
    rank: int = 0,
) -> dict[str, Any]:
    """Load the rank-0 example matrix npz (attn_matrix_exmpl_rank{rank}.npz)."""
    d = _inference_dir(exp_base, model_name, rep, offset, prefix_len, suffix_len)
    f = d / f"attn_matrix_exmpl_rank{rank}.npz"
    if not f.exists():
        raise FileNotFoundError(f"Not found: {f}")
    npz = np.load(f)
    return {k: npz[k] for k in npz.files}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _model_label(key: str) -> str:
    return {"full": "Full", "gated": "Gated", "obo": "Off-by-one", "sink": "Sink"}.get(key, key)


def _model_color(key: str) -> str:
    return {"full": "steelblue", "gated": "darkorange", "obo": "forestgreen", "sink": "crimson"}.get(key, "black")


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    examples: dict[str, dict[int, dict]],
    rep: int,
    layer: int,
    head: int,
    *,
    figsize: tuple[float, float] | None = None,
    vmax: float | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """4-panel heatmap of decode-step × key-position attention weights.

    Parameters
    ----------
    examples : {model_key: {rep: load_example(...)}}
    rep      : repetition bucket to plot
    layer    : 0-based layer index
    head     : 0-based head index
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    model_keys = list(examples.keys())
    n = len(model_keys)
    fig, axes = plt.subplots(1, n, figsize=figsize or (5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, model_keys):
        ex = examples[key][rep]
        mat = ex["matrix"][layer, head].astype(np.float32)   # [T, S]
        prompt_len = int(ex["prompt_len"])

        vm = vmax or float(np.quantile(mat[mat > 0], 0.995)) if mat.max() > 0 else 1.0
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            norm=LogNorm(vmin=1e-4, vmax=vm),
            cmap="viridis",
        )
        ax.axvline(prompt_len - 0.5, color="white", linewidth=0.8, linestyle="--")
        ax.set_title(_model_label(key))
        ax.set_xlabel("Key position")
        if ax is axes[0]:
            ax.set_ylabel("Decode step")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate sink mass if available
        if "sink_mass_matrix" in ex:
            sink = ex["sink_mass_matrix"][layer, head]   # [T]
            mean_sink = float(sink.mean())
            if mean_sink > 1e-3:
                ax.set_title(f"{_model_label(key)}\n(sink={mean_sink:.3f})")

    fig.suptitle(f"rep={rep}  layer={layer}  head={head}", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_sink_strength
# ---------------------------------------------------------------------------

def plot_sink_strength(
    stats: dict[str, dict[int, dict]],
    models: dict[str, str],
    *,
    figsize: tuple[float, float] = (10, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Two-panel plot: BOS attention (pos 0) and explicit virtual-sink mass vs. repetition.

    Panel 1 — mean_attn[:, :, :, 0].mean() across layers/heads/samples (BOS position).
    Panel 2 — sink_mass.mean() across steps/layers/heads/samples (only non-zero for sink/obo).

    Parameters
    ----------
    stats  : {model_key: {rep: load_stats(...)}}
    models : {model_key: human-readable name or exp_name} — used for ordering/labels
    """
    import matplotlib.pyplot as plt

    reps = sorted(next(iter(stats.values())).keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    for key in models:
        if key not in stats:
            continue
        color = _model_color(key)
        label = _model_label(key)

        bos_vals, sink_vals = [], []
        for rep in reps:
            s = stats[key][rep]
            # BOS: mean over samples, layers, heads
            bos_vals.append(float(s["mean_attn"][..., 0].mean()))
            # Virtual sink mass: mean over all dimensions
            sink_vals.append(float(s["sink_mass"].mean()))

        ax1.plot(reps, bos_vals, marker="o", color=color, label=label)
        ax2.plot(reps, sink_vals, marker="o", color=color, label=label)

    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xlabel("Repetitions")
    ax1.set_ylabel("Mean attention to BOS (pos 0)")
    ax1.set_title("BOS attention vs. repetition")
    ax1.legend()

    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xlabel("Repetitions")
    ax2.set_ylabel("Mean virtual-sink mass")
    ax2.set_title("Explicit sink mass vs. repetition\n(0 for full/gated)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_entropy
# ---------------------------------------------------------------------------

def plot_entropy(
    stats: dict[str, dict[int, dict]],
    models: dict[str, str],
    *,
    figsize: tuple[float, float] = (7, 5),
    save_path: str | Path | None = None,
) -> Any:
    """Mean attention entropy vs. repetition bucket (cross-model comparable).

    Entropy is computed over the full distribution including the virtual-sink
    column, so all 4 models are directly comparable.

    Parameters
    ----------
    stats  : {model_key: {rep: load_stats(...)}}
    models : {model_key: ...} — used for ordering/labels
    """
    import matplotlib.pyplot as plt

    reps = sorted(next(iter(stats.values())).keys())
    fig, ax = plt.subplots(figsize=figsize)

    for key in models:
        if key not in stats:
            continue
        ent_vals = []
        for rep in reps:
            s = stats[key][rep]
            ent_vals.append(float(s["entropy"].mean()))
        ax.plot(reps, ent_vals, marker="o", color=_model_color(key), label=_model_label(key))

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Repetitions")
    ax.set_ylabel("Mean entropy (nats)")
    ax.set_title("Attention entropy vs. repetition\n(full distribution, cross-model comparable)")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# plot_gate
# ---------------------------------------------------------------------------

def plot_gate(
    example: dict[str, Any],
    rep: int,
    layer: int,
    *,
    figsize: tuple[float, float] = (8, 4),
    save_path: str | Path | None = None,
) -> Any:
    """Heatmap of the gate scalar (head × decode-step) for the gated model.

    Also shows an effective-attention heatmap for head 0 when both
    gate_matrix and matrix are present in the example.

    Parameters
    ----------
    example : load_example(...) result for the gated model at the given rep
    rep     : repetition bucket (used only for the figure title)
    layer   : 0-based layer index
    """
    import matplotlib.pyplot as plt

    if "gate_matrix" not in example:
        raise ValueError("example does not contain gate_matrix — was it captured with a gated model?")

    gate = example["gate_matrix"][layer].astype(np.float32)   # [H, T]
    n_panels = 2 if "matrix" in example else 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    im0 = axes[0].imshow(gate, aspect="auto", origin="lower", cmap="plasma", vmin=0, vmax=1)
    axes[0].set_title(f"Gate scalar  (rep={rep} layer={layer})")
    axes[0].set_xlabel("Decode step")
    axes[0].set_ylabel("Head")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="sigmoid(gate).mean(hn)")

    if n_panels == 2:
        # Effective attention for head 0: attn × gate (broadcast over key positions)
        attn = example["matrix"][layer, 0].astype(np.float32)   # [T, S]
        g0   = gate[0, :, np.newaxis]                            # [T, 1]
        eff  = attn * g0                                          # [T, S]
        prompt_len = int(example["prompt_len"])
        im1 = axes[1].imshow(eff, aspect="auto", origin="lower", cmap="viridis")
        axes[1].axvline(prompt_len - 0.5, color="white", linewidth=0.8, linestyle="--")
        axes[1].set_title(f"Effective attn head 0 (gate × attn)")
        axes[1].set_xlabel("Key position")
        axes[1].set_ylabel("Decode step")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .columns import Col
from .tokenize_excerpts import SEQ_LEN

INFERENCE_BATCH_SIZE = 4
MIN_K = 0.2  # bottom 20% of token scores


def load_model(ckpt_dir: str, tokenizer_path: str, batch_size: int = INFERENCE_BATCH_SIZE):
    """Load FineWeb LLaMA 1B from a torch_dist checkpoint for single-GPU inference.

    Architecture args are loaded via --use-checkpoint-args.
    TP=1: DCP resharding merges the TP=2 checkpoint shards into a single replica.
    """
    from gpt_builders import gpt_builder
    from model_provider import model_provider
    from megatron.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron

    saved_argv = sys.argv[:]
    sys.argv = [
        'score_perplexity',
        '--use-checkpoint-args',
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1',
        '--micro-batch-size', str(batch_size),
        '--global-batch-size', str(batch_size),
        '--train-iters', '1',
        '--tokenizer-type', 'HuggingFaceTokenizer',
        '--tokenizer-model', tokenizer_path,
        '--load', str(ckpt_dir),
        '--no-load-optim',
        '--no-load-rng',
        '--ckpt-format', 'torch_dist',
        '--dist-ckpt-strictness', 'assume_ok_unexpected',
        '--finetune',
        '--bf16',
        '--transformer-impl', 'transformer_engine',
        '--main-grads-dtype', 'fp32',
    ]
    try:
        initialize_megatron()
        model = get_model(partial(model_provider, gpt_builder), wrap_with_ddp=False)
        load_checkpoint(model, optimizer=None, opt_param_scheduler=None)
        model = model[0]
        model.eval()
        return model
    finally:
        sys.argv = saved_argv


def _score_batch(model, token_ids_batch: list[list[int]], k: float = MIN_K) -> tuple[list[float], list[float]]:
    """Single forward pass → (perplexities, min_k_pp_scores).

    Min-K%++ (Zhang et al. 2024): z-score each token's log-prob against the full
    vocabulary distribution at that position, then average the bottom-k% of z-scores.
    Higher score (less negative) = model confident on hard tokens = likely training data.
    """
    device = next(model.parameters()).device
    tokens = torch.tensor(token_ids_batch, dtype=torch.long, device=device)
    input_ids = tokens[:, :-1]   # [B, S]
    labels = tokens[:, 1:]       # [B, S]
    B, S = input_ids.shape
    position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        logits = model(input_ids, position_ids, attention_mask=None)  # [B, S, V]

    log_probs = torch.log_softmax(logits.float(), dim=-1)  # [B, S, V], float32
    del logits

    # Per-token log-prob of the correct token → perplexity
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, S]
    perplexities = (-token_log_probs).mean(dim=1).exp().tolist()

    # Min-K%++ z-score: normalize each token against its position's vocabulary distribution
    probs = log_probs.exp()                                                          # [B, S, V]
    mu = (probs * log_probs).sum(-1)                                                 # [B, S]
    sigma = ((probs * log_probs.square()).sum(-1) - mu.square()).clamp(min=0).sqrt() # [B, S]
    del probs, log_probs
    z_scores = (token_log_probs - mu) / sigma.clamp(min=1e-8)                       # [B, S]

    k_num = max(1, int(k * S))
    sorted_z, _ = torch.sort(z_scores, dim=1)   # ascending: lowest z-scores first
    min_k_pp = sorted_z[:, :k_num].mean(dim=1).tolist()

    return perplexities, min_k_pp


def score_perplexity_and_min_k_pp(ds, model, batch_size: int = INFERENCE_BATCH_SIZE):
    """Score kept books in one forward pass per batch; add Col.PERPLEXITY and Col.MIN_K_PP."""
    keep_indices = [i for i, keep in enumerate(ds[Col.KEEP]) if keep]

    perplexities_map: dict[int, float] = {}
    min_k_pp_map: dict[int, float] = {}
    t0 = time.time()
    for start in range(0, len(keep_indices), batch_size):
        batch_idx = keep_indices[start:start + batch_size]
        batch_tokens = [ds[i][Col.TOKEN_IDS] for i in batch_idx]
        ppls, mkpps = _score_batch(model, batch_tokens)
        for idx, ppl, mkpp in zip(batch_idx, ppls, mkpps):
            perplexities_map[idx] = ppl
            min_k_pp_map[idx] = mkpp
        done = start + len(batch_idx)
        if done % max(batch_size * 10, 1) == 0 or done == len(keep_indices):
            print(f"  scoring: {done}/{len(keep_indices)}  ({time.time() - t0:.0f}s)")

    perplexities: list[float | None] = [perplexities_map.get(i) for i in range(len(ds))]
    min_k_pp: list[float | None] = [min_k_pp_map.get(i) for i in range(len(ds))]

    for col, vals in [(Col.PERPLEXITY, perplexities), (Col.MIN_K_PP, min_k_pp)]:
        if col in ds.column_names:
            ds = ds.remove_columns([col])
        ds = ds.add_column(col, vals)
    return ds


def write_perplexity_stats(ds, stats_dir: Path):
    import matplotlib.pyplot as plt

    kept = [r for r in ds if r[Col.KEEP] and r[Col.PERPLEXITY] is not None]
    if not kept:
        return
    values = [r[Col.PERPLEXITY] for r in kept]

    stats_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(values, dtype=np.float64)
    pcts = np.percentile(arr, [5, 25, 50, 75, 95])

    path_txt = stats_dir / "perplexity_stats.txt"
    with open(path_txt, "w") as f:
        f.write(f"n={len(arr):,}\n")
        f.write(
            f"min={arr.min():.2f}  p5={pcts[0]:.2f}  p25={pcts[1]:.2f}  "
            f"median={pcts[2]:.2f}  p75={pcts[3]:.2f}  p95={pcts[4]:.2f}  max={arr.max():.2f}\n"
        )
    print(f"Perplexity stats -> {path_txt}")

    log_arr = np.log10(arr)
    bins = np.linspace(log_arr.min(), log_arr.max(), 60)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(log_arr, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)

    pct_styles = [
        ("p5",  "#d62728", ":"),
        ("p25", "#ff7f0e", "--"),
        ("p50", "#2ca02c", "-"),
        ("p75", "#ff7f0e", "--"),
        ("p95", "#d62728", ":"),
    ]
    for (label, color, ls), val in zip(pct_styles, pcts):
        ax.axvline(np.log10(val), color=color, linestyle=ls, linewidth=1.3,
                   label=f"{label}={val:.1f}")

    ax.set_xlabel("Perplexity (log₁₀ scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"Perplexity distribution  n={len(arr):,}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10**x:.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()

    path_png = stats_dir / "perplexity_distribution.png"
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    print(f"Perplexity plot -> {path_png}")

    sorted_kept = sorted(kept, key=lambda r: r[Col.PERPLEXITY])
    for fname, rows in [("perplexity_low.txt", sorted_kept[:100]),
                        ("perplexity_high.txt", sorted_kept[-100:][::-1])]:
        path_ex = stats_dir / fname
        with open(path_ex, "w") as f:
            for r in rows:
                f.write(f"ppl={r[Col.PERPLEXITY]:.1f}  id={r[Col.BOOK_ID]}  title={r[Col.BOOK_TITLE]!r}\n")
                f.write(f"{r[Col.TEXT_EXCERPT]}\n\n")
        print(f"Perplexity examples -> {path_ex}")


def write_min_k_pp_stats(ds, stats_dir: Path):
    import matplotlib.pyplot as plt

    kept = [r for r in ds if r[Col.KEEP] and r[Col.MIN_K_PP] is not None]
    if not kept:
        return
    values = [r[Col.MIN_K_PP] for r in kept]

    stats_dir.mkdir(parents=True, exist_ok=True)
    arr = np.array(values, dtype=np.float64)
    pcts = np.percentile(arr, [5, 25, 50, 75, 95])

    path_txt = stats_dir / "min_k_pp_stats.txt"
    with open(path_txt, "w") as f:
        f.write(f"n={len(arr):,}\n")
        f.write(
            f"min={arr.min():.4f}  p5={pcts[0]:.4f}  p25={pcts[1]:.4f}  "
            f"median={pcts[2]:.4f}  p75={pcts[3]:.4f}  p95={pcts[4]:.4f}  max={arr.max():.4f}\n"
        )
        f.write(f"(Min-{int(MIN_K*100)}%++ z-score; higher = model confident on hard tokens = more likely training data)\n")
    print(f"Min-K%++ stats -> {path_txt}")

    bins = np.linspace(arr.min(), arr.max(), 60)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(arr, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)

    pct_styles = [
        ("p5",  "#d62728", ":"),
        ("p25", "#ff7f0e", "--"),
        ("p50", "#2ca02c", "-"),
        ("p75", "#ff7f0e", "--"),
        ("p95", "#d62728", ":"),
    ]
    for (label, color, ls), val in zip(pct_styles, pcts):
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.3, label=f"{label}={val:.3f}")

    ax.set_xlabel(f"Min-{int(MIN_K*100)}%++ z-score  (higher = more likely training overlap)")
    ax.set_ylabel("Count")
    ax.set_title(f"Min-K%++ distribution  n={len(arr):,}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path_png = stats_dir / "min_k_pp_distribution.png"
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    print(f"Min-K%++ plot -> {path_png}")

    # highest = most suspicious (potential training data); lowest = safest to keep
    sorted_kept = sorted(kept, key=lambda r: r[Col.MIN_K_PP], reverse=True)
    for fname, rows in [("min_k_pp_high.txt", sorted_kept[:100]),
                        ("min_k_pp_low.txt", sorted_kept[-100:][::-1])]:
        path_ex = stats_dir / fname
        with open(path_ex, "w") as f:
            for r in rows:
                f.write(f"min_k_pp={r[Col.MIN_K_PP]:.4f}  id={r[Col.BOOK_ID]}  title={r[Col.BOOK_TITLE]!r}\n")
                f.write(f"{r[Col.TEXT_EXCERPT]}\n\n")
        print(f"Min-K%++ examples -> {path_ex}")


def write_joint_stats(ds, stats_dir: Path):
    """Hexbin scatter of perplexity vs Min-K%++ to show how both metrics co-vary."""
    import matplotlib.pyplot as plt

    kept = [r for r in ds if r[Col.KEEP] and r[Col.PERPLEXITY] is not None and r[Col.MIN_K_PP] is not None]
    if not kept:
        return

    log_ppl = np.log10([r[Col.PERPLEXITY] for r in kept])
    min_k_pp_vals = np.array([r[Col.MIN_K_PP] for r in kept])

    corr = np.corrcoef(log_ppl, min_k_pp_vals)[0, 1]

    stats_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(log_ppl, min_k_pp_vals, gridsize=60, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")

    ax.axvline(np.median(log_ppl), color="#2ca02c", linestyle="--", linewidth=1, label="ppl median")
    ax.axhline(np.median(min_k_pp_vals), color="#ff7f0e", linestyle="--", linewidth=1, label="min-k++ median")

    ax.set_xlabel("Perplexity (log₁₀ scale)")
    ax.set_ylabel("Min-K%++ z-score")
    ax.set_title(f"Perplexity vs Min-K%++  n={len(kept):,}  r={corr:.3f}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{10**x:.0f}"))
    ax.legend(fontsize=8)
    fig.tight_layout()

    path_png = stats_dir / "joint_ppl_min_k_pp.png"
    fig.savefig(path_png, dpi=120)
    plt.close(fig)
    print(f"Joint scoring plot -> {path_png}")


def write_scoring_stats(ds, stats_dir: Path):
    write_perplexity_stats(ds, stats_dir)
    write_min_k_pp_stats(ds, stats_dir)
    write_joint_stats(ds, stats_dir)
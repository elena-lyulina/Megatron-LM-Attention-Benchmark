from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .columns import Col
from .tokenize_excerpts import SEQ_LEN

INFERENCE_BATCH_SIZE = 4


def load_model(ckpt_dir: str, tokenizer_path: str, batch_size: int = INFERENCE_BATCH_SIZE):
    """Initialize Megatron with TP=1 and load the FineWeb model from a torch_dist checkpoint.

    Architecture args (num_layers, hidden_size, swiglu, RoPE params, etc.) are loaded directly
    from the checkpoint via --use-checkpoint-args.  The only thing we override is TP/PP:
    the checkpoint was saved with TP=2, but we run inference with TP=1 so each GPU holds a
    full model replica; Megatron's DCP resharding merges the two shards automatically.

    With DP>1 (multiple GPUs), each rank loads an independent model copy and scores a disjoint
    slice of books — zero inter-GPU communication during forward passes.
    global-batch-size = batch_size * world_size satisfies Megatron's GBS % (MBS * DP) == 0
    validation for any world_size with TP=1 (DP=world_size, GAS=1).
    """
    from gpt_builders import gpt_builder
    from model_provider import model_provider
    from megatron.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    saved_argv = sys.argv[:]
    sys.argv = [
        'score_perplexity',
        # Architecture is read from checkpoint -- no need to repeat it here
        '--use-checkpoint-args',
        # TP=1: each GPU holds a full model replica; DCP resharding merges the TP=2 checkpoint shards
        '--tensor-model-parallel-size', '1',
        '--pipeline-model-parallel-size', '1',
        '--context-parallel-size', '1',
        # Megatron requires these even for inference
        '--micro-batch-size', str(batch_size),
        '--global-batch-size', str(batch_size * world_size),
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


def _score_batch(model, token_ids_batch: list[list[int]]) -> list[float]:
    """Forward pass for a batch of 8193-token sequences; returns per-sequence perplexity."""
    device = next(model.parameters()).device
    tokens = torch.tensor(token_ids_batch, dtype=torch.long, device=device)  # [B, SEQ_LEN+1]
    input_ids = tokens[:, :-1]   # [B, SEQ_LEN]
    labels = tokens[:, 1:]       # [B, SEQ_LEN]
    B, S = input_ids.shape
    position_ids = torch.arange(S, dtype=torch.long, device=device).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        # returns per-token NLL [B, SEQ_LEN]
        per_token_loss = model(input_ids, position_ids, attention_mask=None, labels=labels)

    return per_token_loss.float().mean(dim=1).exp().tolist()


def score_perplexity(ds, model, batch_size: int = INFERENCE_BATCH_SIZE):
    """Score kept books and add Col.PERPLEXITY; skipped books get None.

    With TP=1 and DP>1, each rank scores a disjoint slice of kept books (no inter-GPU
    communication during forward passes). Results are gathered to rank 0 at the end.
    Only rank 0 returns a dataset with the perplexity column added; callers must guard
    downstream writes with rank == 0.
    """
    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world_size = dist.get_world_size() if is_dist else 1

    keep_indices = [i for i, keep in enumerate(ds[Col.KEEP]) if keep]
    my_indices = keep_indices[rank::world_size]

    local_scores: dict[int, float] = {}
    t0 = time.time()
    for start in range(0, len(my_indices), batch_size):
        batch_idx = my_indices[start:start + batch_size]
        batch_tokens = [ds[i][Col.TOKEN_IDS] for i in batch_idx]
        scores = _score_batch(model, batch_tokens)
        for idx, ppl in zip(batch_idx, scores):
            local_scores[idx] = ppl
        done = start + len(batch_idx)
        if done % max(batch_size * 10, 1) == 0 or done == len(my_indices):
            prefix = f"[rank {rank}/{world_size}] " if is_dist else ""
            print(f"  {prefix}perplexity scoring: {done}/{len(my_indices)}  ({time.time() - t0:.0f}s)")

    if is_dist:
        all_scores: list[dict] = [None] * world_size
        dist.all_gather_object(all_scores, local_scores)
    else:
        all_scores = [local_scores]

    if rank != 0:
        return ds

    merged = {k: v for d in all_scores for k, v in d.items()}
    perplexities: list[float | None] = [merged.get(i) for i in range(len(ds))]
    return ds.add_column(Col.PERPLEXITY, perplexities)


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
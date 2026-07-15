"""Shared constants/helpers for the long-sequence position-wise loss plots.

Used by plot_long_gutenberg.py (aggregate, mean/std/count over a bucket) and
plot_individual_sequence.py (raw, one line per sequence).
"""

from __future__ import annotations

import numpy as np

SEQ_LEN = 8192     # training sequence length
SAMPLE_LEN = 8190  # sample content tokens; predicted at positions 0..8189, so 8190 = first suffix token

# Llama-3 tokenizer vocab, padded (confirmed identical across every model's pretraining log:
# "padded vocab (size: 128256) with 0 dummy tokens"). NLL of a uniform random guess over the
# vocab is ln(VOCAB_SIZE); perplexity of that guess is VOCAB_SIZE itself.
VOCAB_SIZE = 128256


def denser_grid(ax):
    """Lighter minor grid between the major gridlines, on both axes (matplotlib's default
    auto-picked minor-tick spacing). Minor tick marks themselves are hidden so only the
    gridlines get denser, not the axis edges."""
    ax.minorticks_on()
    ax.grid(which="minor", color="#ececec", linewidth=0.6)
    ax.tick_params(which="minor", length=0)


def smooth(y, w):
    # Centered rolling mean with correct edge normalization (window shrinks at the ends).
    if not w or w <= 1:
        return y
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")
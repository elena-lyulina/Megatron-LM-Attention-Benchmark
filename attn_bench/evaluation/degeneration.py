"""
Free-generation quality curves from stored greedy continuations (Section 4.1).

Reads the rank*.jsonl of one inference dir (offset_O_prefix_P_suffix_S/rep_R_greedy, as
written by prefix_extraction_inference.py) and computes, per sample, from the stored
`generated_suffix` / `true_suffix` token lists -- CPU only, no model:

  * raw cumulative TTR    TTR(gen[:b]) and TTR(true[:b]) at each boundary b -> [N, B]
  * cumulative TTR ratio  TTR(gen[:b]) / TTR(true[:b]) at each boundary b     -> [N, B]
  * rolling TTR ratio     TTR over a window of --window tokens ending at each
                          position t = window, window+step, ...               -> [N, T]
  * loop onset            first position whose n-gram already occurred earlier
                          in the same sequence (S+1 if none), for the generated
                          and the true continuation, for every n in --ngram:
                          a short n (8) catches the first repeated phrase, a
                          long n (32) the point where the text has become
                          periodic                                            -> [N] per n
  * divergence point      first position where gen != true (S+1 if none)      -> [N]

TTR = |unique tokens| / |tokens|. The ratio to the true continuation at the same length
removes TTR's own length dependence; a value near 1 means "as varied as real text", well
below 1 means the generation has started repeating itself. The true-continuation loop
onset is the false-positive control for the n-gram detector.

Writes one .npz with the per-sample arrays and the axes, and prints the means.

Usage:
    python attn_bench/evaluation/degeneration.py \
        --inference-dir $MEM_BASE/UnseenFineWeb/<exp>/inference/offset_0_prefix_500_suffix_1000/rep_0_greedy \
        --out results/generation-curves/UnseenFineWeb__<exp>.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_BOUNDARIES = (25, 50, 75, 100, 150, 250, 500, 750, 1000)


def ttr(tokens) -> float:
    return len(set(tokens)) / len(tokens) if len(tokens) else float("nan")


def cumulative_ttr(tokens, boundaries) -> np.ndarray:
    """Raw token diversity from the start of a continuation to each boundary."""
    out = np.full(len(boundaries), np.nan)
    for i, b in enumerate(boundaries):
        if b <= len(tokens):
            out[i] = ttr(tokens[:b])
    return out


def cumulative_ttr_ratio(gen, ref, boundaries) -> np.ndarray:
    return cumulative_ttr(gen, boundaries) / cumulative_ttr(ref, boundaries)


def rolling_ttr(tokens, positions, window) -> np.ndarray:
    out = np.full(len(positions), np.nan)
    for i, t in enumerate(positions):
        if t <= len(tokens):
            out[i] = ttr(tokens[t - window:t])
    return out


def loop_onset(tokens, n) -> int:
    """1-based position of the last token of the first n-gram that already occurred
    earlier in `tokens`; len(tokens) + 1 if the sequence never repeats an n-gram."""
    seen = set()
    for i in range(n - 1, len(tokens)):
        gram = tuple(tokens[i - n + 1:i + 1])
        if gram in seen:
            return i + 1
        seen.add(gram)
    return len(tokens) + 1


def divergence_point(gen, ref) -> int:
    """1-based position of the first mismatch; len + 1 if the whole suffix matches."""
    for i, (g, r) in enumerate(zip(gen, ref)):
        if g != r:
            return i + 1
    return min(len(gen), len(ref)) + 1


def load_records(inference_dir: Path) -> list[dict]:
    records = []
    for path in sorted(inference_dir.glob("rank*.jsonl")):
        with open(path) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    if not records:
        raise FileNotFoundError(f"no rank*.jsonl records under {inference_dir}")
    # Older runs predate the sample_idx field; keep file order for those.
    for i, rec in enumerate(records):
        rec.setdefault("sample_idx", i)
    records.sort(key=lambda r: r["sample_idx"])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inference-dir", required=True, type=Path,
                        help="…/offset_O_prefix_P_suffix_S/rep_R_greedy directory with rank*.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output .npz path")
    parser.add_argument("--boundaries", type=int, nargs="+", default=list(DEFAULT_BOUNDARIES),
                        help="Cumulative-TTR boundaries (tokens)")
    parser.add_argument("--window", type=int, default=100, help="Rolling-TTR window (tokens)")
    parser.add_argument("--step", type=int, default=25, help="Rolling-TTR step (tokens)")
    parser.add_argument("--ngram", type=int, nargs="+", default=[8, 16, 32],
                        help="n-gram sizes for the loop-onset detector (one onset array per n)")
    args = parser.parse_args()

    records = load_records(args.inference_dir)
    suffix_len = min(len(r["generated_suffix"]) for r in records)
    boundaries = [b for b in args.boundaries if b <= suffix_len]
    positions = list(range(args.window, suffix_len + 1, args.step))

    n = len(records)
    cum_gen = np.full((n, len(boundaries)), np.nan)
    cum_ref = np.full((n, len(boundaries)), np.nan)
    roll_gen = np.full((n, len(positions)), np.nan)
    roll_ref = np.full((n, len(positions)), np.nan)
    onset_gen = {k: np.zeros(n, dtype=np.int32) for k in args.ngram}
    onset_ref = {k: np.zeros(n, dtype=np.int32) for k in args.ngram}
    diverge = np.zeros(n, dtype=np.int32)
    sample_idx = np.zeros(n, dtype=np.int64)

    for i, rec in enumerate(records):
        gen, ref = rec["generated_suffix"], rec["true_suffix"]
        sample_idx[i] = rec["sample_idx"]
        cum_gen[i] = cumulative_ttr(gen, boundaries)
        cum_ref[i] = cumulative_ttr(ref, boundaries)
        roll_gen[i] = rolling_ttr(gen, positions, args.window)
        roll_ref[i] = rolling_ttr(ref, positions, args.window)
        for k in args.ngram:
            onset_gen[k][i] = loop_onset(gen, k)
            onset_ref[k][i] = loop_onset(ref, k)
        diverge[i] = divergence_point(gen, ref)

    cum_ratio = cum_gen / cum_ref
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # onset_gen / onset_ref keep the first n for backwards compatibility; every n is also
    # stored as onset_gen_n<k> / onset_ref_n<k>.
    per_n = {}
    for k in args.ngram:
        per_n[f"onset_gen_n{k}"] = onset_gen[k]
        per_n[f"onset_ref_n{k}"] = onset_ref[k]
    np.savez(args.out,
             sample_idx=sample_idx, boundaries=np.array(boundaries),
             positions=np.array(positions), window=args.window, step=args.step,
             ngram=np.array(args.ngram), suffix_len=suffix_len,
             cum_gen=cum_gen, cum_ref=cum_ref, cum_ratio=cum_ratio,
             roll_gen=roll_gen, roll_ref=roll_ref,
             onset_gen=onset_gen[args.ngram[0]], onset_ref=onset_ref[args.ngram[0]],
             divergence=diverge, **per_n)

    roll_ratio = roll_gen / roll_ref
    print(f"{args.inference_dir}: {n} samples, suffix {suffix_len}")
    for name, values in (("generated", cum_gen), ("true", cum_ref)):
        print(f"cumulative TTR ({name}, mean):",
              ", ".join(f"{b}:{v:.3f}" for b, v in zip(boundaries, np.nanmean(values, 0))))
    print("cumulative TTR ratio (mean):",
          ", ".join(f"{b}:{v:.3f}" for b, v in zip(boundaries, np.nanmean(cum_ratio, 0))))
    print("rolling TTR ratio (mean):   ",
          ", ".join(f"{t}:{v:.3f}" for t, v in zip(positions, np.nanmean(roll_ratio, 0))
                    if t in (args.window, 250, 500, 750, suffix_len)))
    for k in args.ngram:
        for name, onset in (("generated", onset_gen[k]), ("true", onset_ref[k])):
            share = {t: float((onset > t).mean()) for t in (100, 250, 500, suffix_len)}
            looped = onset <= suffix_len
            median = int(np.median(onset[looped])) if looped.any() else None
            print(f"loop onset ({name}, {k}-gram): loop-free share "
                  + ", ".join(f"@{t}:{s:.2f}" for t, s in share.items())
                  + f"; median onset among looping {median}")
    print(f"divergence point: mean {diverge.mean():.1f}, share never diverging "
          f"{(diverge > suffix_len).mean():.3f}")


if __name__ == "__main__":
    main()

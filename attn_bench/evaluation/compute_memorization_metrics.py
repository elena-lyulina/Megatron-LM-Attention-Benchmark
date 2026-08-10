"""
Step 2: Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z at each hardcoded suffix
boundary, computed purely from Step 1's jsonl (CPU-only, no model). Same auto-discovery/
skip-if-exists shape as compute_generation_quality.py/compute_mauve.py.

Writes one PDM-Results-shaped .pkl per suffix value (boundary or the real suffix length)
under exp_dir/metrics/, named offset_O_prefix_P_suffix_S_policy.pkl; skip-if-done is an
exact-match path check.

Needs Phase A (megatron_inference_backfill.py) run first for NLL/p_z on older results --
text metrics work on un-backfilled records too.

Usage:
    python attn_bench/evaluation/compute_memorization_metrics.py \
        --exprs llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B \
        --base-path $MEM_BASE/SparseGutenberg \
        --save-path $MEM_BASE/SparseGutenberg
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import jit
from verbatim_eval.controlled_expr import Results
from verbatim_eval.my_rouge import _compute_dp_matrix_2d, compute_rouge_l_2d

from attn_bench.evaluation.megatron_inference import find_suffix_dirs

BOUNDARIES = [25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000]


### lcs_norm -- local copy of PDM's _find_lcs, keeping the array instead of a running max ###
# (see PDM/src/verbatim_eval/LCS.py -- its public function discards the array, and a
# mid-fill running-max snapshot is wrong here since the fill order touches columns beyond
# any given boundary before that row finishes. Filling once and reading dp[:k+1,:k+1].max()
# per boundary afterward is correct and ~2.2x cheaper than refilling per boundary at this
# boundary list's scale.)
@jit(nopython=True)
def _lcs_dp_matrix(s1, s2):
    m, n = len(s1), len(s2)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
    return dp


### (n, p)-discoverable extraction, derived from p_z on demand (Hayes et al. 2025 Eq. 2) ###
def n_for_p(p_z: float, p: float) -> float:
    """Number of queries needed for probability p of extracting the true suffix at least once."""
    if p_z <= 0:
        return math.inf
    if p_z >= 1:
        return 1.0
    return math.log(1 - p) / math.log(1 - p_z)


def p_for_n(p_z: float, n: int) -> float:
    """Probability of extracting the true suffix at least once in n queries."""
    return 1 - (1 - p_z) ** n


### PER-SAMPLE METRICS AT A SET OF BOUNDARIES ###
def text_metrics_at_boundaries(true_suffix: list, gen_suffix: list, boundaries: list) -> dict:
    """{boundary: {Rouge-L, lcs_norm, TTR_ref, TTR_gen, token_acc, divergence_point}} --
    one DP fill per metric regardless of how many boundaries are requested.
    """
    usable = [k for k in boundaries if k <= len(true_suffix) and k <= len(gen_suffix)]
    if not usable:
        return {}
    max_k = max(usable)
    true_arr = np.array(true_suffix[:max_k], dtype=np.int32)
    gen_arr = np.array(gen_suffix[:max_k], dtype=np.int32)

    rouge_dp = _compute_dp_matrix_2d(true_arr, gen_arr)
    lcs_dp = _lcs_dp_matrix(true_arr, gen_arr)
    match = true_arr == gen_arr
    cummatch = np.cumsum(match)
    mismatches = np.where(~match)[0]

    result = {}
    for k in usable:
        first_mismatch = mismatches[mismatches < k]
        result[k] = {
            "Rouge-L": compute_rouge_l_2d(rouge_dp[:k + 1, :k + 1]),
            "lcs_norm": float(lcs_dp[:k + 1, :k + 1].max()) / k,
            "TTR_ref": len(set(true_suffix[:k])) / k,
            "TTR_gen": len(set(gen_suffix[:k])) / k,
            "token_acc": float(cummatch[k - 1]) / k,
            "divergence_point": float(first_mismatch[0]) / k if len(first_mismatch) else 1.0,
        }
    return result


def nll_p_z_at_boundaries(ref_nll: list, gen_nll: list, p_z_logprob: list, boundaries: list) -> dict:
    """{boundary: {ref_nll_mean, gen_nll_mean, p_z}} from Step 1's stored per-position arrays."""
    usable = [k for k in boundaries if k <= len(ref_nll)]
    result = {}
    for k in usable:
        result[k] = {
            "ref_nll_mean": float(np.mean(ref_nll[:k])),
            "gen_nll_mean": float(np.mean(gen_nll[:k])),
            "p_z": float(np.exp(np.sum(p_z_logprob[:k]))),
        }
    return result


### PER-REP AGGREGATION (dataset-level match_x + Results-shaped scores/mean/std) ###

def compute_rep_metrics(records: list, boundaries: list) -> dict:
    """records: one rep bucket's jsonl records. Returns {boundary: {metric: {scores, mean, std}}}."""
    per_boundary = defaultdict(lambda: defaultdict(list))
    missing_nll = 0
    for rec in records:
        text_m = text_metrics_at_boundaries(rec["true_suffix"], rec["generated_suffix"], boundaries)
        for k, metrics in text_m.items():
            for name, val in metrics.items():
                per_boundary[k][name].append(val)
        if "ref_nll" in rec:
            nll_m = nll_p_z_at_boundaries(rec["ref_nll"], rec["gen_nll"], rec["p_z_logprob"], boundaries)
            for k, metrics in nll_m.items():
                for name, val in metrics.items():
                    per_boundary[k][name].append(val)
        else:
            missing_nll += 1
    if missing_nll:
        print(f"  {missing_nll}/{len(records)} record(s) missing ref_nll -- NLL/p_z skipped for "
              "them (run backfill first to fill them in)")

    result = {}
    for k, metrics in per_boundary.items():
        result[k] = {}
        for name, values in metrics.items():
            arr = np.array(values)
            result[k][name] = {"scores": arr, "mean": float(arr.mean()), "std": float(arr.std())}
        lcs_arr = np.array(metrics["lcs_norm"])
        n = len(lcs_arr)
        for label, threshold in [("exact_match", 1.0), ("match_75", 0.75), ("match_50", 0.5), ("match_25", 0.25)]:
            count = int((lcs_arr >= threshold).sum())
            result[k][label] = {"scores": count, "mean": (count / n if n else 0.0), "std": 0}
    return result


### DISCOVERY (same directory convention as Step 1 / compute_generation_quality.py) ###

def find_existing_inference_results(inference_dir: Path) -> set:
    # returns pairs of <offset, prefix>
    pairs = set()
    for d in inference_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("offset_"):
            continue
        parts = d.name.split("_")
        try:
            pairs.add((int(parts[1]), int(parts[3])))
        except (IndexError, ValueError):
            continue
    return pairs


def find_reps_with_max_suffix(expr_dir: Path, offset: int, prefix_length: int) -> dict:
    """{rep: (max_suffix_available, rep_dir_path)} across every suffix dir for
    (offset, prefix_length) -- a rep can have data in more than one suffix dir (e.g. after
    an extend), only the largest one matters for how far its boundaries reach."""
    result = {}
    for suffix_prime, d in find_suffix_dirs(expr_dir, offset, prefix_length):
        for rep_dir in d.iterdir():
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                continue
            if not any(rep_dir.glob("rank*.jsonl")):
                continue
            rep = int(rep_dir.name.split("_")[1])
            if rep not in result or suffix_prime > result[rep][0]:
                result[rep] = (suffix_prime, rep_dir)
    return result


def load_rep_records(rep_dir: Path) -> list:
    records = []
    for rank_file in sorted(rep_dir.glob("rank*.jsonl")):
        with open(rank_file) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


### METADATA ###

def append_metrics_metadata(exp_dir: Path, boundary: int, offset: int, prefix_length: int, reps: list) -> None:
    """One file per call under metrics_metadata/, instead of one shared growing list --
    separate (offset, prefix) jobs for the same experiment run concurrently and would
    otherwise race on a single read-modify-write file."""
    meta_dir = exp_dir / "metrics_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    meta_path = meta_dir / f"offset_{offset}_prefix_{prefix_length}_suffix_{boundary}_{timestamp.strftime('%Y%m%dT%H%M%S%f')}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "action": "metrics",
            "offset": offset,
            "prefix_length": prefix_length,
            "suffix_length": boundary,
            "reps": reps,
            "timestamp": timestamp.isoformat(),
        }, f, indent=2)


### MAIN LOOP ###

def process_offset_prefix(exp_name: str, offset: int, prefix_length: int,
                          reps_info: dict, boundaries: list, save_path: Path,
                          policy: str = "greedy", tag: str | None = None, force: bool = False) -> None:
    reachable = set()
    for rep, (max_suffix, _) in reps_info.items():
        reachable.update(k for k in boundaries if k <= max_suffix)
        if max_suffix not in boundaries:
            reachable.add(max_suffix)

    # tag, when set, appends after policy (e.g. "..._greedy_opt.pkl") so a validation run
    # never overwrites the real pkl at the same path -- policy itself stays "greedy",
    # since that describes the actual sampling method, not which analysis code ran.
    suffix_tag = f"_{tag}" if tag else ""

    def pkl_path(k):
        return save_path / exp_name / "metrics" / f"offset_{offset}_prefix_{prefix_length}_suffix_{k}_{policy}{suffix_tag}.pkl"

    boundaries_to_compute = sorted(k for k in reachable if force or not pkl_path(k).exists())
    if not boundaries_to_compute:
        return

    # One DP fill per rep, across every boundary that rep can reach -- not one fill per
    # boundary. Skipped boundaries (already have a pkl) are never even loaded.
    per_rep_metrics = {}
    for rep, (max_suffix, rep_dir) in sorted(reps_info.items()):
        rep_boundaries = [k for k in boundaries_to_compute if k <= max_suffix]
        if not rep_boundaries:
            continue
        records = load_rep_records(rep_dir)
        if not records:
            continue
        per_rep_metrics[rep] = compute_rep_metrics(records, rep_boundaries)

    for k in boundaries_to_compute:
        data = {exp_name: {}}
        reps_included = []
        for rep, by_boundary in per_rep_metrics.items():
            if k in by_boundary:
                data[exp_name].setdefault(rep, {}).setdefault(offset, {}).setdefault(prefix_length, {})[k] = by_boundary[k]
                reps_included.append(rep)
        if not data[exp_name]:
            continue
        results = Results.from_raw_dict(data, policy=policy)
        path = pkl_path(k)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(results.data, f)
        append_metrics_metadata(save_path / exp_name, k, offset, prefix_length, sorted(reps_included))
        print(f"Saved: {path}  (reps={sorted(reps_included)})")


def process_expr(expr: str, base_path: Path, save_path: Path, boundaries: list, args) -> None:
    expr_dir = base_path / expr
    inference_dir = expr_dir / "inference"
    if not inference_dir.exists():
        print(f"[SKIP] inference dir not found: {inference_dir}")
        return

    for offset, prefix_length in sorted(find_existing_inference_results(inference_dir)):
        if args.offsets and offset not in args.offsets:
            continue
        if args.prefix_lengths and prefix_length not in args.prefix_lengths:
            continue
        reps_info = find_reps_with_max_suffix(expr_dir, offset, prefix_length)
        if not reps_info:
            continue
        print(f"\n=== {expr}  offset={offset} prefix={prefix_length}  reps={sorted(reps_info)} ===")
        process_offset_prefix(expr, offset, prefix_length, reps_info, boundaries,
                              save_path, tag=args.tag, force=args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z per suffix boundary")
    parser.add_argument("--exprs", type=str, nargs="+", required=True)
    parser.add_argument("--base-path", type=str, required=True, help="e.g. $MEM_BASE/SparseGutenberg")
    parser.add_argument("--save-path", type=str, required=True, help="Where to write per-suffix .pkl files")
    parser.add_argument("--offsets", type=int, nargs="+", default=None)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=None)
    parser.add_argument("--tag", type=str, default=None,
                        help="Appended after policy in the pkl filename (e.g. --tag opt -> "
                             "..._greedy_opt.pkl), so a validation run never overwrites the "
                             "real pkl at the same path. Omit for the normal, real output.")
    parser.add_argument("--force", action="store_true", help="Recompute even if the pkl already exists")
    args = parser.parse_args()

    for expr in args.exprs:
        process_expr(expr, Path(args.base_path), Path(args.save_path), BOUNDARIES, args)

"""
Step 2: Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z per suffix boundary, computed
purely from Step 1's jsonl (CPU-only, no model).

For one --exp-name, processes every --points pair at --suffix-length: only reps that reach
--suffix-length count, and a point with none of those errors (real run) or is reported as
needed (--dry-run). Writes one PDM-Results-shaped .pkl per suffix boundary <=
--suffix-length under exp_dir/metrics/, named offset_O_prefix_P_suffix_S_policy.pkl.
Skip-if-done compares an existing pkl's reps against the reps reaching --suffix-length
(scratch, then --persistent-storage-path), so a pkl built before REPETITIONS grew gets
recomputed.

Needs Phase A (prefix_extraction_inference_backfill.py) for NLL/p_z on older results; text
metrics work on un-backfilled records too.

Usage:
    python attn_bench/evaluation/compute_memorization_metrics.py \
        --exp-name llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B \
        --base-path $MEM_BASE/SparseGutenberg --save-path $MEM_BASE/SparseGutenberg \
        --points 0:500 --suffix-length 500

--dry-run: reports missing reps per suffix boundary, no writes, prints space-separated
needed offset:prefix pairs to stdout (e.g. measure_mem_all.sh's submit-time check).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numba import jit
from verbatim_eval.controlled_expr import Results
from verbatim_eval.my_rouge import _compute_dp_matrix_2d, compute_rouge_l_2d

from attn_bench.evaluation.prefix_extraction_inference import (
    find_suffix_dirs, parse_points)

SUFFIX_BOUNDARIES = [25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000]


### lcs_norm -- local copy of PDM's _find_lcs, keeping the array instead of a running max ###
# (see PDM/src/verbatim_eval/LCS.py -- its public function discards the array, and a
# mid-fill running-max snapshot is wrong here since the fill order touches columns beyond
# any given suffix boundary before that row finishes. Filling once and reading
# dp[:k+1,:k+1].max() per boundary afterward is correct and ~2.2x cheaper than refilling
# per boundary at this boundary list's scale.)
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


### PER-SAMPLE METRICS AT A SET OF SUFFIX BOUNDARIES ###
def text_metrics_at_suffix_boundaries(true_suffix: list, gen_suffix: list, suffix_boundaries: list) -> dict:
    """{suffix_boundary: {Rouge-L, lcs_norm, TTR_ref, TTR_gen, token_acc, divergence_point}}
    -- one DP fill per metric regardless of how many suffix boundaries are requested.
    """
    usable = [b for b in suffix_boundaries if b <= len(true_suffix) and b <= len(gen_suffix)]
    if not usable:
        return {}
    max_boundary = max(usable)
    true_arr = np.array(true_suffix[:max_boundary], dtype=np.int32)
    gen_arr = np.array(gen_suffix[:max_boundary], dtype=np.int32)

    rouge_dp = _compute_dp_matrix_2d(true_arr, gen_arr)
    lcs_dp = _lcs_dp_matrix(true_arr, gen_arr)
    match = true_arr == gen_arr
    cummatch = np.cumsum(match)
    mismatches = np.where(~match)[0]

    result = {}
    for suffix_boundary in usable:
        first_mismatch = mismatches[mismatches < suffix_boundary]
        result[suffix_boundary] = {
            "Rouge-L": compute_rouge_l_2d(rouge_dp[:suffix_boundary + 1, :suffix_boundary + 1]),
            "lcs_norm": float(lcs_dp[:suffix_boundary + 1, :suffix_boundary + 1].max()) / suffix_boundary,
            "TTR_ref": len(set(true_suffix[:suffix_boundary])) / suffix_boundary,
            "TTR_gen": len(set(gen_suffix[:suffix_boundary])) / suffix_boundary,
            "token_acc": float(cummatch[suffix_boundary - 1]) / suffix_boundary,
            "divergence_point": float(first_mismatch[0]) / suffix_boundary if len(first_mismatch) else 1.0,
        }
    return result


def nll_p_z_at_suffix_boundaries(ref_nll: list, gen_nll: list, p_z_logprob: list, suffix_boundaries: list) -> dict:
    """{suffix_boundary: {ref_nll_mean, gen_nll_mean, p_z}} from Step 1's stored
    per-position arrays."""
    usable = [b for b in suffix_boundaries if b <= len(ref_nll)]
    result = {}
    for suffix_boundary in usable:
        result[suffix_boundary] = {
            "ref_nll_mean": float(np.mean(ref_nll[:suffix_boundary])),
            "gen_nll_mean": float(np.mean(gen_nll[:suffix_boundary])),
            "p_z": float(np.exp(np.sum(p_z_logprob[:suffix_boundary]))),
        }
    return result


### PER-REP AGGREGATION (dataset-level match_x + Results-shaped scores/mean/std) ###

def compute_rep_metrics(records: list, suffix_boundaries: list) -> dict:
    """records: one rep bucket's jsonl records. Returns {suffix_boundary: {metric: {scores, mean, std}}}."""
    per_suffix_boundary = defaultdict(lambda: defaultdict(list))
    missing_nll = 0
    for rec in records:
        text_m = text_metrics_at_suffix_boundaries(rec["true_suffix"], rec["generated_suffix"], suffix_boundaries)
        for suffix_boundary, metrics in text_m.items():
            for name, val in metrics.items():
                per_suffix_boundary[suffix_boundary][name].append(val)
        if "ref_nll" in rec:
            nll_m = nll_p_z_at_suffix_boundaries(rec["ref_nll"], rec["gen_nll"], rec["p_z_logprob"], suffix_boundaries)
            for suffix_boundary, metrics in nll_m.items():
                for name, val in metrics.items():
                    per_suffix_boundary[suffix_boundary][name].append(val)
        else:
            missing_nll += 1
    if missing_nll:
        print(f"  {missing_nll}/{len(records)} record(s) missing ref_nll -- NLL/p_z skipped for "
              "them (run backfill first to fill them in)")

    result = {}
    for suffix_boundary, metrics in per_suffix_boundary.items():
        result[suffix_boundary] = {}
        for name, values in metrics.items():
            arr = np.array(values)
            result[suffix_boundary][name] = {"scores": arr, "mean": float(arr.mean()), "std": float(arr.std())}
        lcs_arr = np.array(metrics["lcs_norm"])
        n = len(lcs_arr)
        for label, threshold in [("exact_match", 1.0), ("match_75", 0.75), ("match_50", 0.5), ("match_25", 0.25)]:
            count = int((lcs_arr >= threshold).sum())
            result[suffix_boundary][label] = {"scores": count, "mean": (count / n if n else 0.0), "std": 0}
    return result


### INFERENCE-REP DISCOVERY (reads Step 1's jsonl on disk) ###

def find_inference_reps(expr_dir: Path, offset: int, prefix_length: int,
                        persistent_expr_dir: Path | None = None) -> dict:
    """{rep: (max_suffix_available, rep_dir_path)} across every suffix dir for
    (offset, prefix_length) -- a rep can have data in more than one suffix dir (e.g. after
    an extend), only the largest one matters for how far it reaches.
    persistent_expr_dir is checked as a fallback for reps expr_dir no longer has."""
    result = {}
    for suffix_prime, d in find_suffix_dirs(expr_dir, offset, prefix_length, persistent_expr_dir):
        for rep_dir in d.iterdir():
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                continue
            if not any(rep_dir.glob("rank*.jsonl")):
                continue
            rep = int(rep_dir.name.split("_")[1])
            if rep not in result or suffix_prime > result[rep][0]:
                result[rep] = (suffix_prime, rep_dir)
    return result


def load_inference_rep_records(rep_dir: Path) -> list:
    records = []
    for rank_file in sorted(rep_dir.glob("rank*.jsonl")):
        with open(rank_file) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    return records


### METADATA ###

def append_metrics_metadata(exp_dir: Path, suffix_boundary: int, offset: int, prefix_length: int, reps: list) -> None:
    """One file per call under metrics_metadata/, instead of one shared growing list --
    separate (offset, prefix) jobs for the same experiment run concurrently and would
    otherwise race on a single read-modify-write file."""
    meta_dir = exp_dir / "metrics_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    meta_path = meta_dir / f"offset_{offset}_prefix_{prefix_length}_suffix_{suffix_boundary}_{timestamp.strftime('%Y%m%dT%H%M%S%f')}.json"
    with open(meta_path, "w") as f:
        json.dump({
            "action": "metrics",
            "offset": offset,
            "prefix_length": prefix_length,
            "suffix_length": suffix_boundary,
            "reps": reps,
            "timestamp": timestamp.isoformat(),
        }, f, indent=2)


### PKL PATHS + DONE-CHECK ###

def build_pkl_path(storage_path: Path, exp_name: str, offset: int, prefix_length: int, suffix_boundary: int,
                   policy: str = "greedy", tag: str | None = None) -> Path:
    suffix_tag = f"_{tag}" if tag else ""
    return storage_path / exp_name / "metrics" / f"offset_{offset}_prefix_{prefix_length}_suffix_{suffix_boundary}_{policy}{suffix_tag}.pkl"


def find_missing_metrics_reps(pkl_paths: list, inference_reps_reaching_suffix_length: set, exp_name: str) -> set:
    """inference_reps_reaching_suffix_length not yet recorded in the metrics pkl -- empty
    means this suffix boundary is already covered. pkl_paths is checked in order (scratch
    first, then persistent store); the first path that exists is the one read. A pkl built
    from a smaller rep set (e.g. before REPETITIONS grew) is correctly reported as still
    missing the new reps."""
    metrics_reps = set()
    existing_path = next((p for p in pkl_paths if p.exists()), None)
    if existing_path is not None:
        with open(existing_path, "rb") as f:
            metrics_reps = set(pickle.load(f).get(exp_name, {}).keys())
    return inference_reps_reaching_suffix_length - metrics_reps


def print_dry_run_report(exp_name: str, offset: int, prefix_length: int,
                         missing_metrics_reps_by_suffix_boundary: dict) -> None:
    done = sorted(b for b, missing in missing_metrics_reps_by_suffix_boundary.items() if not missing)
    needed = sorted(b for b, missing in missing_metrics_reps_by_suffix_boundary.items() if missing)
    print(f"{exp_name} offset={offset} prefix={prefix_length}: "
          f"{len(done)}/{len(missing_metrics_reps_by_suffix_boundary)} suffix boundaries complete",
          file=sys.stderr)
    for suffix_boundary in needed:
        print(f"  needed: suffix_boundary={suffix_boundary}  "
              f"missing_metrics_reps={sorted(missing_metrics_reps_by_suffix_boundary[suffix_boundary])}",
              file=sys.stderr)


### MAIN LOOP ###

def find_missing_metrics_reps_by_suffix_boundary(
        exp_name: str, offset: int, prefix_length: int,
        inference_reps_reaching_suffix_length: set, target_suffix_boundaries: list,
        save_path: Path, persistent_save_path: Path | None, policy: str, tag: str | None) -> dict:
    """Check phase only: for each boundary in target_suffix_boundaries, which of
    inference_reps_reaching_suffix_length are missing from the metrics pkl (scratch, then
    persistent_save_path)."""
    def pkl_candidates(suffix_boundary: int) -> list:
        paths = [build_pkl_path(save_path, exp_name, offset, prefix_length, suffix_boundary, policy, tag)]
        if persistent_save_path is not None:
            paths.append(build_pkl_path(persistent_save_path, exp_name, offset, prefix_length,
                                        suffix_boundary, policy, tag))
        return paths

    return {
        suffix_boundary: find_missing_metrics_reps(
            pkl_candidates(suffix_boundary), inference_reps_reaching_suffix_length, exp_name)
        for suffix_boundary in target_suffix_boundaries
    }


def write_suffix_boundary_pkls(exp_name: str, offset: int, prefix_length: int, inference_reps: dict,
                               inference_reps_reaching_suffix_length: set, suffix_boundaries_to_compute: list,
                               save_path: Path, policy: str, tag: str | None) -> None:
    """Compute+write phase only: one DP fill per qualifying rep, across every suffix
    boundary being computed -- not one fill per boundary. Recomputing a boundary
    regenerates the whole file with the full rep set (not a merge) -- Step 2 is cheap
    enough (CPU-only, no forward pass) that this isn't a real cost concern even across a
    large sweep."""
    per_rep_metrics = {}
    for rep in sorted(inference_reps_reaching_suffix_length):
        _, rep_dir = inference_reps[rep]
        records = load_inference_rep_records(rep_dir)
        if not records:
            continue
        per_rep_metrics[rep] = compute_rep_metrics(records, suffix_boundaries_to_compute)

    for suffix_boundary in suffix_boundaries_to_compute:
        data = {exp_name: {}}
        reps_included = []
        for rep, rep_metrics_by_suffix_boundary in per_rep_metrics.items():
            if suffix_boundary in rep_metrics_by_suffix_boundary:
                data[exp_name].setdefault(rep, {}).setdefault(offset, {}).setdefault(prefix_length, {})[suffix_boundary] = rep_metrics_by_suffix_boundary[suffix_boundary]
                reps_included.append(rep)
        if not data[exp_name]:
            continue
        results = Results.from_raw_dict(data, policy=policy)
        path = build_pkl_path(save_path, exp_name, offset, prefix_length, suffix_boundary, policy, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(results.data, f)
        append_metrics_metadata(save_path / exp_name, suffix_boundary, offset, prefix_length, sorted(reps_included))
        print(f"Saved: {path}  (reps={sorted(reps_included)})")


def process_point(exp_name: str, offset: int, prefix_length: int, suffix_length: int,
                  inference_reps: dict, suffix_boundaries: list, save_path: Path,
                  policy: str = "greedy", tag: str | None = None, force: bool = False,
                  persistent_save_path: Path | None = None, dry_run: bool = False) -> dict | None:
    """One point's worth of work: computes inference_reps_reaching_suffix_length once (not
    per boundary), then only suffix boundaries <= suffix_length -- never beyond, even if
    some rep reaches further.

    If none reach suffix_length: raises ValueError on a real run (Stage 1 just ran on this
    same point, so that's a real bug), or returns None on --dry-run (not ready yet --
    distinct from an empty missing-set, which means "evaluated, nothing missing").

    Otherwise returns {suffix_boundary: missing_metrics_reps}, dry-run or real -- one
    source of truth for both the report and the recompute decision.
    """
    inference_reps_reaching_suffix_length = {
        rep for rep, (max_suffix, _) in inference_reps.items() if max_suffix >= suffix_length
    }

    if not inference_reps_reaching_suffix_length:
        if not dry_run:
            reached = {rep: max_suffix for rep, (max_suffix, _) in inference_reps.items()}
            raise ValueError(
                f"{exp_name} offset={offset} prefix={prefix_length}: no inference rep reaches "
                f"suffix_length={suffix_length} (reps found: {reached or 'none'})"
            )
        print(f"{exp_name} offset={offset} prefix={prefix_length}: no inference rep reaches "
              f"suffix_length={suffix_length} yet -- needs Stage 1 first.", file=sys.stderr)
        return None

    target_suffix_boundaries = sorted({b for b in suffix_boundaries if b <= suffix_length} | {suffix_length})
    missing_metrics_reps_by_suffix_boundary = find_missing_metrics_reps_by_suffix_boundary(
        exp_name, offset, prefix_length, inference_reps_reaching_suffix_length,
        target_suffix_boundaries, save_path, persistent_save_path, policy, tag)

    if dry_run:
        print_dry_run_report(exp_name, offset, prefix_length, missing_metrics_reps_by_suffix_boundary)
        return missing_metrics_reps_by_suffix_boundary

    suffix_boundaries_to_compute = sorted(
        suffix_boundary for suffix_boundary, missing in missing_metrics_reps_by_suffix_boundary.items()
        if force or missing
    )
    if suffix_boundaries_to_compute:
        write_suffix_boundary_pkls(exp_name, offset, prefix_length, inference_reps,
                                   inference_reps_reaching_suffix_length, suffix_boundaries_to_compute,
                                   save_path, policy, tag)

    return missing_metrics_reps_by_suffix_boundary


def process_expr(exp_name: str, base_path: Path, save_path: Path, suffix_boundaries: list, args) -> list:
    """Runs every point in args.points for one experiment. Returns the (offset, prefix)
    points still needing work at args.suffix_length -- only meaningful when args.dry_run;
    a real run's return value is unused."""
    expr_dir = base_path / exp_name
    persistent_expr_dir = Path(args.persistent_storage_path) / exp_name if args.persistent_storage_path else None
    persistent_save_path = Path(args.persistent_storage_path) if args.persistent_storage_path else None

    needed_points = []
    for offset, prefix_length in sorted(set(parse_points(args.points))):
        inference_reps = find_inference_reps(expr_dir, offset, prefix_length, persistent_expr_dir)
        print(f"\n=== {exp_name}  offset={offset} prefix={prefix_length}  "
              f"inference_reps={sorted(inference_reps)} ===")
        missing_metrics_reps_by_suffix_boundary = process_point(
            exp_name, offset, prefix_length, args.suffix_length, inference_reps, suffix_boundaries,
            save_path, tag=args.tag, force=args.force, persistent_save_path=persistent_save_path,
            dry_run=args.dry_run)
        if args.dry_run:
            # None = not even evaluable yet (needed); a dict = check the requested boundary.
            needed = (missing_metrics_reps_by_suffix_boundary is None
                     or missing_metrics_reps_by_suffix_boundary.get(args.suffix_length))
            if needed:
                needed_points.append((offset, prefix_length))
    return needed_points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z per suffix boundary")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Experiment identifier, e.g. llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B "
                             "(matches EXP_NAME/PDM_EXP_NAME in the .slurm scripts) -- always one "
                             "model/variant per invocation, never a list.")
    parser.add_argument("--base-path", type=str, required=True, help="e.g. $MEM_BASE/SparseGutenberg")
    parser.add_argument("--save-path", type=str, required=True, help="Where to write per-suffix .pkl files")
    parser.add_argument("--persistent-storage-path", type=str, default=None,
                        help="Secondary mirror of --base-path, checked as a fallback for both "
                             "inference-data discovery and existing-pkl reads. Never written to.")
    parser.add_argument("--points", nargs="+", required=True,
                        help="offset:prefix pairs to process, matching Step 1's --points format")
    parser.add_argument("--suffix-length", type=int, required=True,
                        help="Target suffix length -- every point's reps must reach at least this "
                             "far (error otherwise). Computes this suffix boundary plus every "
                             "intermediate one, never beyond it even if a rep's data reaches further.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Appended after policy in the pkl filename (e.g. --tag opt -> "
                             "..._greedy_opt.pkl), so a validation run never overwrites the "
                             "real pkl at the same path. Omit for the normal, real output.")
    parser.add_argument("--force", action="store_true", help="Recompute even if the pkl already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what's missing, write nothing, no GPU/model involved.")
    args = parser.parse_args()

    needed_points = sorted(set(process_expr(args.exp_name, Path(args.base_path), Path(args.save_path),
                                            SUFFIX_BOUNDARIES, args)))

    if args.dry_run:
        if not needed_points:
            print(f"All requested points already complete at suffix_length={args.suffix_length} "
                  "-- nothing will run.", file=sys.stderr)
        print(" ".join(f"{o}:{p}" for o, p in needed_points))

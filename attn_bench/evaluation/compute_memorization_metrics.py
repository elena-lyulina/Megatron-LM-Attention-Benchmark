"""
Step 2: Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z per suffix boundary, computed
purely from Step 1's jsonl (CPU-only, no model).

For one --exp-name, processes every --points pair at --suffix-length, writing one
PDM-Results-shaped .pkl per suffix boundary <= --suffix-length under exp_dir/metrics/.
Skip-if-done compares an existing pkl's reps against what Stage 1 has on disk (scratch,
then --persistent-storage-path); --repetitions widens that to also catch reps that were
requested but never generated at Stage 1 at all (see process_point).

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

from attn_bench.evaluation.inference_common import (filter_points_by_doc_length,
                                                    find_suffix_dirs,
                                                    parse_points)

SUFFIX_BOUNDARIES = [25, 50, 75, 100, 150, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000]


### lcs_norm -- local copy of PDM's _find_lcs, keeping the array instead of a running max ###
# (PDM/src/verbatim_eval/LCS.py's public function discards it; filling once and reading
# dp[:k+1,:k+1].max() per boundary is correct and ~2.2x cheaper than refilling per boundary.)
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
    (offset, prefix_length); persistent_expr_dir is a fallback for reps expr_dir lacks."""
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
    """One file per call under metrics_metadata/ -- avoids a race between concurrent jobs
    on a single shared read-modify-write file."""
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


def find_missing_metrics_reps(pkl_paths: list, target_reps: set, exp_name: str) -> set:
    """target_reps not yet recorded in the metrics pkl. pkl_paths is checked in order
    (scratch first, then persistent store); the first existing one is read."""
    metrics_reps = set()
    existing_path = next((p for p in pkl_paths if p.exists()), None)
    if existing_path is not None:
        with open(existing_path, "rb") as f:
            metrics_reps = set(pickle.load(f).get(exp_name, {}).keys())
    return target_reps - metrics_reps


def print_dry_run_summary(exp_name: str, suffix_length: int, point_results: dict) -> None:
    """point_results: {(offset, prefix_length): missing_metrics_reps_by_suffix_boundary or
    None}. One report per exp_name, status at suffix_length only -- that's the boundary
    that decides submit/skip, not every intermediate one."""
    not_ready = sorted(pt for pt, result in point_results.items() if result is None)
    incomplete = {
        pt: result[suffix_length]
        for pt, result in point_results.items()
        if result is not None and result.get(suffix_length)
    }
    done = len(point_results) - len(not_ready) - len(incomplete)

    print(f"{exp_name} @ suffix_length={suffix_length}: "
          f"{done}/{len(point_results)} points complete", file=sys.stderr)
    if not_ready:
        points_str = ", ".join(f"{o}:{p}" for o, p in not_ready)
        print(f"  missing entirely (no inference yet): {points_str}", file=sys.stderr)
    if incomplete:
        parts = [f"{o}:{p} missing reps={sorted(reps)}" for (o, p), reps in sorted(incomplete.items())]
        print(f"  incomplete (some reps missing): {', '.join(parts)}", file=sys.stderr)


### MAIN LOOP ###

def find_missing_metrics_reps_by_suffix_boundary(
        exp_name: str, offset: int, prefix_length: int,
        inference_reps_reaching_suffix_length: set, target_suffix_boundaries: list,
        save_path: Path, persistent_save_path: Path | None, policy: str, tag: str | None,
        requested_reps: set | None = None) -> dict:
    """Check phase only: per boundary, which reps (union of what's on disk and
    requested_reps) are missing from the metrics pkl."""
    target_reps = inference_reps_reaching_suffix_length | (requested_reps or set())

    def pkl_candidates(suffix_boundary: int) -> list:
        paths = [build_pkl_path(save_path, exp_name, offset, prefix_length, suffix_boundary, policy, tag)]
        if persistent_save_path is not None:
            paths.append(build_pkl_path(persistent_save_path, exp_name, offset, prefix_length,
                                        suffix_boundary, policy, tag))
        return paths

    return {
        suffix_boundary: find_missing_metrics_reps(pkl_candidates(suffix_boundary), target_reps, exp_name)
        for suffix_boundary in target_suffix_boundaries
    }


def write_suffix_boundary_pkls(exp_name: str, offset: int, prefix_length: int, inference_reps: dict,
                               inference_reps_reaching_suffix_length: set, suffix_boundaries_to_compute: list,
                               save_path: Path, policy: str, tag: str | None) -> None:
    """Compute+write phase only: one DP fill per rep across every boundary being computed.
    Recomputing a boundary regenerates the whole file (not a merge) -- cheap enough (CPU-only)
    not to matter."""
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
                  persistent_save_path: Path | None = None, dry_run: bool = False,
                  requested_reps: set | None = None) -> dict | None:
    """One point's worth of work, for suffix boundaries <= suffix_length.

    If no rep reaches suffix_length: raises on a real run, returns None on --dry-run.
    requested_reps (--repetitions): on --dry-run, widens the missing-reps check to reps not
    yet on disk at all, so a point needing fresh Stage 1 generation is reported as needed;
    on a real run, raises if a requested rep still isn't available (Stage 1 should have
    produced it in the same job).

    Otherwise returns {suffix_boundary: missing_metrics_reps}."""
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
        return None

    if not dry_run and requested_reps:
        missing_requested = requested_reps - inference_reps_reaching_suffix_length
        if missing_requested:
            raise ValueError(
                f"{exp_name} offset={offset} prefix={prefix_length}: requested rep(s) "
                f"{sorted(missing_requested)} don't reach suffix_length={suffix_length} "
                f"(reps available: {sorted(inference_reps_reaching_suffix_length)}) -- Stage 1 "
                "should have generated them in this same job; check its logs."
            )

    target_suffix_boundaries = sorted({b for b in suffix_boundaries if b <= suffix_length} | {suffix_length})
    missing_metrics_reps_by_suffix_boundary = find_missing_metrics_reps_by_suffix_boundary(
        exp_name, offset, prefix_length, inference_reps_reaching_suffix_length,
        target_suffix_boundaries, save_path, persistent_save_path, policy, tag,
        requested_reps=requested_reps)

    if dry_run:
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


def process_expr(exp_name: str, base_path: Path, save_path: Path, suffix_boundaries: list, args) -> tuple:
    """Runs every point in args.points for one experiment. Returns (needed_points,
    missing_point_reps) -- the (offset, prefix) points still needing work at
    args.suffix_length, and the total count of missing (offset, prefix, rep) units across
    them -- only meaningful when args.dry_run; a real run's return value is unused."""
    expr_dir = base_path / exp_name
    persistent_expr_dir = Path(args.persistent_storage_path) / exp_name if args.persistent_storage_path else None
    persistent_save_path = Path(args.persistent_storage_path) if args.persistent_storage_path else None
    requested_reps = {int(r) for r in args.repetitions.split(",")} if args.repetitions else None

    all_points = sorted(set(parse_points(args.points)))
    all_points = filter_points_by_doc_length(all_points, args.suffix_length, args.max_doc_length)
    needed_points = []
    missing_point_reps = 0
    point_results = {}
    for i, (offset, prefix_length) in enumerate(all_points, 1):
        inference_reps = find_inference_reps(expr_dir, offset, prefix_length, persistent_expr_dir)
        if not args.dry_run:
            print(f"\n=== [{i}/{len(all_points)}] {exp_name}  offset={offset} prefix={prefix_length}  "
                  f"inference_reps={sorted(inference_reps)} ===", file=sys.stderr)
        missing_metrics_reps_by_suffix_boundary = process_point(
            exp_name, offset, prefix_length, args.suffix_length, inference_reps, suffix_boundaries,
            save_path, tag=args.tag, force=args.force, persistent_save_path=persistent_save_path,
            dry_run=args.dry_run, requested_reps=requested_reps)
        if args.dry_run:
            point_results[(offset, prefix_length)] = missing_metrics_reps_by_suffix_boundary
            if missing_metrics_reps_by_suffix_boundary is None:
                # Not evaluable yet (no inference at all) -- every requested rep counts as missing.
                needed_points.append((offset, prefix_length))
                missing_point_reps += len(requested_reps) if requested_reps else 1
            else:
                missing = missing_metrics_reps_by_suffix_boundary.get(args.suffix_length)
                if missing:
                    needed_points.append((offset, prefix_length))
                    missing_point_reps += len(missing)

    if args.dry_run:
        print_dry_run_summary(exp_name, args.suffix_length, point_results)

    return needed_points, missing_point_reps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Rouge-L/lcs_norm/TTR/token_acc/divergence_point/p_z per suffix boundary")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="Experiment identifier, matching EXP_NAME/PDM_EXP_NAME in the .slurm scripts.")
    parser.add_argument("--base-path", type=str, required=True, help="e.g. $MEM_BASE/SparseGutenberg")
    parser.add_argument("--save-path", type=str, required=True, help="Where to write per-suffix .pkl files")
    parser.add_argument("--persistent-storage-path", type=str, default=None,
                        help="Secondary mirror of --base-path, checked as a fallback for both "
                             "inference-data discovery and existing-pkl reads. Never written to.")
    parser.add_argument("--points", nargs="+", required=True,
                        help="offset:prefix pairs to process, matching Step 1's --points format")
    parser.add_argument("--suffix-length", type=int, required=True,
                        help="Target suffix length -- computes this boundary plus every intermediate "
                             "one, never beyond it.")
    parser.add_argument("--max-doc-length", type=int, default=None,
                        help="Skip a point where offset+prefix+suffix exceeds this (same check as "
                             "Stage 1's --max-doc-length). Omit for no filtering.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Appended after policy in the pkl filename (e.g. --tag opt -> "
                             "..._greedy_opt.pkl), so a validation run never overwrites the "
                             "real pkl at the same path. Omit for the normal, real output.")
    parser.add_argument("--force", action="store_true", help="Recompute even if the pkl already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what's missing, write nothing, no GPU/model involved.")
    parser.add_argument("--repetitions", type=str, default=None,
                        help="Comma-separated reps a point is expected to cover, matching Stage 1's "
                             "--repetitions. See process_point for --dry-run vs. real-run behavior.")
    args = parser.parse_args()

    needed_points, missing_point_reps = process_expr(args.exp_name, Path(args.base_path),
                                                     Path(args.save_path), SUFFIX_BOUNDARIES, args)
    needed_points = sorted(set(needed_points))

    if args.dry_run:
        if not needed_points:
            print(f"All requested points already complete at suffix_length={args.suffix_length} "
                  "-- nothing will run.", file=sys.stderr)
        print(" ".join(f"{o}:{p}" for o, p in needed_points))
        print(missing_point_reps)

"""
Unit naming and completion-check logic for inference_perf.py, kept free of
torch/megatron imports so it can run as plain python3 with no container or
GPU allocation -- used both by inference_perf.slurm (checking scratch after
a run, to decide whether to copy to store) and inference_perf_all.sh
(checking store before submitting a job at all).

Usage: python3 inference_perf_units.py --dir <path>   (exits 0 if all done, 1 otherwise)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREFILL_BATCH_SIZE = 20
DECODE_PREFIX_ANCHORS = [50, 5000]
DECODE_BATCH_SIZE = 20
SWEEP_BATCH_SIZES = [1, 5, 10, 20, 40]
SWEEP_PREFIX = 50


def prefill_rel_path() -> str:
    return f"prefill_bs{PREFILL_BATCH_SIZE}.json"


def decode_rel_path(batch_size: int, prefix_length: int) -> str:
    return f"bs{batch_size}_prefix{prefix_length}.json"


def sweep_rel_path(batch_size: int, prefix_length: int) -> str:
    # Separate namespace from decode_rel_path: DECODE_BATCH_SIZE/DECODE_PREFIX_ANCHORS
    # and SWEEP_BATCH_SIZES/SWEEP_PREFIX can coincide (both currently include 20/50),
    # and the two experiments have different decode_steps -- same name would collide.
    return f"sweep_bs{batch_size}_prefix{prefix_length}.json"


def all_units() -> list[str]:
    units = [prefill_rel_path()]
    units += [decode_rel_path(DECODE_BATCH_SIZE, p) for p in DECODE_PREFIX_ANCHORS]
    units += [sweep_rel_path(b, SWEEP_PREFIX) for b in SWEEP_BATCH_SIZES]
    return units


def already_done(output_dir: Path, rel_path: str) -> bool:
    path = output_dir / rel_path
    if not path.exists():
        return False
    with open(path) as f:
        return json.load(f).get("status") == "ok"


def all_done(output_dir: Path) -> bool:
    return all(already_done(output_dir, unit) for unit in all_units())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    done = all_done(Path(args.dir))
    print("done" if done else "not done")
    sys.exit(0 if done else 1)
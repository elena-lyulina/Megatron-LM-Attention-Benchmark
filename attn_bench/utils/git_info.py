"""
Captures the current commit hash + the working-tree diff for W&B logging.

Rules:
  - Always log the commit hash
  - If the tree is dirty AND diff <= MAX_DIFF_LINES: log the diff
  - If the tree is dirty AND diff > MAX_DIFF_LINES: raise RuntimeError
  - If the tree is clean: diff field is None
"""

import subprocess
from pathlib import Path
from typing import Optional

MAX_DIFF_LINES = 100

# Restricting to attn_bench/ to only track relevant changes
DEFAULT_WATCH_PATHS: list[str] = ["attn_bench/"]

# Prevents log / result files from counting towards the status check / dif
DEFAULT_EXCLUDES: list[str] = [
    ":(exclude)*.log",
    ":(exclude)*.out",
    ":(exclude)*.err",
    ":(exclude)results/",
]

# Repo root derived from this file's location, works regardless of cwd
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, cwd=_REPO_ROOT
    )
    return result.stdout.strip()


def get_git_status(
        watch_paths: Optional[list[str]] = None,
        excludes: Optional[list[str]] = None,
) -> bool:
    if watch_paths is None:
        watch_paths = DEFAULT_WATCH_PATHS
    if excludes is None:
        excludes = DEFAULT_EXCLUDES

    status = _run(["git", "status", "--porcelain", "--"] + watch_paths + excludes)
    return bool(status)


def check_git_working_tree(
    watch_paths: Optional[list[str]] = None,
    excludes: Optional[list[str]] = None,
    max_diff_lines: int = MAX_DIFF_LINES,
) -> None:
    if watch_paths is None:
        watch_paths = DEFAULT_WATCH_PATHS
    if excludes is None:
        excludes = DEFAULT_EXCLUDES

    status = get_git_status(watch_paths, excludes)
    if not status:
        return

    diff = _run(["git", "--no-pager", "diff", "HEAD", "--"] + watch_paths + excludes)
    n_lines = diff.count("\n") if diff else 0
    if n_lines > max_diff_lines:
        raise RuntimeError(
            f"Working tree is dirty with {n_lines} diff lines (limit: {max_diff_lines})."
            f" Commit or stash your changes before running the benchmark."
        )


def get_git_info(
        watch_paths: Optional[list[str]] = None,
        excludes: Optional[list[str]] = None,
) -> dict:
    if watch_paths is None:
        watch_paths = DEFAULT_WATCH_PATHS
    if excludes is None:
        excludes = DEFAULT_EXCLUDES

    commit_hash = _run(["git", "rev-parse", "HEAD"])

    status = get_git_status(watch_paths, excludes)
    dirty = bool(status)

    diff: Optional[str] = None
    if dirty:
        diff = _run(["git", "--no-pager", "diff", "HEAD", "--"] + watch_paths + excludes)
    return {
        "commit_hash": commit_hash,
        "dirty": dirty,
        "diff": diff,
    }

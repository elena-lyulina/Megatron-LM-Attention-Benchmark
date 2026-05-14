from __future__ import annotations

from pathlib import Path

from datasets import disable_progress_bars, enable_progress_bars, load_from_disk


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def find_and_load_latest_ckpt(ckpt_dir: Path | None):
    """Return (step_name, dataset) for the latest checkpoint dir, or None."""
    if not ckpt_dir or not ckpt_dir.exists():
        return None
    candidates = sorted((p for p in ckpt_dir.iterdir() if p.is_dir()), reverse=True)
    if not candidates:
        return None
    p = candidates[0]
    print(f"Fast-forward: loading checkpoint {p.name}")
    return p.name, load_from_disk(str(p))


def run_step(fn, ds, ckpt_path: Path | None):
    if ckpt_path and ckpt_path.exists():
        print(f"  [{ckpt_path.name}] loading checkpoint")
        ds = load_from_disk(str(ckpt_path))
        n_removed = ds.cleanup_cache_files()
        if n_removed:
            print(f"  [{ckpt_path.name}] cleaned {n_removed} stale cache files")
        return ds, dir_size_gb(ckpt_path)
    ds = fn(ds)
    if ckpt_path:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        disable_progress_bars()
        ds.save_to_disk(str(ckpt_path))
        enable_progress_bars()
        size = dir_size_gb(ckpt_path)
        print(f"  [{ckpt_path.name}] checkpoint saved ({size:.2f} GB)")
        return ds, size
    return ds, None
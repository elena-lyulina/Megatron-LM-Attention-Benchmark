"""
Download a raw dataset from HuggingFace and save files flat under output-dir.

Usage:
    python download_dataset.py --dataset fineweb-edu-dedup --raw-dir /path/to/raw
    python download_dataset.py --dataset gutenberg-en      --raw-dir /path/to/raw
    python download_dataset.py --dataset nemotron          --raw-dir /path/to/raw
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import fnmatch

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [INFO] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATASETS = {
    "fineweb-edu-dedup": {
        "repo_id": "HuggingFaceTB/smollm-corpus",
        "allow_patterns": "fineweb-edu-dedup/*.parquet",
        "extension": "parquet",
    },
    "gutenberg-en": {
        "repo_id": "common-pile/project_gutenberg",
        "allow_patterns": "v0/documents/*.gz",
        "extension": "gz",
    },
    "nemotron": {
        "repo_id": "nvidia/Nemotron-PII",
        "allow_patterns": "data/*.parquet",
        "extension": "parquet",
    },
}


def flatten_files(output_dir: Path, extension: str) -> None:
    """Move all files with given extension from subdirectories up to output_dir, then remove empty subdirs."""
    for f in list(output_dir.rglob(f"*.{extension}")):
        if f.parent != output_dir:
            dest = output_dir / f.name
            shutil.move(str(f), dest)
            logger.info(f"  moved {f.relative_to(output_dir.parent)} -> {dest.name}")
    for subdir in sorted(output_dir.iterdir(), key=lambda p: len(p.parts), reverse=True):
        if subdir.is_dir():
            try:
                subdir.rmdir()
            except OSError:
                pass


def main(args):
    cfg = DATASETS[args.dataset]
    ext = cfg["extension"]
    output_dir = Path(args.raw_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")

    logger.info(f"Downloading {cfg['repo_id']} to {output_dir} ...")

    if args.n_percent < 100:
        all_files = [
            f for f in list_repo_files(cfg["repo_id"], repo_type="dataset", token=token)
            if fnmatch.fnmatch(f, cfg["allow_patterns"])
        ]
        n = max(1, int(len(all_files) * args.n_percent / 100))
        logger.info(f"Downloading {n}/{len(all_files)} files ({args.n_percent}%) ...")
        for filename in all_files[:n]:
            hf_hub_download(cfg["repo_id"], filename, repo_type="dataset", local_dir=str(output_dir), token=token)
    else:
        snapshot_download(
            repo_id=cfg["repo_id"],
            repo_type="dataset",
            allow_patterns=cfg["allow_patterns"],
            local_dir=str(output_dir),
            token=token,
        )

    logger.info(f"Flattening {ext} files ...")
    flatten_files(output_dir, ext)

    files = sorted(output_dir.glob(f"*.{ext}"))
    logger.info(f"Done: {len(files)} {ext} files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a raw dataset from HuggingFace")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--n-percent", type=int, default=100,
                        help="Download only this percentage of files (1-100)")
    args = parser.parse_args()
    main(args)
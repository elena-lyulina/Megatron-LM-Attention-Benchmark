"""
Download a raw dataset from HuggingFace and save parquet files flat under output-dir.

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

from huggingface_hub import snapshot_download

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
    },
    "gutenberg-en": {
        "repo_id": "manu/project_gutenberg",
        "allow_patterns": "data/en-*.parquet",
    },
    "nemotron": {
        "repo_id": "nvidia/Nemotron-PII",
        "allow_patterns": "data/*.parquet",
    },
}


def flatten_parquets(output_dir: Path) -> None:
    """Move all parquet files from subdirectories up to output_dir, then remove empty subdirs."""
    for parquet in list(output_dir.rglob("*.parquet")):
        if parquet.parent != output_dir:
            dest = output_dir / parquet.name
            shutil.move(str(parquet), dest)
            logger.info(f"  moved {parquet.relative_to(output_dir.parent)} -> {dest.name}")
    for subdir in sorted(output_dir.iterdir(), key=lambda p: len(p.parts), reverse=True):
        if subdir.is_dir():
            try:
                subdir.rmdir()
            except OSError:
                pass


def main(args):
    cfg = DATASETS[args.dataset]
    output_dir = Path(args.raw_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {cfg['repo_id']} to {output_dir} ...")
    snapshot_download(
        repo_id=cfg["repo_id"],
        repo_type="dataset",
        allow_patterns=cfg["allow_patterns"],
        local_dir=str(output_dir),
        token=os.environ.get("HF_TOKEN"),
    )

    logger.info("Flattening parquet files ...")
    flatten_parquets(output_dir)

    parquets = sorted(output_dir.glob("*.parquet"))
    logger.info(f"Done: {len(parquets)} parquet files in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a raw dataset from HuggingFace")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--raw-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
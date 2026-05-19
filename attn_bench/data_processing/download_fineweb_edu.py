"""
Download FineWeb-Edu-dedup parquet files from HuggingFace to local storage.

Usage:
    python download_fineweb_edu.py --output-dir /path/to/raw
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

DATASET_REPO = "HuggingFaceTB/smollm-corpus"
DATASET_CONFIG = "fineweb-edu-dedup"


def main(args):
    logger.info(f"Downloading {DATASET_CONFIG} parquet files to {args.output_dir} ...")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=f"{DATASET_CONFIG}/*.parquet",
        local_dir=args.output_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    logger.info("Download complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu-dedup parquet files")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
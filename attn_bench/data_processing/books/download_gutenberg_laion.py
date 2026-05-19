"""
Download laion/Project-Gutenberg and build a HuggingFace dataset.

Step 1: download english_epub.tar.gz -> <base-dir>/raw/  (skipped if already there)
Step 2: read epubs from tar, extract all metadata, save as HF dataset
        --n-books N  -> <base-dir>/raw-N/     (first N books)
        (no flag)    -> <base-dir>/raw-full/  (all books)

Usage:
  python -m attn_bench.data_processing.books.download_gutenberg_laion \
      --base-dir /path/to/gutenberg-laion [--n-books 7000]
"""
from __future__ import annotations

import argparse
import io
import re
import tarfile
import zipfile
from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


REPO_ID = "laion/Project-Gutenberg"
FILENAME = "english_epub.tar.gz"

def _epub_metadata(epub_data: bytes) -> dict:
    try:
        with zipfile.ZipFile(io.BytesIO(epub_data)) as z:
            container = z.read("META-INF/container.xml").decode("utf-8", errors="replace")
            opf_match = re.search(r'full-path="([^"]+\.opf)"', container)
            if not opf_match:
                return {}
            opf = z.read(opf_match.group(1)).decode("utf-8", errors="replace")

        def get(tag):
            m = re.search(rf"<{tag}[^>]*>([^<]+)</{tag}>", opf)
            return m.group(1).strip() if m else None

        def get_date(event):
            m = re.search(rf'<dc:date[^>]*opf:event="{event}"[^>]*>([^<]+)</dc:date>', opf)
            return m.group(1).strip() if m else None

        identifier = get("dc:identifier")
        id_match = re.search(r'/(\d+)$', identifier) if identifier else None

        contributor_m = re.search(r'<dc:contributor[^>]*>([^<]+)</dc:contributor>', opf)

        meta = {
            "gutenberg_id":   int(id_match.group(1)) if id_match else None,
            "title":          get("dc:title"),
            "author":         get("dc:creator"),
            "language":       get("dc:language"),
            "subjects":       re.findall(r'<dc:subject[^>]*>([^<]+)</dc:subject>', opf),
            "date_published": get_date("publication"),
            "date_converted": get_date("conversion"),
            "source":         get("dc:source"),
            "rights":         get("dc:rights"),
            "contributor":    contributor_m.group(1).strip() if contributor_m else None,
            "identifier":     identifier,
        }
        return {k: v for k, v in meta.items() if v is not None and v != []}
    except Exception:
        return {}


def _read_from_tar(tar_path: Path, n_books: int | None) -> list[dict]:
    books = []
    label = str(n_books) if n_books else "all"
    with tarfile.open(tar_path, "r:gz") as tar, tqdm(total=n_books, desc=f"reading {label} epubs", unit="epub") as pbar:
        for member in tar:
            if n_books is not None and len(books) >= n_books:
                break
            if not member.isfile() or not member.name.endswith(".epub"):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            epub_data = f.read()
            meta = _epub_metadata(epub_data)
            if not meta:
                continue
            books.append({"id": Path(member.name).stem, "epub": epub_data, **meta})
            pbar.update(1)
    return books


def download(base_dir: Path, n_books: int | None):
    raw_dir = base_dir / "raw-tar-gz"
    tar_path = raw_dir / FILENAME
    out_dir = base_dir / (f"raw-hf-{n_books}" if n_books else "raw-hf-full")

    # --- step 1: download tar ---
    if tar_path.exists():
        print(f"TAR already exists, skipping download: {tar_path}")
    else:
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {FILENAME} -> {raw_dir} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        print(f"Done: {tar_path}")

    # --- step 2: build HF dataset ---
    if (out_dir / "dataset_info.json").exists():
        print(f"Dataset already exists, skipping: {out_dir}")
        return

    print(f"\nReading epubs from tar...")
    books = _read_from_tar(tar_path, n_books)
    print(f"Read {len(books):,} books")

    out_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(books)
    ds.save_to_disk(str(out_dir))
    print(f"Saved {len(ds):,} books -> {out_dir}")

    n = len(ds)
    print(f"\n### field fill rate ({n:,} books) ###")
    for col in ds.column_names:
        if col == "epub":
            continue
        empty = sum(1 for x in ds[col] if x is None or x == [])
        print(f"  {col:20s} {n - empty:6,} / {n:,}  ({100*(n-empty)/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Download laion/Project-Gutenberg")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="root dir; tar -> <base-dir>/raw/, dataset -> <base-dir>/raw-<n>/ or raw-full/")
    parser.add_argument("--n-books", type=int, default=None,
                        help="number of books to sample; omit for full dataset")
    args = parser.parse_args()
    download(Path(args.base_dir), args.n_books)


if __name__ == "__main__":
    main()
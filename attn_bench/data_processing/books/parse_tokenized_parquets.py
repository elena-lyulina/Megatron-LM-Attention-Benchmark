"""
Parse DataTrove tokenization logs to extract the parquet files that were
actually opened (and thus at least partially tokenized into the training corpus).

For each task log in <log-dir>/logs/task_*.log, finds lines like:
    "Reading input file train-00233-of-00234.parquet, 1/6"
and collects the unique filenames. The last file per task may be partially
consumed (budget hits mid-file), but was opened, so its content is in training data.

Usage:
    python parse_tokenized_parquets.py \\
        --log-dirs /path/to/datatrove/dump_0 /path/to/datatrove/dump_1 \\
        --output tokenized_parquets.txt
"""

import argparse
import re
from pathlib import Path


def parse_log_dir(log_dir: Path) -> list[str]:
    """Return parquet filenames read by any task in this DataTrove logging dir.

    Handles two layouts:
      - log_dir/logs/task_*.log  (DataTrove default on cluster)
      - log_dir/task_*.log       (flat, e.g. when copied locally)
    """
    pattern = re.compile(r'Reading input file (\S+\.parquet)')
    seen: set[str] = set()
    found: list[str] = []

    logs_subdir = log_dir / 'logs'
    search_dir = logs_subdir if logs_subdir.exists() else log_dir
    log_files = sorted(search_dir.glob('task_*.log'))
    if not log_files:
        print(f"Warning: no task_*.log files found in {search_dir}")
        return []

    for log_file in log_files:
        task_count = 0
        with open(log_file) as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    fname = m.group(1)
                    if fname not in seen:
                        seen.add(fname)
                        found.append(fname)
                        task_count += 1
        if task_count == 0:
            print(f"  Warning: no parquet files found in {log_file.name}")
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dirs', nargs='+', required=True,
                        help='DataTrove logging directories to parse (one per dump)')
    parser.add_argument('--output', required=True,
                        help='Output txt file with one parquet filename per line')
    args = parser.parse_args()

    all_files: list[str] = []
    seen: set[str] = set()
    for log_dir in args.log_dirs:
        files = parse_log_dir(Path(log_dir))
        new = [f for f in files if f not in seen]
        seen.update(new)
        all_files.extend(new)
        print(f"{log_dir}: {len(files)} files read by tasks ({len(files) - len(new)} already seen from previous dump)")

    all_files.sort()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for fname in all_files:
            f.write(fname + '\n')

    print(f"\nTotal: {len(all_files)} unique parquet files -> {out}")


if __name__ == '__main__':
    main()
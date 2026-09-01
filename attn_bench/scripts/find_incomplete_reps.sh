#!/usr/bin/env bash
set -euo pipefail

# Scans a mem-results SparseGutenberg tree for rep_*_greedy dirs whose rank*.jsonl files
# sum to fewer records than expected -- the signature a job killed mid-generation leaves
# behind. Step 1 writes each rank's jsonl incrementally with a flush per record (see
# prefix_extraction_inference.py's run_bucket), but the resume/skip check only tests that
# rank0.jsonl exists and is non-empty, not that every rank finished (_process_rep's
# jsonl_done check) -- so a truncated rep looks "done" and gets silently skipped on
# resubmission, then silently averaged over by compute_memorization_metrics.py (which just
# globs+reads whatever rank*.jsonl records exist, no count check).
#
# Usage: bash find_incomplete_reps.sh [base_dir] [expected_count]
#   base_dir        default: /iopsstor/scratch/cscs/$USER/mem-results/SparseGutenberg
#   expected_count  default: 660 (the fixed SparseGutenberg sample count seen in every run)

BASE_DIR=${1:-/iopsstor/scratch/cscs/$USER/mem-results/SparseGutenberg}
EXPECTED=${2:-660}

echo "Scanning $BASE_DIR for rep_*_greedy dirs with != $EXPECTED total records..."
echo

FOUND=0
while IFS= read -r -d '' rep_dir; do
    total=$(find "$rep_dir" -maxdepth 1 -name 'rank*.jsonl' -exec cat {} + 2>/dev/null | wc -l | tr -d ' ')
    if [ "$total" != "$EXPECTED" ]; then
        FOUND=$((FOUND + 1))
        echo "$rep_dir: $total/$EXPECTED records"
        for rank_file in "$rep_dir"/rank*.jsonl; do
            [ -f "$rank_file" ] && echo "    $(wc -l < "$rank_file" | tr -d ' ')  $(basename "$rank_file")"
        done
    fi
done < <(find "$BASE_DIR" -type d -name 'rep_*_greedy' -print0)

echo
echo "$FOUND incomplete rep dir(s) found."
[ "$FOUND" -gt 0 ] && echo "Delete each with: rm -rf <path>   (then resubmit -- Step 1 will regenerate just that rep)"

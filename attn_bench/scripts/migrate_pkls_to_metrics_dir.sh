#!/usr/bin/env bash
# One-off migration: move each experiment's *.pkl files from directly under exp_dir/ into
# exp_dir/metrics/, matching compute_memorization_metrics.py's pkl_path() convention. Run
# once at the SparseGutenberg/ level -- covers every model dir underneath in one pass.
# Only touches mem-results/SparseGutenberg/; long-fineweb-results/ and
# long-gutenberg-results/ use an unrelated layout. Bash 3.2 compatible.
#
# Usage: migrate_pkls_to_metrics_dir.sh <base_path> [--dry-run]
#   local:  bash migrate_pkls_to_metrics_dir.sh attn_bench/results/mem-results/SparseGutenberg
#   remote: bash migrate_pkls_to_metrics_dir.sh /users/$USER/store/mem-results/SparseGutenberg
set -euo pipefail

BASE="$1"
DRY=${2:-}

TOTAL=0
for exp_dir in "$BASE"/*/; do
    exp_dir="${exp_dir%/}"
    exp_name="$(basename "$exp_dir")"
    moved=0
    for pkl in "$exp_dir"/*.pkl; do
        [ -e "$pkl" ] || continue
        dest="$exp_dir/metrics/$(basename "$pkl")"
        if [ "$DRY" = "--dry-run" ]; then
            echo "[dry-run] $pkl -> $dest"
        else
            mkdir -p "$exp_dir/metrics"
            mv "$pkl" "$dest"
            echo "$pkl -> $dest"
        fi
        moved=$((moved + 1))
    done
    if [ "$moved" -gt 0 ]; then
        if [ "$DRY" = "--dry-run" ]; then
            echo "$exp_name: would move $moved pkl file(s)"
        else
            echo "$exp_name: moved $moved pkl file(s)"
        fi
    fi
    TOTAL=$((TOTAL + moved))
done

echo
echo "Total: $TOTAL pkl file(s) ${DRY:+would be }moved."

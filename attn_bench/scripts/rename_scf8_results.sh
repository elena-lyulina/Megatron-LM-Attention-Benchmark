#!/usr/bin/env bash
# One-off migration for the rope-scaling-factor fix: renames existing checkpoint/result
# directories to the scf8-in-the-middle / -scf1-at-the-end naming scheme (see
# attn_bench/_plans/rope_scaling_fix_plan.md, Step 4). Bash 3.2 compatible (no assoc arrays).
#
# Usage: rename_scf8_results.sh <base_dir> [--dry-run]
#   local:  bash rename_scf8_results.sh attn_bench/results/long-fineweb-results
#   remote: bash rename_scf8_results.sh /users/$USER/store/long-fineweb-results
# Run once per tree: long-fineweb-results, long-gutenberg-results, mem-results/SparseGutenberg.
# (Checkpoint dir renames under pretrain-results use a separate, CKPT_NAME-keyed list --
# see the plan for the exact commands.)
set -euo pipefail

BASE="$1"
DRY=${2:-}

PAIRS=(
  # already correct (computed at the actual trained factor, 8) -- no trailing suffix
  "llama3-1b-full-attn-fineweb40B-gutenberg3B-scf-8|llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B"
  "llama3-1b-full-attn-fineweb40B-long-gutenberg3B-scf-8|llama3-1b-full-attn-scf8-fineweb40B-long-gutenberg3B"
  "llama3-1b-full-attn-fineweb40B-long-split-1024-gutenberg3B-scf-8|llama3-1b-full-attn-scf8-fineweb40B-long-split-1024-gutenberg3B"
  "llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B-scf-8|llama3-1b-full-attn-xdoc-attn-leak-scf8-fineweb40B-gutenberg3B"

  # mismatched (computed with no rope scaling at all, i.e. factor 1) -- archived with -scf1
  "llama3-1b-full-attn-fineweb40B-gutenberg3B|llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-full-attn-fineweb40B-long-gutenberg3B|llama3-1b-full-attn-scf8-fineweb40B-long-gutenberg3B-scf1"
  "llama3-1b-full-attn-fineweb40B-long-split-1024-gutenberg3B|llama3-1b-full-attn-scf8-fineweb40B-long-split-1024-gutenberg3B-scf1"
  "llama3-1b-full-attn-fineweb80B-gutenberg3B|llama3-1b-full-attn-scf8-fineweb80B-gutenberg3B-scf1"
  "llama3-1b-full-attn-goldfish-fineweb40B-gutenberg3B|llama3-1b-full-attn-goldfish-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B|llama3-1b-full-attn-xdoc-attn-leak-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-gated-attn-fineweb40B-gutenberg3B|llama3-1b-gated-attn-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B|llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215|llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-te215-scf1"
  "llama3-1b-sink-attn-fineweb40B-gutenberg3B|llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-scf1"
  "llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215|llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-te215-scf1"
)

# sweep-variant suffixes (mem-results ablations): same base mapping, suffix carried through, -scf1 appended after it
for base_old in \
  "llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215" \
  "llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215"; do
  base_new="llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-te215"
  [ "$base_old" = "llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215" ] && base_new="llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-te215"
  for suffix in _nsinks16 _nsinks32 _nsinks4 _nsinks8 _sscale0 _sscale0.25 _sscale0.5 _sscale0.75 _sscale1.5 _sscale2; do
    PAIRS+=("${base_old}${suffix}|${base_new}${suffix}-scf1")
  done
done

for pair in "${PAIRS[@]}"; do
  old="${pair%%|*}"
  new="${pair##*|}"
  src="$BASE/$old"
  dst="$BASE/$new"
  if [ -d "$src" ]; then
    if [ "$DRY" = "--dry-run" ]; then
      echo "mv '$src' '$dst'"
    else
      mv "$src" "$dst"
      echo "moved: $old -> $new"
    fi
  fi
done

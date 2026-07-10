#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=01:00:00
#SBATCH --job-name=verify-and-split-fineweb-new25
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Step 1: verify the regenerated 0.25 (from separate_fineweb_edu_40B.slurm, written
# to datasets/tokenized_re/) is byte-identical to the live 0.25 partition that
# in-progress training runs are reading from datasets/tokenized/. This confirms
# separate_binary.py reproduced the original split deterministically before we
# trust the regenerated 0.75 remainder.
#
# Step 2 (only runs if step 1 passes): carve a second, disjoint 40B (0.25 of the
# full 160B corpus) out of the regenerated 0.75 remainder, by splitting it again
# 1/3 (new 40B partition) + 2/3 (remainder). The two outputs and the regenerated
# 0.75 parent are then renamed/moved into datasets/tokenized/ as:
#   fineweb-edu-dedup-160B-datatrove_0.25          ← unchanged, live, first quarter
#   fineweb-edu-dedup-160B-datatrove_0.25_2        ← new, second quarter (combine with 0.25 for half the corpus)
#   fineweb-edu-dedup-160B-datatrove_0.5_unseen    ← remaining half, still unused
#   fineweb-edu-dedup-160B-datatrove_0.75_unseen   ← all data beyond the live 0.25 (0.25_2 + 0.5_unseen), kept as one block too
# datasets/tokenized_re/ is then deleted (only the throwaway verification 0.25
# copy is left in it at that point, since the 0.75 was moved out).
#
# Step 5: copy the three new outputs (0.25_2, 0.5_unseen, 0.75_unseen — combined
# ~960GB) to /capstor/store for long-term retention (iopsstor scratch purges
# after 30 days of no access). This is cross-filesystem, unlike the same-filesystem
# renames above, so it actually moves every byte over the network — all ~240
# shard files across the three directories are pooled into one manifest and
# copied via 32 parallel single-file rsyncs rather than 3 whole-directory
# streams, so the copy isn't bottlenecked by the single largest directory.
#
# If step 1 fails, the job exits before anything else runs, so nothing new is
# written and the live 0.25 partition and tokenized_re/ are never touched.
#
# Usage:
#   sbatch attn_bench/submissions/verify_and_split_fineweb_edu_new_25.slurm

echo "START TIME: $(date)"

T_START=$(date +%s)
T_PREV=$T_START
mark() {
    local now=$(date +%s)
    echo "[TIMING] $1: $((now - T_PREV))s"
    T_PREV=$now
}

CONTAINER_ENV=nemo_26.04_te2.15

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

LIVE_025=/iopsstor/scratch/cscs/$USER/datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25
RE_025=/iopsstor/scratch/cscs/$USER/datasets/tokenized_re/fineweb-edu-dedup-160B-datatrove_0.25
RE_075=/iopsstor/scratch/cscs/$USER/datasets/tokenized_re/fineweb-edu-dedup-160B-datatrove_0.75
OUTPUT_DIR=/iopsstor/scratch/cscs/$USER/datasets/tokenized

mkdir -p $MEGATRON_LM_DIR/attn_bench/logs

### STEP 1: verify regenerated 0.25 == live 0.25 (md5sum, per file) ###

echo "Comparing:"
echo "  live: $LIVE_025"
echo "  re:   $RE_025"

files_live=$(cd "$LIVE_025" && find . -type f \( -name "*.idx" -o -name "*.bin" \) | sort)
files_re=$(cd "$RE_025" && find . -type f \( -name "*.idx" -o -name "*.bin" \) | sort)

if [ "$files_live" != "$files_re" ]; then
    echo "FAIL: file lists differ between live and regenerated 0.25"
    diff <(echo "$files_live") <(echo "$files_re")
    exit 1
fi

mismatch=0
while IFS= read -r f; do
    hash_live=$(md5sum "$LIVE_025/$f" | cut -d' ' -f1)
    hash_re=$(md5sum "$RE_025/$f" | cut -d' ' -f1)
    if [ "$hash_live" != "$hash_re" ]; then
        echo "MISMATCH: $f  (live=$hash_live re=$hash_re)"
        mismatch=1
    fi
done <<< "$files_live"

if [ "$mismatch" -ne 0 ]; then
    echo "FAIL: regenerated 0.25 does not match live 0.25 — not touching anything further."
    exit 1
fi

echo "OK: regenerated 0.25 is byte-identical to live 0.25 across $(echo "$files_live" | wc -l) files."
mark "Step 1 (verify)"

### STEP 2: split regenerated 0.75 into a new disjoint 0.25 (1/3) + remainder (2/3) ###

echo "Splitting: $RE_075"
echo "Output:    $OUTPUT_DIR"
echo "Ratios:    0.3333333 (new 40B) + 0.6666667 (remainder)"

srun \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --environment=$CONTAINER_ENV \
    bash -c "
    export PYTHONPATH='${PYTHONPATH}'
    python3 $MEGATRON_LM_DIR/attn_bench/utils/tools/separate_binary.py \
        --input-dir $RE_075 \
        --output-dir $OUTPUT_DIR \
        --ratios 0.3333333 0.6666667 \
        --seed 42 \
        --workers $SLURM_CPUS_PER_TASK"

if [ $? -ne 0 ]; then
    echo "FAIL: separate_binary.py crashed (see .err log for the traceback) — not renaming/moving/copying anything."
    exit 1
fi
mark "Step 2 (split)"

### STEP 3: rename/move outputs into their final names under datasets/tokenized/ ###

mv "$OUTPUT_DIR/fineweb-edu-dedup-160B-datatrove_0.75_0.3333333" \
   "$OUTPUT_DIR/fineweb-edu-dedup-160B-datatrove_0.25_2"
mv "$OUTPUT_DIR/fineweb-edu-dedup-160B-datatrove_0.75_0.6666667" \
   "$OUTPUT_DIR/fineweb-edu-dedup-160B-datatrove_0.5_unseen"
mv "$RE_075" \
   "$OUTPUT_DIR/fineweb-edu-dedup-160B-datatrove_0.75_unseen"
mark "Step 3 (rename/move)"

### STEP 4: clean up tokenized_re/ (only the throwaway verification 0.25 is left in it) ###

rm -rf /iopsstor/scratch/cscs/$USER/datasets/tokenized_re
mark "Step 4 (cleanup tokenized_re)"

### STEP 5: copy the three new outputs to long-term store, parallelized across files ###

STORE_DIR=/users/$USER/store/datasets/tokenized

echo "Copying to store: $STORE_DIR"

MANIFEST=$(mktemp)
for name in fineweb-edu-dedup-160B-datatrove_0.25_2 \
            fineweb-edu-dedup-160B-datatrove_0.5_unseen \
            fineweb-edu-dedup-160B-datatrove_0.75_unseen; do
    src_dir="$OUTPUT_DIR/$name"
    dst_dir="$STORE_DIR/$name"
    ( cd "$src_dir" && find . -type f \( -name "*.idx" -o -name "*.bin" \) ) | while IFS= read -r rel; do
        echo "$src_dir/$rel" "$dst_dir/$rel"
    done
done > "$MANIFEST"

echo "Files to copy: $(wc -l < "$MANIFEST")"

xargs -a "$MANIFEST" -n 2 -P 32 bash -c '
    mkdir -p "$(dirname "$2")"
    rsync -a "$1" "$2"
' _

rm -f "$MANIFEST"
mark "Step 5 (copy to store)"

echo "END TIME: $(date)"
echo "TOTAL TIME: $(($(date +%s) - T_START))s"
echo "New quarter (use with live 0.25 for half the corpus): ${STORE_DIR}/fineweb-edu-dedup-160B-datatrove_0.25_2"
echo "Remaining half, still unseen:                         ${STORE_DIR}/fineweb-edu-dedup-160B-datatrove_0.5_unseen"
echo "Everything beyond the live 0.25, as one block:        ${STORE_DIR}/fineweb-edu-dedup-160B-datatrove_0.75_unseen"
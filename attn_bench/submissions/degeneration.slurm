#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=1:00:00
#SBATCH --job-name=degeneration
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Free-generation degeneration curves (Section 4.1) for every unseen-text generation run:
# runs evaluation/degeneration.py on each <dataset>/<exp>/inference/offset_O_prefix_P_suffix_S/rep_R_greedy
# found under scratch (falling back to store), writing one .npz per run to
# attn_bench/results/degeneration/<dataset>__<exp>.npz. CPU only, no model. Skips runs
# whose .npz already exists unless FORCE=1.
#
# Env vars (all optional):
#   DATASETS  space-separated result subfolders (default: "SparseGutenberg UnseenFineWeb")
#   POINT     offset_O_prefix_P_suffix_S dir name (default: offset_0_prefix_500_suffix_1000)
#   REP       repetition bucket (default: 0)
#   EXPRS     space-separated experiment names to restrict to (default: every exp with the run)
#   NGRAM / WINDOW / STEP  passed through to degeneration.py (defaults "8 16 32" / 100 / 25)
#   FORCE=1   recompute existing .npz
#
# Usage:
#   sbatch attn_bench/submissions/degeneration.slurm
#   # Refresh existing files to include raw cumulative TTR (cum_gen, cum_ref):
#   sbatch --export=ALL,FORCE=1 attn_bench/submissions/degeneration.slurm
#   sbatch --export=DATASETS="UnseenFineWeb",EXPRS="llama3-1b-full-attn-scf1-fineweb40B-gutenberg3B_hf" attn_bench/submissions/degeneration.slurm

echo "START TIME: $(date)"

CONTAINER_ENV=nemo_26.04_te2.15
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
SCRATCH_MEM_BASE=/iopsstor/scratch/cscs/$USER/mem-results
STORE_MEM_BASE=/users/$USER/store/mem-results
OUT_DIR=$MEGATRON_LM_DIR/attn_bench/results/degeneration

DATASETS=${DATASETS:-"SparseGutenberg UnseenFineWeb"}
POINT=${POINT:-offset_0_prefix_500_suffix_1000}
REP=${REP:-0}
EXPRS=${EXPRS:-}
NGRAM=${NGRAM:-"8 16 32"}
WINDOW=${WINDOW:-100}
STEP=${STEP:-25}
FORCE=${FORCE:-}

mkdir -p "$OUT_DIR"
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

# Collect (dataset, exp, inference dir), scratch first, store as fallback for purged runs.
RUNS=()
for DATASET in $DATASETS; do
    for BASE in "$SCRATCH_MEM_BASE" "$STORE_MEM_BASE"; do
        for DIR in "$BASE/$DATASET"/*/inference/"$POINT"/rep_"$REP"_greedy; do
            [ -d "$DIR" ] || continue
            EXP=$(basename "$(dirname "$(dirname "$(dirname "$DIR")")")")
            if [ -n "$EXPRS" ]; then
                case " $EXPRS " in *" $EXP "*) ;; *) continue ;; esac
            fi
            # Skip if this (dataset, exp) was already picked up from scratch.
            case " ${SEEN:-} " in *" $DATASET/$EXP "*) continue ;; esac
            SEEN="${SEEN:-} $DATASET/$EXP"
            RUNS+=("$DATASET|$EXP|$DIR")
        done
    done
done

echo "Found ${#RUNS[@]} runs for point $POINT rep $REP"
[ ${#RUNS[@]} -eq 0 ] && { echo "Nothing to do."; exit 0; }

srun --cpus-per-task $SLURM_CPUS_PER_TASK --environment=$CONTAINER_ENV bash -c "
export PYTHONPATH='$PYTHONPATH'
FAIL=0
for RUN in $(printf '%q ' "${RUNS[@]}"); do
    IFS='|' read -r DATASET EXP DIR <<< \"\$RUN\"
    OUT=$OUT_DIR/\${DATASET}__\${EXP}.npz
    if [ -z '$FORCE' ] && [ -f \"\$OUT\" ]; then echo \"skip (exists): \$OUT\"; continue; fi
    echo \"=== \$DATASET / \$EXP\"
    python $MEGATRON_LM_DIR/attn_bench/evaluation/degeneration.py \
        --inference-dir \"\$DIR\" --out \"\$OUT\" --ngram $NGRAM --window $WINDOW --step $STEP \
        || { echo \"FAILED: \$DATASET / \$EXP\"; FAIL=1; }
done
exit \$FAIL
"
STATUS=$?
echo "END TIME: $(date)  (status $STATUS)"
exit $STATUS

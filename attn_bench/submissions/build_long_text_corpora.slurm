#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-long-text
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --environment=nemo_26.04_te2.15
#SBATCH --partition=normal
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Detokenize the long-context corpora to raw-text jsonl, for the lm-eval perplexity tasks in
# attn_bench/evaluation/lm_eval_tasks/. Writes to store, not scratch: the task YAMLs hardcode
# these paths, so a purge would break every future run.
#
# Env vars:
#   CORPORA     space-separated subset of "gutenberg fineweb_unseen fineweb_seen"
#               (default: all three)
#   MAX_LENGTH  content-token cap per document (default: 32768 = 4x the training seq len)
#   OVERWRITE   rebuild corpora that already exist
#
# Usage: sbatch attn_bench/submissions/build_long_text_corpora.slurm
#        CORPORA=gutenberg OVERWRITE=1 sbatch attn_bench/submissions/build_long_text_corpora.slurm

set -eo pipefail

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}

STORE_TOKENIZED=/users/$USER/store/datasets/tokenized
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
OUT_DIR=/users/$USER/store/datasets/text/long-corpora

GUTENBERG_SRC=$STORE_TOKENIZED/gutenberg_rep_jsonl_long/rep_0_token.jsonl
FINEWEB_UNSEEN_SRC=$STORE_TOKENIZED/fineweb-edu-dedup-160B-datatrove_0.75_unseen_long/long_24576_32768.jsonl
FINEWEB_SEEN_SRC=$STORE_TOKENIZED/fineweb-edu-dedup-160B-datatrove_0.25_long/long_24576_32768.jsonl

CORPORA=${CORPORA:-"gutenberg fineweb_unseen fineweb_seen"}
MAX_LENGTH=${MAX_LENGTH:-32768}
OVERWRITE=${OVERWRITE:-}

OVERWRITE_ARG=""
[ -n "$OVERWRITE" ] && OVERWRITE_ARG="--overwrite"

mkdir -p "$OUT_DIR"

# corpus_kind picks the span rule in the .py (gutenberg drops the prefix, fineweb takes the
# document verbatim minus its BOS).
run_corpus() {
    local name="$1" kind="$2" src="$3"
    if [ ! -f "$src" ]; then
        echo "ERROR: source not found for $name: $src" >&2
        exit 1
    fi
    echo ""
    echo "=== $name  ($kind)  <- $src"
    python3 -u -m attn_bench.data_processing.detokenize_long_corpora \
        --corpus "$kind" \
        --data-file "$src" \
        --tokenizer-path "$TOKENIZER_PATH" \
        --output "$OUT_DIR/$name.jsonl" \
        --max-length "$MAX_LENGTH" \
        $OVERWRITE_ARG
}

for NAME in $CORPORA; do
    case "$NAME" in
        gutenberg)      run_corpus gutenberg_rep0   gutenberg $GUTENBERG_SRC ;;
        fineweb_unseen) run_corpus fineweb_unseen   fineweb   $FINEWEB_UNSEEN_SRC ;;
        fineweb_seen)   run_corpus fineweb_seen     fineweb   $FINEWEB_SEEN_SRC ;;
        *) echo "Unknown corpus: $NAME (expected gutenberg, fineweb_unseen or fineweb_seen)" >&2; exit 1 ;;
    esac
done

echo ""
echo "END TIME: $(date)"
echo "Done. Text corpora in $OUT_DIR"
ls -la "$OUT_DIR"

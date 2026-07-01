#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-gutenberg-long
#SBATCH --output=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/Megatron-LM-Attention-Benchmark/attn_bench/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=2:00:00
#SBATCH --environment=nemo
#SBATCH --no-requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=clariden-jobs-elyulin-aaaauapxybdkdx42ppvef2c2eq@swissai-initiative.slack.com

# Build long-context versions of the Gutenberg memorization probes.
# Regenerates per-book text + excerpt position for the whole raw corpus, matches each
# existing probe by its token sequence, and re-tokenizes a longer span around it.
# CPU-only; uses the original `nemo` env so text (inscriptis/ftfy) reproduces byte-exact.
#
# Usage (from attn_bench/):
#   sbatch submissions/build_gutenberg_long.slurm              # full run
#   STATS_ONLY=1 sbatch submissions/build_gutenberg_long.slurm # length stats only, no token files
# Worker counts are split per phase (regen is light + fast, build-long is tokenization-heavy):
#   NUM_WORKERS=96 BUILD_WORKERS=32 sbatch submissions/build_gutenberg_long.slurm  # defaults

echo "START TIME: $(date)"

### PATHS ###

MEGATRON_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-Attention-Benchmark
TOKENIZER_PATH=/iopsstor/scratch/cscs/$USER/tokenizers/llama-3.2-1b
JSONL_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl
RAW_DIR=/users/$USER/store/datasets/raw/gutenberg-laion/raw-hf-full
OUTPUT_DIR=/users/$USER/store/datasets/tokenized/gutenberg_rep_jsonl_long

export PYTHONPATH=$MEGATRON_DIR:${PYTHONPATH:-}
cd $MEGATRON_DIR


### INSTALL DEPS ###

pip install --quiet -e "$MEGATRON_DIR/attn_bench[data]"


### BUILD ###

python3 -u -m attn_bench.data_processing.books.extract_long_context \
    --jsonl-dir      $JSONL_DIR \
    --raw-dir        $RAW_DIR \
    --tokenizer-path $TOKENIZER_PATH \
    --output-dir     $OUTPUT_DIR \
    --num-workers    ${NUM_WORKERS:-$SLURM_CPUS_PER_TASK} \
    --build-workers  ${BUILD_WORKERS:-32} \
    ${STATS_ONLY:+--stats-only}


echo "END TIME: $(date)"
echo "Done. Output in $OUTPUT_DIR"
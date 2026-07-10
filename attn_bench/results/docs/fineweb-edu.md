# Data Processing

## Step 1: Tokenize FineWeb-Edu

Tokenizes 160B tokens from the FineWeb-Edu-dedup dataset into Megatron binary shards.

Tokenization scripts are copied from [swiss-ai/pretrain](https://github.com/swiss-ai/pretrain) and extended to support token budgets.

**Process:**
1. Download raw parquet files (`../download_fineweb_edu.py`)
2. Split files into balanced dumps (`prepare_dumps.py`): scans parquet files, sorts by size, and distributes them evenly across N dumps, writing `dumps/paths_file_*.txt`
3. Tokenize each dump (`preprocess_megatron_budgeted.py` via datatrove): each worker tokenizes its assigned parquet files and writes a binary shard (`XXXXX_tokens.bin/idx`).
A per-worker token budget stops each worker once it is reached to respect the global token budget.

With the per-worker budget, there will be some overshoot unless the workers communicate with each other.
However, this approach would require cross-node and cross-process communication. 
The latter one showed to be clashing with datatrove's `forkserver` process start method
(datatrove serializes pipeline components with `dill`, which cannot serialize `Manager` proxies or multiprocessing values).
However, the overshoot is very acceptable: at most, each worker processes an extra batch before stopping (~10,000 docs × ~1,003 tokens/doc ≈ 10M tokens).
With 20 workers, it is at most 200M, or just ~0.125% of the 160B target. 

 
**Tokenizer:** LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`).

**Stats (160B budget run, 2 nodes):**
- Tokens processed: 160,159,943,902 (160.16B) — dump_0: 80.08B, dump_1: 80.08B
- Time: ~1.5h per node (dump_0: 5336s, dump_1: 5411s)
- Speed: ~15M tok/s per node (still quite slow, probably due to 2.4GB parquet files, while the recommended size is 0.5GB)
- Workers: 20 per node, batch size: 10000
- Slurm jobs: 1931370 (dump_0, nid007126), 1931371 (dump_1, nid007127)
- Verification passed (logs/1934285.out)

**Scripts:**
- [`prepare_dumps.py`](../../data_processing/tokenization/prepare_dumps.py) — splits parquet files into balanced dumps
- [`preprocess_megatron_budgeted.py`](../../data_processing/tokenization/preprocess_megatron_budgeted.py) — main tokenization script
- [`megatron_tokenizer_budgeted.py`](../../data_processing/tokenization/megatron_tokenizer_budgeted.py) — per-worker budget logic
- [`../../submissions/submit_tokenization_fineweb_edu_datatrove.sh`](../../submissions/submit_tokenization_fineweb_edu_datatrove.sh) — submits one slurm job per dump
- [`../../submissions/tokenize_fineweb_edu_datatrove.slurm`](../../submissions/tokenize_fineweb_edu_datatrove.slurm) — slurm job script

**Results CSV:** [`../../results/tokenization.csv`](../tokenization.csv)

## Step 2: Split FineWeb-Edu into a 40B partition

Splits the 160B tokenized dataset into a 40B partition (ratio 0.25) for training alongside a smaller dataset, and a 120B remainder.
Documents are randomly assigned per shard using Megatron's greedy blending sampler (`build_blending_indices`), preserving the dump structure.

**Stats (job 2256031, 2026-05-15):**
- ratio 0.25 → 40,038,865,413 tokens (40.039B) — `fineweb-edu-dedup-160B-datatrove_0.25/`
- ratio 0.75 → 120,121,078,489 tokens (120.121B) — `fineweb-edu-dedup-160B-datatrove_0.75/`
- total → 160,159,943,902 tokens (160.160B)
- Workers: 16, time: 4min

**Scripts:**
- [`../../../attn_bench/utils/tools/separate_binary.py`](../../utils/tools/separate_binary.py) — splitting script
- [`../../submissions/separate_fineweb_edu_40B.slurm`](../../submissions/separate_fineweb_edu_40B.slurm) — slurm job script

## Step 3: Carve a second 40B partition from the unseen 0.75

Splits the 0.75 remainder again into a second, disjoint 40B partition (`_0.25_2`), so `_0.25` +
`_0.25_2` together give half the corpus for training. (The original `_0.75` had been cleaned up
unused, so it was briefly regenerated and verified byte-identical to the live `_0.25` first —
`separate_binary.py` is deterministic given the same input, ratios, and seed.)

**Stats (job 2715356, 2026-07-10):**
- ratio 0.3333333 → 40,045,014,234 tokens (40.045B) — `fineweb-edu-dedup-160B-datatrove_0.25_2/`
- ratio 0.6666667 → 80,076,064,255 tokens (80.076B) — `fineweb-edu-dedup-160B-datatrove_0.5_unseen/`
- total (= 0.75 remainder) → 120,121,078,489 tokens (120.121B) — `fineweb-edu-dedup-160B-datatrove_0.75_unseen/` (kept as one block too)
- Workers: 32, time: 930s total
- Outputs copied to `/users/$USER/store/datasets/tokenized/` for long-term retention

**Scripts:**
- [`../../submissions/verify_and_split_fineweb_edu_new_25.slurm`](../../submissions/verify_and_split_fineweb_edu_new_25.slurm) — verifies + splits + renames + copies to store
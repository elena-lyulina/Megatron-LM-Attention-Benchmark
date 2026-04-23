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
The latter one showed to be clashing with datatrove's`forkserver` process start method
(datatrove serializes pipeline components with `dill`, which cannot serialize `Manager` proxies or multiprocessing values).
However, the overshoot is very acceptable: at most, each worker processes an extra batch before stopping (~10,000 docs × ~1,003 tokens/doc ≈ 10M tokens).
With 20 workers, it is at most 200M, or just ~0.125% of the 160B target. 


**Tokenizer:** LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`).

**Stats (160B budget run, 2 nodes):**
- Tokens processed: 160,153,550,653 (160.15B) — dump_0: 80.06B, dump_1: 80.09B
- Time: ~1.5h per node (dump_0: 5355s, dump_1: 5489s)
- Speed: ~15M tok/s per node (still quite slow, probably due to 2.4GB parquet files, while the recommended size is 0.5GB)
- Workers: 20 per node, batch size: 10000
- Slurm jobs: 1894737 (dump_0, nid007509), 1894738 (dump_1, nid007382)

**Scripts:**
- [`prepare_dumps.py`](prepare_dumps.py) — splits parquet files into balanced dumps
- [`preprocess_megatron_budgeted.py`](preprocess_megatron_budgeted.py) — main tokenization script
- [`budgeted_tokenizer.py`](megatron_tokenizer_budgeted.py) — per-worker budget logic
- [`../submissions/submit_tokenization_fineweb_edu_datatrove.sh`](../../submissions/submit_tokenization_fineweb_edu_datatrove.sh) — submits one slurm job per dump
- [`../submissions/tokenize_fineweb_edu_datatrove.slurm`](../../submissions/tokenize_fineweb_edu_datatrove.slurm) — slurm job script

**Results CSV:** [`../results/tokenization`](../../results/tokenization)
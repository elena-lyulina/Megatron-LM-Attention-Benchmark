# Data Processing

## Step 1: Tokenize FineWeb-Edu

Tokenizes the FineWeb-Edu-dedup dataset into Megatron binary shards.

**Process:** Raw parquet files are downloaded first (`download_fineweb_edu.py`), then
`tokenize_fineweb_edu_parallel.py` distributes them across workers. Each worker tokenizes
its own subset and writes a separate binary shard (`shard_XXXXX.bin/idx`). A shared token
counter stops all workers once the global budget (160B) is reached.

**Tokenizer:** LLaMA 3.2-1B (`meta-llama/Llama-3.2-1B`) via HuggingFace `AutoTokenizer`.

**Stats (160B budget run):**
- Tokens processed: 160,108,458,502 (160.11B)
- Documents processed: 163,196,552
- Time: 1.34h on 128 CPUs
- Speed: 33.2M tok/s

**Log:** [`../logs/1844299.out`](../logs/1844299.out)
# Models Pretraining: FineWeb-80B + Gutenberg-3B

Running log of the LLaMA 3.2 1B models pretrained on the doubled-fineweb blended dataset, each with a different attention mechanism, to serve as baselines for the memorization study.
New experiments are appended below as they finish.

W&B project: https://wandb.ai/elyulina-thesis/fineweb-80B_gutenberg-3B?nw=nwuserelyulina

---

## Initial training

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (baseline) | `2732335` | 2026-07-11 00:51:49 | 2026-07-11 12:00:54 | 11h 09m 05s | COMPLETED (data exhausted) | 2.3873 | 293.6 (avg) |

W&B run: `llama3-1b-full-attn-fineweb80B-gutenberg3B-2732335` (`9m1zfzlx`).

This run completed cleanly via the data-exhaustion fix — it exited with `[exiting program after consuming all available data at iteration 30098]` and saved a valid checkpoint at step 30098 (no `StopIteration` crash, no resume needed, unlike the initial fineweb40B runs).

Container: `nemo_26.04_te2.15`.

Logs: `attn_bench/logs/2732335.{out,err}`.

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb80B_gutenberg3B.slurm`.

Checkpoint saved at step 30098. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-full-attn-fineweb80B-gutenberg3B/`
Full training config (parallelism, batch size, LR schedule, seed, container, etc.) can be seen in the slurm script as well.

---

## Dataset

Three sources blended proportionally by sequence count (all sequences are 8192 tokens, so sequence count = token count):

| source | tokens | path on cluster |
|---|---|---|
| FineWeb-Edu-Dedup (0.25) | 40,038,865,413 | `datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25` |
| FineWeb-Edu-Dedup (0.25_2) | 40,045,014,234 | `datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25_2` |
| Gutenberg (rep_1_256) | 2,762,833,920 | `datasets/tokenized/gutenberg_rep_1_256` |
| **total** | **82,846,713,567 (~82.8B)** | |

FineWeb: two disjoint 40B partitions of the 160B FineWeb-Edu-Dedup dataset (0.25 and 0.25_2, together half the corpus). See `fineweb-edu.md` for how both partitions were carved out.

Gutenberg: 9 repetition-level buckets (rep 1, 2, 4, 8, 16, 32, 64, 128, 256) from the memorization study pipeline, all included. See `gutenberg_laion_pipeline.md` for the pipeline that produced this dataset.
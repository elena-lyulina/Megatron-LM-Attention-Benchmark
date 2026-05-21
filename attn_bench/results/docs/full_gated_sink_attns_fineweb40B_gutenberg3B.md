# Attention Variants Pretraining: FineWeb-40B + Gutenberg-3B

Four LLaMA 3.2 1B models trained on the same blended dataset, each with a different attention mechanism, to serve as baselines for the memorization study. All runs completed successfully in a single job each (no requeue needed).

W&B project: https://wandb.ai/elyulina-thesis/fineweb-40B_gutenberg-3B?nw=nwuserelyulina

## Runs

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (baseline) | `2327225` | 2026-05-21 07:18 | 2026-05-21 13:23 | 6h 04m 57s | COMPLETED 0 | 2.3824 | 328.2 |
| sink | `2327229` | 2026-05-21 07:28 | 2026-05-21 13:40 | 6h 11m 59s | COMPLETED 0 | 2.3966 | 320.9 |
| off-by-one | `2330335` | 2026-05-21 07:49 | 2026-05-21 14:00 | 6h 11m 21s | COMPLETED 0 | 2.3777 | 323.4 |
| gated | `2327228` | 2026-05-21 07:28 | 2026-05-21 14:12 | 6h 44m 17s | COMPLETED 0 | 2.3691 | 329.4 |

All 4 reached step 15549/15550. The `Force Terminated` lines in `.err` files are normal — Slurm kills remaining processes after the job exits cleanly.

W&B run names: `llama3-1b-{variant}-fineweb40B-gutenberg3B-{job_id}`


Slurm scripts:

- `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm`
- `attn_bench/submissions/pretrain_llama3_1b_sink_attn_fineweb40B_gutenberg3B.slurm`
- `attn_bench/submissions/pretrain_llama3_1b_off_by_one_attn_fineweb40B_gutenberg3B.slurm`
- `attn_bench/submissions/pretrain_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm`

Logs (local copies):

- `attn_bench/logs/2327225.{out,err}` — full attn
- `attn_bench/logs/2327229.{out,err}` — sink attn
- `attn_bench/logs/2330335.{out,err}` — off-by-one attn
- `attn_bench/logs/2327228.{out,err}` — gated attn

---

## Attention variants

All variants share the same architecture and training setup; only the attention softmax changes.

| variant | Megatron flag | description |
|---|---|---|
| full | *(none)* | standard softmax — `softmax(QKᵀ/√d)` |
| gated | `--attention-output-gate` | element-wise gate multiplied onto the attention output |
| sink | `--softmax-type learnable` | learnable sink logit added to denominator — `exp(s) / (exp(s) + Σ exp(xⱼ))` |
| off-by-one | `--softmax-type off-by-one` | sink with fixed logit 0 — `1 / (1 + Σ exp(xⱼ))` |

---

## Dataset

Two sources blended proportionally by sequence count (all sequences are 8192 tokens, so sequence count = token count):

| source | tokens | path on cluster |
|---|---|---|
| FineWeb-Edu-Dedup | 40,038,865,413 | `datasets/tokenized/fineweb-edu-dedup-160B-datatrove_0.25` |
| Gutenberg (rep_1_256) | 2,762,833,920 | `datasets/tokenized/gutenberg_rep_1_256` |
| **total** | **42,801,699,333 (~42.8B)** | |

FineWeb: 0.25 partition of the 160B FineWeb-Edu-Dedup dataset (selected via datatrove partition).

Gutenberg: 9 repetition-level buckets (rep 1, 2, 4, 8, 16, 32, 64, 128, 256) from the memorization study pipeline, all included. See `gutenberg_laion_pipeline.md` for the pipeline that produced this dataset.

---

## Training config

| param | value |
|---|---|
| model | LLaMA 3.2 1B (1.23B params) |
| cluster | CSCS Clariden |
| nodes | 14 × 4 GPUs = 56 GPUs total |
| parallelism | TP=2, PP=1, CP=1, DP=28 |
| seq length | 8192 |
| micro batch size | 4 |
| global batch size | 336 |
| tokens/step | 2,752,512 |
| training steps | 15550 |
| checkpoint interval | 2000 steps |
| precision | bf16 |
| optimizer | AdamW (β₁=0.9, β₂=0.95) |
| learning rate | 4×10⁻⁴ → 4×10⁻⁵ (cosine, 2000 warmup steps) |
| weight decay | 0.01 |
| gradient clipping | 1.0 |
| seed | 28 |
| container | `nemo_26` |

Results saved on cluster under:
`attn_bench/results/pretrain/fineweb-40B_gutenberg-3B/llama3-1b-{variant}-fineweb40B-gutenberg3B/`
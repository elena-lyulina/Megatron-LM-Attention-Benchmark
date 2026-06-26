# Models Pretraining: FineWeb-40B + Gutenberg-3B

Running log of the LLaMA 3.2 1B models pretrained on the same blended dataset, each with a different attention mechanism, to serve as baselines for the memorization study. 
New experiments are appended below as they finish.

W&B project: https://wandb.ai/elyulina-thesis/fineweb-40B_gutenberg-3B?nw=nwuserelyulina

---

## Initial training

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (baseline) | `2327225` | 2026-05-21 07:18 | 2026-05-21 13:23 | 6h 04m 57s | COMPLETED 0 | 2.3824 | 328.2 |
| sink | `2327229` | 2026-05-21 07:28 | 2026-05-21 13:40 | 6h 11m 59s | COMPLETED 0 | 2.3966 | 320.9 |
| off-by-one | `2330335` | 2026-05-21 07:49 | 2026-05-21 14:00 | 6h 11m 21s | COMPLETED 0 | 2.3777 | 323.4 |
| gated | `2327228` | 2026-05-21 07:28 | 2026-05-21 14:12 | 6h 44m 17s | COMPLETED 0 | 2.3691 | 329.4 |

Logs: `attn_bench/logs/2327225.{out,err}` (full), `2327229.{out,err}` (sink), `2330335.{out,err}` (off-by-one), `2327228.{out,err}` (gated).

Slurm scripts:

- full: `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm`
- sink: `attn_bench/submissions/pretrain_llama3_1b_sink_attn_fineweb40B_gutenberg3B.slurm`
- off-by-one: `attn_bench/submissions/pretrain_llama3_1b_off_by_one_attn_fineweb40B_gutenberg3B.slurm`
- gated: `attn_bench/submissions/pretrain_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm`

Checkpoints moved to long-term storage under: `/users/elyulina/store/pretrain-results/llama3-1b-{variant}-fineweb40B-gutenberg3B/`
Full training config (parallelism, batch size, LR schedule, seed, container, etc.) cam be seem in the slurm scripts as well.

**Post-hoc: no checkpoint at step 15549.** All 4 runs trained to step 15549 but did not save the final checkpoint. Root cause: Megatron does not handle `StopIteration` (token budget exhaustion) gracefully — the data iterator raises `StopIteration` on step 15550 inside `train_step`, crashing the process before `checkpoint_and_decide_exit` runs. Last saved checkpoint: step 14000. Steps 14001→15549 were re-run in the resume jobs below.

---

## Resume: step 14000 → 15549

Re-ran the final 1549 steps from the step-14000 checkpoint.

**First attempt** (jobs `2339779`, `2339785`, `2339789`, `2339790`, 2026-05-21) — failed immediately due to `OptimizerParamScheduler` assertion: scripts set `TRAINING_STEPS=15549` but the checkpoint's LR scheduler stores `total_samples = 15550 × 336 = 5224800`; the mismatch `5224464 ≠ 5224800` aborts on load.

**Script changes for the successful resume:**
- `TRAINING_STEPS=15550` — must match original (LR scheduler checkpoint assertion)
- `CHECKPOINT_STEPS=15549` — save at the last valid step
- `--async-save` disabled — sync save ensures the checkpoint at step 15549 completes before the crash at step 15550

| variant | Slurm job | start (CEST) | end (CEST) | run time | W&B status | final lm loss (step 15549) |
|---|---|---|---|---|---|---|
| full | `2340717` | 2026-05-22 02:08:44 | 2026-05-22 02:57:33 | 48m 49s | crashed* | 2.3824 |
| gated | `2340721` | 2026-05-22 02:37:31 | 2026-05-22 03:31:06 | 53m 35s | crashed* | 2.3691 |
| off-by-one | `2340722` | 2026-05-22 02:58:00 | 2026-05-22 03:47:27 | 49m 27s | crashed* | 2.3778 |
| sink | `2340723` | 2026-05-22 03:08:08 | 2026-05-22 03:57:38 | 49m 30s | crashed* | 2.3966 |

\* WandB shows "crashed": the process still exits via unhandled `StopIteration` at step 15550 without calling `wandb.finish()`. Checkpoints at step 15549 are complete and valid. Final losses match the original runs exactly — deterministic resume from the step-14000 checkpoint with the same seed.

Logs: `attn_bench/logs/2340717.{out,err}` (full), `2340721.{out,err}` (gated), `2340722.{out,err}` (off-by-one), `2340723.{out,err}` (sink).

---

## Re-training: sink attention with TE 2.15

Sink attention was re-trained from scratch using TransformerEngine 2.15 (container `nemo_26` → updated container with TE 2.15). All other config identical to the initial training.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| sink | `2403506` | 2026-05-28 02:36 | 2026-05-28 08:35 | 5h 59m | COMPLETED | 2.3796 | ~318–327 |

W&B run: `llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215-2403506` (`xmjqh0ty`).

Container: `nemo_26.04_te2.15` (fixes `softmax_offset` zero init and gradient flow).

Checkpoint saved at step 15549. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm`

Logs: `attn_bench/logs/2403506.{out,err}`.

---

## Full attention with leaking cross-document attention

Full (standard softmax) attention trained **without** intra-document masking, so attention leaks across document boundaries within a packed sequence. Tests the memorization hypothesis against PDM by removing the cross-document isolation that the baseline `full` run has. Config: dropped `--use-packed-seq-params` and `--reset-position-ids` (kept `--eod-mask-loss`); confirmed in the log as `reset_position_ids=False`, `reset_attention_mask=False`, `create_attention_mask=False`, `eod_mask_loss=True`.

This run completed cleanly — the data-exhaustion fix worked: it exited via `[exiting program after consuming all available data at iteration 15549]` and saved a valid checkpoint at step 15549 (no `StopIteration` crash, no resume needed).

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 15549) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (xdoc leak) | `2567002` | 2026-06-19 13:41:49 | 2026-06-19 19:56:26 | 6h 14m 37s | COMPLETED (data exhausted) | 2.4239 | 310.5 (avg) |

W&B run: `llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B-2567002` (project `fineweb-40B_gutenberg-3B`).

Final step lm loss 2.4239 is higher than the masked `full` baseline (2.3824) — cross-document leakage hurts loss, as expected. (No validation set: split `100,0,0`.)

Container: `nemo_26` (not TE 2.15 — same container as the initial training).

Checkpoint saved at step 15549. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_full_attn_xdoc_attn_leak_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/2567002.{out,err}`.

---

## Gated Delta Net (GDN) mixer

LLaMA 3.2 1B backbone with the attention layers replaced by a Gated Delta Net (GDN) linear-attention mixer on all 16 layers — a different sequence mixer rather than a softmax variant. Param-matched to the ~1.236B attention baselines (~1.239B): GDN mixer with 8 K/V heads, `key_head_dim 192` / `value_head_dim 384` (paper ratios 0.75 / 1.5), FFN shrunk from 8192 to `--ffn-hidden-size 5824` to absorb the wider mixer. Config: `attn_bench/data/param_count_configs/gdn_1B_args_8heads_ffn5824.txt`. Like the masked `full` baseline, document boundaries are isolated: `--use-packed-seq-params` resets the GDN recurrent state + conv at every document boundary via `cu_seqlens` (kept `--reset-position-ids` + `--eod-mask-loss`).

This run completed cleanly via the data-exhaustion fix — it exited with `[exiting program after consuming all available data at iteration 15549]` and saved a valid checkpoint at step 15549 (no `StopIteration` crash, no resume needed).

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 15549) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| gated delta net (GDN) | `2613202` | 2026-06-24 23:58:05 | 2026-06-25 05:15:35 | 5h 17m 30s | COMPLETED (data exhausted) | 2.4125 | ~321.3 (avg) |

W&B run: `llama3-1b-gdn-fineweb40B-gutenberg3B-2613202` (project `fineweb-40B_gutenberg-3B`).

Final step lm loss 2.4125 is higher than the masked `full` baseline (2.3824). (No validation set: split `100,0,0`.)

Container: `nemo_26.04_te2.15` (ships `flash-linear-attention` + `causal_conv1d`, required by the GDN layer).

Checkpoint saved at step 15549. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-gdn-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_gdn_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/2613202.{out,err}`.

---

## GDN state carry across batches (r = 0 / 0.5 / 1)

GDN mixer (same param-matched config as above) but **without** `--use-packed-seq-params`, so the recurrent + conv state is not reset at document boundaries (it leaks across docs within a sequence). `--gdn-state-carry-ratio` then controls whether the state is also carried *across batch boundaries*: `0.0` = always reset per batch (vanilla Megatron GDN, xdoc-leak baseline), `1.0` = always carry, `0.5` = carry per sequence with p = 0.5. All three launched together on 2026-06-26.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| carry r=0 | `2622827` | 2026-06-26 03:04:10 | TBD | TBD | RUNNING | TBD | TBD |
| carry r=0.5 | `2622828` | 2026-06-26 03:25:51 | TBD | TBD | RUNNING | TBD | TBD |
| carry r=1 | `2622831` | 2026-06-26 03:55:57 | TBD | TBD | RUNNING | TBD | TBD |

Nodes (14 each, disjoint across the three jobs — recorded for throughput-placement analysis):

- r=0 (`2622827`): `nid[006272,006281,006315,006761,006904,006916,006954,006969,007041,007048,007095,007272,007278,007339]`
- r=0.5 (`2622828`): `nid[006719,006728,006749,006751,006917,007013,007134,007184,007188,007211,007216,007236-007237,007239]`
- r=1 (`2622831`): `nid[006041,006050,006107,007263,007305,007333,007340,007342,007464,007476,007499,007512,007525,007528]`

Earlier GDN tests flagged `nid006742` as unreliable (excluded via `sbatch --exclude=nid006742` on the main GDN runs); it is not in any of the three allocations above.

Note: r=0 shows noticeably lower and jitterier throughput than r=1 (median ~301 vs ~312 TFLOP/s, ~3.4% vs ~0.25% of iters stalling), while lm loss is unaffected (r=0 tracks slightly *below* r=0.5/r=1). The carry code path is not the cause (r=0 disables carry entirely — less work, no extra kernels/recompiles), so this is under investigation as a per-job node-placement artifact rather than a property of the training mode.

Slurm scripts: `attn_bench/submissions/pretrain_llama3_1b_gdn_carry_r{0,0.5,1}_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/2622827.{out,err}` (r=0), `2622828.{out,err}` (r=0.5), `2622831.{out,err}` (r=1).

---

## Attention variants / trained models 

| variant | Megatron flag | description |
|---|---|---|
| full | *(none)* | standard softmax — `softmax(QKᵀ/√d)` |
| gated | `--attention-output-gate` | element-wise gate multiplied onto the attention output |
| sink | `--softmax-type learnable` | learnable sink logit added to denominator — `exp(s) / (exp(s) + Σ exp(xⱼ))` |
| off-by-one | `--softmax-type off-by-one` | sink with fixed logit 0 — `1 / (1 + Σ exp(xⱼ))` |
| full (xdoc leak) | drop `--use-packed-seq-params` + `--reset-position-ids` (keep `--eod-mask-loss`) | standard softmax, but no intra-document masking — attention leaks across document boundaries within a packed sequence |
| gated delta net (GDN) | `--experimental-attention-variant gated_delta_net --linear-attention-freq [1]*16` | GDN linear-attention mixer replaces softmax attention on all layers; FFN shrunk to 5824 to param-match (~1.239B) |

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


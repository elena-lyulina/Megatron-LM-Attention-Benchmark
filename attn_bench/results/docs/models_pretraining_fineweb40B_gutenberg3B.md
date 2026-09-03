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

LLaMA 3.2 1B backbone with the attention layers replaced by a Gated Delta Net (GDN) linear-attention mixer on all 16 layers — a different sequence mixer rather than a softmax variant. Param-matched to the ~1.236B attention baselines (~1.239B): GDN mixer with 8 K/V heads, `key_head_dim 192` / `value_head_dim 384` (paper ratios 0.75 / 1.5), FFN shrunk from 8192 to `--ffn-hidden-size 5824` to absorb the wider mixer. Config: `attn_bench/configs/param_count_configs/gdn_1B_args_8heads_ffn5824.txt`. Like the masked `full` baseline, document boundaries are isolated: `--use-packed-seq-params` resets the GDN recurrent state + conv at every document boundary via `cu_seqlens` (kept `--reset-position-ids` + `--eod-mask-loss`).

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
| carry r=0 | `2622827` | 2026-06-26 03:04:10 | 2026-06-26 08:46:45 | 5h 42m 35s | COMPLETED (data exhausted) | 2.4136 | ~296.7 (avg) |
| carry r=0.5 | `2622828` | 2026-06-26 03:25:51 | 2026-06-26 09:00:18 | 5h 34m 27s | COMPLETED (data exhausted) | 2.4133 | ~303.1 (avg) |
| carry r=1 | `2622831` | 2026-06-26 03:55:57 | 2026-06-26 09:23:42 | 5h 27m 45s | COMPLETED (data exhausted) | 2.4218 | ~310.1 (avg) |

Nodes (14 each, disjoint across the three jobs — recorded for throughput-placement analysis):

- r=0 (`2622827`): `nid[006272,006281,006315,006761,006904,006916,006954,006969,007041,007048,007095,007272,007278,007339]`
- r=0.5 (`2622828`): `nid[006719,006728,006749,006751,006917,007013,007134,007184,007188,007211,007216,007236-007237,007239]`
- r=1 (`2622831`): `nid[006041,006050,006107,007263,007305,007333,007340,007342,007464,007476,007499,007512,007525,007528]`

Earlier GDN tests flagged `nid006742` as unreliable (excluded via `sbatch --exclude=nid006742` on the main GDN runs); it is not in any of the three allocations above.

Note: r=0 shows noticeably lower and jitterier throughput than r=1 (median ~301 vs ~312 TFLOP/s, ~3.4% vs ~0.25% of iters stalling), while lm loss is unaffected (r=0 tracks slightly *below* r=0.5/r=1). The carry code path is not the cause (r=0 disables carry entirely — less work, no extra kernels/recompiles), so this is under investigation as a per-job node-placement artifact rather than a property of the training mode.

W&B runs (project `fineweb-40B_gutenberg-3B`):

- r=0: `llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B-2622827`
- r=0.5: `llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B-2622828`
- r=1: `llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B-2622831`

Checkpoints saved at step 15549. Moved to long-term storage under:

- r=0: `/users/elyulina/store/pretrain-results/llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B/`
- r=0.5: `/users/elyulina/store/pretrain-results/llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B/`
- r=1: `/users/elyulina/store/pretrain-results/llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B/`

Slurm scripts: `attn_bench/submissions/pretrain_llama3_1b_gdn_carry_r{0,0.5,1}_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/2622827.{out,err}` (r=0), `2622828.{out,err}` (r=0.5), `2622831.{out,err}` (r=1).

---

## Goldfish loss (full attention vs. GDN)

Full-attention baseline and the GDN mixer, each re-trained with goldfish loss added on top (same param-matched configs as above): a pseudo-random, hash-based token dropout from the loss (never from the input) to reduce verbatim memorization. Mechanism: for each sample, hash every length-`h` token window; a token is excluded from the loss if `hash < 1/k`. `--goldfish-k 50 --goldfish-h 50` → ~2% of tokens dropped from the loss per sample. See `attn_bench/_plans/goldfish_loss_port_plan.md` for the port details. Document boundaries isolated in both runs (`--use-packed-seq-params` + `--reset-position-ids` + `--eod-mask-loss`), same as the masked `full` / `gated delta net (GDN)` baselines.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (goldfish) | `2710458` | 2026-07-09 23:38 | 2026-07-10 05:25 | 5h 47m | COMPLETED | 2.3835 | ~290.1 |
| gdn (goldfish) | `2710460` | 2026-07-09 23:38 | 2026-07-10 04:55 | 5h 17m | COMPLETED | 2.4126 | ~320.9 |

Note: the full Slurm `.out`/`.err` logs are not available in full for these two jobs — the logs directory was rsynced off scratch mid-run, and the scratch copy was deleted before that was caught. Only a partial (~50%-progress) local copy survives. All figures above come from the corresponding W&B runs instead.

W&B runs (project `fineweb-40B_gutenberg-3B`):

- full (goldfish): `llama3-1b-full-attn-goldfish-fineweb40B-gutenberg3B-2710458` (`7p8xyl6l`)
- gdn (goldfish): `llama3-1b-gdn-goldfish-fineweb40B-gutenberg3B-2710460` (`omr2wira`)

Container: `nemo_26.04_te2.15`.

Slurm scripts: `attn_bench/submissions/pretrain_llama3_1b_full_attn_goldfish_fineweb40B_gutenberg3B.slurm`, `attn_bench/submissions/pretrain_llama3_1b_gdn_goldfish_fineweb40B_gutenberg3B.slurm`.

---

## Filler data: long documents vs. same documents split into 1024-token chunks

Two full-attention runs at the same ~42.8B token budget, holding the Gutenberg filler fixed and varying only the FineWeb-Edu-Dedup filler: `long` uses the longest-document subset of FineWeb-Edu-Dedup as-is (`build_long_dataset.py`/`extract_long_docs.slurm`, whole documents, budget 40B), `long-split-1024` takes that same long-document pool and splits it into fixed 1024-token chunks (`split_long_dataset.slurm`, `chunk_size=1024`, `tail_merge_threshold=256`, each chunk re-wrapped with its own BOS/EOS).

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (long) | `2765350` | 2026-07-15 06:01 | 2026-07-15 12:04 | 6h 03m 14s | COMPLETED | 2.36721 | ~300 |
| full (long-split-1024) | `2765404` | 2026-07-15 07:18 | 2026-07-15 13:02 | 5h 43m 53s | COMPLETED (data exhausted) | 2.465832 | ~291.9 |

Note: the `2765350` (long) Slurm `.out`/`.err` logs are only available up to iteration 2092/15535 locally (checkpoint at step 2000 confirmed) — neither the scratch nor store copies had the full log when checked, cause unclear. Timing/loss/throughput for that row come from W&B instead. `2765404` (long-split-1024) has a full local log: it exited cleanly via `[exiting program after consuming all available data at iteration 15561]`, no crash, checkpoint saved at step 15561.

Final step lm loss for `long` (2.36721) is essentially on par with the masked `full` baseline on the original blend (2.3824, see "Initial training" above), while `long-split-1024` (2.465832) is meaningfully higher — despite an identical token budget and near-identical Gutenberg half, splitting the FineWeb filler into 1024-token chunks costs loss relative to keeping the long documents whole.

Dataset paths (`/iopsstor/scratch/cscs/$USER/datasets/tokenized/`):

- long: `fineweb-edu-dedup-160B-datatrove_long_40B` — 40,000,002,920 tokens
- long-split-1024: `fineweb-edu-dedup-160B-datatrove_long_40B_split1024` — 40,070,239,040 tokens
- Gutenberg (both runs): `gutenberg_rep_1_256` — 2,762,833,920 tokens

Container: `nemo_26.04_te2.15`.

Checkpoints saved at the final step. Moved to long-term storage under:

- long: `/users/elyulina/store/pretrain-results/llama3-1b-full-attn-fineweb40B-long-gutenberg3B/`
- long-split-1024: `/users/elyulina/store/pretrain-results/llama3-1b-full-attn-fineweb40B-long-split-1024-gutenberg3B/`

Slurm scripts: `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb40B-long_gutenberg3B.slurm`, `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb40B-long-split-1024_gutenberg3B.slurm`.

Logs: `attn_bench/logs/2765350.{out,err}` (long, partial), `attn_bench/logs/2765404.err` + `attn_bench/_logs/2765404.out` (long-split-1024, full — moved to `_logs/` for exceeding 3 MB).

---

## Full attention, scf=1 (explicit rope-scaling-factor)

Full attention re-trained with `--rope-scaling-factor` explicitly set to `1` (previously Megatron used hardcoded 8; `--max-position-embeddings` correspondingly dropped 131072 → 8192).
Also changed: `--weight-decay` 0.01 → 0.1, `--lr-warmup-iters` 2000 → 500, distributed/batch config 14 nodes/TP=2/MBS=4/GBS=336 → 8 nodes/TP=1/MBS=3/GBS=288 (same token budget, so `TRAINING_STEPS` 15550 → 18141), container `nemo_26` → `nemo_26.04_te2.15`, and periodic checkpointing added (`CHECKPOINT_STEPS=2000`, previously only saved at the final step).

First attempt (`3073433`) was cancelled after a node failure at iteration 2394 (SIGTERM triggered Megatron's exit-signal checkpoint at iteration 2395, though its confirmation print never flushed to the log before the process was killed); restarted (`3074350`), which loaded that iteration-2395 checkpoint and resumed at iteration 2396 — zero repeated iterations.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| full (scf1) | `3073433` (1h 16m) → `3074350` (7h 17m) | 2026-08-13 12:34 | 2026-08-14 08:52 | 8h 33m combined | cancelled → COMPLETED (data exhausted) | 2.3946 | ~356.8 (avg) |

W&B run: `llama3-1b-full-attn-scf1-fineweb40B-gutenberg3B-3074350` (`14gbs383`, project `fineweb-40B_gutenberg-3B`).

Checkpoint saved at step 18141. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-full-attn-scf1-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_full_attn_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/3073433.{out,err}` (cancelled attempt), `attn_bench/logs/3074350.err` + `attn_bench/_logs/3074350.out` (resume, full — moved to `_logs/` for exceeding 3 MB).

---

## Gated attention, scf=1

Gated attention re-trained with the same scf=1 config as the full-attention run above (`--rope-scaling-factor 1`, 8 nodes / TP=1 / MBS=3 / GBS=288, `TRAINING_STEPS=18141`, container `nemo_26.04_te2.15`).

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| gated (scf1) | `3108434` | 2026-08-18 03:52:54 | 2026-08-18 11:59:50 | 8h 06m 56s | COMPLETED (data exhausted) | 2.3832 | ~360 (avg) |

Note: local `.out`/`.err` logs for job `3108434` are incomplete — the scratch copy was deleted mid-run, before `move_checkpoint_to_store.sh` could archive it (its "slurm log not found" warning confirms this), so the file on disk stops at iteration 1069/18141 with no crash.
Training and checkpointing were unaffected; the figures above come from the run's W&B-captured console output instead.

W&B run: `llama3-1b-gated-attn-scf1-fineweb40B-gutenberg3B-3108434` (`guoowo9z`, project `fineweb-40B_gutenberg-3B`).

Checkpoint saved at step 18141. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-gated-attn-scf1-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_gated_attn_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/3108434.{out,err}` (partial — see note above).

---

## Sink attention, scf=1

Sink attention re-trained with the same scf=1 config as the full-attention run above (`--rope-scaling-factor 1`, 8 nodes / TP=1 / MBS=3 / GBS=288, `TRAINING_STEPS=18141`, container `nemo_26.04_te2.15`).

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| sink (scf1) | `3108550` | 2026-08-18 04:24:28 | 2026-08-18 12:35:07 | 8h 10m 39s | COMPLETED (data exhausted) | 2.3941 | ~359 (avg) |

W&B run: `llama3-1b-sink-attn-scf1-fineweb40B-gutenberg3B-3108550` (`k48e1gw9`, project `fineweb-40B_gutenberg-3B`).

Checkpoint saved at step 18141. Moved to long-term storage under:
`/users/elyulina/store/pretrain-results/llama3-1b-sink-attn-scf1-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_sink_attn_fineweb40B_gutenberg3B_te215.slurm`

Logs: `attn_bench/logs/3108550.err` + `attn_bench/_logs/3108550.out` (full — moved to `_logs/` for exceeding 3 MB).

---

## Sliding window attention (SWA), scf=1

Sliding window (causal, left-only) attention on all layers — not a Gemma2/Mistral-style hybrid with full-attention layers mixed in (`--window-attn-skip-freq` left unset, so every layer uses the window). Same scf=1 config as the full/gated/sink runs above (`--rope-scaling-factor 1`, 8 nodes / TP=1 / MBS=3 / GBS=288, `TRAINING_STEPS=18141`, container `nemo_26.04_te2.15`). Three window sizes launched together on 2026-08-19 via `--window-size $WINDOW_SIZE,0`.

All three completed cleanly via the data-exhaustion fix — exited with `[exiting program after consuming all available data at iteration 18141]`, no crash.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| swa w=4096 | `3119138` | 2026-08-19 00:54:21 | 2026-08-19 08:42:43 | 7h 48m 22s | COMPLETED (data exhausted) | 2.3923 | ~380.1 (avg) |
| swa w=1024 | `3119140` | 2026-08-19 00:54:31 | 2026-08-19 08:08:44 | 7h 14m 13s | COMPLETED (data exhausted) | 2.4013 | ~410.2 (avg) |
| swa w=256 | `3119141` | 2026-08-19 00:54:40 | 2026-08-19 07:53:17 | 6h 58m 37s | COMPLETED (data exhausted) | 2.4305 | ~425.6 (avg) |

Throughput increases as the window shrinks (less attention compute per token), while final loss worsens — the expected quality/speed trade-off of restricting the attention context.

W&B runs (project `fineweb-40B_gutenberg-3B`):

- w=4096: `llama3-1b-swa-w4096-scf1-fineweb40B-gutenberg3B-3119138` (`05fak7uh`)
- w=1024: `llama3-1b-swa-w1024-scf1-fineweb40B-gutenberg3B-3119140` (`a7fkl5p5`)
- w=256: `llama3-1b-swa-w256-scf1-fineweb40B-gutenberg3B-3119141` (`kxayjfpn`)

Checkpoints saved at step 18141, on scratch under `attn_bench/results/pretrain/fineweb-40B_gutenberg-3B/llama3-1b-swa-w{4096,1024,256}-scf1-fineweb40B-gutenberg3B/checkpoints/` — not yet moved to long-term store.

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_sliding_window_attn_fineweb40B_gutenberg3B.slurm`, launched via `sbatch --export=ALL,WINDOW_SIZE={4096,1024,256} attn_bench/submissions/pretrain_llama3_1b_sliding_window_attn_fineweb40B_gutenberg3B.slurm`.

Logs: `attn_bench/logs/3119138.err` + `attn_bench/_logs/3119138.out` (w=4096), `3119140.err` + `_logs/3119140.out` (w=1024), `3119141.err` + `_logs/3119141.out` (w=256) — `.out` files moved to `_logs/` for exceeding 3 MB.

---

## Multi-head Latent Attention (MLA) mixer

LLaMA 3.2 1B backbone with the attention layers replaced by the Multi-head Latent Attention (MLA) block from DeepSeek-V2-Lite (verbatim config): 16 heads, `kv-lora-rank 512`, `qk_nope 128` / `qk_rope 64` / `v 128`, no Q compression (`--q-lora-rank` omitted). FFN shrunk from 8192 to `--ffn-hidden-size 7680` to param-match the ~1.236B attention baselines (~1.238B). Config: `attn_bench/configs/param_count_configs/mla_llama_1B_args_ffn7680.txt`. RoPE forced to plain rotary: `--rope-type rope --no-rope-fusion --rotary-base 500000` (MLA's dataclass default is `yarn`; fused MLA-RoPE is yarn-only and `__post_init__` errors on `rope` + fusion), `--max-position-embeddings 8192`. Document boundaries isolated via `--use-packed-seq-params` + `--reset-position-ids` + `--eod-mask-loss`, same as the masked `full` baseline.

Dist config: 12 nodes / TP=1 / MBS=3 / GBS=288 (DP=48, GAS=2), `TRAINING_STEPS=18141`, `--weight-decay 0.1`, `--lr-warmup-iters 500`, container `nemo_26.04_te2.15`. Same token budget and LR schedule as the other scf=1 runs.

**Note — the args dump shows `rope_scaling_factor = 8.0`, but it is a dead value in these runs.** The MLA/KDA scripts do not pass `--rope-scaling-factor`, so that arg sits at Megatron's default 8.0; the args dump prints every arg whether or not the model reads it. It is never wired in:

- MLA with `--rope-type rope` builds its rotary embedding at `megatron/core/transformer/multi_latent_attention.py:181` — `RotaryEmbedding(qk_pos_emb_head_dim, rotary_percent=..., rotary_base=..., cp_group=...)` — it does **not** pass `rope_scaling` / `rope_scaling_factor` at all, so they fall to the `RotaryEmbedding` class defaults (`rope_scaling=False`, `rope_scaling_factor=8.0`).
- In `RotaryEmbedding.__init__` (`megatron/core/models/common/embeddings/rotary_pos_embedding.py:83`) the scaling is guarded: `if rope_scaling: self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)`. With `rope_scaling=False`, `_apply_scaling` never runs and `rope_scaling_factor` is untouched.
- The non-MLA GPT path (`megatron/core/models/gpt/gpt_model.py:167`) only forwards `rope_scaling=args.use_rope_scaling`, which is `False` here (confirmed in the args dump). The softmax scf=1 baselines instead have `use_rope_scaling=True` + `rope_scaling_factor=1.0`, which reaches `_apply_scaling` but is a mathematical no-op (scale by 1).

Net effect for MLA: plain RoPE, `rotary_base=500000`, `--max-position-embeddings 8192`, no scaling applied — identical in effect to the softmax scf=1 baselines, just reached by a different route. So `scf1` in the experiment name is accurate.

Ran in two slices (all resume slices must use the same 12-node count — distrib-optim checkpoint padding is DP-count-specific):

- **Initial** (`3245261`): `--exit-duration-in-mins 315` → Megatron saved a checkpoint and exited cleanly at iteration 8180 (5h15m, 15 min before the 05:30:00 walltime kill), *not* a crash.
- **Resume** (`3248575`): loaded the iteration-8180 checkpoint and resumed at iteration 8181 — zero repeated iterations — then ran to 18141 and exited cleanly via `[exiting program after consuming all available data at iteration 18141]`. No `--exit-duration-in-mins` this slice.

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| mla (scf1) | `3245261` (initial) → `3248575` (resume) | 2026-08-31 17:34:28 | 2026-09-01 22:56:06 | 5h 15m 56s + 6h 28m 35s | clean time-limit save → COMPLETED (data exhausted) | 2.3972 (final step; ~2.385 avg last 50) | ~173 (avg) |

Throughput (~173) is roughly half the softmax baselines' ~360 — the packed-seq (THD) path is ~1.8x slower even absorbed onto 12 nodes (the THD path, not MLA itself; see the slurm header + job 3245069).

W&B runs (project `fineweb-40B_gutenberg-3B`):

- initial: `llama3-1b-mla-scf1-fineweb40B-gutenberg3B-3245261` (`bu9dmq9m`)
- resume: `llama3-1b-mla-scf1-fineweb40B-gutenberg3B-3248575` (`fkouec5p`)

Checkpoint saved at step 18141, on scratch under `attn_bench/results/pretrain/fineweb-40B_gutenberg-3B/llama3-1b-mla-scf1-fineweb40B-gutenberg3B/checkpoints/` — move to long-term store under:
`/users/elyulina/store/pretrain-results/llama3-1b-mla-scf1-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_mla_fineweb40B_gutenberg3B.slurm`

Logs: `attn_bench/logs/3245261.err` + `attn_bench/_logs/3245261.out` (initial), `attn_bench/logs/3248575.err` + `attn_bench/_logs/3248575.out` (resume) — `.out` files in `_logs/` for exceeding 3 MB.

---

## Kimi Delta Attention (KDA) mixer

LLaMA 3.2 1B backbone with the attention layers replaced by a Kimi Delta Attention (KDA) linear-attention mixer on all 16 layers. 16 K/V heads, `--linear-key-head-dim 128` / `--linear-value-head-dim 128` (symmetric — *not* the GDN-mirrored 192/384, which NaNs fla's `chunk_kda`; jobs 3244703 etc.); `16 × 128 = 2048` total width matches the softmax baseline's `32 × 64`. FFN shrunk from 8192 to `--ffn-hidden-size 6976` to param-match (~1.235B). Config: `attn_bench/configs/param_count_configs/kda_1B_args_16heads_128_128_ffn6976.txt`. `--position-embedding-type none` (KDA ignores positions — the decay gate handles order; no positional params at all), `--max-position-embeddings 8192` (validation only). Document boundaries isolated via `--use-packed-seq-params`, which resets the KDA recurrent state + conv at every `cu_seqlens` boundary.

KDA needs `flash-linear-attention >= 0.5.x`; the `nemo_26.04_te2.15` container ships 0.4.2, whose `chunk_kda` gate + L2-norm are the old unbounded form → `grad_norm = nan` from iteration 1. The script side-installs fla 0.5.2 into a persisted, gitignored pip prefix (`attn_bench/packages-fla`), mirroring swiss-ai/pretrain `kda-common.sh`.

Dist config: 12 nodes / TP=1 / MBS=2 / GBS=288 (DP=48, GAS=3), `TRAINING_STEPS=18141`, `--weight-decay 0.1`, `--lr-warmup-iters 500`, container `nemo_26.04_te2.15`. MBS=2 (not the baseline's 3): `chunk_kda` intermediates OOM at MBS=3 (jobs 3244703/3244987); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` recovers the fragmented reserved memory. `rope_scaling_factor` also shows 8.0 in the args (same dead default as the MLA run — see that note above), but here it is doubly moot: `use_rope_scaling = False` gates the scaling code path off, and `--position-embedding-type none` means no rotary module is built at all — KDA uses no positional encoding, its decay gate carries order.

Completed cleanly in a single job (`3248579`): ran start→finish to iteration 18141, exited via `[exiting program after consuming all available data at iteration 18141]`, checkpoint saved — no crash, no resume. (The NCCL / TCPStore "connection closed" warnings at the tail of the `.err` are normal post-run teardown noise, after `END TIME` printed.)

| variant | Slurm job | start (CEST) | end (CEST) | run time | status | final lm loss (step 18141) | throughput (TFLOP/s/GPU) |
|---|---|---|---|---|---|---|---|
| kda (scf1) | `3248579` | 2026-09-01 16:30:07 | 2026-09-01 23:35:49 | 7h 05m 42s | COMPLETED (data exhausted) | 2.3473 (final step; ~2.335 avg last 50) | ~283 (avg) |

Final loss (~2.35) is the lowest of the scf=1 mixers so far, and throughput (~283) sits below the softmax baselines' ~360 but well above the MLA THD path. KDA's `params norm` runs at ~1328 (vs ~975–987 for MLA / the softmax variants) — a different init/parameter scale, recorded but not investigated.

W&B run: `llama3-1b-kda-scf1-fineweb40B-gutenberg3B-3248579` (`sobnh674`, project `fineweb-40B_gutenberg-3B`).

Checkpoint saved at step 18141, on scratch under `attn_bench/results/pretrain/fineweb-40B_gutenberg-3B/llama3-1b-kda-scf1-fineweb40B-gutenberg3B/checkpoints/` — move to long-term store under:
`/users/elyulina/store/pretrain-results/llama3-1b-kda-scf1-fineweb40B-gutenberg3B/`

Slurm script: `attn_bench/submissions/pretrain_llama3_1b_kda_fineweb40B_gutenberg3B.slurm` (the committed version has since been switched to a 1h30 reservation-slice + `--exit-duration-in-mins 75` resume setup; job `3248579` itself ran with no exit-duration and finished in one slice).

Logs: `attn_bench/logs/3248579.err` + `attn_bench/_logs/3248579.out` (`.out` in `_logs/` for exceeding 3 MB).

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
| GDN carry (r = 0 / 0.5 / 1) | `--gdn-state-carry-ratio {0,0.5,1}`, drop `--use-packed-seq-params` | GDN without doc-boundary state reset (leaks across docs within a sequence); recurrent + conv state additionally carried across batch boundaries with probability r |
| full / GDN + goldfish loss | `--goldfish-k 50 --goldfish-h 50` (stacks on top of `full` or `gated delta net (GDN)`) | hash-based token dropout from the loss only (~2% of tokens), reduces verbatim memorization |
| full (long / long-split-1024 filler) | *(none — same as `full`)* | same standard softmax attention as `full`; only the FineWeb-Edu-Dedup filler dataset differs (longest documents, whole or split into 1024-token chunks), same token budget |
| sliding window attention (SWA), w = 256/1024/4096 | `--window-size {256,1024,4096},0` | causal sliding window on all layers (left=w, right=0), no full-attention layers mixed in |
| multi-head latent attention (MLA) | `--multi-latent-attention --kv-lora-rank 512 --rope-type rope --no-rope-fusion` | DeepSeek-V2-Lite MLA block replaces softmax attention on all layers (16 heads, KV squeezed through a 512 latent); FFN shrunk to 7680 to param-match (~1.238B) |
| kimi delta attention (KDA) | `--experimental-attention-variant kimi_delta_attention --linear-attention-freq [1]*16 --linear-num-key-heads 16 --linear-num-value-heads 16 --linear-key-head-dim 128 --linear-value-head-dim 128 --position-embedding-type none` | KDA linear-attention mixer replaces softmax attention on all layers (symmetric 128/128 heads); FFN shrunk to 6976 to param-match (~1.235B); needs flash-linear-attention ≥ 0.5.x |

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

### FineWeb filler variants (long documents)

Used only by the "Filler data: long documents vs. same documents split into 1024-token chunks" runs above — same Gutenberg half as the blend above, but a different FineWeb-Edu-Dedup filler (longest documents instead of the 0.25 partition):

| source | tokens | path on cluster |
|---|---|---|
| FineWeb-Edu-Dedup, long docs | 40,000,002,920 | `datasets/tokenized/fineweb-edu-dedup-160B-datatrove_long_40B` |
| FineWeb-Edu-Dedup, long docs split into 1024-token chunks | 40,070,239,040 | `datasets/tokenized/fineweb-edu-dedup-160B-datatrove_long_40B_split1024` |
| Gutenberg (rep_1_256) | 2,762,833,920 | `datasets/tokenized/gutenberg_rep_1_256` |

`long`: longest-document subset of FineWeb-Edu-Dedup, selected up to a 40B token budget (`build_long_dataset.py`/`extract_long_docs.slurm`), documents kept whole.

`long-split-1024`: the same long-document pool, split into fixed 1024-token chunks (`split_long_dataset.slurm`, `chunk_size=1024`, `tail_merge_threshold=256`), each chunk re-wrapped with its own BOS/EOS.


# Adding a New Model: Workflow

Checklist for taking a new attention/mixer variant from idea to evaluated results.
See `memorization_measurement.md`, `models_pretraining_fineweb40B_gutenberg3B.md` and
`attn_bench/scripts/llama_checkpoints.sh` for the details each step points at.

## 1. Implement + test

- Add correctness tests under `attn_bench/tests/` (parameter-disturbance + cross-doc
  isolation, see `reference_attn_test_harness` conventions: `(model) -> bool` checks
  registered and run via `--tests`).
- Add a 1-GPU smoke-test slurm (`attn_bench/submissions/test_<variant>.slurm`, ~5 min)
  that runs the tests. Must PASS before training.

## 2. Final pretrain slurm

- Copy the nearest existing `pretrain_llama3_1b_*.slurm`.
- Naming: `pretrain_llama3_1b_<variant>_<data-tag>.slurm` (`<data-tag>` = the dataset
  blend it trains on, e.g. `fineweb40B_gutenberg3B`).
- Container: always `nemo_26.04_te2.15` (`nemo_26` crashes on any `megatron.core`
  import — see `reference_nemo26_nvrx_crash`).
- `sbatch` it. Note the job ID(s) (resumes get their own IDs too).

## 3. Move checkpoint + logs to store

Once the job completes:

```bash
bash attn_bench/scripts/move_checkpoint_to_store.sh <EXP_NAME> [JOB_ID ...]
```

Rsyncs the whole experiment folder (checkpoints, wandb/tensorboard logging, debug/,
triggers/) from scratch to `/users/$USER/store/pretrain-results/<EXP_NAME>/` as one
unit, drops the slurm `.out`/`.err` logs under `<EXP_NAME>/slurm-logs/` in that same
folder, and verifies the copy (`du -sh` + file-count diff). Does **not** delete the
scratch copy — remove it yourself once confirmed.

## 4. Pull logs locally, update docs, commit

- Pull `.out`/`.err` logs to your machine with the `cplogs` script (a custom local script, `.local/bin/cplogs`).
- Append a run entry (job ID, timing, final loss, throughput, checkpoint path) to
  `models_pretraining_fineweb40B_gutenberg3B.md`.
- Commit the new slurm scripts + doc update.

## 5. Update W&B

Add the new run to the report:
https://wandb.ai/elyulina-thesis/fineweb-40B_gutenberg-3B/reports/Llama-1B-pre-trained-on-Fineweb-edu-40B-Gutenberg-3B-with-different-attention-mechanisms--VmlldzoxNzM0ODgxOA

## 6. Confirm inference works before measuring anything

Standard softmax variants decode out of the box. A **different sequence mixer** needs its
own cached-decode path verified first — see `memorization_measurement.md` § "Adding a new
attention variant", steps 1-2. Precedents: `_plans/gdn_inference_plan.md` (GDN),
`_plans/kda_packed_and_inference_plan.md` + `attn_bench/tests/test_kda_inference.py` (KDA:
`_decode` porting a `fused_recurrent_*` + `causal_conv1d_update` path, parity-tested against
a cacheless quadratic oracle). Do this on a short run before trusting any number from step 8.

## 7. Register the model once

Add one entry to `attn_bench/scripts/llama_checkpoints.sh` (`MODELS` + a `model_config()`
case: `EXP_NAME`, `CKPT_NAME` if it differs, `MEGATRON_EXTRA` flags not restored by
`--use-checkpoint-args`, `NEEDS_TRITON` for fla/triton mixers (GDN, KDA), `NEEDS_FLA_052` if
the mixer needs the side-installed flash-linear-attention 0.5.2 (KDA — the container's 0.4.2
NaNs `chunk_kda`), `IS_SINK_FAMILY` for sink-logit variants (resource policy + config
selection), `NEEDS_UNFUSED_DECODE` if the decode path needs `--attention-backend unfused`,
and `HAS_ROPE=0` for a variant with no rotary embeddings (GDN, KDA — skips the HF-conversion
rope sanity check). This is the single source of truth for every sweep and puller below —
nothing else needs to change.

## 8. Run the eval sweeps

`measure_mem_all.sh` nests `--offsets` × `--prefixes`, drops points where
`offset + prefix + suffix > --max-doc-length`, dual-checks scratch+store for what's already
done, and submits **one bundled `measure_mem.slurm` job per model** (one checkpoint load,
multi-point Stage 1). Megatron backend is the default and is what the dashboard uses; a
recurrent mixer needing flash-linear-attention 0.5.2 is handled by `NEEDS_FLA_052` from
step 7 (the `--backend hf` path is only for softmax families — see
`_plans/kda_mla_hf_checkpoint_conversion_plan.md`, "megatron is faster + exact for KDA").

### Dashboard grid — the one to run

`attn_bench/dashboard/export_data.py` is the source of truth for exactly which points,
suffixes and repetitions the published memorization dashboard plots. Keep this in sync:

- `CANDIDATE_OFFSETS  = [0, 50, 150, 250, 500, 1000, 2000, 3971, 5942, 7892]`
- `CANDIDATE_PREFIXES = [50, 250, 500, 1000, 2000, 3971, 5942, 7892]`
- `REPS               = [0, 1, 16, 32, 64, 128, 256]`   — **not** the full 10-bucket set
- `SUFFIXES           = [25, 50, 75, 100, 150, 250]`   — all from **one suffix=250 run**;
  Stage 2 writes a metrics pkl per reachable boundary
- `MAX_DOC_LENGTH     = 8192`

```bash
# from attn_bench/ -- --dry-run first, then drop it to submit
bash submissions/measure_mem_all.sh --models <tag> --offsets 0 50 150 250 500 1000 2000 3971 5942 7892 --prefixes 50 250 500 1000 2000 3971 5942 7892 --suffix 250 --repetitions 0,1,16,32,64,128,256 --max-doc-length 8192 --time 05:00:00 --dry-run

bash submissions/long_gutenberg_inference_all.sh
bash submissions/long_fineweb_inference_all.sh
```

`--time 05:00:00` overrides measure_mem.slurm's 50-min default. The bundled per-model job
does one checkpoint load then all feasible points sequentially — the full dashboard grid is
~10-15 h of compute for a recurrent mixer (KDA ~30 tok/s/GPU, ~40+ points × 7 reps), so it
will **not** finish in one job. Keep `--time` short (~5 h) so it actually schedules, and
just re-run the call (no `--dry-run`) a few times: each pass submits only still-missing
`(point, rep)` combos and Stage 1 self-skips completed points, so it converges over 2-3
jobs. Softmax mixers are far faster and usually finish in one 5 h job. Omit `--models <tag>`
to (re)sweep every model in `llama_checkpoints.sh`.

The wider offset grid from `notebooks/mem_metrics_2.ipynb`
(`OFFSETS = [0, 1, 5, 12, 25, 50, 100, 500, 1000, 2000, 3000]`, `SUFFIXES = [50, 500]`) is
kept for that notebook's finer-resolution analysis near offset 0 — run it separately if you
need those plots; it is not what the dashboard uses.

Each sweep iterates `llama_checkpoints.sh`, skips combinations already complete, and
writes to scratch only -- scratch is purged only every ~2 weeks, so there's no rush to
promote results. Skip-checks look at both scratch and store (`--persistent-storage-path`),
so results already on store from a previous sync are still found even after scratch has
been purged. Sync scratch to store yourself when you want it (e.g.
`scripts/copy_mem_results_to_store.sh <exp1> [exp2 ...]`; don't do this from a compute
node). Pull results locally:

```bash
bash attn_bench/scripts/pull_mem_results.sh <tag>      # megatron backend -> no --backend hf
bash attn_bench/scripts/pull_long_inference_results.sh
```

## 9. Add the model to the dashboard

Add `'<tag>'` to `MODELS` in `attn_bench/dashboard/export_data.py` (and a colour +
display name to `attn_bench/plotting/model_registry.py` if not already there), then
regenerate the per-suffix JSONs:

```bash
cd attn_bench/dashboard && python export_data.py
```

Commit the new `dashboard/data/<tag>__s*.json` files.

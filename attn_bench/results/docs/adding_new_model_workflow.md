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

- Pull `.out`/`.err` logs to your machine with the `cplogs` script.
- Append a run entry (job ID, timing, final loss, throughput, checkpoint path) to
  `models_pretraining_fineweb40B_gutenberg3B.md`.
- Commit the new slurm scripts + doc update.

## 5. Update W&B

Add the new run to the report:
https://wandb.ai/elyulina-thesis/fineweb-40B_gutenberg-3B/reports/Llama-1B-pre-trained-on-Fineweb-edu-40B-Gutenberg-3B-with-different-attention-mechanisms--VmlldzoxNzM0ODgxOA

## 6. Confirm inference works before measuring anything

Standard softmax variants decode out of the box. A **different sequence mixer**
(e.g. GDN) needs its own cached-decode path verified first — see
`memorization_measurement.md` § "Adding a new attention variant", steps 1-2, and
`_plans/gdn_inference_plan.md` for the GDN precedent. Do this on a short run before
trusting any number from step 7.

## 7. Register the model once

Add one entry to `attn_bench/scripts/llama_checkpoints.sh` (`MODELS` + a `model_config()`
case: `EXP_NAME`, `CKPT_NAME` if it differs, `MEGATRON_EXTRA` flags not restored by
`--use-checkpoint-args`, `NEEDS_TRITON` for GDN-style mixers, `IS_SINK_FAMILY` for sink-logit
variants (resource policy + config selection) and `NEEDS_UNFUSED_DECODE` if the model's decode
path needs `--attention-backend unfused`). This is the single source of truth for
every sweep and puller below — nothing else needs to change.

## 8. Run the eval sweeps

```bash
# from attn_bench/
bash submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 500 1000 2000 5000
bash submissions/long_gutenberg_inference_all.sh
bash submissions/long_fineweb_inference_all.sh
```

Each sweep iterates `llama_checkpoints.sh`, skips combinations already done on store, and
writes to scratch before copying to store via the shared `scripts/scratch_to_store.sh`
helper (never write to capstor from a compute node). Pull results locally:

```bash
bash attn_bench/scripts/pull_mem_results.sh
bash attn_bench/scripts/pull_long_inference_results.sh
```

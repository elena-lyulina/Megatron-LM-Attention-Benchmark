# Memorization Measurement

How we measure verbatim memorization for the pretrained attention variants, and
how the pieces fit together. 

## The idea

For now, all models are LLaMA-3.2-1B trained on the **same** FineWeb-40B + Gutenberg-3B
blend; they differ only in their attention mechanism. In the blend, Gutenberg
books are duplicated at controlled repetition counts (rep = 1, 2, 4, …, 256), so
the same text is seen a known number of times during training.

Memorization is probed by feeding a model a **prefix** of a Gutenberg excerpt,
letting it **greedily generate a suffix**, and measuring how close the generated
suffix is to the true continuation. More training repetitions → more verbatim
memorization. The research question is which attention mechanism memorizes more
(and how attention flow differs between memorized and non-memorized samples).

Models under test (see `models_pretraining_fineweb40B_gutenberg3B.md` for training
details and the variant → Megatron-flag table):

| variant | note |
|---|---|
| `full` | standard softmax baseline |
| `gated` | `--attention-output-gate` |
| `sink` | `--softmax-type learnable` (learnable sink logit) |
| `off-by-one` | `--softmax-type off-by-one` (fixed sink logit 0) |
| `full (xdoc leak)` | standard softmax, no intra-document masking |
| `GDN` | Gated Delta Net linear-attention mixer |
| `GDN-carry r=0/0.5/1` | GDN with recurrent state carried across batches |

Checkpoints live under `/users/elyulina/store/pretrain-results/llama3-1b-<variant>-fineweb40B-gutenberg3B/`.

## Vocabulary: offset / prefix / suffix / repetition

A single inference "point" is defined by four knobs:

- **repetition** — which Gutenberg rep bucket the excerpt is drawn from
  (`rep_0` … `rep_256`). Always swept *inside* a single job
  (`--repetitions 0,1,2,4,8,16,32,64,128,256`). `rep_0` = held-out books the
  model never trained on (control).
- **offset** — where in the document the excerpt starts.
  `offset == 0`: excerpt begins at the document start, BOS is already token 0.
  `offset > 0`: excerpt begins mid-document, so BOS (`128000`) is prepended.
- **prefix length** — how many tokens of context the model is given before it
  starts generating.
- **suffix length** — how many tokens it generates (and how many gold tokens we
  score against). Fixed at **500** for the metric runs, **50** for attention
  capture.

For each sample: `excerpt = input_ids[offset : offset + prefix + suffix]`, the
first `prefix` tokens are the prompt, the last `suffix` tokens are the gold
continuation.

## The measurement pipeline — `measure_mem_*.slurm`

One SLURM script per variant
(`measure_mem_llama3_1b_{full,gated,off_by_one,sink}_attn_*.slurm`), each running
two `srun` stages for a single `(offset, prefix, suffix)` point (offset/prefix
come from `--export`, suffix is hardcoded to 500):

**Stage 1 — sparse inference** (`evaluation/megatron_inference_sparse.py`, 4 GPUs via torchrun)
- Loads the `torch_dist` checkpoint **directly** with `--use-checkpoint-args` (no
  HF conversion — HF doesn't support the custom attentions; DCP resharding merges
  the TP shards on the fly; architecture flags are restored from the checkpoint).
- For each rep bucket, for each sample: a single forward pass computes the
  reference NLL on the gold suffix (`compute_nll`), then `greedy_generate` decodes
  the suffix from the prefix using a `StaticInferenceContext` KV cache.
- Writes one `rank{N}.jsonl` per GPU under
  `inference/offset_O_prefix_P_suffix_S/rep_R_greedy/`, in PDM-compatible format.

**Stage 2 — metric aggregation** (PDM `verbatim_eval/main.py --mode sparse`)
- Reads the jsonls and writes
  `SparseGutenberg/<exp>/offset_O_prefix_P_suffix_S_greedy.pkl` of LCS / Rouge-L
  summary statistics.

Output root: `MEM_BASE=/users/elyulina/store/mem-results`, namespace
`SparseGutenberg/<exp_name>/`.

### Per-sample metrics (written to the jsonl)

Computed in `run_bucket` / `text_metrics`:

| field | meaning |
|---|---|
| `lcs_norm` | normalized longest common **substring** — verbatim memorization |
| `Rouge-L` | longest common **subsequence** overlap |
| `TTR_ref`, `TTR_gen` | type-token ratio of gold / generated suffix (diversity) |
| `nll_mean`, `nll_std`, `perplexity` | on the **generated** suffix |
| `ref_nll_mean`, `ref_nll_std`, `ref_perplexity` | on the **gold** suffix |
| `prefix`, `true_suffix`, `generated_suffix` | raw token ids |

LCS comes from PDM's `verbatim_eval.LCS.find_longest_common_substrings`; Rouge-L
from `verbatim_eval.my_rouge` (DP matrix + `compute_rouge_l_2d`).

## The grid sweep — `measure_mem_all.sh`

Submits every variant listed in its `JOBS` array for every `offset × prefix`
combination, one SLURM job per `(variant, offset, prefix)`. Each `JOBS` entry is
`SCRIPT|EXP_NAME|EXTRA_EXPORTS`: the explicit `EXP_NAME` drives the skip-check, and
`EXTRA_EXPORTS` carries per-job `--export` params — which is how the four GDN
variants reuse one parametrized script via `GDN_VARIANT` (`base`, `carry-r0`,
`carry-r0.5`, `carry-r1`).

```bash
# from attn_bench/
bash submissions/measure_mem_all.sh --offsets 0 --prefixes 50 100 250 1000 1500 2000 3000 4000 5000
bash submissions/measure_mem_all.sh --offsets 0 500 1000 --prefixes 500
bash submissions/measure_mem_all.sh --force --offsets 0 --prefixes 500   # re-submit even if done
```

Each job internally sweeps all repetition buckets. This is the grid that lets us
plot memorization vs prefix length (how much context is needed before the model
"locks on" to a memorized continuation) and vs offset (does position in the
document matter).

**Skip-if-done.** For each combination the driver reads `EXP_NAME` out of the
variant script and checks for the Stage-2 pkl
(`…/SparseGutenberg/$EXP_NAME/offset_O_prefix_P_suffix_500_greedy.pkl`) on store;
if it exists, the job is **not** submitted. This matters because a submitted job
isn't free — it still grabs a 4-GPU node and starts the container even when there
is nothing to do. Pass `--force` to submit regardless (re-measure / overwrite).
The marker is the *final* artifact, so a half-finished point (jsonls but no pkl)
is re-submitted.

Defense in depth: even if a redundant job *is* submitted,
`megatron_inference_sparse.py` checks for existing results (`results_already_complete`)
**before** loading the checkpoint and exits early if everything is on disk — so the
expensive model load is skipped, not just the per-rep generation. The submit-time
check above still matters because the node allocation + container start are paid as
soon as the job runs, regardless.

## Adding a new attention variant

To measure memorization for a newly trained variant:

1. **Confirm inference works.** The eval greedy-generates with a
   `StaticInferenceContext` KV cache (`megatron_inference_sparse.py`). Standard
   softmax variants (full, gated, sink, off-by-one) decode out of the box. A
   *different sequence mixer* may need its own cached-decode path first — e.g. GDN
   originally raised `NotImplementedError` on any `inference_context` and needed a
   conv + recurrent state cache implemented and verified against a quadratic oracle
   before any number could be trusted (see `_plans/gdn_inference_plan.md`,
   `tests/test_gdn_inference.py`).
2. **Confirm checkpoint loading.** Inference loads the `torch_dist` checkpoint with
   `--use-checkpoint-args` (no HF conversion). Verify which architecture flags that
   restores and which must be re-passed explicitly via `--megatron-extra-args` —
   the inference docstring warns some `store_true`/custom flags aren't restored
   correctly, which is why the gated/sink/obo scripts re-pass
   `--attention-output-gate` / `--softmax-type …`. Check this on a short run.
3. **Create `measure_mem_<variant>_*.slurm`.** Copy the nearest existing variant
   script and set: `EXP_NAME` (+ `MODEL_DIR`), `CONTAINER_ENV` (e.g. GDN needs
   `nemo_26.04_te2.15` for `flash-linear-attention` + `causal_conv1d`), and the
   `--megatron-extra-args` for the variant. Keep suffix hardcoded to 500. If the
   variant uses a non-attention mixer with Triton kernels (GDN), give each rank a
   per-rank node-local `TRITON_CACHE_DIR` (the `torchrun --no-python` wrapper in the
   GDN script) — a shared cache races and crashes. A parametrized script (one file,
   several checkpoints via an export like `GDN_VARIANT`) is fine.
4. **Register in `measure_mem_all.sh`.** Add a `JOBS` entry
   `SCRIPT|EXP_NAME|EXTRA_EXPORTS` — otherwise the grid sweep won't include the
   variant. `EXP_NAME` must be explicit (it drives the skip-check); `EXTRA_EXPORTS`
   holds any per-job `--export` params (e.g. `GDN_VARIANT=carry-r1`), or is empty.
5. **Note: `--capture-attention` is softmax-only.** The capture hooks target
   `TEDotProductAttention`; a non-attention mixer (e.g. GDN) has none, so the hooks
   silently never fire. Memorization metrics still work; attention-map capture does
   not apply.

## Attention-score capture — `capture_attn_*.slurm` (`--capture-attention`)

Same inference path, but `evaluation/attn_capture.py` registers a forward-pre-hook
on every `TEDotProductAttention`, **recomputes** the softmax from q/k/v (pure
observation — model output is unchanged), and averages the full `[L, H, S, S]`
attention maps into **10 Rouge-L buckets** (`[0,0.1) … [0.9,1.0]`). This lets us
compare attention flow between memorized (high Rouge-L) and non-memorized samples.

Three file families are written per rank at the run-level inference dir
(aggregated across **all** repetition buckets):

| file | contents |
|---|---|
| `attn_scores_rouge_l_{NN-MM}_rank{N}.npz` | mean softmax-weight map `[L,H,S,S]`. Rows **not** renormalized: for sink/off-by-one the row-sum deficit is the virtual-sink mass; BOS attention is column 0. |
| `norm_attn_rouge_l_{NN-MM}_rank{N}.npz` | Kobayashi norm-based map `n(i,j) = α_ij · ‖v_j ⊙ g_i‖₂` (`g` = sigmoid output gate, 1 for non-gated). |
| `gating_scores_rank{N}.npz` | per-(bucket, layer, head) gate-value histogram — **gated model only**. |

Nothing is collapsed at capture time: BOS attention, sink mass, row entropy, etc.
are all recoverable from the saved maps at whatever granularity you want.

Constraint: `prefix + suffix ≤ 600` (maps are O(S²) per layer/head). Capture runs
use prefix 50 / suffix 50. Capturing **regenerates** every rep (the maps need the
forward passes), so `rank*.jsonl` is rewritten; a run-level marker
(`attn_scores_rouge_l_09-10_rank{N}.npz`) decides resume.

Plotting helpers: `evaluation/plot_attention_patterns.py`
(`load_maps`, `load_all_maps`, `plot_map`, `plot_full_grid`,
`plot_gating_distribution`).

## Generation-quality metrics (PDM-derived, separate jobs)

These run off already-written inference dirs — they auto-discover
`(offset, prefix, suffix, rep, policy)` from the directory structure and
skip-if-output-exists:

| job | script | metric | output |
|---|---|---|---|
| `generation_quality.slurm` | `evaluation/compute_generation_quality.py` | **distinct-n** (token-level diversity) + **perplexity** under a reference HF model (Qwen2.5-1.5B) | `..._distinct_n.json`, `..._perplexity.json` |
| `mauve.slurm` | `evaluation/compute_mauve.py` | **MAUVE** — distribution-level similarity of generated vs true suffixes | `..._mauve.json` |

Both take `--export=EXPRS="exp1 exp2 …"`. `compute_mauve.py` is adapted from
PDM's `verbatim_eval/compute_mauve.py`.

## Role of the PDM repo

PDM (Xu et al., 2025) is an **external** repo we don't own. The memorization
metrics reuse its machinery:
`verbatim_eval.{LCS, my_rouge, main, utils.load_inference_data}`. We cannot push
to it — local changes are preserved as `attn_bench/utils/PDM_patch.txt` and
applied to a fresh clone with `patch -p1 --ignore-whitespace` (see
`PDM-workflow.md`). On the cluster: `PDM_DIR=/users/elyulina/scratch/PDM`, added
to `PYTHONPATH` alongside its `src/`.

## Extras

- **`--sink-scale`** (`patch_sink_scale`, sink / off-by-one only): at inference,
  scales the learned sink logit (`offset_new = offset_trained + log(scale)`;
  `1` = identity, `>1` strengthens the sink, `<1` weakens it) to probe how the
  sink affects memorization. Appends `_sscale{X}` to the experiment path and dumps
  the original per-head offsets to `sink_scale_metadata.json`.
- **`cross_doc_attn.slurm`** — *not* a memorization metric. It's a correctness
  test (on a tiny model) that the block-diagonal cross-document mask actually
  isolates documents end-to-end. It underpins the premise of the `xdoc-leak`
  experiment (does cross-document attention leakage increase memorization?).
- **GDN checkpoint-state init (planned).** The current GDN measure runs
  zero-initialise the recurrent + conv state. The `r0.5` / `r1` carry checkpoints
  also persist their carried state (per-rank `gdn_state_carry_rank<N>.pt`,
  layout-locked to training MBS/DP/seed). Seeding inference from those — to ask
  whether the learned cross-batch memory changes what's memorized — is a future
  `CHKP_STATES`-style knob (suffix `_chkpstates` on the experiment name, like
  `_sscale`). It needs both a `gated_delta_net.py` change (prefill from a supplied
  `initial_state`) and an aggregation rule (the saved per-sequence states don't
  match the inference batch), so it's deferred, not yet implemented.

## Directory layout (output)

```
/users/elyulina/store/mem-results/SparseGutenberg/<exp_name>/
├── inference/
│   └── offset_O_prefix_P_suffix_S/
│       ├── rep_R_greedy/rank{0..3}.jsonl          # per-sample metrics
│       ├── attn_scores_rouge_l_{NN-MM}_rank{N}.npz   # capture (run-level)
│       ├── norm_attn_rouge_l_{NN-MM}_rank{N}.npz
│       └── gating_scores_rank{N}.npz                 # gated only
├── offset_O_prefix_P_suffix_S_greedy.pkl          # LCS / Rouge-L summary
├── offset_O_prefix_P_suffix_S_greedy_distinct_n.json
├── offset_O_prefix_P_suffix_S_greedy_perplexity.json
└── offset_O_prefix_P_suffix_S_greedy_mauve.json
```

## At a glance — which script does what

| script | purpose |
|---|---|
| `measure_mem_<variant>_*.slurm` | one `(offset, prefix)` point for one model: inference + LCS/Rouge-L |
| `measure_mem_all.sh` | grid driver: submits all variants × offsets × prefixes (skips combos whose pkl exists; `--force` overrides) |
| `capture_attn_<variant>_*.slurm` | capture `[L,H,S,S]` attention maps bucketed by Rouge-L |
| `generation_quality.slurm` | distinct-n + reference-model perplexity |
| `mauve.slurm` | MAUVE score |
| `cross_doc_attn.slurm` | correctness test of cross-document masking (not a metric) |
| `evaluation/megatron_inference_sparse.py` | the inference engine (greedy gen + per-sample metrics) |
| `evaluation/attn_capture.py` | attention-map capture into Rouge-L buckets |
| `evaluation/plot_attention_patterns.py` | plotting the captured maps |
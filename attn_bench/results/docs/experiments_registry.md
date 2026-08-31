# Experiments Registry

Every analysis experiment run so far: what it asks, what code/data it uses, what it
plots, what we found, and what's still open. One entry per experiment, not per model —
several entries can point at the same notebook when that notebook covers more than one
question (most of them do).

## Index

Category is one of **memorization comparison** (fixed outcome metric, varying treatment),
**behavioral** (black-box loss/output under a condition), or **mechanistic** (internals —
attention weights, recurrent state). It's a label on the finding, not the notebook as a
whole — see the note below the table.

| Experiment | Category | Notebook(s) (current) |
|---|---|---|
| Attention variant vs memorization | memorization comparison | `scf8/attn_types.ipynb` |
| xdoc-attn-leak vs memorization | memorization comparison | `scf8/xdoc_leak.ipynb` |
| Goldfish loss vs memorization | memorization comparison | `scf8/goldfish_loss.ipynb` |
| GDN state-carry policy vs memorization | memorization comparison | `gdn_variants.ipynb` |
| FineWeb80B vs memorization | memorization comparison | `scf8/data_amount.ipynb` |
| Sink value ablation | memorization comparison | `scf8/sink_variants.ipynb` |
| Attention pattern analysis | mechanistic | `scf8/attention_patterns.ipynb` |
| Long-context extrapolation (aggregate) | behavioral | `scf8/long_inference.ipynb` |
| GDN recurrent state norm | mechanistic | `gdn_variants.ipynb` |
| Long-context individual-sequence loss peak analysis | behavioral | `scf8/long_inference_individual.ipynb` |
| Training data length effect (long vs split-1024) | memorization comparison + behavioral | `scf8/data_length.ipynb` |
| SWA (sliding-window attention) vs memorization | memorization comparison | `swa_variants.ipynb` |
| Suffix-boundary sensitivity | memorization comparison | `suffix_boundaries.ipynb` |
| scf1-trained model family comparison | memorization comparison | `mem_results_scf1.ipynb` |
| Extended metrics across models | memorization comparison | `extended_metrics.ipynb` |
| RoPE-scaling correctness check | n/a (correctness check) | `scf1_vs_scf8.ipynb` |
| Backend agreement + scf1 training-config check | n/a (infra correctness check) | `backend_comparison.ipynb` |
| Generated examples viewer | n/a (qualitative/secondary tooling) | `generated_examples.ipynb` |
| Document / packed-chunk length distributions | n/a (data characterization) | `training_data.ipynb` |
| Chinchilla budget planning (not an experiment) | n/a (planning) | `chinchilla_budget.ipynb` |

`attn_types`/`sink_variants`/`xdoc_leak`/`data_amount`/`data_length`/`goldfish_loss`/
`attention_patterns`/`long_inference`/`long_inference_individual` live under
`notebooks/scf8/` (scf8-only families, or structural ablations staying at scf8 by choice);
`gdn_variants` (no scf8/scf1 split for GDN) stays at the top level. `extended_metrics.ipynb`
and `generated_examples.ipynb` aren't listed as separate rows above since they're
presentation views of experiments already covered elsewhere, not new questions — see their
own entries below. Data lives in `results/mem-results/SparseGutenberg/` (current) and
`results/mem-results/SparseGutenberg-legacy/` (older, 4-model + sink-ablation data).

---

## Attention variant vs memorization

**Category:** memorization comparison.

**What it does:** Core question of the study — feed a Gutenberg excerpt prefix, let the
model greedily generate the suffix, measure how close it is to the true continuation.
Main cross-mechanism comparison: full, learnable sink, gated, GDN. Off-by-one is
deliberately not included here — it lives in "Sink value ablation" below, since the sink
family gets its own deeper comparison (off-by-one vs learnable vs value-ablated) rather
than one more line on the main chart. Also compares across repetition counts (rep 0...256)
and (offset, prefix, suffix) combinations.

**Code:** `results/mem-results/SparseGutenberg/`, loaded via `plotting/data_loading.py`,
plotted via `plotting/heatmap.py` (Rouge-L/Hayes heatmaps), `plotting/rouge_l.py`
(distributions), `plotting/utils.py` (paired-metric lineplots) — mostly called from
`extended_metrics.ipynb` now, Rouge-L itself from `scf8/attn_types.ipynb`.

**Plots:** Rouge-L, LCS, match-rate (threshold buckets + exact match), divergence point,
token-level accuracy, paired PPL/NLL/TTR, Rouge-L distributions (per-snippet histograms +
rep x Rouge-L-bin heatmaps). Main comparison: Rouge-L at offset=0, prefix=500, suffix=500
across all variants.

**Findings:**
- Full-attn: prefix=500 keeps memorization high across offset (a strong anchor is directly
  attendable); prefix=50 decays with offset (weak anchor, no run-up to lock onto).
- Cross-model ranking (which mechanism memorizes most/least overall) — TODO, revisit the
  main comparison plot.

**Todo:**
- Off-diagonal of the prefix x suffix grid: run both (50, 500) and (500, 50), not just the
  diagonal, to separate the anchor-length effect from the continuation-length effect.
- Fixed-`suffix_start` anti-diagonal sweep (`suffix_start = offset + prefix` held constant,
  prefix/offset traded off against each other) — disentangles anchor length from cold-start
  depth, which the current offset=0 / fixed-prefix sweeps don't isolate.

## xdoc-attn-leak vs memorization

**Category:** memorization comparison.

**What it does:** Tests whether removing cross-document boundaries increases
memorization, for both architectures — the mechanism differs per architecture but the
question is the same. `full-attn-xdoc-attn-leak` was trained without cross-document
attention masking (packed docs can attend across their boundary), to test whether that
leakage explains higher memorization numbers than PDM/Xu et al. reported for standard
full-attn. For GDN, the equivalent manipulation isn't an attention mask — it's whether
the recurrent state resets at document boundaries: plain `gdn` resets state at every
document (and batch) boundary (no leak, analogous to masked full-attn); `gdn-carry-r0`
does **not** reset state between documents, only between sequences/batches — the same
underlying "no document boundary" property as xdoc-attn-leak, just implemented via state
continuity instead of attention. So `gdn-carry-r0` already *is* the GDN xdoc-leak
counterpart — no separate model needs to be trained.

**Code:** `results/mem-results/SparseGutenberg/llama3-1b-full-attn-xdoc-attn-leak-...`
(full-attn pair) and `llama3-1b-gdn-fineweb40B-gutenberg3B` vs
`llama3-1b-gdn-carry-r0-...` (GDN pair), same `plotting/heatmap.py`/`rouge_l.py` path as
the main comparison, plus a focused Rouge-L-vs-prefix heatmap (offset=0, suffix=500) in
`scf8/xdoc_leak.ipynb` -- a quick presentation-cut view of the same experiment.

**Findings:**
- Full-attn pair: xdoc-leak's apparent strength at short scale (prefix=50, suffix=50,
  small offset) is short-range **local copying**, not verbatim memorization — it
  collapses once a real long continuation is demanded (even at offset=0,
  prefix=500/suffix=500). Supports the leak hypothesis in the other direction from what
  was expected: leaking attention memorizes **less** verbatim than masked full-attn, not
  more.
- GDN pair (`gdn` vs `gdn-carry-r0`) — TODO, not yet analyzed through this lens
  specifically (the state-carry effect on the offset x rep interaction is written up under
  "GDN state-carry policy vs memorization" above, but reading it explicitly as a
  leak/no-leak comparison hasn't been done).

**Todo:**
- Firm up the full-attn finding above with more rigorous verification — currently one
  read of the plots, not yet a proven result.
- Analyze the `gdn` vs `gdn-carry-r0` pair as the leak/no-leak comparison, and check
  whether it agrees with the full-attn direction (leak -> less verbatim memorization).

## Goldfish loss vs memorization

**Category:** memorization comparison.

**What it does:** Does goldfish loss (randomly dropping some tokens from the loss to
suppress verbatim memorization) reduce memorization relative to standard loss, for both
full-attn and GDN?

**Code:** `results/mem-results/SparseGutenberg/llama3-1b-full-attn-goldfish-...` and
`llama3-1b-gdn-goldfish-...`, `plotting/heatmap.py`/`rouge_l.py`.

**Findings:** TODO — not yet written down, revisit the plots for these two models
specifically.

**Todo:** write up the goldfish-vs-non-goldfish comparison once revisited.

## GDN state-carry policy vs memorization

**Category:** memorization comparison.

**What it does:** Does how the GDN recurrent state is seeded across batches (always reset
to zero vs carried from the previous batch, at carry probability r=0/0.5/1) change
memorization behavior, especially the offset x repetition interaction?

**Code:** `results/mem-results/SparseGutenberg/llama3-1b-gdn-*`, `plotting/heatmap.py`/`rouge_l.py`.

**Findings (strong result):** GDN memorization is non-monotonic in repetition count — mid-rep
buckets memorize *more* than high-rep buckets — but only at offset > 0 (monotonic at
offset=0; absent for attention models; present only for the GDN family). Mechanism:
repetition sharpens GDN's recurrent state into an attractor rigidly keyed to the zero-state
start at position 0. High rep -> sharper, more position-brittle -> can't re-lock from a wrong
cold start when offset > 0. Mid rep -> broader/softer basin -> the prefix tokens can still pull
the state back onto the trajectory despite the offset. Attention doesn't have this failure
mode (it attends to prefix tokens directly, no state-init to overfit).
Causally confirmed via the carry variants: base GDN resets state at every doc/batch boundary
-> always sees the excerpt from a point-mass zero-init -> largest dip. Carry variants
(r=0/0.5/1) see the excerpt from varied non-zero initial states -> memory generalizes over
init state -> smaller dip. Chunk-phase (the chunked kernel's internal chunk boundaries) ruled
out as a cause — the effect is purely about missing run-up content / state initialization.

**Todo:**
- Warmup experiment (designed, not run): prepend unscored context before the scored prefix
  to un-cold-start the recurrent state, and check whether the bump survives. Related warmup
  (the excerpt's own true run-up) should erase the bump; unrelated warmup (self-generated or
  unseen FineWeb-Edu text) should preserve it. This is the decisive test of the state-init
  mechanism vs a training-weight explanation.

## FineWeb80B vs memorization

**Category:** memorization comparison.

**What it does:** Does training on 80B FineWeb tokens instead of 40B (same Gutenberg rep
schedule) change memorization at fixed rep counts?

**Code:** `results/mem-results/SparseGutenberg/llama3-1b-full-attn-fineweb80B-...`,
`plotting/heatmap.py`/`rouge_l.py`, plus the same folded-in presentation heatmap as xdoc-leak above.

**Findings:** TODO — not yet written down.

**Todo:** write up the 40B-vs-80B comparison.

## Sink value ablation

**Category:** memorization comparison.

**What it does:** For the sink-family models (learnable sink, off-by-one), the model was
trained with one sink configuration but the number/strength of sink tokens is swapped at
**inference time** (the model never trained with these other values) — does that change
memorization or generation quality?

**Code:** `results/mem-results/SparseGutenberg-legacy/llama3-1b-{off-by-one,sink}-...`
(older data version — this ablation was only run against the original 4-model set, never
re-run against the current `SparseGutenberg` tree), `plotting/sink.py`, `plotting/heatmap.py`.

**Plots:** Rouge-L, PPL/TTR vs number of virtual sink tokens (off-by-one) or `sscale`
(learnable sink), generation examples side by side, generation-quality (Qwen2.5-1.5B-judged
perplexity + distinct-n) from `compute_generation_quality.py` output.

**Findings:** TODO — not yet written down, revisit the plots.

**Todo:** decide whether this ablation should be rerun against the current (`SparseGutenberg`)
model set — currently only compares against the 4 original models.

## Attention pattern analysis

**Category:** mechanistic.

**What it does:** How does attention flow differ between memorized and non-memorized text,
across the 4 base attention variants (full, gated, off-by-one, sink)?

**Code:** `results/mem-results/SparseGutenberg-legacy/` (older data version, 4 models only),
`plotting/attention_patterns.py`.

**Plots:** mean attention to the first token (sink strength) per layer, one line per model;
query x key attention heatmaps for a chosen layer/head, high- vs low-memorization Rouge-L
buckets side by side; gating-score distribution for the gated model only (memorized vs not).

**Findings:** TODO — not yet written down. The notebook runs end to end with real captured
data (6,600 samples/model across all 4 models); the gap is that nobody has interpreted the
plots yet, not missing data.

**Todo:** revisit the plots and write up findings; decide whether to extend this to the
newer model set (GDN, goldfish, xdoc-leak, 80B) — currently only the 4 original models
have attention-map captures.

## Long-context extrapolation (aggregate)

**Category:** behavioral.

**What it does:** Position-wise loss on long-context Gutenberg/FineWeb-Edu sequences —
does loss blow up past the training sequence length (8192), and does state-carrying GDN
degrade more gracefully than attention or state-resetting GDN?

**Code:** `results/long-gutenberg-results/`, `results/long-fineweb-results/`,
`results/data/gutenberg-long/lengths.jsonl`; `plotting/long_inference.py` (loaders moved
to `plotting/data_loading.py`, shared constants/helpers to `plotting/utils.py`).

**Plots:** data-coverage curve (fraction of sequences still present at each position);
per-repetition-bucket position-wise NLL for every model, with a sequence-count panel
underneath; FineWeb-Edu seen-vs-unseen partitions; sink/off-by-one TP=1-vs-TP=4 and
old-workaround-vs-redone-run correctness checks (moves to `sink_variants` on the notebook
split, not `long_inference` — sink/off-by-one only, not a cross-model comparison).

**Findings:**
- Sink/off-by-one correctness checks both pass: TP=1 vs TP=4 agree over the shared 0-12k
  region, and the redone full-length run (TP=1, fused attention, one forward pass) agrees
  with the old TP=4/generation workaround to <0.1% mean NLL over the shared 0-20k region —
  the eval-scripts perf fix doesn't change the math.
- Whether attention explodes past 8192 while state-carrying GDN stays flat — TODO, revisit
  the main position-wise-loss plot.

**Todo:**
- SP vs no-SP inference equivalence check (baselines were trained with TP=2+SP, evaluated
  at TP=1/no SP) — not yet done.

## GDN recurrent state norm

**Category:** mechanistic.

**What it does:** Does the GDN recurrent state stay bounded past the training sequence
length, or does it keep growing? If `||S||` grows past `seq_len`, that plausibly explains
a loss blowup in the aggregate long-context extrapolation entry above: the state goes
out-of-distribution. Only the GDN variants carry a state, so attention baselines drop out
here.

**Code:** same source as long-context extrapolation above
(`results/long-gutenberg-results/` `rep_{R}_state.npz` files); loaders
(`load_long_inference_state_norm`, `load_long_inference_state_norm_by_layer`) in
`plotting/data_loading.py`, plots (`plot_state_norm_panel`, `plot_state_norm_by_layer`) in
`plotting/gdn_state.py`.

**Plots:** state norm (Frobenius, averaged over layers/heads) vs token position, one cell
per repetition bucket, all GDN variants overlaid; per-layer breakdown for a single
repetition bucket (16 head-averaged layer lines, one subplot per GDN model).

**Findings:** TODO — not yet written down, revisit the state-norm plots (does the norm
plateau, keep growing, or differ between reset vs carry variants past `seq_len`?).

**Todo:** none beyond writing up the findings above.

## Long-context individual-sequence loss peak analysis

**Category:** behavioral.

**What it does:** Per-sequence NLL/true-token-rank inspection, to check whether an anomaly
seen in the aggregate long-context view is real or an averaging artifact.

**Code:** `results/long-gutenberg-results/` `*_individual.jsonl` files (from
`--store-individual`), `plotting/long_individual_sequence.py`.

**Plots:** per-sequence NLL + true-token-rank (+ token-text strip when the window is short
enough), ranked by squared NLL difference or single-position max difference between two
models/configs.

**Findings:** The ~3k-position loss bump seen in the aggregate plot is **real, not an
averaging artifact**: at rep=0, 660/660 sequences show a step of size >0.3 nats (position
min=1501/median=3460/max=4999); at rep=256, also 660/660, step position
min=3287/median=3570/max=4999. The bump gets later and sharper at high repetition.

**Todo:** none open right now.

## Training data length effect (long-whole vs split-1024 Gutenberg)

**Category:** memorization comparison + behavioral (two angles, see below).

**What it does:** Broader question than any single tool below — does training on long
Gutenberg documents kept whole vs the same documents chopped into 1024-token segments
(`split_long_dataset.slurm`) change model behavior? Two analysis angles so far, at very
different levels of completeness:
- **Long-context loss** (done): `llama3-1b-full-attn-fineweb40B-long-gutenberg3B` vs
  `...-long-split-1024-gutenberg3B`, evaluated on the same long-Gutenberg eval task, so any
  loss difference traces the training-document structure, not the eval.
- **Memorization** (barely started): each model has exactly one data point in
  `mem-results/SparseGutenberg` (`offset_0_prefix_500_suffix_500`), not a full sweep — no
  analysis has been done on this angle yet.

**Code:** both angles live in `scf8/data_length.ipynb`. Long-context angle:
`results/long-gutenberg-results/`, `plotting/long_inference.py` +
`plotting/long_individual_sequence.py` (aggregate position-wise loss, then ranked
per-sequence divergence). Memorization angle: `results/mem-results/SparseGutenberg/
llama3-1b-full-attn-fineweb40B-long*`, `plotting/heatmap.py` (not yet run beyond the
single existing data point).

**Plots (long-context angle):** aggregate position-wise loss (both models overlaid);
per-sequence NLL ranked by mean-squared difference and by single largest pointwise jump
between the two models; a few random + a few chosen high-divergence sequences plotted in
detail.

**Findings:** TODO — not yet written down, revisit the ranked-divergence plots for the
long-context angle. Memorization angle has no findings yet (no sweep run).

**Todo:**
- Write up what the long-context loss comparison actually shows.
- Run the full (offset, prefix, suffix) sweep for both models in `mem-results` and analyze
  the memorization angle in `scf8/data_length.ipynb`'s own memorization section.
- No FineWeb-Edu long-inference data yet for `full-long-docs`/`full-short-docs` (`full-long`/
  `full-long-split-1024` in `llama_checkpoints.sh`) -- checkpoints were mid-training, now
  resumed. Once training finishes: `bash submissions/long_fineweb_inference_all.sh` (already
  covers both, skips everything else already done), then
  `bash scripts/pull_long_inference_results.sh --models full-long full-long-split-1024`.

## SWA (sliding-window attention) vs memorization

**Category:** memorization comparison.

**What it does:** How does memorization change with SWA window size (w256/w1024/w4096 vs
unbounded `full`, all scf1-trained) across offset, prefix, and a `suffix_start`-fixed
anti-diagonal that disentangles anchor length (prefix) from discarded run-up depth
(offset)? Directly tests the "prefix >= window" hypothesis -- does memorization collapse
once the verbatim anchor no longer fits inside the attention window? Also a full
offset x prefix grid at suffix=250, discovered off disk rather than hand-listed. Most
points beyond the offset=0/prefix=500 baseline only exist via the HF inference backend
(not yet captured through Megatron), with automatic fallback.

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`
(`plot_models_heatmap_panel`, all four `panels=` modes including `'diagonal'`),
`plotting/data_loading.py` (`discover_offset_prefix_points` for the disk-driven grid).

**Findings:** The anti-diagonal sweep (fixed `suffix_start=1000`) found collapse
boundaries between prefix=750 (collapsed) and prefix=1000 (works) for full/w4096/w1024,
and between prefix=250 (collapsed) and prefix=500 (works) for w256 -- consistent with the
window size itself as the threshold (w256's own window is 256, sitting inside the 250-500
collapse band). This is a first read that motivated densifying the sweep near those
boundaries (including prefix=256 exactly), not yet a written-up, confirmed conclusion.

**Todo:**
- Write up whether "prefix >= window" cleanly predicts the collapse boundary now that the
  densified points are in.
- Extend Megatron-backend capture to close the HF-only gap (currently only the
  offset=0/prefix=500 baseline has Megatron data; everything else relies on HF, unverified
  against Megatron the way `backend_comparison.ipynb` does for `full-scf1`).

## Suffix-boundary sensitivity

**Category:** memorization comparison.

**What it does:** For one fixed (offset=0, prefix=500) generation run on `full-scf1`,
sliced at every reachable suffix boundary (25 to 1000 tokens) -- how sensitive is
memorization to how much of the continuation is actually scored? Two metrics: Rouge-L, and
the Hayes et al. 2025 (n, p)-discoverable-extraction rate (fraction of samples with
`1-(1-p_z)^n >= p`, computed on the fly from stored `p_z` scores rather than a metric
already in `Results`).

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`
(`plot_heatmap_per_model`, including `metric='hayes'`).

**Findings:** TODO — not yet written down, revisit the Rouge-L and Hayes-extraction
heatmaps.

**Todo:** write up the suffix-boundary sensitivity curve; consider extending past the one
model currently covered.

## scf1-trained model family comparison

**Category:** memorization comparison.

**What it does:** Single reference-point (offset=0, prefix=500, suffix=500) Rouge-L
comparison across every model actually retrained (not just re-inferred) at scf1 --
full/gated/sink and the three SWA window sizes -- plus the GDN baseline (RoPE scaling
doesn't affect GDN, so no scf1 variant exists or is needed, see `scf8_variant()` in
`model_registry.py`). This is the main current-generation cross-family view. Also an
offset-swept version of the same comparison (fixed prefix/suffix=500) for a narrower
6-model subset (drops one SWA window size).

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`
(`plot_models_heatmap`, `plot_models_heatmap_panel`).

**Findings:** TODO — not yet written down.

**Todo:** write up the cross-family ranking, both at the single reference point and across
offsets.

## Extended metrics across models

**Category:** memorization comparison.

**What it does:** Companion to "Attention variant vs memorization" above (Rouge-L itself
lives per-family, e.g. in `attn_types.ipynb`) -- for full-scf1/sink-scf1/gated-scf1/
swa-w1024-scf1/gdn, how do the *other* metrics compare: LCS, match-rate (25/50/75%
thresholds + exact match), divergence point, token-level accuracy, paired PPL/NLL/TTR, and
the Rouge-L distribution itself (per-snippet histograms + repetition x Rouge-L-bin
heatmaps)?

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`,
`plotting/utils.py` (`plot_lineplot_panel`), `plotting/rouge_l.py` (`plot_rouge_hist`,
`plot_rouge_heatmap`).

**Findings:** TODO — not yet written down, revisit each metric section's plots.

**Todo:** write up findings per metric. Note: prefix/suffix coverage is uneven across
these 5 models (some sweep points only exist for `gdn`), so most panels are restricted to
`fixed_prefix=500` (fully covered by all 5) rather than the full `PREFIXES` list; the
prefix-dim panels use a `shared_prefixes` filter for the same reason.

## Document / packed-chunk length distributions

**Category:** n/a (data characterization).

**What it does:** Data characterization, not a model comparison — document length and
training-time packed-chunk-length distributions for the pretraining data blends.

**Code:** `results/datasets/analysis/doc-lengths/`, `results/datasets/packed_chunk_lengths/`
(produced by `data_processing/dataset_doc_stats.py` / `packed_chunk_stats.py`). All plotting
is inline in the notebook (no shared `plotting/` module — flagged in the reorg plan to move
out).

**Findings:** (these numbers are the finding — pure data characterization)
- FineWeb-Edu (160B full): mean doc length 992.5 tokens, p50=607, p99=7,358, min/max 29/266,969.
- Long FineWeb-Edu subset (40B): mean 7,566.9, p50=5,560, p99=33,639 — genuinely long docs, as
  intended for the long-context extrapolation eval.
- Gutenberg: every doc is exactly 8,192 tokens (std=0) by construction.
- Token budget: docs >=3,650 tokens are 25% of total FineWeb-Edu tokens (40.03B of 160.2B)
  despite being a small fraction of documents.
- Packed chunk lengths (one training epoch): FineWeb-Edu chunks mean 885.3/p50=567 (close to
  raw doc lengths — packing barely changes short docs); long-FineWeb chunks mean 3,933.8,
  p90/p95/p99 all saturate at 8,192 (the seq_len cap) — most long docs get cut into
  full-length chunks; ~0.02% of FineWeb chunks are length-1 slivers from docs straddling a
  sample boundary.

**Todo:** none open right now.

## RoPE-scaling correctness check

**Category:** n/a (correctness check).

**What it does:** Sanity-checks `rename_scf8_results.sh`'s RoPE-scaling fix (RoPE scaling
was silently disabled at inference despite being enabled during training, because
`--use-checkpoint-args` reverted it to its `False` default) by comparing Rouge-L across
three groups at the same fixed point (offset=0, prefix=500, suffix=500): the old
mismatched runs (trained scf=8, inferred scf=1), the fixed reruns (everything scf=8), and
the current generation (everything scf=1, for the 3 families retrained so far --
full/gated/sink). Behavioral angle: long FineWeb-Edu (seen partition) position-wise loss,
old vs scf8-fixed, for the 4 families that have both generations captured.

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`
(`plot_models_heatmap_panel`, `panels='models'`); `results/long-fineweb-results/`,
`plotting/long_inference.py` (`plot_loss_panel`).

**Findings:** TODO — not yet written down, revisit the `panels='models'` Rouge-L
comparison and the FineWeb loss overlay.

**Todo:** write up whether the fix changed anything measurable, and by how much.

## Backend agreement + scf1 training-config check

**Category:** n/a (infra correctness check).

**What it does:** Two related sanity checks in one notebook. (1) Does the Megatron
inference backend agree with the HF backend for the same checkpoint (`full-scf1`, run
through both) -- needed before trusting the HF-only sweeps elsewhere (e.g.
`swa_variants.ipynb`'s prefix/diagonal panels, which exist almost entirely via HF). (2) How
does `full-scf1` -- a genuinely separate training run from `full-scf8` (different RoPE
scaling factor, batch size, TP degree, weight decay, warmup, and total steps, not an
isolated RoPE ablation) -- compare to `full-scf8`/`full-scf8-scf1` on Rouge-L across
offsets.

**Code:** `results/mem-results/SparseGutenberg/`, `plotting/heatmap.py`
(`plot_models_heatmap_panel`), `model_registry.model_folder(key, 'hf')` for the
backend-tagged label + `load_mem_results_scores_grid_by_name`.

**Findings:** TODO — not yet written down.

**Todo:** confirm backend agreement explicitly (currently just eyeballed on the same
plot, no numeric diff check the way `compare_nll` does for long-inference); write up the
`full-scf1`-vs-`full-scf8` gap and how much of it is RoPE scaling vs the other config diffs.

## Generated examples viewer

**Category:** n/a (qualitative/secondary tooling).

**What it does:** Qualitative side-by-side prefix / reference-suffix / generated-suffix
tables -- not a new experiment, a presentation layer for comparisons already covered
elsewhere. Three sections: main cross-family comparison (full/sink/gated/swa-w1024/gdn, one
table per repetition), GDN generations by offset (one shared prefix/suffix span per
repetition with colored inline markers at each offset's boundary, since the windows mostly
overlap), sink sscale-ablation generations (off-by-one and learnable sink, each sscale
value its own column).

**Code:** `results/mem-results/SparseGutenberg/` (main + GDN sections),
`results/mem-results/SparseGutenberg-legacy/` (sink section -- same tree "Sink value
ablation" above reads, since the sink sscale sweep was never rerun against the current
tree), `plotting/generated_examples.py`.

**Findings:** n/a — qualitative viewer, not a results-bearing comparison on its own.

**Todo:** local data gap, not a code bug -- the main-comparison section excludes every
model at every repetition ("no data ... fewer than 5 samples available"): this machine's
local `SparseGutenberg` mirror only has the small `metrics/` pickles synced for most
models, not the much larger raw per-sample `inference/` text records this notebook needs
-- only `gdn` has those synced locally. The sink section shows every experiment as
"Missing (not yet run)" -- same symptom, but unconfirmed whether it's the same local-sync
gap or these `-scf1`-suffixed sscale folders genuinely don't exist on the cluster at all
(plausible, since off-by-one sink is separately noted as never retrained under later
generations). Both need checking against a fuller local sync (or running directly on the
cluster) before this notebook actually shows anything.

## Chinchilla budget planning (not an experiment)

**Category:** n/a (planning).

**What it does:** Pure planning/math tool, no `results/` dependency. For a candidate total
token budget (Gutenberg + FineWeb-Edu filler) and model size, computes the Chinchilla ratio
(multiple of 20 tokens/param); also the reverse (budget needed to hit a target ratio), and a
search for "coincidences" where different (budget, model) pairs land on the same ratio
(because the three model sizes' pairwise ratios are fixed constants).

**Code:** self-contained, no shared data/plotting dependencies.

**Findings:** n/a — planning tool, not a results-bearing experiment.

**Todo:** none.
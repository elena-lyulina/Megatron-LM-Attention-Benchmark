# Memorization Observations

Running log of empirical patterns noticed while exploring the memorization dashboard
(`attn_bench/dashboard/`, published at
https://elena-lyulina.github.io/Megatron-LM-Attention-Benchmark/). Source data:
`attn_bench/dashboard/data/*.json`, one file per model (`full-scf1`, `gated-scf1`,
`sink-scf1`, `swa-w4096-scf1`, `swa-w1024-scf1`, `swa-w256-scf1`, `gdn`, `gdn-xdl`,
`gdn-xdl-xsl-0.5`, `gdn-xdl-xsl`), each holding the offset x prefix grid for reps
0/1/16/32/64/128/256. See `memorization_measurement.md` for how the numbers are produced.

Every section below is split into **Observation** (what the data shows, not up for
debate) and **Possible explanations** (our current best guesses at why, explicitly
labeled as hypotheses — not verified unless stated). Keep them separate: an
explanation being wrong doesn't make the observation wrong.

## Measurement setup — what "offset" actually means

Needed context before any explanation below makes sense. From
`attn_bench/evaluation/prefix_extraction_inference.py:65-72,385` and
`load_rep_bucket`:

- `excerpt = document_ids[offset : offset + prefix + suffix]`. Tokens `0..offset` are
  **not part of the model's input at all** — not a visible gap the model reads and
  discards, just genuinely absent from the forward pass.
- `needs_bos = offset > 0`: for `offset > 0` a synthetic BOS token is prepended in
  front of the excerpt (the excerpt doesn't naturally start with one). For
  `offset == 0` no prepending is needed — the excerpt already starts at the
  document's real BOS.
- `position_ids = arange(0, S-1)` always — position IDs are **reset to zero** for
  every excerpt regardless of how deep `offset` actually is in the document. The
  model has no positional signal telling it "I'm 5000 tokens into a document";
  every excerpt looks, positionally, like a document opening.

So `offset` is not "extra visible distractor context between the anchor and the
generation point" — it's **how much of the document's true preceding content is
discarded, replaced by a synthetic, position-reset BOS**. During training the model
only ever sees a document continuously from its real position 0; `offset > 0` at
eval time is a construction the model never encounters in training. This reframes
every "offset hurts memorization" observation below as fundamentally a
train/eval-distribution-mismatch question, not an attention-distance/copy-dilution
one. (An earlier round of this investigation described offset as the latter — that
framing is wrong and superseded by this section.)

---

## 1. Full attention, gated, sink, and swa-w4096 are dominated by offset, not prefix

**Observation.** For these four models, increasing `offset` collapses ROUGE-L
almost regardless of `prefix`, and increasing `prefix` at fixed large offset barely
moves it. E.g. full-scf1, rep=256: off=50→250 collapses ROUGE-L from 0.59→0.29 even
at prefix=5942; at off=2000, ROUGE-L stays at 0.17-0.18 across the entire prefix
range 50→7892. Same shape for gated-scf1, sink-scf1, swa-w4096-scf1.

**Possible explanations (hypotheses).**
- Given the corrected offset semantics above: larger offset means more of the
  document's true preceding context is thrown away and replaced with a synthetic,
  position-reset BOS. During training these models always process a document
  continuously from its real start; a large-offset excerpt is a bigger departure
  from anything seen in training, regardless of how long the remaining `prefix` is.
  Prefix length alone doesn't recreate the missing history.
- Not verified: why prefix fails to compensate even when it's very long (e.g.
  prefix=5942 real tokens still only gets to 0.18 ROUGE-L at offset=2000). One
  guess is that recall is keyed more to "how close is this excerpt to a genuine,
  in-distribution training-time context" than to raw prefix length — but this
  hasn't been tested against, e.g., an ablation that removes the position-reset or
  BOS-prepend to isolate which piece of the mismatch matters most.

## 2. swa-w1024 and swa-w256 show a sharp prefix threshold, past which offset barely matters

**Observation.** For swa1024/swa256, holding offset fixed at any large value
(≥2000, offset dominates elsewhere) and sweeping prefix: ROUGE-L stays near
baseline (~0.16-0.29) below a threshold, then jumps sharply past it. swa1024
(window=1024): prefix=1000 → 0.26-0.29, prefix=2000 → 0.63-0.90 (crossing 0.5
around prefix≈1400-1700). swa256 (window=256): prefix=250 → 0.16-0.20, prefix=500 →
0.71-0.83 (crossing 0.5 around prefix≈350-450). Past the threshold, offset stops
mattering almost entirely — swa1024, prefix=2000, rep=256: ROUGE-L is ~0.99 across
offset=50 through offset=7892.

**Possible explanations (hypotheses).**
- Given the corrected offset semantics: a sliding-window model, even during normal
  training on a real, continuous document, never attends further back than its
  window size — it never gets the "full history back to position 0" that full
  attention enjoys. So a synthetic, position-reset, BOS-prepended excerpt at eval
  time is a *smaller* departure from its normal training experience than it is for
  full attention, provided the given (uncut) prefix is at least as long as its
  training window — at that point the eval scenario looks statistically like an
  ordinary local window it seen constantly during training, independent of what
  document depth it actually came from. This would explain the threshold shape and
  offset-independence past it.
- This does not explain observation 3 below (swa-w4096 does *not* show the same
  offset-independence-past-threshold pattern, despite window=4096 being smaller
  than several tested prefixes) — flagged as an open contradiction, not resolved.
- Also unexplained: why the crossover sits at ≈1.4-1.8× the window size rather than
  exactly at the window size.

## 3. swa-w4096 does not show the sharp threshold that swa-w1024/swa-w256 show

**Observation.** swa-w4096, rep=256, off=2000 (< window=4096): ROUGE-L only reaches
0.17→0.21 as prefix goes 50→5942 — nothing like swa1024's 0.16→0.77 or swa256's
0.16→1.00 at the identical (off=2000, prefix) cells, even though swa4096's window
is larger than both. off=50 (near-zero offset) does reach 0.64 at prefix=7892, but
that's driven by offset≈0 (see observation 6), not by prefix crossing the window.

**Possible explanations (hypotheses).** None that fit cleanly yet. Naively,
observation 2's window-familiarity story predicts swa4096 should show the *same*
threshold effect, just at a larger prefix — it doesn't show it at all in the tested
range. Open question, not resolved by anything in this doc.

## 4. exact_match hits a hard, near-universal zero at the document's literal last tokens, despite near-total memorization by every other metric

**Observation.** At the grid corner where `offset + prefix = max_doc_length -
suffix` (7942 here — the suffix is literally the document's last 250 tokens),
`exact_match = 0.0` across every attention-family model (full, gated, sink,
swa4096, swa1024, swa256) at every repetition count, at every one of the four
corner cells tested. At the same cells, `rouge_l` and `divergence_point` are near
1.0 at high rep — e.g. swa1024 rep=256, off=50/prefix=7892: rouge_l=0.9947,
divergence_point=0.9944, exact_match=0.0. `divergence_point` is a per-document
metric (`first_mismatch_index/suffix_length`, or exactly 1.0 only for a perfect
match); given `exact_match=0` (0/660 documents perfect) and a mean divergence_point
at ~99.4% of the theoretical ceiling for an imperfect document (249/250=0.996 for a
250-token suffix), the arithmetic bounds at most ~1-2 of the 660 documents as
diverging early — the rest are individually matching 248-249 of 250 tokens and
failing only on the last one or two. GDN does not show this hard zero at the same
corner cell (e.g. gdn rep=64, off=50/prefix=7892: exact_match=0.79, not 0).

**Possible explanations (hypotheses).**
- The token immediately after a document's true end (8192 tokens, exactly the
  measurement's `max_doc_length`) isn't real content of *this* document — it's
  whatever follows in the corpus/packing, effectively arbitrary relative to this
  document's own text. So it's the one position an otherwise near-perfectly
  memorized document can't be completed on, and a single wrong token anywhere
  zeroes an all-or-nothing metric like exact_match.
- Why GDN doesn't hit the same hard zero at that corner is not explained — flagged
  as open.

## 5. Teacher-forced probabilistic extraction (hayes_n10_p*) tells a different story than greedy exact_match at the same corner

**Observation.** At the document-boundary corner from observation 4, swa1024/swa256
show `hayes_n10_p50 ≈ 0.988-0.989` (98.8-98.9% of the 660 documents would be
extractable within 10 sampled queries at ≥50% confidence) and `hayes_n10_p75 ≈
0.91-0.93`, despite `exact_match = 0.0`. `hayes_n10_p99` (near-certainty) drops to
~0.08-0.09. Full attention at the *same* corner shows the opposite: hayes drops to
exactly 0.0 at all four thresholds (p25 through p99) once offset≥2000, while
exact_match is also 0 there but for a genuinely different reason (real,
offset-driven decay, not a last-token quirk — see observation 1). Away from the
boundary corner, full attention's hayes_n10_p50 generally runs *below*
exact_match (e.g. off=50/prefix=1000, rep=256: exact_match=0.36 vs
hayes_p50=0.22).

**Possible explanations (hypotheses).**
- `divergence_point`/`p_z` (teacher-forced probability of the true suffix) compounds
  multiplicatively over all 250 suffix tokens, while greedy `exact_match` only
  needs the true token to be the argmax at each step (which can hold even at
  modest, non-dominant probability). This can make hayes stricter than exact_match
  away from a hard failure point, but far more lenient right at the doc-boundary
  corner, where the *entire* 249/250-token near-perfect match compounds a p_z that
  clears the p50 bar even though the one uncertain final token never wins the
  greedy argmax against a competing candidate. Back-solving the p50/p99 thresholds
  suggests the true final token typically carries roughly 7-37% probability under
  the model — high enough to often win under repeated sampling, not high enough to
  win every greedy decode.
- Full attention's hayes going to exactly 0.0 away from the corner (not just low)
  is offered as evidence that observation 1's decay is a genuine collapse in
  probability mass, not a decoding artifact — but this hasn't been checked at
  intermediate offsets, only at off≥2000.

## 6. off=0 is a uniformly near-ceiling case, across every model and every repetition count

**Observation.** At `offset=0` (no synthetic BOS, no discarded context — the
excerpt is a genuine document opening), ROUGE-L is markedly higher than any other
offset even at small prefix and low rep. E.g. full-scf1, rep=32, off=0/prefix=50:
0.41 vs off=50/prefix=50: 0.16 (same prefix length, only offset differs). At
rep=256, off=0 reaches ~1.00 ROUGE-L across the entire prefix range for every
model tested.

**Possible explanations (hypotheses).** Consistent with the corrected offset
semantics: off=0 requires no synthetic BOS and no discarded context — it's exactly
the scenario the model was trained on (document start, real BOS, continuous
position IDs from 0), so this isn't really testing memorization robustness at all,
more the model's baseline in-distribution completion ability at the true start of
a repeatedly-seen document.

## 7. TTR_gen sits at a fixed, low floor whenever nothing real is retrieved, independent of model or repetition count

**Observation.** In cells where ROUGE-L is at baseline (offset≥2000, prefix≤250),
`ttr_gen` sits at ~0.27-0.35 across full attention, GDN, and swa256, across rep=0
through rep=256 — barely moves with repetition count. `ttr_ref` (natural text) is
~0.61 in the same cells. This floor is essentially flat regardless of model
architecture or how many times the document was trained on.

**Possible explanations (hypotheses).** Greedy decoding under high uncertainty
(nothing memorized, no real content to retrieve) degenerates toward repetitive,
low-lexical-diversity text — a generic property of greedy decoding rather than
anything specific to this benchmark or these architectures. Useful mainly as a
sanity check: it confirms the ~0.16-0.18 ROUGE-L floor seen throughout is a real,
stable no-signal baseline and not measurement noise.

## 8. All four GDN leak variants look alike at low-to-mid repetition, then split sharply at rep=256

**Observation.** At rep=32-64, `gdn`, `gdn-xdl`, `gdn-xdl-xsl`, and
`gdn-xdl-xsl-0.5` all show a similar pattern — a broad diagonal where both offset
and prefix matter — distinct from full attention's offset-dominated decay
(observation 1). At rep=256, off=1000/prefix=5942: `gdn`=0.25, `gdn-xdl-xsl-0.5`
=0.33 (both low, closer to full attention's decay), `gdn-xdl`=0.94,
`gdn-xdl-xsl`=0.95 (both high, far-reaching memorization). The split is not
monotonic in "how much state leaks" — 0% and 100% cross-batch carry both give high
memorization; 50% gives low memorization, same as 0% infrastructure-disabled
carry.

**Possible explanations (hypotheses).** See observation 10 for the corrected
mechanism (a repetition-dependent effect, not a static capability difference). No
settled explanation yet for why exactly-50% carry produces low memorization rather
than something intermediate between the 0% and 100% cases — flagged as open.

## 9. The GDN leak split (observation 8) shows up in teacher-forced probability too, not just greedy decoding

**Observation.** At the same off=1000/prefix=5942/rep=256 cell,
`hayes_n10_p50`: `gdn`=0.005, `gdn-xdl-xsl-0.5`=0.036 (both low), `gdn-xdl`=0.80,
`gdn-xdl-xsl`=0.78 (both high) — tracking rouge_l/exact_match closely.

**Possible explanations (hypotheses).** Since this shows up in teacher-forced
probability (not dependent on what greedy decoding happens to output), the split
is a property of the trained weights/representations, not a sampling or decoding
artifact.

## 10. gdn and gdn-xdl are identical at rep 0-1, then diverge with repetition — gdn peaks around rep=32 and then declines

**Observation.** At off=1000/prefix=5942 (and confirmed at off=2000/prefix=5942):
rep=0/1, `gdn` and `gdn-xdl` are indistinguishable (~0.17 both). From rep=16
onward they diverge: `gdn` rises to a peak around rep=32 (0.59-0.69) then
*declines* monotonically through rep=64/128/256 (down to 0.25). `gdn-xdl` dips
slightly at rep=64 then climbs monotonically past `gdn` and keeps rising to
rep=256 (0.83-0.94).

**Verified fact relevant to any explanation:** `gdn` and `gdn-xdl` have identical
`--gdn-state-carry-ratio 0.0` — they do not differ in cross-batch state
persistence at all (checked directly against `attn_bench/submissions/*.slurm` and
`megatron/core/ssm/gated_delta_net.py`). Their only training-time difference is
whether `--use-packed-seq-params` is set (`gdn`: yes; `gdn-xdl`: no) — which
controls whether the recurrent state resets at document (EOD) boundaries inside a
packed training sequence. Also verified: Gutenberg probe documents are always
exactly 8192 tokens, filling an entire training sequence on their own — never
packed with another document. So this flag only has a literal effect on how the
model experiences the FineWeb-Edu filler portion of training (which likely *is*
packed multiple-documents-per-sequence), not on how it processes a Gutenberg probe
document itself.

**Possible explanations (hypotheses).**
- The rep-0/1 identity rules out a static "gdn-xdl is a generically better
  long-range SSM" explanation — if that were true it should already show an
  advantage on a single presentation of a long document, and it doesn't.
- Best current guess: the only real training difference (per-document state resets
  during the FineWeb-Edu filler majority of training, present for `gdn` and absent
  for `gdn-xdl`) doesn't create a fixed capability gap, but changes how each
  model's weight-level memorization *responds to repeated exposure* — e.g. `gdn`'s
  per-doc-reset training regime may make its verbatim-recall pathway more
  susceptible to being overwritten/interfered with as training continues past the
  point of peak memorization (a forgetting-like effect), while `gdn-xdl`'s
  never-reset-within-a-packed-sequence regime may build a more
  interference-resistant encoding. Not verified — would need checking whether the
  same rep=32 peak / rep=256 decline shows up for `gdn` on other repeated probe
  documents, not just this cell.

## 11. Reference perplexity: all four GDN variants decay like SWA, not like full attention — the leak setting only decides whether the saturation is complete or partial

**Observation.** `ppl_ref` (perplexity of the true suffix under the model — lower
means more confident/familiar) at rep=256, sweeping prefix at fixed large offset.
Plateau value at prefix=5942, off=1000 (baseline/no-signal ppl_ref ≈ 1.0):

| model | plateau ppl_ref |
|---|---|
| full-scf1 | 4.20 |
| gdn | 1.27 |
| gdn-xdl-xsl-0.5 | 1.29 |
| swa-w1024-scf1 | 1.01 |
| gdn-xdl | 1.01 |
| gdn-xdl-xsl | 1.01 |

All four GDN variants — including `gdn` (no leak) and `gdn-xdl-xsl-0.5` (50%
carry), the two that looked "full-attention-like" in rouge_l (observation 8) —
plateau within 20-30% of baseline (1.20-1.30), far closer to swa1024's 1.01 than to
full attention's 4.20. This holds across offsets 500/1000/2000, not just one cell:
e.g. at off=2000, prefix=5942, gdn=1.30 and gdn-xdl-xsl-0.5=1.44 vs full=5.10.
`gdn-xdl`/`gdn-xdl-xsl` go further and reach *full* saturation (~1.01-1.03,
matching SWA almost exactly), while `gdn`/`gdn-xdl-xsl-0.5` reach a *partial*
saturation — much better than full attention, but not all the way to baseline. So
the right characterization is: **all four GDN variants share SWA's saturating
decay shape, not full attention's persistently-elevated one** — the leak setting
only decides whether that saturation closes the gap completely or leaves a
residual ~20-30% penalty. (This corrects an earlier draft of this section, which
mischaracterized `gdn`/`gdn-xdl-xsl-0.5` as resembling full attention because they
don't hit exactly 1.0 — the magnitudes make clear they're much closer to SWA.)
This shape is present already at rep=32 too (unlike the rouge_l split in
observation 8, which only emerges at rep=256) — worth noting since it means the
"GDN looks SWA-shaped" property in perplexity is not purely a high-repetition
effect, even though the specific full-vs-partial-saturation split is.

**Possible explanations (hypotheses).**
- GDN's recurrent state is a fixed-size, compressed summary — structurally closer
  to a bounded window (SWA) than to full attention's unbounded, exact access to
  every prior token. That architectural similarity (bounded working memory) could
  explain why GDN's perplexity decay shape resembles SWA's saturating curve rather
  than full attention's shape, independent of any repetition/leak effects — and
  would explain why this shows up already at rep=32, unlike the rouge_l split.
- Why `gdn-xdl`/`gdn-xdl-xsl` close the remaining ~20-30% gap to true baseline
  while `gdn`/`gdn-xdl-xsl-0.5` don't is the same open question as observations
  8-10 — not resolved here, just confirmed to be visible in a second, independent
  metric.

## 12. rep_0/rep_1 reference perplexity is high (30-55x baseline) even at the document's own start — genuine zero-exposure control, not a labeling bug

**Observation.** At off=0 (document start, no synthetic BOS/context-discard
issues from observation 1 in play), `ppl_ref` at rep=0 and rep=1 stays far above
baseline (ppl≈1.0) across the *entire* prefix range — full-scf1: rep=0 goes
55.9→37.2 (prefix 50→3971), rep=1 goes 43.6→31.1 over the same range; gdn: rep=0
goes 53.4→42.0, rep=1 goes 42.6→34.1. Both reps are dramatically higher than
rep=16 (2.1-5.1) or rep=32 (1.08-1.4) at the same cells. rep=1 is consistently,
meaningfully lower than rep=0 at every prefix value tested, for both models.

**Verified fact:** `rep_0` books are a genuine zero-exposure control, not a
mislabeled "seen once" bucket. Checked directly against the data-curation
pipeline: `attn_bench/data_processing/books/filter_and_build_buckets.py` splits
filtered books into 10 equal groups; 9 are labeled `bucket_rep ∈
{1,2,4,8,16,32,64,128,256}` and go into the training-corpus list, the 10th is
written to a *separate* `unseen_buckets.jsonl` that the corpus writer
(`write_megatron_books.py`) never reads. `write_megatron_books.py:57-60` writes
each training book literally `rep` times total — `rep=N` is the total appearance
count in the corpus, not N extra copies on top of some base inclusion. Held-out
books were additionally pre-filtered for zero n-gram overlap against FineWeb-40B,
so they can't leak in through the filler portion either. So rep=0 truly means
zero total exposures, and the rep=1-vs-rep=0 gap above is real single-exposure
learning, not noise.

**Possible explanations (hypotheses).**
- The remaining question is why ppl_ref is still 30-55x baseline at rep=0/1 even
  at the document's true start with generous prefix, rather than something closer
  to a "normal" fluent-but-uncertain language-model perplexity. Best guess:
  `ppl_ref` measures confidence in the *exact* original wording of a specific
  passage, not general fluency — narrative prose has real per-token entropy among
  many statistically plausible phrasings, and a 1B model reading Gutenberg-style
  prose zero or one times has no way to have converged on this author's specific
  word choices. Not verified against, e.g., what this model's perplexity looks
  like on ordinary non-probe held-out text of similar style, which would be the
  natural baseline to compare against rather than assuming ppl≈1.0 is the
  reasonable expectation.

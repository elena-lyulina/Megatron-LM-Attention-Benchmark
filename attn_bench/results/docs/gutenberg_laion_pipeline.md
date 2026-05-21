# Gutenberg LAION Preprocessing Pipeline

Source: `laion/Project-Gutenberg` (HuggingFace). Goal: produce one 8,192-token excerpt per book, deduplicated and clean, for use as copyright-risk probes in a memorization study.

Pipeline code: `attn_bench/data_processing/books/main_gutenberg_laion.py`

## Threshold tuning

All thresholds were first validated on a 7k-book sample (~10% of the corpus, `raw-hf-7000`). At each step the pipeline writes stats files and HTML diff examples to the `stats/` directory; thresholds were adjusted manually by inspecting those examples and sampled outputs until the kept/dropped split looked correct. Once stable, the parameters were applied unchanged to the full dataset.

## Full-dataset run (2026-05-13)

56,982 books in → 26,814 kept (47%). Two Slurm jobs on CSCS Clariden (`nemo` env), 1 node, 96 CPUs, scripts in `attn_bench/submissions/`:

- `2123565` — `prepare_gutenberg_laion_full.slurm`, steps 1–18, 2026-05-11 18:48→~19:48, logs: `attn_bench/logs/2123565.{out,err}`
- `2125605` — same script resumed from step-18 checkpoint, step 19 + final output, 2026-05-11 21:03→21:30, logs: `attn_bench/logs/2125605.{out,err}`

Step compute times (from tqdm, excludes checkpoint save/load I/O):

| step | time |
|---|---|
| 01 init_columns | <1s |
| 02 extract_text | 29s |
| 03 dedup_id | 24s |
| 04 dedup_title | 31s |
| 05 add_title_embeddings | 34s |
| 06 build_title_clusters | 25s |
| 07 strip_gutenberg | 8s |
| 08 normalize_text | **4m 51s** (ftfy is slow) |
| 09 set_content_bounds | 7s |
| 10 mark_too_short | 1s |
| 11 verify_no_gutenberg | 8s |
| 12 compute_content_chunk_sigs | 2m 11s |
| 13 dedup_content_minhash | **26m 31s** (LSH load 20m04s + dedup 5m57s + mark 30s) |
| 14 sample_window | 4s |
| 15 find_excerpt_start | 17s |
| 16 tokenize_excerpt | 43s |
| 17 verify_tokenization | 33s |
| 18 compute_excerpt_chunk_sigs | 18s |
| 19 dedup_excerpts_minhash | **17m 24s** (LSH load 6m12s + dedup 10m41s + mark 31s) |

The dominant costs are the sequential LSH steps (13 and 19) — O(n) loading and O(n²) candidate search — and normalize_text (ftfy). Everything else is cheap.

---

## Steps

### 1. init_columns
Normalise schema to internal `Col` fields; filter non-English (`language` field must be `en/english/eng`); drop books with no title. Non-English books are rare in this dataset — only 1 dropped in the full run.

### 2. extract_text
Decode epub bytes → plain text via `inscriptis` (HTML→text), respecting the OPF spine order so chapters come out in reading order. Books with empty result are dropped (`empty_text`).

### 3. dedup_id
Drop exact duplicate Gutenberg IDs (sequential, keep first). 767 dropped in full run — mostly re-uploads with different epub packaging.

### 4. dedup_title
Drop exact duplicate normalised titles (case/punctuation-folded, keep first). 923 dropped.

### 5. add_title_embeddings
Embed all book titles with `all-MiniLM-L6-v2` (384-dim) to enable semantic dedup. GPU if available.

### 6. build_title_clusters
Agglomerative clustering on title embeddings (cosine distance, average linkage, threshold=**0.25**); keep only the first book per cluster, drop the rest as `dedup_title_cluster`.

Threshold 0.25 was determined empirically: cluster outputs were manually inspected at several values to find the point where same-book variants (different editions, translations filed under slightly different titles) reliably group together without collapsing distinct books.

This was the single largest filter at full scale: **14,516 dropped** (25% of the original corpus). The high count reflects the Gutenberg dataset's many duplicated editions.

### 7. strip_gutenberg
Regex-strip Project Gutenberg license headers and footers from the first/last 25,000 characters (`GUTENBERG_ZONE`). Zone is 25k because epub license chapters run ~15k characters; 25k gives a safe margin. Multiple header and footer patterns are matched (start-of-ebook markers, "produced by", "e-text prepared by", etc.).

### 8. normalize_text
`ftfy` for encoding fixes → NFKC unicode normalisation → collapse trailing whitespace per line → collapse 3+ consecutive newlines to 2. Keeps the text structurally intact while removing noise introduced by epub-to-text conversion.

### 9. set_content_bounds
Compute a "content zone" within each book to avoid front and back matter:

- `margin_start = max(20_000, text_len × 0.10)`
- `margin_end   = max(20_000, text_len × 0.20)`

The 20k minimum skips ~7 pages on each side (at ~2,800 chars/page), covering typical front matter (copyright notices, dedications, tables of contents). The end margin is larger (20%) because Gutenberg books often have substantial back matter — appendices, notes, author bibliographies. The percentage kicks in once the book is long enough that a fixed 20k would be less than 10%/20% of the text.

### 10. mark_too_short
Drop books whose content zone is shorter than `WINDOW_CHARS = 45,000` characters. This means a book needs at least ~85k total characters (~30 pages) to survive after stripping margins. **8,594 dropped.**

`WINDOW_CHARS = 45,000` was chosen to give comfortable headroom above the 8,192-token target: at ~3–4 chars/token, 45k characters ≈ 11–15k tokens, well above the 8,193 stored per excerpt.

### 11. verify_no_gutenberg
Drop books where "Project Gutenberg" still appears in the content zone after stripping (step 7). These are books that reference PG in their actual text (e.g. anthology introductions). **56 dropped.**

### 12. compute_content_chunk_sigs
Compute MinHash signatures over 600-word non-overlapping chunks of each book's content zone. 5-grams of normalised words, 128 permutations.

### 13. dedup_content_minhash
LSH-based near-duplicate detection across books. A book is marked `content_minhash_duplicate` if **≥ 1 chunk** matches a previously seen book at Jaccard ≥ **0.3**.

Parameters:
- `CHUNK_WORDS = 600`, `CHUNK_NGRAM_SIZE = 5`, `CHUNK_NUM_PERM = 128`
- `CHUNK_SIM_THRESHOLD = 0.3`, `MIN_CHUNK_MATCHES = 1`

A single matching 600-word chunk at 0.3 Jaccard is sufficient to drop the book. The target was the same book appearing from multiple sources (mirror uploads, re-packaged editions) or books sharing large repeated passages. 5-grams absorb minor typos between versions. **3,261 dropped.**

### 14. sample_window
Pick a random 45,000-character window inside the content zone. Sampling is deterministic per book ID (seeded by `BOOK_ID`) so the run is reproducible.

### 15. find_excerpt_start
Align the window start to a sentence boundary using NLTK punkt. The window start is likely mid-sentence; punkt tokenizes the first 3,000 characters and we skip to the **second** sentence (index 1) — sentence 0 is the initial fragment, sentence 1 is the first complete sentence. Books where no second sentence boundary is found within 3,000 characters are dropped (`no_excerpt_start`). **42 dropped.**

### 16. tokenize_excerpt
Tokenize `WINDOW_CHARS` starting from the sentence-aligned position using the LLaMA 3.2-1B tokenizer, with BOS prepended and EOS appended (matching Megatron's `TemplateProcessing`). Store exactly `SEQ_LEN + 1 = 8,193` tokens (Megatron reads `seq_length+1` and splits into `input[:-1]` / `labels[1:]`). Books that don't produce enough tokens are dropped (`not_enough_tokens`). **119 dropped.**

### 17. verify_tokenization
Round-trip check: decode token IDs → re-encode → verify identical count and sequence. Also verifies exactly one BOS at position 0 and one EOS at the last position. Failures (`token_round_trip`) indicate pathological text. **24 dropped.**

### 18. compute_excerpt_chunk_sigs
Compute MinHash signatures over 100-word chunks of the decoded excerpt text. 5-grams, 40 permutations.

`EXCERPT_NUM_PERM = 40` (not the default 128): datasketch's LSH internally picks `b=40, r=1` for `threshold=0.04` regardless of `num_perm` — extra permutations are unused by the LSH index. Using 40 gives 3.8× faster loading with identical results.

### 19. dedup_excerpts_minhash
Two-stage excerpt deduplication:
1. LSH candidate retrieval at `EXCERPT_LSH_THRESHOLD = 0.04`
2. Exact Jaccard check per candidate chunk pair; drop if any chunk reaches `EXCERPT_JACCARD_THRESHOLD = 0.05`

Thresholds were calibrated on real duplicate pairs found at Jaccard ≈ 0.055 on 100-word chunks. Setting LSH below 0.04 proved unreliable; the exact Jaccard pass at 0.05 filters false positives from the broad LSH net. **1,863 dropped.**

---

## Filter summary (full dataset)

| filter | dropped |
|---|---|
| dedup_title_cluster | 14,516 |
| too_short | 8,594 |
| content_minhash_duplicate | 3,261 |
| excerpt_minhash_duplicate | 1,863 |
| duplicate_title | 923 |
| duplicate_id | 767 |
| not_enough_tokens | 119 |
| token_round_trip | 24 |
| project_gutenberg | 56 |
| no_excerpt_start | 42 |
| empty_text | 2 |
| non_english | 1 |
| **total kept** | **26,814 / 56,982 (47%)** |

---

## Step 20: perplexity + Min-K%++ scoring

Scoring code: `attn_bench/data_processing/books/score_perplexity.py`

Scored with the 1B LLaMA trained on FineWeb-Edu-160B (`llama3-1b-fineweb160B`). Two metrics per excerpt:

- **Perplexity**: cross-entropy loss exponentiated. Scores all tokens in one forward pass.
- **Min-20%++ (Zhang et al. 2024)**: z-score each token log-prob against the full vocabulary distribution at that position; take the mean of the bottom 20% z-scores. Higher (less negative) = model is confident on the hard tokens = likely overlap with training data.

The full 26,814-book scoring job is currently running. A 7k-book pilot (steps run on `raw-hf-7000`) was scored first to examine the distribution and understand what falls at each tail. Charts and per-example outputs are in `attn_bench/results/data/gutenberg-laion/raw-hf-7000-output/stats/20_score_perplexity_min_k_pp/`.

### Pilot distribution observations (7k books)

Stats files: `attn_bench/results/data/gutenberg-laion/raw-hf-7000-output/stats/20_score_perplexity_min_k_pp/`

**Low-perplexity tail** (pilot range ~15.7–54.3): these are texts the FineWeb-Edu model predicts easily — either because they resemble web content or because they are structurally repetitive/simple. Examples seen:
- Bibles and religious texts
- Tables and lists
- US law and legal documents
- Recipe books
- Some poems (simple/formulaic)
- Educational non-fiction (political, historical, art history)

**High-perplexity tail** (pilot range ~1,094–20,600+): texts the model finds unpredictable — unusual language, non-standard orthography, non-English passages. Examples seen:
- Poems with archaic or highly stylised language
- Lists of names
- Plays in Early Modern / Middle English
- Text with heavy phonetic spelling / dialectal orthography (e.g. *"um noshuns are like gitting struk with litening"*)
- Archaic English (*"Spake Hagen of Troneg: 'Never fear stirreth nor stayeth me!'"*)
- Passages in French, German, Esperanto, Latin, Finnish, Passamaquoddy
- Scots/dialect verse (*"Quakyng throu dreid, ruschit furth, or scho wald stent"*)
- Plays labelled things like "An Irishman's Difficulties with the Dutch Language"

**High Min-K%++ tail** (`min_k_pp_high.txt` — most likely seen by the FineWeb model during training): Bible, nutrition and food lists, recipes and cookbooks, books about America/Americans, educational non-fiction (history, art, species), comedies, poems, architecture, philosophy, lectures, studies, Catholic World publications, letters, autobiography, essays, preventable diseases, Encyclopaedia, Algernon Charles Swinburne, De Re Metallica, Tannhäuser, Mining Laws of Ohio, how to grow mushrooms. These are all web-adjacent or widely reproduced reference/instructional texts — exactly what FineWeb-Edu contains.

**Low Min-K%++ tail** (`min_k_pp_low.txt` — least likely to overlap with training data): dominated by plays and dialogs with lots of listed character names. Also: Shakespeare (multiple editions), A Tale of Salt Lake City, A Legend of Old Persia and Other Poems, Lord Byron, educational books, dictionaries (English proverbs, German idioms), Encyclopaedia Britannica, Jamaican song and story, Esperanto, verse, footnotes, River-Names of Europe, Four Mystery Plays. Same typo-heavy / archaic-English examples as in the high-perplexity tail also appear here.

---

## Step 21: FineWeb-Edu containment check (2026-05-15)

Checks how many of the 26,816 Gutenberg excerpts appear in FineWeb-Edu-dedup (200 parquet files, 24 workers). Uses 13-gram hashing to detect overlap.

Job: `2254409`, runtime: 19:05→21:05 (~2h). Output: `.../gutenberg-laion/hf-full-output/sampled_containment.jsonl`

| coverage threshold | contaminated excerpts |
|---|---|
| any match (≥1 ngram) | 18,780 / 26,816 (70.0%) |
| >1% of ngrams | 8,782 / 26,816 (32.7%) |
| >5% of ngrams | 5,018 / 26,816 (18.7%) |
| >10% of ngrams | 4,212 / 26,816 (15.7%) |
| >25% of ngrams | 3,300 / 26,816 (12.3%) |
| >50% of ngrams | 2,544 / 26,816 (9.5%) |

70% have at least one 13-gram match (expected for any large web corpus), but coverage drops sharply — only 9.5% have >50% of their ngrams matched, suggesting most hits are incidental rather than substantial overlap.

Scripts: `attn_bench/data_processing/books/check_fineweb_containment.py`, `attn_bench/submissions/check_fineweb_containment.slurm`

---

## Planned: perplexity-based filtering (step 22)

After the full scoring run completes, apply bilateral cutoffs informed by the pilot observations:

- **Low-perplexity cutoff**: removes contamination (texts resembling FineWeb-Edu) and structurally trivial content (bibles, legal tables, recipe lists). The FineWeb model predicts these easily.
- **High-perplexity cutoff**: removes non-English passages, heavy dialect/phonetic text, and otherwise anomalous excerpts that would introduce noise into the memorization probes.
- **Decontamination via Min-K%++**: flags excerpts the model likely saw during training, complementing the low-perplexity cut.

Specific thresholds TBD after examining the full-dataset distribution; pilot observations above guide where to look.

## Megatron tokenization (2026-05-19)

Script: `attn_bench/submissions/write_gutenberg_in_megatron_format.slurm`  
Input: `repetition_buckets.jsonl` (660 books × 9 rep levels)  
Output: `/iopsstor/scratch/cscs/elyulina/datasets/tokenized/gutenberg_rep_1_256`  
Job: `2305689`, runtime: 09:59→10:02 (~3 min)

One `rep_N_tokens.bin/.idx` pair per repetition level. Each sequence is 8,192 tokens (BOS + content + EOS).


All verification checks passed (lengths, BOS/EOS position and count, repetition counts).

SUMMARY
  total sequences  : 374,563
  total unique books (across all buckets): 6,597
  total tokens     : 3,068,420,096  (3.07B)
  ALL CHECKS PASSED
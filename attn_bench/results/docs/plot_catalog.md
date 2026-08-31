# Plot Catalog

Every module in `attn_bench/plotting/`, one line each: what it covers, key functions.
Signatures/params live in the code's own docstrings — this is just "where do I find X."

- **`data_loading.py`** — every loader, family-prefixed (`load_mem_results_*`,
  `load_long_inference_*`, `load_long_individual_sequence*`,
  `load_attention_patterns*`/`load_attention_gating`,
  `load_doc_lengths`/`load_packed_chunk_length_hist`), plus `select_models` (the shared
  `models=` filter every plot function below takes).
- **`model_registry.py`** — model-name -> results-folder registry (`MODELS`,
  `MODEL_COLORS`), per-family results-base paths, `model_folder`/`scf8_variant`/
  `sink_variant` helpers.
- **`heatmap.py`** — every heatmap: `plot_models_heatmap`/`_panel` (columns=models),
  `plot_heatmap_per_model`/`_panel` (columns=swept offset/prefix/suffix, one model per
  heatmap). Metric-agnostic, optional Wilcoxon-vs-reference p-values.
- **`rouge_l.py`** — Rouge-L-specific: `plot_mem_bucket_counts`, `plot_rouge_hist`,
  `plot_rouge_heatmap` (rep x Rouge-L-bin distribution).
- **`utils.py`** — `plot_lineplot`/`plot_lineplot_panel` (a metric, or paired ref/gen
  metrics, vs repetition), plus shared scaffolding (panel-grid layout, `suptitle_centered`,
  `smooth`, `denser_grid`).
- **`attention_patterns.py`** — mechanistic: `plot_first_token_attention_avg`/`_by_bucket`
  (attention-sink strength), `plot_map`/`plot_bucket_maps`/`plot_full_attn_maps_panel`
  (query x key heatmaps), `plot_gating_distribution` (gated model only).
- **`gdn_state.py`** — GDN recurrent state norm: `plot_state_norm_panel` (overview),
  `plot_state_norm_by_layer` (per-layer breakdown).
- **`long_inference.py`** — aggregate long-context loss: `plot_loss_panel`,
  `plot_coverage` (data-coverage curve), `compare_nll` (correctness-check table, not a plot).
- **`long_individual_sequence.py`** — detailed per-sequence inspection: `plot_individual_panel`,
  `filter_sequences`/`score_sequences` (pick which sequences are worth plotting).
- **`doc_lengths.py`** — document/packed-chunk length distributions: histograms, overlays,
  token-budget plots, doc-vs-chunk comparison.
- **`generated_examples.py`** — qualitative HTML tables: `show_examples`/`_by_exp` (side
  by side across models), `show_examples_by_offset` (one model, offsets compared with a
  shared marked span).
- **`memorization_overlap.py`** — `plot_sequence_overlap_heatmap`: do different models
  memorize the same sequences?
- **`sink.py`** — sink-family-specific: trained softmax-offset heatmaps, PPL/TTR line
  plots, generation-quality comparisons, `plot_sink_nll_comparison` (multi-config overlay +
  correctness table).
- **`prefix_suffix.py`** — offset x prefix 3D surface / 2D contour panels feeding the
  memorization dashboard.

**Inline plotting not in a `.py` module** (deliberate scratch exception, not a gap):
`mem_plotting_style.ipynb`'s early-stage offset x prefix visualization experiments —
anything that earned its keep already graduated into `prefix_suffix.py`'s two functions
above.

"""Canonical model-name -> results-folder registry, plus per-family results-base paths."""

from pathlib import Path

# Anchored to this file's own location (attn_bench/plotting/ -> attn_bench/), not the
# caller's cwd -- these used to be plain '../results/...' literals, which only resolved
# correctly for a notebook sitting exactly one level below attn_bench/ (i.e. directly in
# notebooks/); any notebook nested deeper (e.g. notebooks/scf8/) silently got an empty
# results dir instead of an error, since Path('../results/...') is still a valid (just
# wrong) path one level short.
_ATTN_BENCH_DIR = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _ATTN_BENCH_DIR / 'results'

MEM_RESULTS_DIR = str(_RESULTS_DIR / 'mem-results/SparseGutenberg')
MEM_RESULTS_DIR_OLD = str(_RESULTS_DIR / 'mem-results/SparseGutenberg-legacy')  # older, 4-model + sink-ablation data

LONG_GUTENBERG_RESULTS_DIR = _RESULTS_DIR / 'long-gutenberg-results'
LONG_INFERENCE_CONFIG = 'all_samples_full_tokens'  # the standard capture, used by almost every model
GUTENBERG_LONG_LENGTHS_PATH = _RESULTS_DIR / 'data/gutenberg-long/lengths.jsonl'

LONG_FINEWEB_RESULTS_DIR = _RESULTS_DIR / 'long-fineweb-results'
LONG_FINEWEB_CONFIG = '660_samples_full_tokens'
LONG_FINEWEB_KEY = 'long_24576_32768'  # must match extract_long_docs.py's --min/--max-length
LONG_FINEWEB_PARTITION_DIRS = {
    'seen': 'fineweb-edu-dedup-160B-datatrove_0.25_long',
    'unseen': 'fineweb-edu-dedup-160B-datatrove_0.75_unseen_long',
}
DOC_LENGTHS_DIR = _RESULTS_DIR / 'datasets/analysis/doc-lengths'
PACKED_CHUNK_LENGTHS_DIR = _RESULTS_DIR / 'datasets/packed_chunk_lengths'

MODELS = {
    'full': 'llama3-1b-full-attn-fineweb40B-gutenberg3B',
    'gated': 'llama3-1b-gated-attn-fineweb40B-gutenberg3B',
    'learn-sink': 'llama3-1b-sink-attn-fineweb40B-gutenberg3B-te215',
    'off-by-one': 'llama3-1b-off-by-one-attn-fineweb40B-gutenberg3B-te215',
    'gdn': 'llama3-1b-gdn-fineweb40B-gutenberg3B',
    'gdn-xdl': 'llama3-1b-gdn-carry-r0-fineweb40B-gutenberg3B',  # GDN counterpart of full-xdoc-leak: state not reset between docs
    'gdn-xdl-xsl-0.5': 'llama3-1b-gdn-carry-r0.5-fineweb40B-gutenberg3B',
    'gdn-xdl-xsl': 'llama3-1b-gdn-carry-r1-fineweb40B-gutenberg3B',
    'full-xdl': 'llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B',
    'full-goldfish': 'llama3-1b-full-attn-goldfish-fineweb40B-gutenberg3B',
    'gdn-goldfish': 'llama3-1b-gdn-goldfish-fineweb40B-gutenberg3B',
    'full-80B': 'llama3-1b-full-attn-fineweb80B-gutenberg3B',
    'full-long-docs': 'llama3-1b-full-attn-fineweb40B-long-gutenberg3B',
    'full-short-docs': 'llama3-1b-full-attn-fineweb40B-long-split-1024-gutenberg3B',
    # RoPE-scaling-fix rerun (inference_common.py now passes --use-rope-scaling explicitly,
    # see load_megatron_model) -- fineweb only, no gutenberg counterpart.
    'full-scf-8': 'llama3-1b-full-attn-fineweb40B-gutenberg3B-scf-8',
    'full-xdl-scf-8': 'llama3-1b-full-attn-xdoc-attn-leak-fineweb40B-gutenberg3B-scf-8',
    'full-long-docs-scf-8': 'llama3-1b-full-attn-fineweb40B-long-gutenberg3B-scf-8',
    'full-short-docs-scf-8': 'llama3-1b-full-attn-fineweb40B-long-split-1024-gutenberg3B-scf-8',
    # mem-results rename (da76e69e1): -scf8 = fixed rerun, -scf8-scf1 = old mismatched run,
    # archived under this name. `full`/`gated`/`learn-sink` above still point at their
    # pre-rename (now-nonexistent) folders -- intentionally not migrated.
    'full-scf8': 'llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B',
    'full-scf8-scf1': 'llama3-1b-full-attn-scf8-fineweb40B-gutenberg3B-scf1',
    'gated-scf8': 'llama3-1b-gated-attn-scf8-fineweb40B-gutenberg3B',
    'gated-scf8-scf1': 'llama3-1b-gated-attn-scf8-fineweb40B-gutenberg3B-scf1',
    'learn-sink-scf8': 'llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-te215',
    'learn-sink-scf8-scf1': 'llama3-1b-sink-attn-scf8-fineweb40B-gutenberg3B-te215-scf1',
    'off-by-one-scf8': 'llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-te215',
    'off-by-one-scf8-scf1': 'llama3-1b-off-by-one-attn-scf8-fineweb40B-gutenberg3B-te215-scf1',
    'full-xdl-scf8': 'llama3-1b-full-attn-xdoc-attn-leak-scf8-fineweb40B-gutenberg3B',
    'full-xdl-scf8-scf1': 'llama3-1b-full-attn-xdoc-attn-leak-scf8-fineweb40B-gutenberg3B-scf1',
    'full-goldfish-scf8': 'llama3-1b-full-attn-goldfish-scf8-fineweb40B-gutenberg3B',
    'full-goldfish-scf8-scf1': 'llama3-1b-full-attn-goldfish-scf8-fineweb40B-gutenberg3B-scf1',
    'full-80B-scf8': 'llama3-1b-full-attn-scf8-fineweb80B-gutenberg3B',
    'full-80B-scf8-scf1': 'llama3-1b-full-attn-scf8-fineweb80B-gutenberg3B-scf1',
    'full-long-docs-scf8': 'llama3-1b-full-attn-scf8-fineweb40B-long-gutenberg3B',
    'full-long-docs-scf8-scf1': 'llama3-1b-full-attn-scf8-fineweb40B-long-gutenberg3B-scf1',
    'full-short-docs-scf8': 'llama3-1b-full-attn-scf8-fineweb40B-long-split-1024-gutenberg3B',
    'full-short-docs-scf8-scf1': 'llama3-1b-full-attn-scf8-fineweb40B-long-split-1024-gutenberg3B-scf1',
    # full-scf1: trained AND inferred with rope-scaling-factor=1 -- a separate training run,
    # not an isolated RoPE-scaling ablation of full-scf8 (see the pretrain slurm script for the
    # other hyperparameters that also differ). Folder here is the Megatron-backend results dir;
    # use model_folder(key, 'hf') for the same checkpoint through the HF inference backend.
    'full-scf1': 'llama3-1b-full-attn-scf1-fineweb40B-gutenberg3B',
    # scf1-trained counterparts of gated/sink/SWA -- same scf1 training config as full-scf1
    # (see models_pretraining_fineweb40B_gutenberg3B.md), not RoPE-scaling ablations of the
    # scf8 runs above.
    'gated-scf1': 'llama3-1b-gated-attn-scf1-fineweb40B-gutenberg3B',
    'sink-scf1': 'llama3-1b-sink-attn-scf1-fineweb40B-gutenberg3B',
    'swa-w256-scf1': 'llama3-1b-swa-w256-scf1-fineweb40B-gutenberg3B',
    'swa-w1024-scf1': 'llama3-1b-swa-w1024-scf1-fineweb40B-gutenberg3B',
    'swa-w4096-scf1': 'llama3-1b-swa-w4096-scf1-fineweb40B-gutenberg3B',
    # scf1-trained non-softmax / non-MHA architectures. MLA is still softmax attention (latent
    # KV compression); KDA and the Qwen hybrid are delta-rule linear attention like GDN. The
    # Qwen hybrid runs GDN on 12 layers and gated softmax attention on 4 (--linear-attention-freq 4).
    'mla': 'llama3-1b-mla-scf1-fineweb40B-gutenberg3B',
    'kda': 'llama3-1b-kda-scf1-fineweb40B-gutenberg3B',
    'qwen': 'llama3-1b-hybrid-qwen-scf1-fineweb40B-gutenberg3B',
}

# Backend -> folder-name suffix appended to a MODELS entry (mirrors measure_mem.slurm's own
# PDM_EXP_NAME suffixing -- keep in sync). Use model_folder() below to resolve one, or pass
# backend=[...] to load_mem_results_scores_grid for priority-ordered fallback across backends.
BACKEND_SUFFIXES = {'megatron': '', 'hf': '_hf'}


def model_folder(key, backend='megatron'):
    """Folder name for `key` under one specific backend -- e.g. model_folder('full-scf1', 'hf')
    for the HF-backend results dir of the same checkpoint. For priority-ordered fallback across
    backends instead of pinning one, pass backend=[...] directly to load_mem_results_scores_grid."""
    return MODELS[key] + BACKEND_SUFFIXES[backend]

# One hue per model family, shade for the variant within it -- so "which family" reads at
# a glance across every plot. Colourblind-safe, no black. Keys match MODELS.
MODEL_COLORS = {
    # full-attention family -- blues (+ cyan for goldfish)
    'full':             '#08519c',
    'full-xdl':   '#6A8EEB',
    'full-goldfish':    '#17BECF',
    'full-80B':         '#3182bd',
    # sink family (softmax-denominator variants) -- greens
    'off-by-one':       '#74c476',
    'learn-sink':       '#238b45',
    'gated':            '#e6cc00',
    # GDN family -- pinks/reds
    'gdn':              '#EB0EE7',
    'gdn-xdl':    '#FFA6F4',
    'gdn-xdl-xsl-0.5':   '#9C1F1F',
    'gdn-xdl-xsl':     '#EB7D0E',
    'gdn-goldfish':     '#8B4513',
    # long-vs-split-1024 training-document-structure pair -- teals
    'full-long-docs':   '#005F73',
    'full-short-docs':  '#0A9396',
    # scf-8 rerun -- same hues as their counterparts above
    'full-scf-8':             '#08519c',
    'full-xdl-scf-8':         '#6A8EEB',
    'full-long-docs-scf-8':   '#005F73',
    'full-short-docs-scf-8':  '#0A9396',
    # scf8 rerun #2 (mem-results) -- same hues as their family above
    'full-scf8':              '#08519c',
    'full-scf8-scf1':         '#08519c',
    'gated-scf8':             '#e6cc00',
    'gated-scf8-scf1':        '#e6cc00',
    'learn-sink-scf8':        '#238b45',
    'learn-sink-scf8-scf1':   '#238b45',
    'off-by-one-scf8':        '#74c476',
    'off-by-one-scf8-scf1':   '#74c476',
    'full-xdl-scf8':          '#6A8EEB',
    'full-xdl-scf8-scf1':     '#6A8EEB',
    'full-goldfish-scf8':     '#17BECF',
    'full-goldfish-scf8-scf1': '#17BECF',
    'full-80B-scf8':          '#3182bd',
    'full-80B-scf8-scf1':     '#3182bd',
    'full-long-docs-scf8':    '#005F73',
    'full-long-docs-scf8-scf1': '#005F73',
    'full-short-docs-scf8':   '#0A9396',
    'full-short-docs-scf8-scf1': '#0A9396',
    'full-scf1':              '#08519c',
    'gated-scf1':             '#e6cc00',
    'sink-scf1':              '#238b45',
    # SWA family -- purples, darker shade for a wider (less restrictive) window
    'swa-w256-scf1':          '#cbc9e2',
    'swa-w1024-scf1':         '#9e9ac8',
    'swa-w4096-scf1':         '#54278f',
    # MLA -- own hue (burnt sienna); linear-attention newcomers share the GDN pink family,
    # lighter shade for the hybrid. All three checked for CVD/normal-vision separation
    # against every colour above.
    'mla':                    '#B15928',
    'kda':                    '#C51B7D',
    'qwen':                   '#F06BA8',
}


def scf8_variant(base_key):
    """(pre_fix_key, scf8_key) for a base family name: its -scf8-scf1/-scf8 pair if one's in
    MODELS, else base_key for both -- e.g. GDN, which the RoPE-scaling fix doesn't touch."""
    pre_fix_key, scf8_key = f'{base_key}-scf8-scf1', f'{base_key}-scf8'
    if pre_fix_key in MODELS and scf8_key in MODELS:
        return pre_fix_key, scf8_key
    return base_key, base_key


def scf8_variants(base_keys):
    """Flat, de-duplicated list of every scf8_variant() key for `base_keys`, in order -- pass
    as `models=` to the loader when you'll plot both the pre-fix and scf8 side by side."""
    seen, out = set(), []
    for base in base_keys:
        for key in scf8_variant(base):
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def sscale_label(base_key, s):
    """The key sink_variant() gives scale s under -- kept separate so callers can select a
    subset of an already-loaded sink_variant() dict by scale value instead of by string."""
    return f'{base_key} sscale=1 (default)' if s == 1 else f'{base_key} sscale={s}'


def sink_variant(base_key, sscales=(0, 0.25, 0.5, 0.75, 1.5, 2), results_base=MEM_RESULTS_DIR_OLD):
    """{label: exp_name} for one model's sink-scale ablation. sscale=1 is the plain
    checkpoint (no suffix); other values normally read `_sscale{s}`, falling back to the
    older `_nsinks{s}` folder name if that's what's on disk (same ablation, older naming).
    Every constructed name also falls back to a trailing `-scf1` if that's what's on disk --
    rename_scf8_results.sh archived the RoPE-scaling-mismatched sscale sweeps that way.
    """
    base_folder = MODELS[base_key]

    def _resolve(name):
        if (Path(results_base) / name).exists() or not (Path(results_base) / f'{name}-scf1').exists():
            return name
        return f'{name}-scf1'

    out = {}
    for s in sorted(set(sscales) | {1}):  # sort 1 in with the rest instead of always-first
        if s == 1:
            out[sscale_label(base_key, 1)] = _resolve(base_folder)
            continue
        sscale_dir = Path(results_base) / f'{base_folder}_sscale{s}'
        nsinks_dir = Path(results_base) / f'{base_folder}_nsinks{s}'
        suffix = 'nsinks' if not sscale_dir.exists() and nsinks_dir.exists() else 'sscale'
        out[sscale_label(base_key, s)] = _resolve(f'{base_folder}_{suffix}{s}')
    return out
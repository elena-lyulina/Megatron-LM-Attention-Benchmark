"""Qualitative side-by-side example display: prefix / reference suffix / each model's
generated suffix, plus the per-sample metrics that were computed from them. Needs the raw
text records (data_loading.load_mem_results_text_records), not just the scores."""

from IPython.display import HTML, display

from attn_bench.plotting.data_loading import load_mem_results_text_records
from attn_bench.plotting.model_registry import MODELS

_CELL_STYLE  = 'padding:6px 8px;vertical-align:top;max-width:280px;min-width:200px;text-align:left;'
_SCROLL_STYLE = (
    'max-height:160px;overflow-y:auto;white-space:pre-wrap;overflow-wrap:break-word;'
    'font-size:11px;font-family:monospace;color:#111;line-height:1.4;text-align:left;'
)
_METRIC_STYLE = 'font-size:10px;font-family:monospace;color:#111;text-align:left;line-height:1.8;'
_SEP_STYLE    = 'border:none;border-top:1px solid #999;margin:4px 0;'
_TH_STYLE     = 'padding:6px 10px;text-align:left;background:#ddd;color:#111;font-size:12px;'
_N_METRIC_LINES = 5


def _esc(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _td_text(text, bg):
    return (f'<td style="background:{bg};{_CELL_STYLE}">'
            f'<div style="{_SCROLL_STYLE}">{_esc(text)}</div></td>')


def _td_metrics(lines, bg):
    content = '<br>'.join(lines) + f'<hr style="{_SEP_STYLE}">'
    return f'<td style="background:{bg};{_CELL_STYLE}"><div style="{_METRIC_STYLE}">{content}</div></td>'


def _fmt(v):
    return f'{v:.3f}'


def _load_examples(tok, exp_name, rep, indices, results_base, prefix, suffix, offset, policy):
    data = load_mem_results_text_records(exp_name, rep, results_base, prefix, suffix, offset, policy)
    if len(data) <= max(indices):
        return None  # fewer samples than requested -- caller excludes this model and says so
    return [{
        'prefix':  tok.decode(data[i]['prefix'],           skip_special_tokens=True),
        'ref':     tok.decode(data[i]['true_suffix'],      skip_special_tokens=True),
        'gen':     tok.decode(data[i]['generated_suffix'], skip_special_tokens=True),
        'Rouge-L': data[i]['Rouge-L'],
        'LCS':     data[i]['lcs_norm'],
        'PPL':     data[i]['perplexity'],
        'NLL':     data[i]['nll_mean'],
        'TTR_gen': data[i]['TTR_gen'],
        'Ref_PPL': data[i]['ref_perplexity'],
        'Ref_NLL': data[i]['ref_nll_mean'],
        'TTR_ref': data[i]['TTR_ref'],
    } for i in indices]


def show_examples(tok, rep, models, sample_indices, results_base,
                  prefix=500, suffix=500, offset=0, policy='greedy', col_bg=None):
    """HTML table: prefix / reference suffix / each model's generated suffix + per-example
    metrics, for one repetition bucket.

    models: list of names from model_registry.MODELS.
    col_bg: optional {name: hex} background per column ('prefix', 'ref', and each model
    name); defaults to a light grey for anything not given -- these are meant to be pale
    backgrounds behind black text, not MODEL_COLORS' saturated line colors.
    """
    exp_names = {name: MODELS[name] for name in models}
    show_examples_by_exp(tok, rep, exp_names, sample_indices, results_base,
                        prefix, suffix, offset, policy, col_bg)


def _wrap_table(rows, rep):
    table = (
        f'<details style="margin-top:24px;width:100%">'
        f'<summary style="cursor:pointer;font-size:16px;font-weight:bold;padding:4px 0">'
        f'repetition = {rep}</summary>'
        # onwheel: trackpad two-finger horizontal swipes report a real deltaX, but Jupyter's
        # own output-area scroll handling can swallow it before the browser's default
        # overflow-x scrolling kicks in -- forwarding deltaX to scrollLeft ourselves means
        # you can pan by hovering anywhere over the table, not just by grabbing the
        # scrollbar. Vertical-only scrolls (deltaX == 0) are left alone.
        f'<div style="width:100%;overflow-x:auto" '
        f'onwheel="if(event.deltaX){{this.scrollLeft+=event.deltaX;event.preventDefault();}}">'
        f'<table style="border-collapse:collapse;border:1px solid #ccc;width:max-content;max-width:none">'
        + ''.join(rows) +
        '</table></div>'
        f'</details>'
    )
    display(HTML(table))


def show_examples_by_exp(tok, rep, exp_names, sample_indices, results_base,
                         prefix=500, suffix=500, offset=0, policy='greedy', col_bg=None):
    """Same as show_examples, but keyed by arbitrary {label: exp_name} pairs instead of
    model_registry names -- for labels that don't have their own MODELS entry, e.g.
    inference-time sink-scale variants of one base model."""
    col_bg = col_bg or {}
    data_by_model = {}
    for label, exp_name in exp_names.items():
        examples = _load_examples(tok, exp_name, rep, sample_indices, results_base, prefix, suffix, offset, policy)
        if examples is None:
            print(f'{label}: no data for rep={rep} (fewer than {len(sample_indices)} samples available) -- excluded')
        else:
            data_by_model[label] = examples
    if not data_by_model:
        return

    cols = ['prefix', 'ref suffix'] + [f'gen suffix: {label}' for label in data_by_model]
    header = ''.join(f'<th style="{_TH_STYLE}">{c}</th>' for c in cols)

    rows = [f'<tr>{header}</tr>']
    for idx in range(len(sample_indices)):
        ref_ex = next(iter(data_by_model.values()))[idx]

        metric_cells = (
            _td_metrics(['&nbsp;'] * _N_METRIC_LINES, col_bg.get('prefix', '#f0f0f0'))
            + _td_metrics([
                '&nbsp;',
                '&nbsp;',
                f'Ref_PPL: {_fmt(ref_ex["Ref_PPL"])}',
                f'Ref_NLL: {_fmt(ref_ex["Ref_NLL"])}',
                f'TTR_ref: {_fmt(ref_ex["TTR_ref"])}',
            ], col_bg.get('ref', '#c8e6c9'))
        )
        for name, examples in data_by_model.items():
            e = examples[idx]
            metric_cells += _td_metrics([
                f'Rouge-L: {_fmt(e["Rouge-L"])}',
                f'LCS:     {_fmt(e["LCS"])}',
                f'PPL:     {_fmt(e["PPL"])}',
                f'NLL:     {_fmt(e["NLL"])}',
                f'TTR_gen: {_fmt(e["TTR_gen"])}',
            ], col_bg.get(name, '#f5f5f5'))

        text_cells = (
            _td_text(ref_ex['prefix'], col_bg.get('prefix', '#f0f0f0')) +
            _td_text(ref_ex['ref'],    col_bg.get('ref', '#c8e6c9'))
        )
        for name, examples in data_by_model.items():
            text_cells += _td_text(examples[idx]['gen'], col_bg.get(name, '#f5f5f5'))

        rows.append(f'<tr style="border-top:2px solid #aaa">{metric_cells}</tr>')
        rows.append(f'<tr>{text_cells}</tr>')

    _wrap_table(rows, rep)


_OFFSET_PALETTE = [
    ('#1565c0', '#e3f2fd'),  # blue
    ('#2e7d32', '#e8f5e9'),  # green
    ('#e65100', '#fff3e0'),  # orange
    ('#6a1b9a', '#f3e5f5'),  # purple
    ('#ad1457', '#fce4ec'),  # pink
    ('#00838f', '#e0f7fa'),  # teal
]  # (marker/saturated, pale background) pairs, one per offset


def _default_offset_colors(offsets):
    return {o: _OFFSET_PALETTE[i % len(_OFFSET_PALETTE)] for i, o in enumerate(offsets)}


def _offset_marker(label, color):
    return (f'<span style="background:{color};color:#fff;font-size:9px;font-family:monospace;'
            f'padding:1px 5px;margin:0 2px;border-radius:3px;white-space:nowrap;">{label}</span>')


def _load_records(exp_name, rep, indices, results_base, prefix, suffix, offset, policy):
    data = load_mem_results_text_records(exp_name, rep, results_base, prefix, suffix, offset, policy)
    return [data[i] for i in indices]


def _marked_span_html(tok, full_ids, start, end, marks, colors):
    """Decode full_ids[start:end], with a colored marker inserted right after each mark's
    position that falls inside [start, end]. marks: (position, offset, label) tuples --
    ties at the same position render in the given order."""
    marks = sorted((m for m in marks if start <= m[0] <= end), key=lambda m: m[0])
    segments = []
    cursor = start
    for pos, offset, label in marks:
        segments.append(_esc(tok.decode(full_ids[cursor:pos], skip_special_tokens=True)))
        segments.append(_offset_marker(label, colors[offset][0]))
        cursor = pos
    segments.append(_esc(tok.decode(full_ids[cursor:end], skip_special_tokens=True)))
    return ''.join(segments)


def show_examples_by_offset(tok, rep, exp_name, offsets, sample_indices, results_base,
                            prefix=500, suffix=500, policy='greedy', label=None, offset_colors=None):
    """Same data as show_examples, but columns are offsets of one model/exp_name instead of
    several models. Offsets slide the (prefix, suffix) window forward in the same passage, so
    small offsets overlap almost entirely -- rather than repeat near-duplicate prefix/ref
    blocks per offset, this renders ONE shared prefix span and ONE shared ref-suffix span
    (union across offsets), with a colored marker at each offset's boundary, and only the
    (differing) generations as separate columns.

    offsets: sorted ascending internally, regardless of the order passed in.
    """
    offsets = sorted(offsets)
    label = label or exp_name
    colors = offset_colors or _default_offset_colors(offsets)
    min_off, max_off = offsets[0], offsets[-1]

    records_by_offset = {
        o: _load_records(exp_name, rep, sample_indices, results_base, prefix, suffix, o, policy)
        for o in offsets
    }

    cols = ['prefix (shared span, offset boundaries marked)',
            'ref suffix (shared span, offset boundaries marked)'] + \
           [f'{label} gen (offset={o})' for o in offsets]
    header = ''.join(f'<th style="{_TH_STYLE}">{c}</th>' for c in cols)

    rows = [f'<tr>{header}</tr>']
    for idx in range(len(sample_indices)):
        min_rec = records_by_offset[min_off][idx]
        prefix_ids = list(min_rec['prefix'])
        full_ids = prefix_ids + list(min_rec['true_suffix'])
        if max_off > min_off:
            full_ids += list(records_by_offset[max_off][idx]['true_suffix'][-(max_off - min_off):])

        start_of = {o: o - min_off for o in offsets}
        boundary_of = {o: start_of[o] + len(prefix_ids) for o in offsets}

        prefix_marks = (
            [(start_of[o], o, f'&#9654; offset={o} starts') for o in offsets] +
            [(boundary_of[o], o, f'&#9660; offset={o} ends') for o in offsets]
        )
        prefix_html = _marked_span_html(tok, full_ids, 0, boundary_of[max_off], prefix_marks, colors)

        suffix_marks = [(boundary_of[o], o, f'&#9660; offset={o} starts') for o in offsets]
        suffix_html = _marked_span_html(tok, full_ids, boundary_of[min_off], len(full_ids), suffix_marks, colors)

        metric_cells = (
            _td_metrics(['&nbsp;'] * 8, '#f0f0f0') +
            _td_metrics(['&nbsp;'] * 8, '#c8e6c9')
        )
        for o in offsets:
            e = records_by_offset[o][idx]
            metric_cells += _td_metrics([
                f'Rouge-L: {_fmt(e["Rouge-L"])}',
                f'LCS:     {_fmt(e["lcs_norm"])}',
                f'PPL:     {_fmt(e["perplexity"])}',
                f'NLL:     {_fmt(e["nll_mean"])}',
                f'TTR_gen: {_fmt(e["TTR_gen"])}',
                f'Ref_PPL: {_fmt(e["ref_perplexity"])}',
                f'Ref_NLL: {_fmt(e["ref_nll_mean"])}',
                f'TTR_ref: {_fmt(e["TTR_ref"])}',
            ], colors[o][1])

        text_cells = (
            f'<td style="background:#f0f0f0;{_CELL_STYLE}"><div style="{_SCROLL_STYLE}">{prefix_html}</div></td>'
            f'<td style="background:#c8e6c9;{_CELL_STYLE}"><div style="{_SCROLL_STYLE}">{suffix_html}</div></td>'
        )
        for o in offsets:
            gen_text = tok.decode(records_by_offset[o][idx]['generated_suffix'], skip_special_tokens=True)
            text_cells += _td_text(gen_text, colors[o][1])

        rows.append(f'<tr style="border-top:2px solid #aaa">{metric_cells}</tr>')
        rows.append(f'<tr>{text_cells}</tr>')

    _wrap_table(rows, rep)
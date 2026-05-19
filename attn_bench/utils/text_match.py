import argparse
import hashlib
import os
import re
import difflib
from pathlib import Path
from typing import Optional

from datasketch import MinHash

# Palette: cohesive blue-green family, soft pastels
_COLORS = [
    "#bbdefb",  # pale blue
    "#b2dfdb",  # pale teal
    "#c8e6c9",  # pale green
    "#b2ebf2",  # pale cyan
    "#90caf9",  # blue
    "#80cbc4",  # teal
    "#a5d6a7",  # green
    "#80deea",  # cyan
    "#64b5f6",  # medium blue
    "#4db6ac",  # medium teal
]


def _tokenize(text: str) -> list[str]:
    # Split into word-tokens and whitespace/punctuation tokens, preserving originals
    return re.findall(r"\S+|\s+", text)


def _norm(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _word_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def _find_matching_blocks(words1: list[str], words2: list[str], min_match: int = 5):
    """Return list of (i, j, n) matching blocks at word level using normalized comparison."""
    norm1 = [_norm(w) for w in words1]
    norm2 = [_norm(w) for w in words2]
    sm = difflib.SequenceMatcher(None, norm1, norm2, autojunk=False)
    blocks = []
    for i, j, n in sm.get_matching_blocks():
        if n >= min_match:
            blocks.append((i, j, n))
    return blocks


def _build_highlighted_html(tokens: list[str], highlights: dict[int, tuple[str, float, int, int]]) -> str:
    # highlights: word_idx -> (color, opacity, block_num, offset_in_block)

    # Pre-compute word index for each token position (-1 for whitespace tokens)
    tok_word_idx = []
    wi = 0
    for tok in tokens:
        if tok.strip():
            tok_word_idx.append(wi)
            wi += 1
        else:
            tok_word_idx.append(-1)

    def prev_word_idx(t):
        return next((tok_word_idx[pt] for pt in range(t - 1, -1, -1) if tokens[pt].strip()), -1)

    def next_word_idx(t):
        return next((tok_word_idx[nt] for nt in range(t + 1, len(tokens)) if tokens[nt].strip()), -1)

    def same_block(wi_a, wi_b):
        return wi_a in highlights and wi_b in highlights and highlights[wi_a][2] == highlights[wi_b][2]

    parts = []
    for t, tok in enumerate(tokens):
        if tok.strip():
            wi = tok_word_idx[t]
            if wi in highlights:
                color, opacity, block_num, offset = highlights[wi]
                style = f"background:{color};opacity:{opacity:.2f};padding:3px 2px;cursor:pointer"
                parts.append(f'<span style="{style}" data-block="{block_num}" data-idx="{offset}">{tok}</span>')
            else:
                parts.append(tok)
        else:
            pw = prev_word_idx(t)
            nw = next_word_idx(t)
            if pw in highlights and nw in highlights and same_block(pw, nw) and "\n" not in tok:
                color, opacity = highlights[pw][0], highlights[pw][1]
                parts.append(f'<span style="background:{color};opacity:{opacity:.2f};padding:3px 0">{tok}</span>')
            else:
                parts.append(tok.replace("\n", "<br>"))
    return "".join(parts)


def _ngrams(words: list[str], n: int = 5) -> frozenset[str]:
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def _compute_metrics(words1: list[str], words2: list[str], blocks: list, ngram_size: int = 5) -> dict:
    norm1 = [_norm(w) for w in words1 if _norm(w)]
    norm2 = [_norm(w) for w in words2 if _norm(w)]

    set1 = set(norm1)
    set2 = set(norm2)
    intersection = set1 & set2
    union = set1 | set2
    jaccard_unigram = len(intersection) / len(union) if union else 0.0
    overlap = len(intersection) / min(len(set1), len(set2)) if set1 and set2 else 0.0

    ng1 = _ngrams(norm1, ngram_size)
    ng2 = _ngrams(norm2, ngram_size)
    ng_inter = ng1 & ng2
    ng_union = ng1 | ng2
    jaccard_ngram = len(ng_inter) / len(ng_union) if ng_union else 0.0

    matched1 = sum(n for _, _, n in blocks)
    matched2 = sum(n for _, _, n in blocks)
    coverage1 = matched1 / len(words1) if words1 else 0.0
    coverage2 = matched2 / len(words2) if words2 else 0.0

    return {
        f"Jaccard ({ngram_size}-gram)": jaccard_ngram,
        "Jaccard (unigram)": jaccard_unigram,
        "Overlap coeff": overlap,
        "Coverage A": coverage1,
        "Coverage B": coverage2,
        "Matched blocks": len(blocks),
        "Matched words": matched1,
    }


def compare_texts(
    text1: str,
    text2: str,
    output_folder: Path,
    filename: str,
    label1: str = "Text 1",
    label2: str = "Text 2",
    min_match_words: int = 5,
    word_sim_threshold: float = 0.75,
    description: str = "",
) -> str:
    """
    Compare two texts and render a side-by-side HTML with matching passages highlighted.
    Matching blocks are highlighted in the same color in both panels.
    Within each block, per-word similarity controls highlight opacity.

    Returns the path to the generated HTML file.
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    words1 = [t for t in tokens1 if t.strip()]
    words2 = [t for t in tokens2 if t.strip()]

    blocks = _find_matching_blocks(words1, words2, min_match=min_match_words)

    # highlights[panel][word_idx] = (color, opacity)
    h1: dict[int, tuple[str, float]] = {}
    h2: dict[int, tuple[str, float]] = {}

    for block_num, (i, j, n) in enumerate(blocks):
        color = _COLORS[block_num % len(_COLORS)]
        for offset in range(n):
            w1 = words1[i + offset]
            w2 = words2[j + offset]
            sim = _word_similarity(w1, w2)
            if sim >= word_sim_threshold:
                opacity = 0.4 + 0.6 * sim  # 0.4 at threshold, 1.0 at exact match
                h1[i + offset] = (color, opacity, block_num, offset)
                h2[j + offset] = (color, opacity, block_num, offset)

    metrics = _compute_metrics(words1, words2, blocks)
    html1 = _build_highlighted_html(tokens1, h1)
    html2 = _build_highlighted_html(tokens2, h2)

    html = _render_html(html1, html2, label1, label2, metrics, description=description)

    output_folder.mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_folder, filename if filename.endswith(".html") else filename + ".html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


def compare_text_files(
    path1: Path,
    path2: Path,
    output_folder: str,
    filename: str,
    label1: Optional[str] = None,
    label2: Optional[str] = None,
    min_match_words: int = 5,
    word_sim_threshold: float = 0.75,
) -> str:
    """Like compare_texts but reads inputs from file paths. Labels default to filenames."""
    return compare_texts(
        text1=path1.read_text(encoding="utf-8"),
        text2=path2.read_text(encoding="utf-8"),
        output_folder=output_folder,
        filename=filename,
        label1=label1 or path1.name,
        label2=label2 or path2.name,
        min_match_words=min_match_words,
        word_sim_threshold=word_sim_threshold,
    )


def _render_html(html1: str, html2: str, label1: str, label2: str, metrics: dict, description: str = "") -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    metrics_html = "".join(
        f'<tr><td class="metric-name">{k}</td><td class="metric-value">{fmt(v)}</td></tr>'
        for k, v in metrics.items()
    )
    description_html = f'<div class="description-bar">{description}</div>\n' if description else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{label1} vs {label2}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital@0;1&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Inter', system-ui, sans-serif; background: #f0f4f8; height: 100vh; display: flex; flex-direction: column; }}

  header {{
    padding: 16px 28px;
    background: #fff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    display: flex;
    align-items: stretch;
    gap: 0;
    z-index: 10;
  }}
  .panel-label {{
    flex: 1;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #374151;
    padding: 6px 12px;
    border-radius: 8px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
  }}
  .panel-label:first-child {{ margin-right: 12px; }}
  .panel-label span {{
    display: block;
    font-size: 11px;
    font-weight: 400;
    color: #9ca3af;
    margin-top: 2px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }}

  .panels {{ display: flex; flex: 1; overflow: hidden; gap: 12px; padding: 12px; }}
  .panel {{
    flex: 1;
    overflow-y: auto;
    background: #fff;
    border-radius: 10px;
    padding: 32px 36px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .panel p {{
    font-family: 'Lora', Georgia, serif;
    font-size: 15px;
    line-height: 2;
    white-space: pre-wrap;
    word-wrap: break-word;
    color: #1f2937;
  }}

  span[data-block] {{ transition: outline 0.1s; }}
  span.active {{
    background: #1d4ed8 !important;
    opacity: 1 !important;
    color: #fff !important;
  }}

  .metrics-bar {{
    padding: 10px 28px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
  }}
  .metrics-bar table {{ border-collapse: collapse; font-size: 13px; }}
  .metrics-bar tr {{ display: inline-block; margin-right: 24px; }}
  .metric-name {{ font-weight: 600; color: #374151; }}
  .metric-name::after {{ content: ":"; }}
  .metric-value {{ color: #374151; padding-left: 4px; }}

  .description-bar {{
    padding: 8px 28px;
    background: #fffbeb;
    border-bottom: 1px solid #fde68a;
    font-size: 12px;
    color: #78350f;
    white-space: pre-wrap;
    font-family: 'Inter', monospace;
  }}
</style>
</head>
<body>
<header>
  <div class="panel-label">{label1}<span>Text A</span></div>
  <div class="panel-label">{label2}<span>Text B</span></div>
</header>
{description_html}<div class="metrics-bar"><table><tbody>{metrics_html}</tbody></table></div>
<div class="panels">
  <div class="panel" id="panel-0"><p>{html1}</p></div>
  <div class="panel" id="panel-1"><p>{html2}</p></div>
</div>
<script>
  const panels = [document.getElementById('panel-0'), document.getElementById('panel-1')];

  document.querySelectorAll('span[data-block]').forEach(span => {{
    span.addEventListener('click', () => {{
      document.querySelectorAll('span.active').forEach(s => s.classList.remove('active'));
      const block = span.dataset.block;
      const idx = span.dataset.idx;
      const myPanel = span.closest('.panel');
      const otherPanel = panels.find(p => p !== myPanel);
      const match = otherPanel.querySelector(`span[data-block="${{block}}"][data-idx="${{idx}}"]`);
      span.classList.add('active');
      if (match) {{
        match.classList.add('active');
        // Scroll other panel so match appears at same visual height as clicked word
        const clickedY = span.getBoundingClientRect().top - myPanel.getBoundingClientRect().top;
        const matchAbsPos = match.getBoundingClientRect().top - otherPanel.getBoundingClientRect().top + otherPanel.scrollTop;
        otherPanel.scrollTop = matchAbsPos - clickedY;
      }}
    }});
  }});
</script>
</body>
</html>"""


if __name__ == "__main__":
    from datetime import datetime
    text_dir = Path("/Users/Elena.Lyulina/PycharmProjects/swiss-ai/Megatron-LM-Attention-Benchmark/attn_bench/results/data/test/text-match/texts")
    pairs = [
        ("marriage", "analytical-studies.txt", "the-physiology-of-marriage-complete.txt"),
        ("bartleby", "bartleby-the-scrivener-a-story-of-wall-street.txt", "the-piazza-tales.txt"),
        ("shakespeare", "shakespeares-sonnets.txt", "the-complete-works-of-william-shakespeare.txt"),
        ("types", "twelve-types.txt", "varied-types.txt")
    ]
    output = text_dir / "matches"

    for pair in pairs:
        name, filename1, filename2 = pair
        path1 = text_dir / filename1
        path2 = text_dir / filename2

        filename = f"sim-0d438_{name}_book-ids-6600-6601_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        out = compare_text_files(path1, path2, output, filename)
        print(out)

    # --- CLI ---
    # parser = argparse.ArgumentParser(description="Compare two text files side-by-side with matching passages highlighted.")
    # parser.add_argument("path1", help="Path to first text file")
    # parser.add_argument("path2", help="Path to second text file")
    # parser.add_argument("output_folder", help="Folder to write the HTML file into")
    # parser.add_argument("filename", help="Output filename (with or without .html)")
    # parser.add_argument("--label1", default=None, help="Label for first text (default: filename)")
    # parser.add_argument("--label2", default=None, help="Label for second text (default: filename)")
    # parser.add_argument("--min-match-words", type=int, default=5, help="Minimum block length to highlight (default: 5)")
    # parser.add_argument("--word-sim-threshold", type=float, default=0.75, help="Per-word similarity threshold for highlighting within a block (default: 0.75)")
    # args = parser.parse_args()
    #
    # out = compare_text_files(
    #     path1=args.path1,
    #     path2=args.path2,
    #     output_folder=args.output_folder,
    #     filename=args.filename,
    #     label1=args.label1,
    #     label2=args.label2,
    #     min_match_words=args.min_match_words,
    #     word_sim_threshold=args.word_sim_threshold,
    # )
    # print(out)


"""
Detokenize the long-context corpora to raw-text jsonl, for the lm-eval perplexity tasks.

They are stored as `input_ids` only, but lm-eval's `loglikelihood_rolling` takes text: it
re-tokenizes what it is given, and `word_perplexity` divides by the raw text's word count.
The decoded span matches what the position-wise inference scripts score, minus the BOS
lm-eval prepends itself; for FineWeb it is the same seeded 660-document subsample.

Usage:
    python -m attn_bench.data_processing.detokenize_long_corpora \
        --corpus gutenberg --tokenizer-path $TOKENIZER_PATH \
        --data-file $GUTENBERG_LONG_JSONL_DIR/rep_0_token.jsonl \
        --output $TEXT_DIR/gutenberg_rep0.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from attn_bench.evaluation.inference_common import BOS_TOKEN_ID
from attn_bench.evaluation.long_inference import sample_lines

FINEWEB_DEFAULT_MAX_SAMPLES = 660   # matches long_fineweb_inference.py
DEFAULT_MAX_LENGTH = 32768          # 4x the training sequence length

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


### SPAN SELECTION ###

def gutenberg_span(record: dict) -> tuple[list[int], str]:
    """[sample, suffix] token span of one long Gutenberg record, plus its book id."""
    start = record["sample_offset"]
    length = record["sample_len"] + record["extra_suffix_len"]
    tokens = record["input_ids"][start:start + length]
    assert len(tokens) == length, f"short record: {len(tokens)} != {length}"
    return tokens, str(record["book_id"])


def fineweb_span(record: dict) -> tuple[list[int], str]:
    """One long FineWeb-Edu document, leading BOS dropped, plus its doc id."""
    tokens = record["input_ids"]
    if tokens and tokens[0] == BOS_TOKEN_ID:
        tokens = tokens[1:]
    return tokens, record["doc_id"]


SPANS = {"gutenberg": gutenberg_span, "fineweb": fineweb_span}


### MAIN ###

def build(data_file: Path, corpus: str, tokenizer, max_length: int, max_samples: int | None):
    """Decode every selected record; return (rows, stats). Rows are {doc_id, text}.

    The faithfulness check is whether the text is a fixed point, decode(encode(text)) == text.
    A handful of documents also re-segment, harmlessly: the stored ids are non-canonical where
    the corpora were tokenized in pieces, so BPE re-merges across those seams.
    """
    # Must be off: the Llama-3.2 config sets it True, which deletes the space before punctuation
    # ("Dry Season .5" -> "Dry Season.5"), corrupting the text and the word count alike.
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
    span_fn = SPANS[corpus]
    rows, token_lengths, word_counts, byte_counts = [], [], [], []
    unstable, resegmented, deltas = 0, 0, []

    for line in sample_lines(data_file, max_samples):
        tokens, doc_id = span_fn(json.loads(line))
        tokens = tokens[:max_length]
        text = decode(tokens)

        reencoded = tokenizer.encode(text, add_special_tokens=False)
        if decode(reencoded) != text:
            unstable += 1
        if reencoded != tokens:
            resegmented += 1
            deltas.append(abs(len(reencoded) - len(tokens)) / len(tokens))

        rows.append({"doc_id": doc_id, "text": text})
        token_lengths.append(len(tokens))
        # lm-eval's own expression, without .strip(): word_perplexity's denominator.
        word_counts.append(len(re.split(r"\s+", text)))
        byte_counts.append(len(text.encode("utf-8")))

    lengths = np.array(token_lengths)
    stats = {
        "corpus": corpus,
        "source_file": str(data_file),
        "max_length": max_length,
        "max_samples": max_samples,
        "n_docs": len(rows),
        "total_tokens": int(lengths.sum()),
        "total_words": int(sum(word_counts)),
        "total_bytes": int(sum(byte_counts)),
        "token_length": {
            "min": int(lengths.min()), "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            **{f"p{p}": int(v) for p, v in zip(PERCENTILES, np.percentile(lengths, PERCENTILES))},
        },
        "roundtrip": {
            "text_not_a_fixed_point": unstable,   # must be 0; anything else means a lossy decode
            "resegmented_docs": resegmented,      # informational, see build()
            "token_delta_share": {
                "mean": float(np.mean(deltas)) if deltas else 0.0,
                "p50": float(np.median(deltas)) if deltas else 0.0,
                "max": float(np.max(deltas)) if deltas else 0.0,
            },
        },
    }
    return rows, stats


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, choices=sorted(SPANS),
                   help="Which span rule to apply (see module docstring)")
    p.add_argument("--data-file", required=True, help="Tokenized jsonl to decode")
    p.add_argument("--tokenizer-path", required=True, help="Llama-3.2-1B tokenizer directory")
    p.add_argument("--output", required=True, help="Output text jsonl; stats land beside it")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help=f"Cap each document to this many content tokens (default: {DEFAULT_MAX_LENGTH})")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Subsample to this many documents, same fixed seed as the position-wise "
                        f"sweep. Default: all for gutenberg, {FINEWEB_DEFAULT_MAX_SAMPLES} for fineweb.")
    p.add_argument("--overwrite", action="store_true", help="Rebuild even if --output exists")
    args = p.parse_args()
    if args.max_samples is None and args.corpus == "fineweb":
        args.max_samples = FINEWEB_DEFAULT_MAX_SAMPLES
    return args


def main():
    args = parse_args()
    out_path = Path(args.output)
    stats_path = out_path.with_name(out_path.stem + "_stats.json")
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} already exists -- pass --overwrite to rebuild. Exiting.")
        return

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    rows, stats = build(Path(args.data_file), args.corpus, tokenizer,
                        args.max_length, args.max_samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"\nWrote {len(rows)} documents -> {out_path}")
    print(f"Wrote stats -> {stats_path}")
    rt = stats["roundtrip"]
    if rt["text_not_a_fixed_point"]:
        print(f"WARNING: {rt['text_not_a_fixed_point']}/{stats['n_docs']} documents are NOT a "
              f"decode/encode fixed point -- the decode lost information. lm-eval scores the text, "
              f"so do not trust these perplexities until this is 0.")
    else:
        print(f"Decode is faithful: all {stats['n_docs']} documents are fixed points. "
              f"BPE re-segments {rt['resegmented_docs']} (mean "
              f"{rt['token_delta_share']['mean']:.4%} of tokens).")


if __name__ == "__main__":
    main()

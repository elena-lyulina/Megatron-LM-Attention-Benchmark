"""
Check whether the datatrove tokenizer (tokenizers.Tokenizer) adds BOS/EOS
for the Llama 3.2 tokenizer — mirrors the exact call in BudgetedMegatronDocumentTokenizer.

Usage:
    python check_special_tokens.py --tokenizer-path /path/to/llama-3.2-1b
"""

import argparse
from pathlib import Path
from tokenizers import Tokenizer

DOCS = [
    "Hello, this is a test document.",
    "Another document to check boundaries.",
]


def main(args):
    tok = Tokenizer.from_file(str(Path(args.tokenizer_path) / "tokenizer.json"))
    bos_id = tok.token_to_id("<|begin_of_text|>")
    eos_id = tok.token_to_id("<|end_of_text|>")
    print(f"Tokenizer vocab size: {tok.get_vocab_size()}")
    print(f"BOS token id: {bos_id}  (<|begin_of_text|>)")
    print(f"EOS token id: {eos_id}  (<|end_of_text|>)")
    print()

    encodings = tok.encode_batch(DOCS)
    for i, (doc, enc) in enumerate(zip(DOCS, encodings)):
        ids = enc.ids
        has_bos = ids[0] == bos_id
        has_eos = ids[-1] == eos_id
        print(f"Doc {i}: {doc!r}")
        print(f"  ids (first 5): {ids[:5]}")
        print(f"  ids (last  5): {ids[-5:]}")
        print(f"  BOS at start:  {has_bos}")
        print(f"  EOS at end:    {has_eos}")
        print()

    has_bos_all = all(enc.ids[0] == bos_id for enc in encodings)
    has_eos_all = all(enc.ids[-1] == eos_id for enc in encodings)
    print("=" * 50)
    print(f"RESULT — BOS added: {has_bos_all}, EOS added: {has_eos_all}")
    if not has_eos_all:
        print("WARNING: no EOS between documents — --eod-mask-loss / --reset-attention-mask will have no effect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", required=True)
    main(parser.parse_args())
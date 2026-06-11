# Computes distinct-n (token-level) and perplexity under a reference HF model
# for sparse Gutenberg inference results.
# Same auto-discovery and skip-if-exists pattern as compute_mauve.py.
# Outputs per (offset, prefix, suffix, policy) group:
#   offset_X_prefix_Y_suffix_Z_policy_distinct_n.json    {rep: {"1": score, "2": score, "3": score}}
#   offset_X_prefix_Y_suffix_Z_policy_perplexity.json    {rep: score}

import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verbatim_eval.utils import load_inference_data


# ### distinct-n ###

def distinct_n(sequences, n):
    """Fraction of unique token n-grams across all sequences."""
    all_ngrams = []
    for seq in sequences:
        all_ngrams.extend(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    if not all_ngrams:
        return 0.0
    return round(len(set(all_ngrams)) / len(all_ngrams), 4)


def compute_distinct_n_for_rep(data, ns=(1, 2, 3)):
    seqs = data['generated_suffix']
    return {str(n): distinct_n(seqs, n) for n in ns}


# ### perplexity ###

def compute_perplexity_for_rep(data, ref_model, ref_tokenizer, src_tokenizer, max_length=512, batch_size=16):
    """Average per-token perplexity of generated suffixes under ref_model."""
    generated_ids = data['generated_suffix']
    texts = [src_tokenizer.decode(ids) for ids in generated_ids]

    total_nll = 0.0
    total_tokens = 0

    ref_model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = ref_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(ref_model.device)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100

            out = ref_model(**enc, labels=labels)
            # out.loss is mean NLL over non-padded tokens; recover sum
            n_tokens = (labels != -100).sum().item()
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    return round(math.exp(total_nll / total_tokens), 3)


# ### discovery (shared with compute_mauve.py) ###
# looks for all combinations of offset, prefix, suffix inside the model's folder

def discover_groups(inference_dir):
    groups = defaultdict(list)
    for param_dir in sorted(inference_dir.iterdir()):
        if not param_dir.is_dir() or not param_dir.name.startswith("offset_"):
            continue
        parts = param_dir.name.split("_")
        try:
            offset, prefix, suffix = int(parts[1]), int(parts[3]), int(parts[5])
        except (IndexError, ValueError):
            continue
        for rep_dir in sorted(param_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                continue
            rparts = rep_dir.name.split("_", 2)
            try:
                rep, policy = int(rparts[1]), rparts[2]
            except (IndexError, ValueError):
                continue
            if not list(rep_dir.glob("rank*.jsonl")):
                continue
            groups[(offset, prefix, suffix, policy)].append(rep)
    return {k: sorted(v) for k, v in groups.items()}


def load_ref_model(model_name):
    print(f"Loading reference model {model_name} ...")
    ref_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    return ref_model, ref_tokenizer


def process_group(expr_dir, inference_dir, offset, prefix, suffix, policy, reps, args, src_tokenizer, ref_state):
    stem = f"offset_{offset}_prefix_{prefix}_suffix_{suffix}_{policy}"
    dn_file = expr_dir / f"{stem}_distinct_n.json"
    ppl_file = expr_dir / f"{stem}_perplexity.json"

    need_dn = not dn_file.exists()
    need_ppl = not ppl_file.exists()
    if not need_dn and not need_ppl:
        print(f"[SKIP] already computed: {expr_dir.name} {stem}")
        return

    print(f"\n=== {expr_dir.name}  offset={offset} prefix={prefix} suffix={suffix} policy={policy} reps={reps} ===")

    if need_ppl and ref_state['model'] is None:
        ref_state['model'], ref_state['tokenizer'] = load_ref_model(args.ref_model)

    dn_scores, ppl_scores = {}, {}
    for rep in reps:
        data = load_inference_data(
            base_dir=str(inference_dir),
            offset=offset, len_prefix=prefix, len_suffix=suffix, rep=rep, policy=policy,
        )
        if need_dn:
            dn_scores[str(rep)] = compute_distinct_n_for_rep(data)
            print(f"  distinct-n (rep={rep}): {dn_scores[str(rep)]}")
        if need_ppl:
            ppl_scores[str(rep)] = compute_perplexity_for_rep(
                data, ref_state['model'], ref_state['tokenizer'], src_tokenizer,
                max_length=args.perplexity_max_length, batch_size=args.perplexity_batch_size,
            )
            print(f"  perplexity (rep={rep}): {ppl_scores[str(rep)]}")

    for file, scores in [(dn_file, dn_scores), (ppl_file, ppl_scores)]:
        if scores:
            with open(file, "w") as f:
                json.dump(scores, f, indent=4)
            print(f"Saved: {file}")


def process_expr(expr, base_path, args, src_tokenizer, ref_state):
    expr_dir = base_path / expr
    inference_dir = expr_dir / "inference"
    if not inference_dir.exists():
        print(f"[SKIP] inference dir not found: {inference_dir}")
        return

    groups = discover_groups(inference_dir)
    if not groups:
        print(f"[SKIP] no inference data found under {inference_dir}")
        return

    for (offset, prefix, suffix, policy), reps in groups.items():
        if args.offsets and offset not in args.offsets:
            continue
        if args.prefix_lengths and prefix not in args.prefix_lengths:
            continue
        if args.suffix_lengths and suffix not in args.suffix_lengths:
            continue
        process_group(expr_dir, inference_dir, offset, prefix, suffix, policy, reps, args, src_tokenizer, ref_state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute distinct-n and perplexity for inference results")
    parser.add_argument("--exprs", type=str, nargs="+", required=True)
    parser.add_argument("--base-path", type=str, required=True, help="e.g. $MEM_BASE/SparseGutenberg")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer used during inference (for decoding)")
    parser.add_argument("--ref-model", type=str, default="Qwen/Qwen2.5-1.5B", help="Reference model for perplexity")
    parser.add_argument("--offsets", type=int, nargs="+", default=None, help="Filter by offset values; if omitted, all discovered offsets are used")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=None, help="Filter by prefix lengths; if omitted, all discovered values are used")
    parser.add_argument("--suffix-lengths", type=int, nargs="+", default=None, help="Filter by suffix lengths; if omitted, all discovered values are used")
    parser.add_argument("--perplexity-max-length", type=int, default=512)
    parser.add_argument("--perplexity-batch-size", type=int, default=16)
    args = parser.parse_args()

    src_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    src_tokenizer.model_max_length = 200_000
    src_tokenizer.pad_token_id = src_tokenizer.eos_token_id

    ref_state = {'model': None, 'tokenizer': None}  # lazy-loaded on first group that needs perplexity

    for expr in args.exprs:
        process_expr(expr, Path(args.base_path), args, src_tokenizer, ref_state)
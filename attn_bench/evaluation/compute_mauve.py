# Copied from PDM/src/verbatim_eval/compute_mauve.py and adapted for this project:
#   - accepts a list of experiment names and auto-discovers (offset, prefix, suffix, rep, policy)
#     from the inference directory structure, no need to pass them explicitly
#   - skips experiments where the output JSON already exists
#   - all reps for a given (offset, prefix, suffix, policy) are collected into one JSON
#     named offset_X_prefix_Y_suffix_Z_policy_mauve.json, saved alongside the existing .pkl
#   - --tokenizer-path arg instead of hardcoded HF model

import os
import json
from pathlib import Path
from collections import defaultdict
from evaluate import load as load_metric
from verbatim_eval.utils import load_inference_data
from transformers import AutoTokenizer
import argparse


def compute_mauve_for_rep(data, tokenizer, rep, sample_size=None):
    os.environ['OPENBLAS_NUM_THREADS'] = '128'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'

    true_suffix = data['true_suffix']
    generated_suffix = data['generated_suffix']
    if sample_size is not None and sample_size < len(true_suffix):
        true_suffix = true_suffix[:sample_size]
        generated_suffix = generated_suffix[:sample_size]

    print(f"  Decoding {len(true_suffix)} samples...")
    true_texts = [tokenizer.decode(ids) for ids in true_suffix]
    generated_texts = [tokenizer.decode(ids) for ids in generated_suffix]

    mauve = load_metric('mauve')
    result = mauve.compute(predictions=generated_texts, references=true_texts, device_id=0, verbose=True)
    score = round(float(result.mauve), 3)
    print(f"  MAUVE (rep={rep}): {score}")
    return score


def discover_groups(inference_dir):
    """
    Scan inference_dir for offset_*/rep_* subdirs and return a dict:
      (offset, prefix, suffix, policy) -> sorted list of rep ints
    """
    groups = defaultdict(list)
    for param_dir in sorted(inference_dir.iterdir()):
        if not param_dir.is_dir() or not param_dir.name.startswith("offset_"):
            continue
        # parse offset_0_prefix_500_suffix_500
        parts = param_dir.name.split("_")
        try:
            offset = int(parts[1])
            prefix = int(parts[3])
            suffix = int(parts[5])
        except (IndexError, ValueError):
            continue
        for rep_dir in sorted(param_dir.iterdir()):
            if not rep_dir.is_dir() or not rep_dir.name.startswith("rep_"):
                continue
            # parse rep_0_greedy
            rparts = rep_dir.name.split("_", 2)
            try:
                rep = int(rparts[1])
                policy = rparts[2]
            except (IndexError, ValueError):
                continue
            if not list(rep_dir.glob("rank*.jsonl")):
                continue
            groups[(offset, prefix, suffix, policy)].append(rep)
    return {k: sorted(v) for k, v in groups.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MAUVE scores for sparse Gutenberg inference results")
    parser.add_argument("--exprs", type=str, nargs="+", required=True, help="Experiment names")
    parser.add_argument("--base-path", type=str, required=True, help="Base path (e.g. $MEM_BASE/SparseGutenberg)")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to tokenizer directory")
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.model_max_length = 200_000
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_path = Path(args.base_path)

    for expr in args.exprs:
        expr_dir = base_path / expr
        inference_dir = expr_dir / "inference"
        if not inference_dir.exists():
            print(f"[SKIP] inference dir not found: {inference_dir}")
            continue

        groups = discover_groups(inference_dir)
        if not groups:
            print(f"[SKIP] no inference data found under {inference_dir}")
            continue

        for (offset, prefix, suffix, policy), reps in groups.items():
            out_file = expr_dir / f"offset_{offset}_prefix_{prefix}_suffix_{suffix}_{policy}_mauve.json"
            if out_file.exists():
                print(f"[SKIP] already computed: {out_file}")
                continue

            print(f"\n=== {expr}  offset={offset} prefix={prefix} suffix={suffix} policy={policy} reps={reps} ===")
            scores = {}
            for rep in reps:
                data = load_inference_data(
                    base_dir=str(inference_dir),
                    offset=offset,
                    len_prefix=prefix,
                    len_suffix=suffix,
                    rep=rep,
                    policy=policy,
                )
                scores[str(rep)] = compute_mauve_for_rep(
                    data=data,
                    tokenizer=tokenizer,
                    rep=rep,
                    sample_size=args.sample_size,
                )

            with open(out_file, "w") as f:
                json.dump(scores, f, indent=4)
            print(f"Saved: {out_file}")

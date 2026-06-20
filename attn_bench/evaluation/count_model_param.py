"""
Count the parameters of a Megatron model built from a flag file.

The file holds Megatron flags (like a slurm MODEL_ARGS array). See examples at
attn_bench/data/param_count_examples/. The model is built through the real
gpt_builder so the count is exact. Prints the total, a category breakdown
(embeddings / mlp / mixer -- mixer is attention/GDN/etc. by subtraction, the row
that differs across architectures), then the per-param breakdown (same-named
params summed across layers). Compare two models by diffing runs.

Needs a GPU + transformer_engine + fla + causal_conv1d (run in the container),
and a distributed rank env -- launch under torchrun --nproc_per_node=1:

    torchrun --nproc_per_node=1 attn_bench/evaluation/count_model_param.py args_gdn.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from megatron.core import parallel_state as mpu
from megatron.training.arguments import (
    core_transformer_config_from_args,
    parse_args,
    validate_args,
)
from megatron.training.global_vars import set_global_variables

# Repo root holds the top-level gpt_builders module and megatron/.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from gpt_builders import gpt_builder  # noqa: E402

# NullTokenizer vocab size (llama3). Only shifts embedding/output size; pick the
# same value when comparing two models so the embedding rows match.
VOCAB_SIZE = 128256


def _read_flags(path: str) -> list[str]:
    """Read a flag file, dropping blank lines and '#' comments."""
    import shlex
    flags: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            flags.extend(shlex.split(line))
    return flags


def build_model(config_path: str):
    # Overrides appended last so argparse keeps them:
    #  - NullTokenizer: no tokenizer files needed.
    #  - all parallel sizes 1: count the full model, not a per-rank shard.
    overrides = [
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(VOCAB_SIZE),
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1",
        "--expert-model-parallel-size", "1",
    ]
    sys.argv = [sys.argv[0]] + _read_flags(config_path) + overrides

    args = parse_args(ignore_unknown_args=True)
    # These don't affect param count; they only exist to satisfy validate_args.
    for key, value in {"micro_batch_size": 1, "global_batch_size": 1,
                       "train_iters": 1, "seq_length": 2048,
                       "max_position_embeddings": 2048}.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    validate_args(args)

    # Single-rank init (launched under torchrun --nproc_per_node=1).
    set_global_variables(args, build_tokenizer=True)  # also sets padded_vocab_size
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    mpu.initialize_model_parallel(1, 1)

    config = core_transformer_config_from_args(args)
    config.perform_initialization = False  # counting shapes, skip RNG init
    return args, gpt_builder(args, pre_process=True, post_process=True, config=config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="file with Megatron flags for the model")
    cli = ap.parse_args()

    args, model = build_model(cli.config)

    # Category counts (embeddings/head, mlp/ffn) + the per-param breakdown, with
    # the layer index collapsed so each layer's same-named params are summed.
    # "mlp" matches the linears and the pre-mlp norm (fused or not); anything not
    # embeddings/mlp is the mixer by subtraction -- robust without knowing what
    # each architecture calls its params, and the row that differs across them.
    # Attention-block norms count as mixer; only the standalone final_layernorm
    # is a stray there (tiny, and identical across models so it cancels).
    cats = {"embeddings + output head": 0, "mlp / ffn": 0, "mixer (rest)": 0}
    grouped: dict[str, int] = {}
    for name, p in model.named_parameters():
        if "embedding" in name or "output_layer" in name:
            cats["embeddings + output head"] += p.numel()
        elif "mlp" in name:
            cats["mlp / ffn"] += p.numel()
        else:
            cats["mixer (rest)"] += p.numel()
        key = re.sub(r"\.\d+\.", ".*.", name)
        grouped[key] = grouped.get(key, 0) + p.numel()
    total = sum(cats.values())

    tied = not getattr(args, "untie_embeddings_and_output_weights", False)
    w = max(len(k) for k in list(cats) + list(grouped) + ["TOTAL"])
    lines = [f"=== Parameter count: {Path(cli.config).stem} ===",
             f"(embeddings {'tied' if tied else 'untied'} with output head)",
             "",
             f"{'TOTAL':<{w}}  {total:>15,}",
             "",
             "By category:"]
    for k, v in cats.items():
        lines.append(f"{k:<{w}}  {v:>15,}")
    lines += ["", "By param (layer index collapsed):"]
    for k in sorted(grouped):
        lines.append(f"{k:<{w}}  {grouped[k]:>15,}")

    report = "\n".join(lines)
    print("\n" + report)

    # Write next to the args file as <args_file_stem>_param_counts.
    out_path = Path(cli.config).with_name(Path(cli.config).stem + "_param_counts")
    out_path.write_text(report + "\n")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
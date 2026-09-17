"""Paired complete-vs-truncated GDN state comparison on Gutenberg excerpts.

For every selected book and truncated point (offset O, prefix P) the model is run twice,
teacher-forced on the same 249-token suffix at g = O+P:

    full:      x[0 : g+S]                (the matched complete-context control)
    truncated: [BOS] + x[O : g+S]

Both branches consume identical text from original position O onwards, so the recurrent
state after c common-prefix tokens (c in 64/256/1024/P) and after t true suffix tokens
(t in 5/50/248) can be compared within the same checkpoint: norms, distance, relative
distance, cosine, and the same for the recurrent output at that token. Suffix NLL, argmax
match and true-vs-best-other margin are kept per position so the state gap can be related
to the loss penalty and the first teacher-forced argmax mismatch per book.

States are read the same way as gdn_state_norm.py (segment the chunk kernel and carry the
state), but cut at the requested boundaries instead of every N tokens, and kept per book
instead of averaged. Only scalar statistics are saved; the reference states live on the GPU
for one book at a time. Single GPU, batch 1, Megatron checkpoint backend.

Plan: attn_bench/_plans/gdn_paired_state_probe_plan.md

Usage (via torchrun, 1 GPU):
    torchrun --nproc_per_node=1 attn_bench/evaluation/gdn_paired_state.py \
        --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $OUT_DIR --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 32,64,256 --points 50:2000 [--max-books 4] [--test] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from attn_bench.evaluation.inference_backend import MegatronBackend
from attn_bench.evaluation.inference_common import (BOS_TOKEN_ID,
                                                    find_rep_paths,
                                                    parse_points)

SEED = 20260917
PREFIX_EVENTS = (64, 256, 1024)  # common-prefix tokens consumed; P itself is always added
SUFFIX_EVENTS = (5, 50, 248)     # true suffix tokens consumed
EPS = 1e-8                       # denominator floor for relative distance / cosine


### DATA ###

def select_books(n_rows: int, rep: int, max_books: int | None) -> list[int]:
    """First max_books rows of one seeded permutation per bucket, so a 4-book test run, a
    64-book pilot and a full run are nested subsets of each other. Sorted for file order."""
    order = np.random.default_rng(np.random.SeedSequence([SEED, rep])).permutation(n_rows)
    n = n_rows if max_books is None else min(max_books, n_rows)
    return sorted(order[:n].tolist())


def load_books(path: Path, sample_ids: list[int]) -> list[tuple[int, list[int]]]:
    wanted = set(sample_ids)
    books = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                books.append((i, json.loads(line)["input_ids"]))
    assert len(books) == len(sample_ids), f"{path.name}: {len(books)} of {len(sample_ids)} rows found"
    return books


def count_rows(path: Path) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def token_checksum(tokens: list[int]) -> int:
    return zlib.crc32(np.asarray(tokens, dtype=np.int64).tobytes())


def build_pair(tokens: list[int], offset: int, prefix: int, suffix: int):
    """Full and truncated scored sequences plus the kernel-input boundaries (number of input
    tokens consumed) at which the state is read in each branch. Boundaries are aligned: the
    i-th full boundary and the i-th truncated boundary end on the same original token."""
    g = offset + prefix
    assert offset > 0, "offset 0 has no missing history, nothing to pair"
    assert tokens[0] == BOS_TOKEN_ID, "bucket rows are expected to start with BOS"
    assert len(tokens) >= g + suffix, f"row too short: {len(tokens)} < {g + suffix}"

    full = tokens[:g + suffix]
    truncated = [BOS_TOKEN_ID] + tokens[offset:g + suffix]

    prefix_counts = [c for c in PREFIX_EVENTS if c < prefix] + [prefix]
    suffix_counts = [t for t in SUFFIX_EVENTS if t <= suffix - 1]
    events = [("prefix", c, offset + c) for c in prefix_counts] + [("suffix", t, g + t) for t in suffix_counts]
    full_b = [offset + c for c in prefix_counts] + [g + t for t in suffix_counts]
    trunc_b = [1 + c for c in prefix_counts] + [1 + prefix + t for t in suffix_counts]

    # model input is seq[:-1]; every boundary must fit in it and end on the same token
    assert full_b[-1] <= len(full) - 1 and trunc_b[-1] <= len(truncated) - 1
    for fb, tb in zip(full_b, trunc_b):
        assert full[fb - 1] == truncated[tb - 1], "misaligned boundary"
    assert full[-suffix:] == truncated[-suffix:], "suffix labels differ"
    return full, truncated, full_b, trunc_b, events


### OBSERVER ###

class StateProbe:
    """Swaps gated_delta_rule on every GatedDeltaNet for a wrapper that runs the kernel in
    segments ending at self.boundaries and keeps each segment's final state and the
    recurrent output of its last token. Set boundaries, run one forward, call take()."""

    def __init__(self, model):
        from megatron.core.ssm.gated_delta_net import GatedDeltaNet
        self.modules = [m for m in model.modules() if isinstance(m, GatedDeltaNet)]
        assert self.modules, "model has no GatedDeltaNet layers"
        assert not any(m.training for m in self.modules), "model must be in eval mode"
        self.layer_ids = [m.layer_number for m in self.modules]
        self.boundaries = None
        self._originals = {}
        self._states = {}
        self._outputs = {}

    def install(self):
        assert not self._originals, "already installed"
        for m in self.modules:
            self._originals[m.layer_number] = m.gated_delta_rule
            m.gated_delta_rule = self._wrap(m.gated_delta_rule, m.layer_number)

    def restore(self):
        for m in self.modules:
            if m.layer_number in self._originals:
                m.gated_delta_rule = self._originals[m.layer_number]
        self._originals = {}
        self._states = {}
        self._outputs = {}

    def take(self):
        """Returns states [L, E, H, K, V] and outputs [L, E, H, V] (float32, on GPU) for the
        last forward and clears the buffers."""
        n = len(self.boundaries)
        for lid in self.layer_ids:
            assert lid in self._states and self._states[lid].shape[0] == n, f"layer {lid}: incomplete capture"
        states = torch.stack([self._states[lid] for lid in self.layer_ids])
        outputs = torch.stack([self._outputs[lid] for lid in self.layer_ids])
        self._states = {}
        self._outputs = {}
        return states, outputs

    def _wrap(self, real_fn, layer_number):
        def wrapper(query, key, value, *, g, beta, initial_state=None,
                    output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
            assert cu_seqlens is None and initial_state is None, "packed input / initial state not supported"
            assert query.shape[0] == 1, "batch size must be 1"
            assert self.boundaries, "set probe.boundaries before the forward"
            seq_len = query.shape[1]
            wanted = set(self.boundaries)
            assert max(wanted) <= seq_len, f"boundary beyond input length {seq_len}"
            ends = sorted(wanted | {seq_len})

            outs, states, out_at = [], [], []
            state = None
            start = 0
            for end in ends:
                out_s, state = real_fn(
                    query[:, start:end], key[:, start:end], value[:, start:end],
                    g=g[:, start:end], beta=beta[:, start:end],
                    initial_state=state, output_final_state=True,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=None,
                )
                outs.append(out_s)
                if end in wanted:
                    states.append(state[0].float().clone())   # [H, K, V]
                    out_at.append(out_s[0, -1].float().clone())  # [H, V]
                start = end
            self._states[layer_number] = torch.stack(states)
            self._outputs[layer_number] = torch.stack(out_at)
            return torch.cat(outs, dim=1), (state if output_final_state else None)
        return wrapper


### FORWARD ###

@torch.no_grad()
def score(backend, tokens: list[int], suffix: int) -> dict:
    """One teacher-forced forward; per-position suffix NLL, argmax match and margin
    (true logit minus best other logit), plus the first argmax mismatch index K
    (K == suffix when every position matches)."""
    device = backend.device
    seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    inputs, labels = seq[:, :-1], seq[:, 1:]
    pos = torch.arange(inputs.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    logits = backend.forward_logits(inputs, pos, fp32_output=False)
    suffix_logits = logits[0, -suffix:].float()  # slice first, then cast (see compute_nll)
    del logits
    suffix_labels = labels[0, -suffix:]

    nll = -F.log_softmax(suffix_logits, dim=-1).gather(1, suffix_labels[:, None]).squeeze(1)
    true_logit = suffix_logits.gather(1, suffix_labels[:, None]).squeeze(1)
    others = suffix_logits.scatter(1, suffix_labels[:, None], float("-inf"))
    margin = true_logit - others.max(dim=1).values
    argmax_match = suffix_logits.argmax(dim=1) == suffix_labels
    mismatch = (~argmax_match).nonzero()
    first_mismatch = int(mismatch[0]) if len(mismatch) else suffix
    return {
        "nll": nll.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "argmax_match": argmax_match.cpu().numpy(),
        "first_mismatch": first_mismatch,
    }


def paired_stats(state_full, state_trunc, out_full, out_trunc) -> dict:
    """Per [L, E, H] scalars comparing the truncated branch to the full one."""
    sf, st = state_full.flatten(-2), state_trunc.flatten(-2)
    norm_full, norm_trunc = sf.norm(dim=-1), st.norm(dim=-1)
    distance = (st - sf).norm(dim=-1)
    out_norm_full, out_norm_trunc = out_full.norm(dim=-1), out_trunc.norm(dim=-1)
    return {
        "norm_full": norm_full,
        "norm_trunc": norm_trunc,
        "distance": distance,
        "relative_distance": distance / norm_full.clamp_min(EPS),
        "cosine": (sf * st).sum(-1) / (norm_full * norm_trunc).clamp_min(EPS),
        "near_zero": (norm_full < EPS) | (norm_trunc < EPS),  # relative_distance/cosine unreliable
        "out_norm_full": out_norm_full,
        "out_norm_trunc": out_norm_trunc,
        "out_distance": (out_trunc - out_full).norm(dim=-1),
    }


### RUN ###

def result_path(output_dir: Path, rep: int, offset: int, prefix: int) -> Path:
    return output_dir / f"rep_{rep}_offset_{offset}_prefix_{prefix}.npz"


def result_done(locations, rep, offset, prefix, sample_ids) -> bool:
    """Done when a file in any location already holds every requested book."""
    for loc in locations:
        if loc is None:
            continue
        path = result_path(loc, rep, offset, prefix)
        if path.exists() and set(sample_ids) <= set(np.load(path)["sample_idx"].tolist()):
            return True
    return False


def run_point(backend, probe, books, offset, prefix, suffix, desc) -> dict:
    per_book = []
    events = None
    t0 = time.perf_counter()
    for n, (sample_idx, tokens) in enumerate(books, 1):
        full, truncated, full_b, trunc_b, events = build_pair(tokens, offset, prefix, suffix)
        probe.boundaries = full_b
        sc_full = score(backend, full, suffix)
        state_full, out_full = probe.take()
        probe.boundaries = trunc_b
        sc_trunc = score(backend, truncated, suffix)
        state_trunc, out_trunc = probe.take()
        stats = {k: v.cpu().numpy() for k, v in paired_stats(state_full, state_trunc, out_full, out_trunc).items()}
        del state_full, state_trunc, out_full, out_trunc
        per_book.append((sample_idx, token_checksum(tokens), sc_full, sc_trunc, stats))
        if n % 20 == 0 or n == len(books):
            print(f"  {desc}: {n}/{len(books)} books, {(time.perf_counter() - t0) / n:.2f} s/book", flush=True)

    out = {
        "sample_idx": np.array([b[0] for b in per_book]),
        "token_checksum": np.array([b[1] for b in per_book], dtype=np.uint32),
        "layer": np.array(probe.layer_ids),
        "event_kind": np.array([e[0] for e in events]),
        "event_count": np.array([e[1] for e in events]),
        "event_position": np.array([e[2] for e in events]),
        "offset": offset, "prefix": prefix, "suffix": suffix,
        "seconds_per_book": (time.perf_counter() - t0) / len(books),
    }
    # token stats: [N, 2, S] with axis 1 = (full, truncated); state stats: [N, L, E, H]
    for key in ("nll", "margin", "argmax_match"):
        out[key] = np.stack([np.stack([b[2][key], b[3][key]]) for b in per_book])
    out["first_mismatch"] = np.array([[b[2]["first_mismatch"], b[3]["first_mismatch"]] for b in per_book])
    for key in per_book[0][4]:
        out[key] = np.stack([b[4][key] for b in per_book])
    return out


def run_test(backend, probe, books, offset, prefix, suffix, n_books) -> dict:
    """Instrumented vs plain forward on the first n_books: the segmented kernel must give
    the same suffix losses and argmax decisions as one unsegmented call."""
    rows = []
    for sample_idx, tokens in books[:n_books]:
        full, truncated, full_b, trunc_b, _ = build_pair(tokens, offset, prefix, suffix)
        for branch, seq, bounds in (("full", full, full_b), ("truncated", truncated, trunc_b)):
            probe.restore()
            plain = score(backend, seq, suffix)
            probe.install()
            probe.boundaries = bounds
            instrumented = score(backend, seq, suffix)
            probe.take()
            d = np.abs(plain["nll"] - instrumented["nll"])
            rows.append({
                "sample_idx": sample_idx, "branch": branch,
                "max_abs_nll_diff": float(d.max()), "mean_abs_nll_diff": float(d.mean()),
                "argmax_disagreements": int((plain["argmax_match"] != instrumented["argmax_match"]).sum()),
                "first_mismatch_plain": plain["first_mismatch"],
                "first_mismatch_instrumented": instrumented["first_mismatch"],
                "mean_nll_plain": float(plain["nll"].mean()),
            })
            print(f"  test {branch} sample {sample_idx}: max|dNLL|={d.max():.2e} mean|dNLL|={d.mean():.2e} "
                  f"argmax changes={rows[-1]['argmax_disagreements']} "
                  f"K plain/instr={plain['first_mismatch']}/{instrumented['first_mismatch']}", flush=True)
    return {"rows": rows, "max_abs_nll_diff": max(r["max_abs_nll_diff"] for r in rows),
            "argmax_disagreements": sum(r["argmax_disagreements"] for r in rows)}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


### CLI ###

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--experiment-path", required=True, help="Output dir (scratch)")
    p.add_argument("--persistent-storage-path", default=None,
                   help="Store mirror, checked as a fallback by the done-check. Never written to.")
    p.add_argument("--data-folder", required=True, help="Directory of rep_*_token.jsonl bucket files")
    p.add_argument("--repetitions", default="32,64,256", help="Comma-separated buckets")
    p.add_argument("--points", nargs="+", default=["50:2000"],
                   help="Truncated offset:prefix pairs; the full control (0, offset+prefix) is run alongside")
    p.add_argument("--suffix-length", type=int, default=249)
    p.add_argument("--max-books", type=int, default=None,
                   help="First N books of the seeded per-bucket permutation (default: all)")
    p.add_argument("--test", action="store_true",
                   help="Compare instrumented vs plain forward on the first 2 books of the first "
                        "rep/point before running; writes test.json")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Report resolved books/boundaries/work, no model load")
    p.add_argument("--container-env", default=None)
    p.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "single-rank script"
    reps = sorted(int(r) for r in args.repetitions.split(","))
    points = parse_points(args.points)
    suffix = args.suffix_length
    output_dir = Path(args.experiment_path)
    store_dir = Path(args.persistent_storage_path) if args.persistent_storage_path else None
    locations = (output_dir, store_dir)

    ### resolve work ###
    bucket_paths = {int(p.stem.split("_")[1]): p for p in find_rep_paths(Path(args.data_folder), set(reps))}
    assert set(bucket_paths) == set(reps), f"buckets missing under {args.data_folder}: {set(reps) - set(bucket_paths)}"
    sample_ids_by_rep = {rep: select_books(count_rows(bucket_paths[rep]), rep, args.max_books) for rep in reps}
    work = []  # (rep, offset, prefix, sample_ids, done)
    for rep in reps:
        sample_ids = sample_ids_by_rep[rep]
        for offset, prefix in points:
            done = result_done(locations, rep, offset, prefix, sample_ids) and not args.overwrite
            work.append((rep, offset, prefix, sample_ids, done))

    for rep, offset, prefix, sample_ids, done in work:
        g = offset + prefix
        tokens = (2 * g + suffix - 1) + (1 + prefix + suffix - 1)
        print(f"rep={rep} offset={offset} prefix={prefix}: {len(sample_ids)} books, "
              f"{tokens * len(sample_ids):,} input tokens -> {'done' if done else 'needed'}")
    if args.dry_run:
        _, first_tokens = load_books(bucket_paths[work[0][0]], work[0][3][:1])[0]
        _, _, full_b, trunc_b, events = build_pair(first_tokens, work[0][1], work[0][2], suffix)
        print(f"events for point {work[0][1]}:{work[0][2]}: {events}")
        print(f"full boundaries {full_b}, truncated boundaries {trunc_b}")
        return
    if all(w[4] for w in work):
        print("All requested results already present -- skipping checkpoint load.")
        return

    ### model ###
    t0 = time.perf_counter()
    backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    backend.load_model()
    load_seconds = time.perf_counter() - t0
    probe = StateProbe(backend.model)
    print(f"model loaded in {load_seconds:.0f} s; {len(probe.layer_ids)} GDN layers")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ckpt_dir": args.ckpt_dir,
            "container_env": args.container_env,
            "git_commit": git_commit(),
            "numpy_version": np.__version__,
            "seed": SEED,
            "repetitions": reps,
            "points": points,
            "suffix_length": suffix,
            "max_books": args.max_books,
            "sample_idx": {str(rep): ids for rep, ids in sample_ids_by_rep.items()},
            "prefix_events": PREFIX_EVENTS,
            "suffix_events": SUFFIX_EVENTS,
            "eps": EPS,
            "model_load_seconds": load_seconds,
        }, f, indent=2)

    try:
        if args.test:
            rep, offset, prefix, sample_ids, _ = work[0]
            books = load_books(bucket_paths[rep], sample_ids[:2])
            print(f"\n### TEST: instrumented vs plain forward (rep={rep} offset={offset} prefix={prefix}) ###")
            report = run_test(backend, probe, books, offset, prefix, suffix, n_books=2)
            with open(output_dir / "test.json", "w") as f:
                json.dump(report, f, indent=2)
            print(f"test: max|dNLL|={report['max_abs_nll_diff']:.2e} argmax changes={report['argmax_disagreements']}")
            probe.restore()

        probe.install()
        for rep, offset, prefix, sample_ids, done in work:
            if done:
                print(f"Skipping rep={rep} offset={offset} prefix={prefix} (already done)")
                continue
            books = load_books(bucket_paths[rep], sample_ids)
            desc = f"rep={rep} offset={offset} prefix={prefix}"
            out = run_point(backend, probe, books, offset, prefix, suffix, desc)
            path = result_path(output_dir, rep, offset, prefix)
            np.savez(path, **out)
            print(f"  done {desc} -> {path}", flush=True)
    finally:
        probe.restore()

    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()

"""Paired complete-vs-truncated recurrent-state comparison on Gutenberg excerpts.

GDN or KDA checkpoints (the gdn_ in the file and results names predates KDA support).

For every selected book and truncated point (offset O, prefix P) the model is compared
teacher-forced on the same 249-token suffix at g = O+P:

    full:      x[0 : g+S]                (the matched complete-context control)
    truncated: [BOS] + x[O : g+S]

Both branches consume identical text from original position O onwards, so the recurrent
state after c common-prefix tokens (c in 63/255/1023/P) and after t true suffix tokens
(t in 5/50/64/128/192/248) can be compared within the same checkpoint: norms, distance,
relative distance, cosine, and the same for the recurrent output at that token. NLL, argmax
match and true-vs-best-other margin are kept at every position of both branches (aligned on
the original token index), so the state gap can be related to the loss penalty and the first
teacher-forced argmax mismatch per book, and -- the model being causal -- the same traces
give the loss penalty of every shorter prefix at this offset without another run.

The same causality shares forwards across points: per book one full forward at the largest
g serves every point's control, and one truncated forward per offset at its largest prefix
serves every point at that offset (thesis report section 12). Output is still one file per
(rep, point).

States are read the same way as gdn_state_norm.py (segment the chunk kernel and carry the
state), but cut at the requested boundaries instead of every N tokens, and kept per book
instead of averaged. Cutting the kernel anywhere but on its 64-token chunk grid changes the
bf16 tiling and perturbs the losses by ~1e-2 nats/token (jobs 3422614/3422619), so only
boundaries on the grid (or at the input end) are cut inside the scoring forward, which is
then bit-exact; every other boundary gets its own shortened forward over input[:b] whose
final state is the exact state at b. Any offset:prefix works; with offset % 64 == 1 and
prefix % 64 == 63 (the truncated branch is shifted by its replacement BOS) only the early
suffix events need extra forwards, hence the odd-looking event counts and points such as
65:1983 instead of 50:2000. --dry-run prints the forwards per book. Only scalar statistics
are saved; the reference states live on the GPU for one book at a time. Single GPU,
batch 1, Megatron checkpoint backend.

Plan: attn_bench/_plans/gdn_paired_state_probe_plan.md

Usage (via torchrun, 1 GPU):
    torchrun --nproc_per_node=1 attn_bench/evaluation/gdn_paired_state.py \
        --ckpt-dir $MODEL_DIR/checkpoints --tokenizer-path $TOKENIZER_PATH \
        --experiment-path $OUT_DIR --data-folder $GUTENBERG_JSONL_DIR \
        --repetitions 32,64,128,256 --points 65:511 65:1983 [--max-books 4] [--test] [--dry-run]
"""

from __future__ import annotations

import argparse
import importlib
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
CHUNK = 64                        # fla chunk size; cuts inside a forward must land on this grid
PREFIX_EVENTS = (63, 255, 1023)   # common-prefix tokens consumed; P itself is always added
SUFFIX_EVENTS = (5, 50, 64, 128, 192, 248)  # true suffix tokens consumed; 248 is the last input token
EPS = 1e-8                        # denominator floor for relative distance / cosine
# Section-12 main diagnostic set, moved onto the chunk grid (offset % 64 == 1, prefix % 64 == 63)
DEFAULT_POINTS = ["65:511", "65:1983", "1025:5951", "65:7871", "3969:3967", "5953:1983"]


### DATA ###

def select_books(n_rows: int, rep: int, max_books: int | None) -> list[int]:
    """First max_books rows of one seeded permutation per bucket, so a 4-book test run, a
    165-book run and a full run are nested subsets of each other. Sorted for file order."""
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


### POINTS AND EVENTS ###

def point_events(offset: int, prefix: int, suffix: int) -> list[tuple]:
    """(kind, count, full_boundary, truncated_boundary) per state snapshot of one point. A
    boundary is the number of input tokens consumed; the i-th full and truncated boundaries
    end on the same original token, and the full boundary is its original position."""
    g = offset + prefix
    prefix_counts = [c for c in PREFIX_EVENTS if c < prefix] + [prefix]
    suffix_counts = [t for t in SUFFIX_EVENTS if t <= suffix - 1]
    return ([("prefix", c, offset + c, 1 + c) for c in prefix_counts]
            + [("suffix", t, g + t, 1 + prefix + t) for t in suffix_counts])


def build_sequences(tokens: list[int], points: list[tuple[int, int]], suffix: int):
    """Shared model inputs for one book: the full branch at the largest g with the union of
    every point's full boundaries, and per offset one truncated branch at that offset's
    largest prefix with the union of its points' truncated boundaries."""
    assert tokens[0] == BOS_TOKEN_ID, "bucket rows are expected to start with BOS"
    g_max = max(o + p for o, p in points)
    assert len(tokens) >= g_max + suffix, f"row too short: {len(tokens)} < {g_max + suffix}"
    full = tokens[:g_max + suffix]
    full_b = sorted({e[2] for o, p in points for e in point_events(o, p, suffix)})
    truncated = {}
    for offset in sorted({o for o, _ in points}):
        assert offset > 0, "offset 0 has no missing history, nothing to pair"
        p_max = max(p for o, p in points if o == offset)
        seq = [BOS_TOKEN_ID] + tokens[offset:offset + p_max + suffix]
        bounds = sorted({e[3] for o, p in points if o == offset for e in point_events(o, p, suffix)})
        truncated[offset] = (seq, bounds)
    return full, full_b, truncated


def split_boundaries(boundaries: list[int], n_inputs: int):
    """Boundaries that can be cut inside the scoring forward without changing its numbers
    (chunk grid or input end) vs those that need a shortened forward of their own."""
    grid = [b for b in boundaries if b % CHUNK == 0 or b == n_inputs]
    extra = [b for b in boundaries if b not in grid]
    return grid, extra


def describe_forwards(tokens, points, suffix) -> tuple[list[str], int]:
    """Per-branch summary lines and the total input tokens per book (main + shortened)."""
    full, full_b, truncated = build_sequences(tokens, points, suffix)
    lines, total = [], 0
    for name, seq, bounds in [("full", full, full_b)] + [(f"offset {o}", s, b) for o, (s, b) in truncated.items()]:
        grid, extra = split_boundaries(bounds, len(seq) - 1)
        total += len(seq) - 1 + sum(extra)
        lines.append(f"  {name}: {len(seq) - 1} input tokens, cuts {grid}, {len(extra)} shortened forwards at {extra}")
    return lines, total


### OBSERVER ###

# Recurrent families the probe can read: module class -> the instance attribute holding the
# prefill chunk kernel. Both kernels take q/k/v/g/beta as [B, T, ...] and return
# (output, final_state [B, H, K, V]); everything else (KDA's A_log/dt_bias/flags) is passed
# through untouched.
STATE_FAMILIES = {
    "GDN": ("megatron.core.ssm.gated_delta_net", "GatedDeltaNet", "gated_delta_rule"),
    "KDA": ("megatron.core.ssm.kimi_delta_attention", "KimiDeltaAttention", "kda_rule"),
}
SEQUENCE_ARGS = ("q", "k", "v", "g", "beta")  # kernel arguments sliced along the token dim


class StateProbe:
    """Swaps the chunk kernel on every recurrent layer (GDN or KDA, see STATE_FAMILIES) for a
    wrapper that runs it in segments ending at self.boundaries and keeps each segment's final
    state and the recurrent output of its last token. Set boundaries, run one forward, call
    take()."""

    def __init__(self, model):
        found = {}
        for family, (module_name, class_name, attr) in STATE_FAMILIES.items():
            cls = getattr(importlib.import_module(module_name), class_name)
            modules = [m for m in model.modules() if isinstance(m, cls)]
            if modules:
                found[family] = (modules, attr)
        assert found, "model has no GDN/KDA layers"
        assert len(found) == 1, f"mixed recurrent families not supported: {sorted(found)}"
        (self.family, (self.modules, self.attr)), = found.items()
        assert not any(m.training for m in self.modules), "model must be in eval mode"
        self.layer_ids = [m.layer_number for m in self.modules]
        self.boundaries = None
        self._originals = {}
        self._states = {}
        self._outputs = {}

    def install(self):
        assert not self._originals, "already installed"
        for m in self.modules:
            real_fn = getattr(m, self.attr)
            self._originals[m.layer_number] = real_fn
            setattr(m, self.attr, self._wrap(real_fn, m.layer_number))

    def restore(self):
        for m in self.modules:
            if m.layer_number in self._originals:
                setattr(m, self.attr, self._originals[m.layer_number])
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
        def wrapper(*args, **kwargs):
            # Both kernels take q, k, v, g, beta first; GDN passes q/k/v positionally and
            # KDA everything by keyword. Split them off and pass the rest through untouched.
            assert len(args) <= len(SEQUENCE_ARGS), "unexpected positional kernel arguments"
            seq = dict(zip(SEQUENCE_ARGS, args))
            seq.update({name: kwargs.pop(name) for name in SEQUENCE_ARGS if name in kwargs})
            assert len(seq) == len(SEQUENCE_ARGS), f"missing kernel arguments: {set(SEQUENCE_ARGS) - set(seq)}"
            assert kwargs.get("cu_seqlens") is None and kwargs.pop("initial_state", None) is None, \
                "packed input / initial state not supported"
            output_final_state = kwargs.pop("output_final_state", False)
            q = seq["q"]
            assert q.shape[0] == 1, "batch size must be 1"
            assert self.boundaries, "set probe.boundaries before the forward"
            seq_len = q.shape[1]
            wanted = set(self.boundaries)
            assert max(wanted) <= seq_len, f"boundary beyond input length {seq_len}"
            ends = sorted(wanted | {seq_len})

            outs, states, out_at = [], [], []
            state = None
            start = 0
            for end in ends:
                out_s, state = real_fn(
                    **{name: tensor[:, start:end] for name, tensor in seq.items()},
                    **kwargs, initial_state=state, output_final_state=True,
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
def score(backend, tokens: list[int], softmax_chunk: int = 2048) -> dict:
    """One teacher-forced forward; NLL, argmax match and margin (true logit minus best other
    logit) at every input position, not only the suffix: the model is causal, so the trace
    at position i is also the loss this branch would have for any shorter prefix ending
    there. Softmax runs over position chunks (bf16 logits, cast per chunk) to keep the fp32
    copy bounded."""
    device = backend.device
    seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    inputs, labels = seq[:, :-1], seq[:, 1:]
    n = inputs.shape[1]
    pos = torch.arange(n, dtype=torch.long, device=device).unsqueeze(0)
    logits = backend.forward_logits(inputs, pos, fp32_output=False)[0]  # [n, V] bf16

    nll = torch.empty(n, dtype=torch.float32, device=device)
    margin = torch.empty(n, dtype=torch.float32, device=device)
    argmax_match = torch.empty(n, dtype=torch.bool, device=device)
    for c0 in range(0, n, softmax_chunk):
        c1 = min(c0 + softmax_chunk, n)
        chunk = logits[c0:c1].float()
        lab = labels[0, c0:c1, None]
        nll[c0:c1] = -F.log_softmax(chunk, dim=-1).gather(1, lab).squeeze(1)
        true_logit = chunk.gather(1, lab).squeeze(1)
        margin[c0:c1] = true_logit - chunk.scatter(1, lab, float("-inf")).max(dim=1).values
        argmax_match[c0:c1] = chunk.argmax(dim=1) == lab.squeeze(1)
    del logits
    return {
        "nll": nll.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "argmax_match": argmax_match.cpu().numpy(),
    }


def align_trace(full: np.ndarray, truncated: np.ndarray, offset: int, fill) -> np.ndarray:
    """Stack the two branches' per-position traces on the full branch's axis, where index i
    is the prediction of original token x[i+1]. The truncated input is [BOS] + x[offset:],
    so its index q predicts x[offset+q] and lands at i = offset-1+q; earlier positions do
    not exist in that branch and get `fill`."""
    aligned = np.full((2, len(full)), fill, dtype=full.dtype)
    aligned[0] = full
    aligned[1, offset - 1:] = truncated
    return aligned


def first_mismatch(argmax_match: np.ndarray, suffix: int) -> int:
    """First argmax mismatch inside the last `suffix` positions; == suffix if none."""
    miss = np.flatnonzero(~argmax_match[-suffix:])
    return int(miss[0]) if len(miss) else suffix


@torch.no_grad()
def forward_only(backend, tokens: list[int]) -> None:
    device = backend.device
    seq = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    pos = torch.arange(seq.shape[1], dtype=torch.long, device=device).unsqueeze(0)
    backend.forward_logits(seq, pos, fp32_output=False)


def run_branch(backend, probe, seq: list[int], boundaries: list[int]):
    """Score one branch and read its state at every boundary, bit-exactly: grid boundaries
    are cut inside the scoring forward, the rest come from a shortened forward over
    seq[:b] (the model input for the first b tokens) whose final state is the state at b.
    Returns the score dict and {boundary: state [L, H, K, V]}, {boundary: output [L, H, V]}."""
    grid, extra = split_boundaries(boundaries, len(seq) - 1)
    probe.boundaries = grid or [len(seq) - 1]
    sc = score(backend, seq)
    st, out = probe.take()
    order = sorted(probe.boundaries)  # the wrapper records in ascending order
    states = {b: st[:, order.index(b)] for b in grid}
    outputs = {b: out[:, order.index(b)] for b in grid}
    for b in extra:
        probe.boundaries = [b]
        forward_only(backend, seq[:b])
        s_b, o_b = probe.take()
        states[b], outputs[b] = s_b[:, 0], o_b[:, 0]
    return sc, states, outputs


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


def extract_point(full, sc_full, st_full, out_full, seq, sc_tr, st_tr, out_tr, offset, prefix, suffix):
    """One point's record from the shared branches: paired state stats at its events and
    its token traces cut to g+S-1 (the last S positions are its suffix)."""
    events = point_events(offset, prefix, suffix)
    for _, _, fb, tb in events:
        assert full[fb - 1] == seq[tb - 1], "misaligned boundary"
    n = offset + prefix + suffix - 1
    assert full[n - suffix + 1:n + 1] == seq[n - offset - suffix + 2:n - offset + 2], "suffix labels differ"
    sf = torch.stack([st_full[e[2]] for e in events], dim=1)
    st = torch.stack([st_tr[e[3]] for e in events], dim=1)
    of = torch.stack([out_full[e[2]] for e in events], dim=1)
    ot = torch.stack([out_tr[e[3]] for e in events], dim=1)
    stats = {k: v.cpu().numpy() for k, v in paired_stats(sf, st, of, ot).items()}
    trace = {}
    for key, fill in (("nll", np.nan), ("margin", np.nan)):
        trace[key] = align_trace(sc_full[key][:n], sc_tr[key][:n - offset + 1], offset, fill)
    trace["argmax_match"] = align_trace(sc_full["argmax_match"][:n].astype(np.int8),
                                        sc_tr["argmax_match"][:n - offset + 1].astype(np.int8), offset, -1)
    trace["first_mismatch"] = np.array([first_mismatch(sc_full["argmax_match"][:n], suffix),
                                        first_mismatch(sc_tr["argmax_match"][:n - offset + 1], suffix)])
    return events, stats, trace


def run_rep(backend, probe, books, points, suffix, desc) -> dict:
    """All points of one bucket, sharing forwards per book. Returns {point: output dict}."""
    per_point = {pt: [] for pt in points}
    events_by_point = {}
    t0 = time.perf_counter()
    for n, (sample_idx, tokens) in enumerate(books, 1):
        full, full_b, truncated = build_sequences(tokens, points, suffix)
        sc_full, st_full, out_full = run_branch(backend, probe, full, full_b)
        for offset, (seq, bounds) in truncated.items():
            sc_tr, st_tr, out_tr = run_branch(backend, probe, seq, bounds)
            for o, p in points:
                if o != offset:
                    continue
                events, stats, trace = extract_point(
                    full, sc_full, st_full, out_full, seq, sc_tr, st_tr, out_tr, o, p, suffix)
                events_by_point[(o, p)] = events
                per_point[(o, p)].append((sample_idx, token_checksum(tokens), stats, trace))
            del st_tr, out_tr
        del st_full, out_full
        if n % 20 == 0 or n == len(books):
            print(f"  {desc}: {n}/{len(books)} books, {(time.perf_counter() - t0) / n:.2f} s/book", flush=True)
    seconds_per_book = (time.perf_counter() - t0) / len(books)

    outputs = {}
    for (offset, prefix), records in per_point.items():
        events = events_by_point[(offset, prefix)]
        out = {
            "sample_idx": np.array([r[0] for r in records]),
            "token_checksum": np.array([r[1] for r in records], dtype=np.uint32),
            "layer": np.array(probe.layer_ids),
            "event_kind": np.array([e[0] for e in events]),
            "event_count": np.array([e[1] for e in events]),
            "event_position": np.array([e[2] for e in events]),
            "offset": offset, "prefix": prefix, "suffix": suffix,
            "points_in_run": np.array(points),
            "seconds_per_book": seconds_per_book,
        }
        # token stats: [N, 2, g+S-1] with axis 1 = (full, truncated), axis 2 = prediction of
        # x[i+1]; the truncated branch starts at i = offset-1 (NaN / -1 before). Suffix = last S.
        # state stats: [N, L, E, H]
        for key in records[0][3]:
            out[key] = np.stack([r[3][key] for r in records])
        for key in records[0][2]:
            out[key] = np.stack([r[2][key] for r in records])
        outputs[(offset, prefix)] = out
    return outputs


def run_test(backend, probe, books, points, suffix) -> dict:
    """Instrumented vs plain forward on each shared branch of the given books: the scoring
    forward with its grid cuts must give the same losses and argmax decisions as one
    unsegmented call, and a shortened forward ending on a grid boundary must give the same
    state as the cut."""
    rows = []
    for sample_idx, tokens in books:
        full, full_b, truncated = build_sequences(tokens, points, suffix)
        branches = [("full", full, full_b)] + [(f"offset {o}", s, b) for o, (s, b) in truncated.items()]
        for branch, seq, bounds in branches:
            probe.restore()
            plain = score(backend, seq)
            probe.install()
            instrumented, states, _ = run_branch(backend, probe, seq, bounds)
            grid, extra = split_boundaries(bounds, len(seq) - 1)
            state_diff = None
            if grid:
                probe.boundaries = [grid[0]]
                forward_only(backend, seq[:grid[0]])
                s_short, _ = probe.take()
                state_diff = float((s_short[:, 0] - states[grid[0]]).abs().max())
            d = np.abs(plain["nll"] - instrumented["nll"])
            k_plain = first_mismatch(plain["argmax_match"], suffix)
            k_instr = first_mismatch(instrumented["argmax_match"], suffix)
            rows.append({
                "sample_idx": sample_idx, "branch": branch,
                "max_abs_nll_diff": float(d.max()), "mean_abs_nll_diff": float(d.mean()),
                "argmax_disagreements": int((plain["argmax_match"] != instrumented["argmax_match"]).sum()),
                "first_mismatch_plain": k_plain, "first_mismatch_instrumented": k_instr,
                "mean_suffix_nll_plain": float(plain["nll"][-suffix:].mean()),
                "grid_cuts": grid, "extra_forwards": extra,
                "shortened_vs_cut_max_state_diff": state_diff,
            })
            print(f"  test {branch} sample {sample_idx}: max|dNLL|={d.max():.2e} mean|dNLL|={d.mean():.2e} "
                  f"argmax changes={rows[-1]['argmax_disagreements']} K plain/instr={k_plain}/{k_instr} "
                  f"shortened-vs-cut max|dstate|={state_diff}", flush=True)
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
    p.add_argument("--repetitions", default="32,64,128,256", help="Comma-separated buckets")
    p.add_argument("--points", nargs="+", default=DEFAULT_POINTS,
                   help="Truncated offset:prefix pairs; the full control (0, offset+prefix) is run "
                        "alongside. offset %% 64 == 1 and prefix %% 64 == 63 keeps the extra forwards "
                        "to the early suffix events (see --dry-run)")
    p.add_argument("--suffix-length", type=int, default=249)
    p.add_argument("--max-books", type=int, default=None,
                   help="First N books of the seeded per-bucket permutation (default: all)")
    p.add_argument("--test", action="store_true",
                   help="Compare instrumented vs plain forward on the first 2 books of the first "
                        "rep before running; writes test.json")
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
    assert len(set(points)) == len(points), "duplicate points"
    suffix = args.suffix_length
    output_dir = Path(args.experiment_path)
    store_dir = Path(args.persistent_storage_path) if args.persistent_storage_path else None
    locations = (output_dir, store_dir)

    ### resolve work ###
    bucket_paths = {int(p.stem.split("_")[1]): p for p in find_rep_paths(Path(args.data_folder), set(reps))}
    assert set(bucket_paths) == set(reps), f"buckets missing under {args.data_folder}: {set(reps) - set(bucket_paths)}"
    sample_ids_by_rep = {rep: select_books(count_rows(bucket_paths[rep]), rep, args.max_books) for rep in reps}
    # per rep: the points still missing (a rep is run once for all of them; done files are kept)
    needed = {}
    for rep in reps:
        needed[rep] = [pt for pt in points
                       if args.overwrite or not result_done(locations, rep, *pt, sample_ids_by_rep[rep])]
        for pt in points:
            print(f"rep={rep} offset={pt[0]} prefix={pt[1]}: {len(sample_ids_by_rep[rep])} books -> "
                  f"{'needed' if pt in needed[rep] else 'done'}")

    if args.dry_run:
        rep = reps[0]
        _, first_tokens = load_books(bucket_paths[rep], sample_ids_by_rep[rep][:1])[0]
        lines, per_book = describe_forwards(first_tokens, points, suffix)
        print(f"forwards per book for points {points}:")
        print("\n".join(lines))
        n_books = sum(len(sample_ids_by_rep[rep]) for rep in reps if needed[rep])
        print(f"{per_book:,} input tokens per book (main + shortened), {n_books} books needed -> {per_book * n_books:,} tokens")
        return
    if not any(needed.values()):
        print("All requested results already present -- skipping checkpoint load.")
        return

    ### model ###
    t0 = time.perf_counter()
    backend = MegatronBackend(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    backend.load_model()
    load_seconds = time.perf_counter() - t0
    probe = StateProbe(backend.model)
    print(f"model loaded in {load_seconds:.0f} s; {len(probe.layer_ids)} {probe.family} layers")

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
            rep = reps[0]
            books = load_books(bucket_paths[rep], sample_ids_by_rep[rep][:2])
            print(f"\n### TEST: instrumented vs plain forward (rep={rep}, points {points}) ###")
            report = run_test(backend, probe, books, points, suffix)
            with open(output_dir / "test.json", "w") as f:
                json.dump(report, f, indent=2)
            print(f"test: max|dNLL|={report['max_abs_nll_diff']:.2e} argmax changes={report['argmax_disagreements']}")
            probe.restore()

        probe.install()
        for rep in reps:
            if not needed[rep]:
                print(f"Skipping rep={rep} (all points already done)")
                continue
            books = load_books(bucket_paths[rep], sample_ids_by_rep[rep])
            outputs = run_rep(backend, probe, books, points, suffix, desc=f"rep={rep}")
            for (offset, prefix), out in outputs.items():
                if (offset, prefix) not in needed[rep]:
                    continue
                path = result_path(output_dir, rep, offset, prefix)
                np.savez(path, **out)
                print(f"  done rep={rep} offset={offset} prefix={prefix} -> {path}", flush=True)
    finally:
        probe.restore()

    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()

# Fast-kernel support for HF inference, per attention family

Goal: convert every trained attention variant (full/sink/gated/GDN/SWA) to HF and get real
fused-kernel inference speed, not just eager fallback. This records what each family
actually needs and whether the current cluster (GH200, `aarch64` host + Hopper/SM90 GPU)
can support it, so we rebuild the container at most once instead of per-family.

Container snapshot used throughout: `attn_bench/data/nemo_26.04_te2.15_env_info.txt`
(`flash_attn 2.7.4.post1+nv26.2`, `transformers 4.57.3`, `torch 2.11.0a0`, CUDA 13.1,
`nvidia-cutlass-dsl 4.3.4`, no `kernels` package installed).

## full (baseline, already working)

No open question — `full.py` is implemented, HF conversion validated, plain
`flash_attention_2`/native TE fused attention both work already.

## sink (learnable softmax offset)

Needs a fused kernel that accepts a per-head sink/bias term appended to the attention
logits before softmax and dropped after (Megatron's `SoftmaxOne`, HF's `s_aux` in
`gpt_oss`). Checked every option that could provide this on GH200:

| Option | Has sink support | Runs on `aarch64` | Verdict |
| --- | --- | --- | --- |
| Container's pip `flash_attn` (2.7.4.post1+nv26.2) | No | — (native) | confirmed via `check_container_env.slurm` — `s_aux` not in `flash_attn_varlen_func` signature |
| `kernels-community/vllm-flash-attn3` (Hub kernel gpt_oss uses) | Yes (`s_aux` param, verified in source) | **No** — 19/19 build variants are `x86_64-linux` only | dead end for this cluster |
| `kernels-community/flash-attn3` (Hub kernel, no `vllm-` prefix) | **No** — checked `flash_attn_interface.py` source, no `s_aux`/sink param | Yes, has `aarch64-linux` variants (torch28/29 × cu126-129) | has the arch, missing the feature |
| Dao-AILab/flash-attention mainline, classic C++/pybind interface | No — open upstream issue [#1797](https://github.com/Dao-AILab/flash-attention/issues/1797), still unresolved a year later | n/a | not available anywhere in this interface |
| Dao-AILab/flash-attention **`flash_attn.cute` module** (CUTLASS Python DSL, `main` branch) | **Yes** — `flash_fwd_sm90.py` has a `learnable_sink` param, confirmed in current source (`softmax.finalize(sink_val=...)`) | Not confirmed either way — no `aarch64` mention in the repo at all, but this module targets **GPU compute capability (SM90/Hopper)**, not host CPU ISA, and is JIT/source-compiled like Triton rather than shipped as prebuilt per-arch wheels (needs `nvidia-cutlass-dsl>=4.2.0`, container already has `4.3.4`) | most promising lead, but empirically unverified |

Two things work in our favor for the `cute` path specifically:
- We only need the **forward pass** (inference only, model already trained in Megatron)
  — `flash_fwd_sm90.py` has `learnable_sink`; the backward-pass gaps discussed in the
  upstream issue don't matter here.
- SM90 = Hopper = what GH200's GPU actually is, regardless of the Grace CPU being
  `aarch64`. CUTE-DSL kernels JIT-compile like Triton (see GDN below), not prebuilt
  wheels tied to host CPU architecture — the same reason GDN's Triton kernels already
  run fine on this cluster despite no `aarch64` wheel ever being published for them.

**Update, tested empirically (`attn_bench/scripts/check_sink_kernel.slurm`, jobs 3116197 →
3116697)**: `aarch64`/GH200/Hopper is NOT the blocker. Across six runs, `flash_attn.cute`
imports cleanly, the JIT pipeline reaches real MLIR/Hopper (SM90) kernel generation every
time, and the public API (`flash_attn_func`/`flash_attn_varlen_func`) exposes
`learnable_sink` exactly as expected. What actually blocks a full run: `main`'s
`flash_attn/cute/utils.py::fmax()` calls `nvvm.fmax(a, b, ...)` with two positional args,
but the installed binding's real signature is `nvvm.fmax(res, a, b, ...)` — a `res`
(result-type) argument short by one. Confirmed general, not sink-specific (a plain
no-sink forward call hits the identical crash, since `online_softmax`'s row-max reduction
runs on every forward pass). Confirmed independent of `nvidia-cutlass-dsl` version --
tried 4.2.0, 4.3.4 (container default), and 4.6.2 (upstream's own current floor as of
their 2026-08-15 "Relax CUTLASS DSL requirement" commit) -- identical failure on all three.

Checked upstream's own CI to rule out "we're holding it wrong": `main`'s GPU-correctness
job (`fa4-correctness-and-benchmark`) only targets **Blackwell (b200)**, not Hopper, and
has shown `cancelled` (not `success`) on every one of the last 3 commits -- so there is no
currently-passing GPU-correctness signal on `main` for any architecture, let alone the
Hopper/SM90 path (`flash_fwd_sm90.py`) sink attention needs. This is genuinely
under-validated upstream code we hit an untested seam in, not a GH200-specific problem or
a mistake on our end.

**Resolved (job 3116933)**: patched and confirmed working. The right `res` value turned out
to be `T.f32()` (`T` = `cutlass.cutlass_dsl.T`, already imported in `utils.py`) --
found by comparing against the *installed* `nvidia-cutlass-dsl==4.6.2` package's own
`cute.arch.fmax` (the function Blackwell's kernel calls), which already passes
`T.f32()` as the first positional arg to `nvvm.fmax`. Note: `NVIDIA/cutlass`'s GitHub
`main` branch does *not* show this fix -- it's only present in what's actually bundled in
the PyPI wheel, so the installed package was the only reliable ground truth here, not a
GitHub read. Patch preserved at `attn_bench/utils/flash_attention_patch.txt` (PDM-style,
see `flash-attention-cute-workflow.md`), verified with `patch -p1 --ignore-whitespace`
against a fresh clone. `check_sink_kernel.slurm` now confirms a real forward pass:

```
forward pass OK, output shape (1, 128, 4, 64), dtype torch.bfloat16
RESULT: flash_attn_func(..., learnable_sink=...) works on NVIDIA GH200 120GB.
```

**Correctness validated too (job 3117001, `attn_bench/scripts/check_sink_kernel_correctness.slurm`)**:
running was not enough on its own -- upstream has no passing Hopper GPU-correctness CI
signal, so a numerically wrong kernel could have passed the run-only check just as easily.
Compared `flash_attn.cute`'s `learnable_sink` output against a plain-PyTorch reference
implementing Megatron's exact sink formula (`fused_softmax.py`'s scale -> causal mask ->
append sink as-is -> softmax over the extended dim -> drop sink column), same random
q/k/v/sink fed to both, no checkpoint needed. The reference itself was cross-checked
against `torch.nn.functional.scaled_dot_product_attention` first (sink neutralized via
`sink=-inf`, mathematically degenerating `SoftmaxOne` to plain softmax) to rule out a bug
in the reference before trusting the comparison:

```
reference formula sanity check vs SDPA (sink=-inf, i.e. no-op): max abs diff 0.000809
fraction close (atol=0.001, rtol=0.02): 0.9998
max abs diff: 0.006310
RESULT: CORRECT -- 99.98% of values close (need >= 99%).
```

Sink attention's fused-kernel path is fully cleared now -- runs on GH200, and matches
Megatron's math. `sink.py` (the `attn_families/` conversion module) is unblocked.

## gated (attention_output_gate)

**No kernel work needed at all.** `attention.py:1352-1356` (`_apply_output_gate`) is a
plain `x * torch.sigmoid(gate.float())`, applied *after* the attention call returns —
gating never touches the attention kernel itself. This is structurally identical to HF's
`Qwen3NextAttention` (`attn_output * torch.sigmoid(gate)`, `modeling_qwen3_next.py`),
which also gates post-hoc outside the attention interface. Any standard
`flash_attention_2`/`flash_attention_3`/`sdpa` implementation works unmodified; the gate
is just an extra elementwise op in plain PyTorch regardless of which attention kernel is
used underneath. Conversion work here is state-dict/config wiring only, same shape as
`full.py`.

## SWA (sliding window attention, planned)

**No kernel work needed at all — the easy family.** Megatron already exposes it as a
plain config field (`transformer_config.py:229`, `window_size: Optional[Tuple[int, int]]`)
passed straight through to TE's fused `DotProductAttention`, nothing experimental like
sink's custom softmax term. `window_size` has been a standard flash-attn parameter since
early FA2 (2023) and is present in every kernel signature checked this session, including
the `aarch64`-available `kernels-community/flash-attn3` build and this container's own
pip `flash_attn`. HF's flash-attention integration treats it as first-class too —
`modeling_flash_attention_utils.py`'s `_hf_api_to_flash_mapping` maps `sliding_window`
(HF config field) to `window_size` (kernel param) generically for any model, the same
mechanism `gpt_oss`/Mistral/Gemma2 already rely on for alternating local/global layers.
Conversion work is pure config/state-dict wiring, same shape as `full.py`.

## GDN (Gated Delta Net)

**Already running on this cluster today, no rebuild needed.** Both
`megatron/core/ssm/gated_delta_net.py` and HF's `qwen3_next` import the same
`fla.ops.gated_delta_rule` functions (`chunk_gated_delta_rule`,
`fused_recurrent_gated_delta_rule`). `fla`'s kernels are Triton-based — JIT-compiled at
runtime for whatever GPU is present, not distributed as prebuilt per-architecture wheels
— so there's no `x86_64`-only trap like `vllm-flash-attn3`. This project's own GDN
training jobs already exercise this path successfully on GH200 — the only issue hit was
a stale shared Triton cache (job 2570257), fixed with a per-rank/per-job cache dir, not
an architecture-support problem.

## Net effect on the rebuild question

No container rebuild needed for any of the five families. `sink` was the only one with a
real open kernel question, and it's fully resolved: `flash_attn.cute` (FA4) runs
`learnable_sink` on GH200 today with a one-line patch (`attn_bench/utils/flash_attention_patch.txt`)
applied to a persistent clone (see `flash-attention-cute-workflow.md`), and its output was
validated against Megatron's exact sink formula (99.98% of values within tolerance). Gated
and SWA need nothing special; GDN already works. Next step: implement `sink.py`.

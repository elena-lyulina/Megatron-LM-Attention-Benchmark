# Gated Delta Net (GDN) config — references & outcomes

Goal: configure a GDN model in Megatron that is faithful to the GDN paper's
architecture while matching the Llama-3.2-1B parameter count. This doc records
where each architectural number comes from and what follows for our config.

## Heads & dimensions (the math)

Every projection slices its output into `num_heads` heads of `head_dim` each, so
`total_dim = num_heads × head_dim`. GDN keeps the **value path wider than the key path**,
so there are two widths:
```
key_dim   = num_heads × head_k_dim     # q/k projection width
value_dim = num_heads × head_v_dim     # v projection width
head_v_dim = 2 × head_k_dim            # GDN design: each value head is 2× a key head 
```
The `0.75 / 1.5` ratios are just these totals divided by hidden, defined by the authors:
`key_dim/hidden = 0.75`, `value_dim/hidden = 1.5`.

The recurrent state per head is an outer-product matrix `head_k_dim × head_v_dim`
(an associative memory); a wider value side = more write capacity per head.

How each repo sets every model parameter. `expand_k` / `expand_v` are not model parameters;
they're just multipliers used to compute the dims, shown inline where they apply.

| Parameter         | NVlabs (official)                                  | fla                                    | Megatron                           |
| ----------------- |----------------------------------------------------| -------------------------------------- | ---------------------------------- |
| `hidden_size`     | passed                                             | passed                                 | passed                             |
| `num_key_heads`   | passed (`num_heads`)                               | passed (`num_heads`)                   | passed (`num_key_heads`)           |
| `num_value_heads` | = `num_heads`                                      | = `num_heads`                          | passed (`num_value_heads`)         |
| `conv_size`       | passed (4)                                         | passed (4)                             | passed (`linear_conv_kernel_dim`)  |
| **`key_dim`**     | `0.75 × hidden` (`0.75 is expand_k`)               | `num_heads × head_dim`                 | `num_key_heads × key_head_dim`     |
| **`value_dim`**   | `1.5 × hidden` (`1.5 is expand_v`) `= 2 × key_dim` | `num_heads × head_v_dim = 2 × key_dim` | `num_value_heads × value_head_dim` |
| `head_k_dim`      | `key_dim / num_heads`                              | passed (`head_dim`)                    | passed (`key_head_dim`)            |
| `head_v_dim`      | `2 × head_k_dim`                                   | `2 × head_dim` (`expand_v`)            | passed (`value_head_dim`)          |

**Invariant (NVlabs and fla repos): the value path is 2× the key path** — `head_v_dim = 2 × head_k_dim`,
so `value_dim = 2 × key_dim`. 

Example (hidden=2048, 12 heads): `12 × 128 = 1536` key (0.75hidden), `12 × 256 = 3072` value (1.5hidden = 2× key).

### Concrete values — references vs our options

`NVlabs 1.3B`: the only real 1.3B model. `fla 1.3B`: same model in fla's `head_dim`/`expand_v`
notation (identical numbers — confirms the two notations agree). `Megatron def.`: the generic
`--linear-*` defaults. `Opt A/B/C`: **our candidates at hidden=2048 with the paper's 0.75/1.5
ratios** (`key_dim=1536`, `value_dim=3072`), differing only in head count.

| Parameter           | NVlabs 1.3B | fla 1.3B | Megatron def. | Opt A (9h) | Opt B (8h) | Opt C (12h) |
| ------------------- | ----------- | -------- | ------------- | ---------- | ---------- | ----------- |
| `hidden_size`       | 2400        | 2400     | (you set)     | 2048       | 2048       | 2048        |
| `num_key_heads`     | 9           | 9        | 16            | 9          | 8          | 12          |
| `num_value_heads`   | 9           | 9        | 32            | 9          | 8          | 12          |
| `conv_size`         | 4           | 4        | 4             | 4          | 4          | 4           |
| **`key_dim`**       | 1800        | 1800     | 2048          | 1536       | 1536       | 1536        |
| **`value_dim`**     | 3600        | 3600     | 4096          | 3072       | 3072       | 3072        |
| `head_k_dim`        | 200         | 200      | 128           | 170.7 ✗    | 192        | 128         |
| `head_v_dim`        | 400         | 400      | 128           | 341.3 ✗    | 384        | 256         |
| **`state / layer`** | 720,000     | 720,000  | 524,288       | 524,288 ✗  | 589,824    | 393,216     |

**Option A (keep the paper's 9 heads) is infeasible at hidden=2048**: `1536/9` and `3072/9`
aren't integers (the paper's hiddens 1536/2400 divide cleanly; 2048 doesn't). So the workable
choices are **Opt B (8 heads, 192/384)** and **Opt C (12 heads, 128/256)**, both keeping the
0.75/1.5 ratios and `value = 2× key`. Opt B has the larger state (590K vs 393K) → more memory.

`state / layer` is the recurrent memory — `num_value_heads` independent 2D matrices, shape
`[num_value_heads, head_k_dim, head_v_dim]`. Since `value_dim = num_value_heads × head_v_dim`,
the count collapses and it simplifies to **`state = head_k_dim × value_dim`** (head count
drops out). (Megatron repeat-interleaves key/query up to `num_value_heads`,
`gated_delta_net.py:380`.) For symmetric heads this is also `key_dim × value_dim / num_heads`.

Notes:
- **Megatron `key_dim`/`value_dim` (2048/4096) are independent of hidden** (`= num_heads ×
  head_dim`). At hidden=2400 that's `expand_k≈0.85`, `expand_v≈1.71` — bigger than the
  paper's `0.75 / 1.5`, so the default mixer is oversized.
- Both reach `value = 2× key`, but differently: the paper uses **2× head dim** (400 vs 200)
  with **equal head counts** (9/9); Megatron defaults use **equal head dims** (128/128)
  with **2× value heads** (32 vs 16).
- **State is `head_k_dim × value_dim`, so big state comes from a big `head_k_dim` and/or
  `value_dim` — not from head count** (it cancels). NVlabs 1.3B is largest (720K) thanks to
  its big `head_k_dim` (200) even with a smaller `value_dim`; Megatron reaches 524K via a big
  `value_dim` (4096) despite `head_k_dim`=128. "Fewer heads ⇒ bigger state" holds *only at
  fixed totals* (e.g. paper ratios @hidden=2048: 8 heads → 590K vs 12 heads → 393K, all from
  `head_k_dim` 192 vs 128). For our memorization study a bigger state is desirable → favor a
  larger `head_k_dim`, i.e. the paper's few-large-heads design.

## Sources

### 1. Paper — *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR 2025)
https://arxiv.org/abs/2412.06464
locally: `attn_bench/papers/GATED DELTA NETWORKS - IMPROVING MAMBA 2 WITH DELTA RULE.pdf`
- Architecture is **qualitative only** (§3.4 + Fig. 1): Llama macro-arch; q/k/v via linear
  proj → short conv → SiLU; L2-norm on q/k; output norm + gating then output proj.
  **Expansion ratios / head counts are not stated — code only.**
- Training recipe: 100B tokens FineWeb-Edu, AdamW lr 4e-4, wd 0.1, grad-clip 1.0,
  Llama2 tokenizer (vocab 32000), seq 4K, SWA window 2K (hybrids).

### 2. Official code — NVlabs/GatedDeltaNet (paper authors, ground truth)
https://github.com/NVlabs/GatedDeltaNet
locally: `~/PycharmProjects/GatedDeltaNet` — `lit_gpt/gated_delta_net.py`, `lit_gpt/config.py`
- `model.py:288` builds the layer as `GatedDeltaNet(hidden_size=config.n_embd)` — **only
  hidden_size is passed**, so all model sizes share the layer defaults (`expand_k=0.75`,
  `expand_v=1.5`, `num_heads=9`, `conv=4`, no bias) and vary only hidden + layer count.
- `config.n_head` (12/16) is for the **attention** layers in hybrids, NOT GDN.
- Real named configs (pure GDN = `gated_delta_per_layer=1`):

  | model | hidden | layers | intermediate | key_dim (0.75h) | value_dim (1.5h) |
  |---|---|---|---|---|---|
  | 0.4B | 1536 | 11 | 6144 (=4×h) | 1152 | 2304 |
  | 1.3B | 2400 | 16 | 5888 (≈2.45×h) | 1800 | 3600 |

### 3. flash-linear-attention (fla) — same authors; kernels Megatron uses
https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
locally: `~/PycharmProjects/flash-linear-attention` — `fla/layers/gated_deltanet.py`
- Same architecture in `head_dim`/`expand_v` notation (see tables above). Its
  `hidden=2048` / 6-head default is a **placeholder — no real 1.3B config here**.
- Megatron's GDN kernels come from this library (requires `fla` + `causal_conv1d`).

### 4. Megatron-core — the implementation we run
`megatron/core/ssm/gated_delta_net.py`, `transformer_config.py`,
`megatron/core/models/gpt/experimental_attention_variant_module_specs.py`
- GDN internals are **hardcoded** (no flags): gating, dt_bias, A_log, qk-L2-norm=True,
  bias=False, conv_bias=False, A_init=(1,16).
- Flags that reach GDN: `--linear-*` (shape), `--normalization`/`--norm-epsilon` (norms),
  `--swiglu`→silu (activation).
- `--linear-*` defaults are generic and oversized vs the paper (see values table above).

## Canonical GDN architecture (what all the code agrees on)

Independent of model size:
- **`key_dim = 0.75 × hidden`**, **`value_dim = 1.5 × hidden`**
- **`value_head_dim = 2 × key_head_dim`**, **`num_value_heads = num_key_heads`** (no GQA)
- **conv kernel = 4**
- head *count* fixed (9 in official code); head *dim* scales with hidden
- head count is ~irrelevant to param count (only tiny per-head vectors scale with it);
  the mixer size is set by the expand factors (total key/value dims).

## What follows for our config (hidden=2048, Llama-1B backbone)

- 2048 is not divisible by the official 9-head scheme (0.75×2048=1536 not /9), so we
  keep the **ratios** and pick a clean head count → **Option A**:
  ```
  --linear-num-key-heads    12
  --linear-num-value-heads  12
  --linear-key-head-dim     128
  --linear-value-head-dim   256   # value_dim=3072=1.5h, key_dim=1536=0.75h, value=2×key
  --linear-conv-kernel-dim  4
  ```
- Per-layer mixer params @hidden=2048: paper-ratio ≈ 25M, Megatron-default ≈ 34M,
  Llama softmax attn ≈ 10.5M. GDN mixer is ~2.4× attention (inherent to linear attn).
- **Match Llama-1B param count by shrinking `--ffn-hidden-size`** to absorb the larger
  mixer — this is exactly what the paper's 1.3B does (intermediate 5888 ≈ 2.45×h, not 4×).
- Keep `--normalization RMSNorm` (paper uses RMSNorm throughout — the one norm knob that
  would deviate if wrong).

## Open / TODO
- **Run param count on every option** (full model, all blocks together) to see how each
  behaves end-to-end, not just the mixer:
  - Opt B: 8 heads, 192/384 (paper ratios, biggest *free* state ≈ 590K) ← current lean
  - Opt C: 12 heads, 128/256 (paper ratios, state ≈ 393K)
  - key=2048, 8 heads: 256/512 (bigger-than-paper mixer, state ≈ 1.05M, costs params)
  - (Megatron default 16/32 for reference — big mixer but only 524K state)
- Then solve `--ffn-hidden-size` per option to land each on the Llama-3.2-1B param count.
- Decide final config (head count + whether to spend params on a bigger key_dim for state).
- Naming cleanup: the config in "What follows" below is **12 heads = table's Opt C** (NOT
  Opt A, which is the infeasible 9-head one) — relabel, or switch it to Opt B.
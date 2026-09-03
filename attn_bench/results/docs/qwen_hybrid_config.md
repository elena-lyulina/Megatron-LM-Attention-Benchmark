# Qwen-style hybrid (gated attention + GDN) — architecture research & config

Goal for **T3** (thesis endgame plan): add one hybrid model that interleaves a
**linear mixer (GDN)** with **gated softmax attention**, on the same LLaMA-3.2-1B
backbone as every other memorization baseline, param-matched to ~1.236 B. This doc
records what the real "gated + linear" hybrid (the Qwen3.5 / 3.8 family) looks
like, what we keep vs. drop, and the parameter-budget math for our replica.

---

## 1. What the real model is — the Qwen `gated-attn + GDN` hybrid

One architecture, three release names:

- **Qwen3-Next-80B-A3B** (Sep 2025, `model_type: qwen3_next`) — the origin. MoE
  only, no dense variant.
- **Qwen3.5** (`model_type: qwen3_5`) — same token-mixing recipe, **adds dense
  variants** (0.8B, 9B, 27B). Exposes the gate as `attn_output_gate: true`.
- **Qwen3.8** — still `model_type: qwen3_5`, still `attn_output_gate` + GDN 3:1
  for the mainline dense/MoE models (27B verified: no sparse keys). Only the
  experimental **Qwen3.8-Flash-Next** (Qwen4 preview) swaps the softmax half for
  QSA (Qwen Sparse Attention) — **out of scope**, we do not touch it.

Every one of them: `full_attention_interval: 4` → `layer_types` = 3 ×
`linear_attention` then 1 × `full_attention`, repeated (full attention at layer
indices 3, 7, 11, …). `hidden_act` silu (SwiGLU), RMSNorm `eps` 1e-6, GDN
`linear_conv_kernel_dim` 4, `partial_rotary_factor` 0.25 and `rope_theta` 1e7 on
the softmax layers, an MTP head (`mtp_num_hidden_layers: 1`). All are shipped as
VL models (`Qwen3_5ForConditionalGeneration` + a `vision_config`); the numbers
below are the `text_config`.

| field | Qwen3-Next 80B | **Qwen3.5-0.8B** | Qwen3.5-9B | Qwen3.8-27B |
|---|---|---|---|---|
| `hidden_size` / `num_hidden_layers` | 2048 / 48 | **1024 / 24** | 4096 / 32 | 5120 / 64 |
| linear : full layers | 36 : 12 | 18 : 6 | 24 : 8 | 48 : 16 |
| softmax `num_attention_heads` / `num_key_value_heads` / `head_dim` | 16 / 2 / 256 | 8 / 2 / 256 | 16 / 4 / 256 | 24 / 4 / 256 |
| `attn_output_gate` | (baked in) | true | true | true (`output_gate_type` swish) |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 32 | 16 / 16 | 16 / 32 | 16 / 48 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | 128 / 128 | 128 / 128 | 128 / 128 |
| ⇒ `key_dim` / `value_dim` vs hidden | 1.0× / 2.0× | 2.0× / 2.0× | 0.5× / 1.0× | 0.4× / 1.2× |
| `intermediate_size` | 5120 (+MoE 512 exp) | 3584 | 12288 | 17408 |
| `tie_word_embeddings` | false | **true** | false | false |
| `vocab_size` | 151 936 | 248 320 | 248 320 | 248 320 |

Key points:

- **"Gated attention"** = standard softmax attention (GQA) **plus a per-head
  sigmoid output gate** — exactly our `--attention-output-gate` (`gated` baseline).
  Qwen additionally uses *partial* rotary (0.25); our gated baseline uses full RoPE.
- **GDN head geometry is a fixed 128/128 head_dim at every model size**, with
  `linear_num_key_heads` **pinned at 16**; only the *value head count* grows
  (16 → 32 → 48). So the key path is always `16 × 128 = 2048`, independent of
  hidden. This matches neither the GDN paper (scale head_dim, 9 heads) nor our
  pure-GDN baseline (8 heads, 192/384, ratios 0.75/1.5). See
  `gated-delta-net-config.md`.
- **Closest analogue: `Qwen/Qwen3.5-0.8B`** — dense, tied embeddings, sub-1B, the
  exact `gated-attn + GDN` 3:1 recipe. Confirms this hybrid is a real deployed
  design at our scale, not a down-scaled 80B MoE.
- **Scale, MoE, MTP, the vision tower, `mrope` are not part of the replication.**
  What we replicate is the *interleaving pattern* — which mixer sits where, in
  what ratio.
- Related hybrids for later (T4): **Kimi Linear** = KDA + MLA (3:1); **Gemma2/3** =
  SWA + full (1:1). Same construction, different components.

Sources (raw `config.json`):
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json),
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json),
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json),
[Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json).

---

## 2. What we build — "our gated + our GDN", same backbone

Design principle: the hybrid must be a **clean interleave of two models we have
already trained**, so the only new variable is *mixing them per-layer*. We reuse
both component configs verbatim and change nothing else:

| component | source config | per-layer mixer params @hidden 2048 |
|---|---|---|
| gated softmax attention | `gated_attn_llama_1B_args_ffn7488.txt` (Llama GQA 32 Q / 8 KV, `kv_channels` 64, `--attention-output-gate`, full RoPE scf=1) | 14,682,112 |
| GDN mixer | `gdn_1B_args_8heads_ffn5824.txt` (8 K/V heads, 192/384, paper ratios, conv 4, `pos-emb none`) | 25,225,616 |

(Per-layer figures are the `mixer (rest)` category from each `_param_counts` file,
minus the shared 2,048 `final_layernorm`, divided by 16. Each includes that
layer's one 2,048-param input RMSNorm — fused into `in_proj` for GDN, into
`linear_qkv` for gated attention.)

**Layer pattern, 16 layers, Qwen 3:1** → `[1,1,1,0] × 4` = **12 GDN + 4 gated
attention**, gated layers at indices 3, 7, 11, 15.
Megatron flag: `--linear-attention-freq 4` (integer form reproduces
`(i+1)%4==0 → SDPA` exactly), or the explicit list `"([1]*3+[0]*1)*4"`.

`--attention-output-gate` is a global config flag read only by `SelfAttention`, so
in the hybrid it gates **only** the 4 softmax layers; the 12 GDN layers are
untouched (GDN's own gate is separate and hardcoded). Verified against
`megatron/core/transformer/attention.py` + `experimental_attention_variant_module_specs.py`.

### 2.1 Parameter budget

Everything except the mixer and the FFN is fixed and identical to the whole family:

| block | params | note |
|---|---|---|
| embeddings + output head (tied, vocab 128256) | 262,668,288 | same for all baselines |
| final layernorm | 2,048 | |
| MLP (SwiGLU, 16 layers) | `98,304 × ffn_hidden + 32,768` | fc1 `2·h·ffn` + fc2 `h·ffn`, ×16, + pre-MLP norm |
| hybrid mixer (12 GDN + 4 gated) | `12 × 25,225,616 + 4 × 14,682,112` = **361,435,840** | |

Target (softmax `full` baseline, and the band every sibling lands in):
**1,235,814,400**. Solve for `ffn_hidden`:

```
ffn* = (1,235,814,400 − 262,668,288 − 2,048 − 32,768 − 361,435,840) / 98,304
     = 6222.2                       → analytic exact-match FFN
```

Round to the house convention (multiple of 64):

| `--ffn-hidden-size` | TOTAL params | Δ vs 1,235,814,400 |
|---|---|---|
| **6208** (97 × 64) | **1,234,410,176** | **−0.114 %** |
| 6272 (98 × 64) | 1,240,701,632 | +0.395 % |

**Recommendation: `--ffn-hidden-size 6208`** (−0.11 %, comfortably inside the
sibling band: GDN +0.24 %, gated −0.17 %, MLA +0.19 %, KDA −0.05 %).

Non-embedding param match is equivalent here (embeddings are byte-identical across
the family): non-emb target 973,146,112, hybrid@6208 = 971,741,888 (−0.14 %).

### 2.2 Deliberate deviations from the Qwen recipe (all to stay comparable to our own baselines)

| Qwen (`qwen3_5` family) | our hybrid | why |
|---|---|---|
| GDN: `linear_num_key_heads` pinned at 16, `head_dim` 128/128, value-head count 16–48 | 8/8 heads, 192/384 (paper ratios 0.75/1.5) | match our pure-GDN baseline exactly |
| gated attn 8–24 Q / 2–4 KV, `head_dim` 256 | Llama GQA 32 Q / 8 KV, `head_dim` 64 | match our gated baseline exactly |
| partial RoPE 0.25 on attn layers, `theta` 1e7 | full RoPE, `theta` 5e5, scf=1 | match our gated baseline |
| MoE (Qwen3-Next / large MoE variants) | dense | whole family is dense 1B |
| MTP head (`mtp_num_hidden_layers: 1`) | none | not part of any of our baselines |
| 24–64 layers, 3:1 | 16 layers, 3:1 → 12 GDN + 4 attn | Llama-3.2-1B depth |
| `full_attention_interval` = 4 | `--linear-attention-freq 4` | **kept** — this is the thing being replicated |

---

## 3. Config & training plan (T3a / T3b)

**T3a — param-count config. ✅ DONE.**
`attn_bench/configs/param_count_configs/hybrid_qwen_1B_args_ffn6208.txt` — GDN
token-mixer block from `gdn_1B_args_8heads_ffn5824.txt` with
`--linear-attention-freq 4` (int → SDPA at layers 3/7/11/15), plus the gated
baseline's SDPA-layer flags (`--group-query-attention --num-query-groups 8
--attention-output-gate --position-embedding-type rope --rotary-base 500000
--use-rope-scaling --rope-scaling-factor 1 --max-position-embeddings 8192`), and
`--ffn-hidden-size 6208`. `count_model_param.slurm` ran →
`hybrid_qwen_1B_args_ffn6208_param_counts`: **TOTAL 1,234,410,176** (matches the
prediction, −0.114% vs `full`). Per-param dump confirms the split —
`in_proj.weight` 226,885,632 / (2048×9232) = **12 GDN**;
`linear_qkv.weight` 41,943,040 / (2048×5120) = **4 gated-attn**; `conv1d`/`A_log`/
`out_proj` all divide by 12; no positional params. `--group-query-attention` +
`gated_delta_net` + `rope` pass `validate_args` together (model built).

`--position-embedding-type rope` is model-wide (the SDPA layers need it; pure-GDN
uses `none`); GDN ignores the rotary tensor and RoPE has no params, so it is a
no-op for the 12 GDN layers — confirmed by the count (no stray positional params).

**T3b — pretrain slurm. ✅ Written.**
`attn_bench/submissions/pretrain_llama3_1b_hybrid_qwen_fineweb40B_gutenberg3B.slurm`
(+ `_test.slurm`). Forked from the **gated-attn scf=1** script (already has
`--attention-output-gate` + RoPE-scaling + 8 nodes / MBS=3 / GBS=288 / 18141 steps
/ wd 0.1 / warmup 500 / container `nemo_26.04_te2.15`); grafted in the GDN mixer
args + `--linear-attention-freq 4` + the per-rank TRITON cache
(`reference_gdn_triton_cache_crash`), `--ffn-hidden-size 6208`, `EXP_NAME`
`llama3-1b-hybrid-qwen-scf1-fineweb40B-gutenberg3B`, `--time 12:00:00`.
`--use-packed-seq-params` resets **both** the GDN recurrent/conv state and the
softmax attention mask at every document boundary. Container fla 0.4.2 is fine for
GDN — **no** 0.5.2 side-install (that is KDA-only).

Dist risk: MBS=3 with 12 GDN layers is untested (every prior GDN run used MBS=2).
Fallbacks in the script header: A = `--recompute-granularity selective` (keeps
8 nodes / MBS=3), B = `--nodes=12` + `MBS=2` (GBS=288 and 18141 steps unchanged →
no impact on the memorization comparison).

**T3c — smoke test.**
`pretrain_llama3_1b_hybrid_qwen_fineweb40B_gutenberg3B_test.slurm` — exact prod
config, `--time 00:15:00`, one srun (KDA/MLA `_test` structure): entry point
`pretrain_gpt_native.py --tests xdoc_mask/xdoc_loss/xdoc_position_ids/gdn/gated=pass
sink=fail` runs the wiring suites on the first forward step (Summary block; does
not fail the job) then falls through to real training until walltime.
`CHECKPOINTING_ARGS` kept + `CHECKPOINT_STEPS=50` so a `torch_dist` save of the
mixed GDN+attn checkpoint is exercised; no `--exit-signal-handler`. Run it first —
it answers the MBS=3 memory question and gives a throughput read.

**T3e (later, the real risk)** — hybrid inference: per-layer KV cache (4 softmax
layers) and recurrent state (12 GDN layers) must coexist in
`inference_backend.py`. Blocks the Qwen memorization sweep (T3f).

---

## 4. Sanity-check numbers (for `count_model_param` diff)

Per-layer mixer, hidden 2048, from the `_param_counts` files:

| mixer | per-layer params | breakdown |
|---|---|---|
| full softmax attn | 10,487,808 | qkv `2048×3072` + proj `2048×2048` + in-norm `2048` |
| gated attn (+output gate) | 14,682,112 | qkv `2048×5120` (gate adds `2048×2048`) + proj `2048×2048` + in-norm `2048` |
| GDN (8h, 192/384) | 25,225,616 | in_proj `2048×9232` + out_proj `3072×2048` + conv1d `6144×4` + out-norm `384` + in-norm `2048` + A_log/dt_bias `8+8` |

Hybrid mixer total = `12 × 25,225,616 + 4 × 14,682,112 = 302,707,392 + 58,728,448
= 361,435,840`.

MLP@ffn6208 = `98,304 × 6208 + 32,768 = 610,304,000`.

TOTAL@ffn6208 = `262,668,288 + 2,048 + 610,304,000 + 361,435,840 = 1,234,410,176`.

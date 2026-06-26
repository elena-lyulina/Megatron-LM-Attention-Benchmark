# GDN State-Carry Throughput Investigation

**Question.** In the production GDN state-carry runs (`2622827` r=0, `2622828` r=0.5, `2622831` r=1), r=0 — the standard no-carry / no-packed-seq path — showed **lower and much jitterier throughput** than r=1, monotonically with the carry ratio. Was this caused by the carry setting (or our code / the allocator / expert imbalance), or by something environmental?

**Short answer.** It was **node placement.** The three production runs were separate jobs on three disjoint 14-node sets; r=0 simply drew a worse set. On a single shared allocation the carry ratio shows **no consistent throughput effect** (it even reverses). Not the carry code, not the allocator, not our fork, and not expert imbalance (the model is dense — no MoE).

---

## Controlled bench — job `2624605`

Four passes run **sequentially in one 14-node allocation** (same nodes, same container `nemo_26.04_te2.15`), 300 iters each, so node placement is held constant and only the variable under test changes per pass. Config is copied verbatim from `pretrain_llama3_1b_gdn_carry_r0_fineweb40B_gutenberg3B.slurm`; the only deltas are the short `--train-iters`, fresh init (no ckpt load/save), and the per-pass knob.

Script: `attn_bench/submissions/bench_gdn_carry_throughput.slurm`
Logs: `attn_bench/logs/2624605.{out,err}` · per-pass: `attn_bench/results/pretrain-bench/gdn-carry-throughput-bench/{r0,r1,r0_expseg,upstream}-2624605/pass.out`

| WandB run | Description & why included | xdoc attn | Impl | Carries state across batches | Avg throughput (TFLOP/s/GPU) | Avg iter time (ms) | CV |
|---|---|---|---|---|---|---|---|
| `bench-r0-2624605` | GDN, state-carry **OFF** (ratio 0.0) — state zeroed each batch. The "standard Megatron" no-carry path; the variant that looked slowest/jitteriest in production, so the baseline to reproduce. | No | Local (fork) | **No** (ratio 0.0) | **300.1** | **1292** | **3.0%** |
| `bench-r1-2624605` | GDN, state-carry **ON** (ratio 1.0) — recurrent+conv state carried across every batch. Looked fastest/smoothest in production; tests whether carry-on is what smooths throughput. | No | Local (fork) | **Yes** (ratio 1.0) | 288.9 | 1349 | 9.2% |
| `bench-r0_expseg-2624605` | Same as r0 (carry OFF) **+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**. Tests whether r0's production jitter was CUDA-allocator fragmentation/churn. | No | Local (fork) | No (ratio 0.0) | 295.5 | 1316 | 6.2% |
| `bench-upstream-2624605` | **Untouched upstream Megatron** (fresh clone), GDN no-carry, no packed-seq. Tests whether the carry wrapper code — even when disabled — adds overhead vs vanilla; i.e. "is it our code?" | No | Upstream | No (native, no carry feature) | 297.0 | 1308 | 5.1% |

**CV** = coefficient of variation = `std(iteration time) / mean(iteration time) × 100%`, over the 300 per-iter elapsed times (first 20 warmup iters dropped). It is a unitless **jitter** measure — how much each step's duration wobbles around the average, independent of absolute speed (that's the mean / throughput). Low CV = steady; high CV = bursty/stally.

Notes on the columns:
- **xdoc attn = No for all** — these are GDN linear-attention, not the softmax xdoc-attention-leak experiment. (All four are also unpacked, so GDN *state* leaks across docs within a sequence — but that is identical across passes, not a variable.)
- **Throughput and iter time are inverses** (TFLOP/s = work ÷ time); same story from two sides.

---

## Conclusion

On identical nodes the production ordering **does not reproduce — it reverses**: r=0 is now the *fastest and steadiest* (300 TFLOP/s, CV 3.0%) and r=1 the *slowest and jitteriest* (CV 9.2%). Therefore:

- **Not the carry setting** — the r0-vs-r1 gap vanished and flipped on shared nodes. (Loss is also unaffected; r=0 trains best.)
- **Not the allocator** — `expandable_segments` did not help; plain r0 was already cleanest.
- **Not our code** — upstream ≈ r0_expseg, mid-pack.
- **Not expert imbalance** — the model is dense (GDN mixer + dense SwiGLU FFN, no MoE/router).

→ The production r=0/r=0.5/r=1 throughput differences were **node-placement variance** (three jobs, three disjoint node sets). The "intrinsic-looking" signals in the production logs (a monotonic stall-free floor, persistent per-bucket jitter) are exactly what a consistently-slower node set produces, and only the same-nodes control disambiguates them.

**Caveats.** One bench run; r1's 9.2% is partly one unlucky 12-min pass on the shared nodes. Same nodes but different times (sequential), so it controls hardware but not perfectly for time-varying I/O. The reversal nonetheless makes a carry-causal story untenable. Repeat 2–3× for an airtight result.

# Memorization Measurement Grid

Running log of `measure_mem_all.sh` / `measure_mem.slurm` job batches (see
`memorization_measurement.md` for the pipeline itself). New batches are
appended below as they finish.

---

## Batch: scf=1 variants, suffix=250, 2026-08-24

Each point sweeps the same 6 reps (`1,16,32,64,128,256`), so grid size = points × 6.

| variant | Slurm job | start (CEST) | end (CEST) | run time | points (done/grid) | point-reps processed | point-reps skipped | status |
|---|---|---|---|---|---|---|---|---|
| full-scf1 | `3169111` | 01:26 | 03:02 | 1h 36m | 54/54 | 99 | 225 | COMPLETED |
| swa-w256-scf1 | `3169212` | 01:41 | 03:21 | 1h 40m | 58/58 | 101 | 247 | COMPLETED |
| swa-w1024-scf1 | `3169214` | 01:41 | 04:07 | 2h 26m | 58/58 | 161 | 187 | COMPLETED |
| swa-w4096-scf1 | `3169216` | 01:42 | 04:28 | 2h 45m | 58/58 | 174 | 174 | COMPLETED |
| sink-attn-scf1 | `3169096` | 01:22 | 06:07 | 4h 45m | 58/58 | 348 | 0 | COMPLETED |
| gated-attn-scf1 | `3169170` | 01:34 | 06:33 | 4h 58m | 58/58 | 348 | 0 | COMPLETED |
| gdn | `3169304` | 02:02 | 02:50 | 47m | 6/63 (+1 partial) | 37 | 0 | **FAILED (OOM)** |
| gdn-carry-r1 | `3169305` | 02:02 | 02:50 | 47m | 6/63 (+1 partial) | 37 | 0 | **FAILED (OOM)** |

"Point-reps skipped" = `(point, rep)` pairs already covered by a prior run at
suffix ≥ 250 (see suffix-reuse in `memorization_measurement.md`) — expected
and cheap, not an error. sink/gated had no prior results to reuse, hence 0
skipped.

**GDN failure.** Both GDN jobs (Megatron backend — no cached results to skip,
runs inference from scratch) died on POINT 7/63 (`offset=0 prefix=5942
suffix=250 rep=1`, 660-sequence batch):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 59.16 GiB.
```

in `module.py:406` (`float16_to_fp32`), converting the full-batch logits to
fp32. Points 1–6 (prefix 50→3971) fit fine; the fixed batch size of 660 stops
fitting once prefix reaches 5942. The whole SLURM step aborts on the OOM, so
points 7–63 never ran for either GDN variant — no results beyond prefix=3971
for `gdn` / `gdn-carry-r1`.

Logs: `attn_bench/logs/3169111.{out,err}`, `3169212.{out,err}`,
`3169214.{out,err}`, `3169216.{out,err}`, `3169304.{out,err}`,
`3169305.{out,err}`; `3169096.out` + `_logs/3169096.err`, `3169170.out` +
`_logs/3169170.err` (`.err` moved to `_logs/` for exceeding 3 MB).

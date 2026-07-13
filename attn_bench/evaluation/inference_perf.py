"""
Raw prefill / decode timing profiler, per attention architecture.

Random token ids (content doesn't matter for timing); everything stored raw,
no averaging/discarding at collection time. Each (batch_size, prefix_length)
combo writes its own file under --output-dir; already-finished ones (status
"ok") are skipped on rerun -- see inference_perf_units.py for unit naming
and the completion check used by the slurm wrapper.

Usage (single GPU, via torchrun):
    torchrun --nproc_per_node=1 attn_bench/evaluation/inference_perf.py \
        --ckpt-dir $MODEL_DIR/checkpoints \
        --tokenizer-path $TOKENIZER_PATH \
        --output-dir $SCRATCH_DIR \
        --model-tag full
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch

from attn_bench.evaluation.inference_common import (greedy_generate,
                                                    load_megatron_model)
from attn_bench.evaluation.inference_perf_units import (DECODE_BATCH_SIZE,
                                                        DECODE_PREFIX_ANCHORS,
                                                        PREFILL_BATCH_SIZE,
                                                        SWEEP_BATCH_SIZES,
                                                        SWEEP_PREFIX,
                                                        already_done,
                                                        decode_rel_path,
                                                        prefill_rel_path,
                                                        sweep_rel_path)

### SWEEP CONSTANTS (not naming -- see inference_perf_units.py for that) ###

PREFILL_LENGTHS = [50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]
PREFILL_REPEATS = 5

DECODE_STEPS = 8192
DECODE_FLUSH_EVERY = 250

SWEEP_DECODE_STEPS = 100


### TIMING ###

class TimingCollector:
    """torch.cuda.Event pairs around greedy_generate's prefill/decode callbacks.

    No sync during the run (decode is already serialized by its own data
    dependency, so syncing per step would only add overhead). elapsed_time()
    is computed lazily at flush time.
    """

    def __init__(self, meta: dict, out_path: Path | None = None, flush_every: int | None = None):
        self.out_path = out_path
        self.meta = meta
        self.flush_every = flush_every
        self.events: list[torch.cuda.Event] = []

    def mark_start(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)

    def prefill_callback(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)

    def decode_step_callback(self, step_t: int):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append(e)
        if self.flush_every is not None and (step_t + 1) % self.flush_every == 0:
            self.flush("running")
            print(f"  bs{self.meta.get('batch_size')}_prefix{self.meta.get('prefix_length')}: "
                  f"step {step_t + 1}/{self.meta.get('decode_steps')}")

    def elapsed_ms(self) -> list[float]:
        torch.cuda.synchronize()
        return [self.events[i].elapsed_time(self.events[i + 1]) for i in range(len(self.events) - 1)]

    def flush(self, status: str):
        ms = self.elapsed_ms()
        record = {
            **self.meta,
            "status": status,
            "prefill_ms": ms[0] if ms else None,
            "decode_step_ms": ms[1:],
        }
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(record, f)
        tmp.replace(self.out_path)


def vocab_size(model) -> int:
    from megatron.core.utils import unwrap_model
    return unwrap_model(model).shared_embedding_or_output_weight().shape[0]


def random_prompt(batch_size: int, prefix_length: int, vocab: int, device) -> torch.Tensor:
    return torch.randint(0, vocab, (batch_size, prefix_length), dtype=torch.long, device=device)


### EXPERIMENTS ###

def run_prefill_sweep(model, vocab: int, device, output_dir: Path, base_meta: dict):
    rel_path = prefill_rel_path()
    if already_done(output_dir, rel_path):
        print(f"Skipping prefill sweep -- already done: {output_dir / rel_path}")
        return

    out_path = output_dir / rel_path
    repeat_ms: dict[str, list[float]] = {}
    for length in PREFILL_LENGTHS:
        times = []
        for _ in range(PREFILL_REPEATS):
            prompt = random_prompt(PREFILL_BATCH_SIZE, length, vocab, device)
            collector = TimingCollector(meta={})
            collector.mark_start()
            greedy_generate(model, prompt, suffix_length=1,
                            prefill_callback=collector.prefill_callback)
            times.append(collector.elapsed_ms()[0])
            del prompt
        repeat_ms[str(length)] = times
        print(f"  prefill length={length}: {times}")

    record = {
        **base_meta,
        "batch_size": PREFILL_BATCH_SIZE,
        "prefix_lengths": PREFILL_LENGTHS,
        "repeat_ms": repeat_ms,
        "status": "ok",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f)


def run_decode_unit(model, vocab: int, device, output_dir: Path, base_meta: dict,
                    rel_path: str, batch_size: int, prefix_length: int, decode_steps: int,
                    flush_every: int | None):
    if already_done(output_dir, rel_path):
        print(f"Skipping {rel_path} -- already done")
        return

    out_path = output_dir / rel_path
    meta = {
        **base_meta,
        "batch_size": batch_size,
        "prefix_length": prefix_length,
        "decode_steps": decode_steps,
    }
    collector = TimingCollector(meta, out_path, flush_every=flush_every)
    prompt = random_prompt(batch_size, prefix_length, vocab, device)
    try:
        collector.mark_start()
        greedy_generate(model, prompt, suffix_length=decode_steps + 1,
                        prefill_callback=collector.prefill_callback,
                        decode_step_callback=collector.decode_step_callback)
        collector.flush("ok")
        print(f"  {rel_path}: done")
    except torch.cuda.OutOfMemoryError:
        print(f"  {rel_path}: OOM -- saving partial results")
        collector.flush("oom")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {rel_path}: error ({e}) -- saving partial results")
        collector.flush("error")
        torch.cuda.empty_cache()
    finally:
        del prompt
        torch.cuda.empty_cache()


### METADATA / MAIN ###

def write_run_metadata(output_dir: Path, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True, help="Where this model's result files are read from and written to")
    parser.add_argument("--model-tag", required=True, help="Model tag, e.g. full/gated/sink/gdn")
    parser.add_argument("--container-env", default=None)
    parser.add_argument("--megatron-extra-args", nargs=argparse.REMAINDER, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    base_meta = {
        "model_tag": args.model_tag,
        "ckpt_dir": args.ckpt_dir,
        "container_env": args.container_env,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_run_metadata(output_dir, base_meta)

    model = load_megatron_model(args.ckpt_dir, args.tokenizer_path, args.megatron_extra_args)
    device = next(model.parameters()).device
    vocab = vocab_size(model)

    print(f"=== Prefill sweep ({args.model_tag}) ===")
    run_prefill_sweep(model, vocab, device, output_dir, base_meta)

    print(f"=== Long decode ({args.model_tag}) ===")
    for prefix_length in DECODE_PREFIX_ANCHORS:
        run_decode_unit(model, vocab, device, output_dir, base_meta,
                        rel_path=decode_rel_path(DECODE_BATCH_SIZE, prefix_length),
                        batch_size=DECODE_BATCH_SIZE, prefix_length=prefix_length,
                        decode_steps=DECODE_STEPS, flush_every=DECODE_FLUSH_EVERY)

    print(f"=== Batch-size sweep ({args.model_tag}) ===")
    for batch_size in SWEEP_BATCH_SIZES:
        run_decode_unit(model, vocab, device, output_dir, base_meta,
                        rel_path=sweep_rel_path(batch_size, SWEEP_PREFIX),
                        batch_size=batch_size, prefix_length=SWEEP_PREFIX,
                        decode_steps=SWEEP_DECODE_STEPS, flush_every=None)

    print(f"\nAll done. Results in: {output_dir}")


if __name__ == "__main__":
    main()
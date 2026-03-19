# Architectural Determinants of Memorization: A Systematic Benchmark of Attention Mechanisms

Benchmarking suite for attention mechanisms and their implementations on NVIDIA GH200 (Hopper) clusters.

## Structure

```
attn_bench/
├── kernels/        # Implementation of attention mechanisms
│   ├── full/
│   ├── sliding_window/
│   ├── sparse_dilated/
│   └── gated/
├── training/       # Megatron integration layer (model, train step, distributed init)
├── benchmarks/     # Correctness and performance benchmarks
│   ├── correctness/
│   └── performance/
├── configs/        # YAML configuration files
├── results/        # Benchmark outputs
│   ├── correctness/
│   └── performance/
├── scripts/        # SLURM job launchers
├── analysis/       # Notebooks for plotting and result analysis
├── utils/          # Shared utilities (git info capture, result writing)
└── submit.py       # Preflight checks + sbatch submission
```
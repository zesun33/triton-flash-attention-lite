# FlashAttention Lite (OpenAI Triton)

Modern GPU kernel development with the Triton compiler.

> **Status:** roadmap stub. Notes and source files listed below are **planned** and not in-tree yet.

## Goals

This repo will cover:
- Triton compiler and Python-like GPU programming
- FlashAttention algorithm (tiling Q, K, V in SRAM)
- Block-level memory management
- Online softmax computation
- Memory-efficient attention for long sequences

## Prerequisites

- **Triton**: `pip install triton`
- **NVIDIA GPU**: Compute Capability 7.0+ (Volta or newer)
- **Python 3.8+**

## Learning Roadmap (Planned)

### Phase 1: Theory (Notes)

| # | File | Topic | Status |
|---|------|-------|--------|
| 0 | `notes/00_triton_basics.md` | Triton vs CUDA, blocks, programming model | planned |
| 1 | `notes/01_flash_attention_algorithm.md` | FlashAttention tiling and online softmax | planned |
| 2 | `notes/02_memory_efficiency.md` | HBM reduction, IO complexity | planned |

### Phase 2: Implementation

| # | File | Concept | Status |
|---|------|---------|--------|
| 1 | `src/01_pytorch_attention.py` | Baseline PyTorch attention | planned |
| 2 | `src/02_triton_matmul.py` | Simple Triton matmul (warmup) | planned |
| 3 | `src/03_triton_softmax.py` | Triton fused softmax | planned |
| 4 | `src/04_flash_attention.py` | Full FlashAttention kernel | planned |

### Phase 3: Benchmarking

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | `benchmarks/benchmark.py` | Compare latency and memory | planned |

## Planned Directory Structure

```
triton-flash-attention-lite/
├── README.md
├── requirements.txt
├── src/
│   ├── 01_pytorch_attention.py
│   ├── 02_triton_matmul.py
│   ├── 03_triton_softmax.py
│   └── 04_flash_attention.py
├── notes/
│   ├── 00_triton_basics.md
│   ├── 01_flash_attention_algorithm.md
│   └── 02_memory_efficiency.md
└── benchmarks/
    └── benchmark.py
```

## Expected Results (Target)

For sequence length N=4096, head_dim=64:

| Implementation | Time | Memory | Speedup |
|----------------|------|--------|---------|
| PyTorch Standard | ~50ms | O(N²) | 1× |
| Triton FlashAttention | ~10ms | O(N) | 5× |

## References

- [Triton Documentation](https://triton-lang.org/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

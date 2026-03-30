# FlashAttention Lite (OpenAI Triton)

A learning-focused project to understand modern GPU kernel development using Triton compiler.

## 🎯 Learning Goals

By completing this project, you will understand:
- Triton compiler and Python-like GPU programming
- FlashAttention algorithm (tiling Q, K, V in SRAM)
- Block-level memory management
- Online softmax computation
- Memory-efficient attention for long sequences

## ⚠️ Prerequisites

- **Triton**: `pip install triton`
- **NVIDIA GPU**: Compute Capability 7.0+ (Volta or newer)
- **Python 3.8+**

## 📚 Learning Roadmap

### Phase 1: Theory (Notes)

| # | Note | Topic | Status |
|---|------|-------|--------|
| 0 | [00_triton_basics.md](notes/00_triton_basics.md) | Triton vs CUDA, blocks, programming model | ⬜ |
| 1 | [01_flash_attention_algorithm.md](notes/01_flash_attention_algorithm.md) | FlashAttention tiling and online softmax | ⬜ |
| 2 | [02_memory_efficiency.md](notes/02_memory_efficiency.md) | HBM reduction, IO complexity | ⬜ |

### Phase 2: Implementation

| # | File | Concept | Prereq Notes | Status |
|---|------|---------|--------------|--------|
| 1 | [01_pytorch_attention.py](src/01_pytorch_attention.py) | Baseline PyTorch attention | - | ⬜ |
| 2 | [02_triton_matmul.py](src/02_triton_matmul.py) | Simple Triton matmul (warmup) | 0 | ⬜ |
| 3 | [03_triton_softmax.py](src/03_triton_softmax.py) | Triton fused softmax | 0 | ⬜ |
| 4 | [04_flash_attention.py](src/04_flash_attention.py) | Full FlashAttention kernel | 0, 1 | ⬜ |

### Phase 3: Benchmarking

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | [benchmark.py](benchmarks/benchmark.py) | Compare latency and memory | ⬜ |

## 📁 Directory Structure

```
triton-flash-attention-lite/
├── README.md
├── requirements.txt
├── src/
│   ├── 01_pytorch_attention.py   # Baseline
│   ├── 02_triton_matmul.py       # Warmup: simple matmul
│   ├── 03_triton_softmax.py      # Fused softmax
│   └── 04_flash_attention.py     # Full FlashAttention
├── notes/
│   ├── 00_triton_basics.md
│   ├── 01_flash_attention_algorithm.md
│   └── 02_memory_efficiency.md
└── benchmarks/
    └── benchmark.py
```

## 🔧 Setup & Run

```bash
# Install Triton
pip install triton

# Run baseline
python src/01_pytorch_attention.py

# Run FlashAttention
python src/04_flash_attention.py

# Benchmark
python benchmarks/benchmark.py --seq_len 4096
```

## 📊 Expected Results

For sequence length N=4096, head_dim=64:

| Implementation | Time | Memory | Speedup |
|----------------|------|--------|---------|
| PyTorch Standard | ~50ms | O(N²) | 1x |
| Triton FlashAttention | ~10ms | O(N) | 5x |

## 🔗 Relevance

**Industry Skills**: Triton is used by Meta, OpenAI for production kernels. Understanding compiler-based GPU programming is a "unicorn skill."

## 📖 References

- [Triton Documentation](https://triton-lang.org/)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [OpenAI Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)

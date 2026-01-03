# FlashAttention Lite (OpenAI Triton)

## Goal
Implement a simplified, fused Attention kernel (`Q @ K.T @ V`) using Triton to demonstrate understanding of modern AI compiler optimization and block-level memory management.

## Roadmap
- [ ] **01_pytorch_attention.py**: Baseline implementation using `torch.matmul` and `torch.softmax`.
- [ ] **02_triton_fused_attention.py**: Custom Triton kernel implementing the FlashAttention tiling logic.
    - [ ] Load Q, K, V blocks into SRAM.
    - [ ] Compute Softmax normalization online.
    - [ ] Store output blocks.
- [ ] **benchmark.py**: Compare latency and memory usage vs PyTorch Baseline on long sequences (N=4096+).

## References
*   [Triton Documentation](https://triton-lang.org/)
*   [FlashAttention Paper](https://arxiv.org/abs/2205.14135)

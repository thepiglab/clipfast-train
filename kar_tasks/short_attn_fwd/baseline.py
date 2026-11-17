"""Baseline: PyTorch's FlashAttention-2 forward (what SDPA runs on sm_120), which also returns LSE."""
import torch


def kernel_fn(q, k, v):
    BH, N, D = q.shape
    q4, k4, v4 = (t.view(1, BH, N, D) for t in (q, k, v))
    return torch.ops.aten._scaled_dot_product_flash_attention(q4, k4, v4, 0.0, False, False, scale=0.125)[0].view(BH, N, D)

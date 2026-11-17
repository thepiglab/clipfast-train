"""Baseline: what PyTorch runs today on sm_120: FlashAttention-2 backward (aten._scaled_dot_product_flash_attention
backward, which also launches its dq-convert kernel)."""
import torch

SCALE = 0.125


def kernel_fn(q, k, v, o, lse, do):
    BH, N, D = q.shape
    q4, k4, v4, o4, do4 = (t.view(1, BH, N, D) for t in (q, k, v, o, do))
    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        do4, q4, k4, v4, o4, lse.view(1, BH, N), None, None, N, N, 0.0, False,
        torch.empty(2, dtype=torch.int64, device=q.device), torch.empty((), dtype=torch.int64, device=q.device),
        scale=SCALE)
    return torch.stack((dq.view(BH, N, D), dk.view(BH, N, D), dv.view(BH, N, D)))

"""What autograd runs: cuBLAS mm -> dact (bf16), then aten.gelu_backward(dact, pre)."""
import torch


def kernel_fn(dy, w2, pre):
    return torch.ops.aten.gelu_backward(dy @ w2, pre)

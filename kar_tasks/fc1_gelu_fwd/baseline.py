"""What PyTorch runs: cuBLAS addmm (bias in the epilogue) -> pre, then a GELU kernel reading pre -> act."""
import torch


def kernel_fn(x, w, b):
    out = torch.empty(2, x.shape[0], w.shape[0], device=x.device, dtype=torch.bfloat16)
    torch.addmm(b, x, w.t(), out=out[0])
    torch.ops.aten.gelu.out(out[0], out=out[1])
    return out

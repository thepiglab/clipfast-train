"""Opt-in FP8 forward GEMMs for training (e4m3, rowwise scales = per token for X, per output channel for W).

Forward:  y = (Xq * sx) (Wq * sw)^T + b via torch._scaled_mm (FP8 tensor cores, 825 vs 437 TFLOP/s measured).
Backward: bf16 GEMMs on the saved bf16 X and bf16 W (dX = dY W, dW = dY^T X), so gradients keep bf16 accuracy;
only the forward activations carry FP8 rounding (current scaling, no amax history: nothing stateful).
Under torch.compile the amax/scale/cast of X fuses into the producing LayerNorm kernel.
"""
import torch

E4M3_MAX = 448.0


def _q_rows(t):
    """t [R, K] float -> (e4m3 [R, K], fp32 scale [R, 1]) with t ~= q * s."""
    s = t.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / E4M3_MAX
    return (t.float() / s).to(torch.float8_e4m3fn), s


class _FP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1])
        xq, sx = _q_rows(x2)
        Wb = W.to(torch.bfloat16)
        wq, sw = _q_rows(Wb)
        y = torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw.t(), bias=None if b is None else b.to(torch.bfloat16),
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(x2.to(torch.bfloat16), Wb)
        ctx.shp, ctx.has_b, ctx.wdt = shp, b is not None, W.dtype
        return y.view(*shp[:-1], W.shape[0])

    @staticmethod
    def backward(ctx, dy):
        xb, Wb = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(torch.bfloat16)
        dx = (dy2 @ Wb).view(ctx.shp) if ctx.needs_input_grad[0] else None
        dW = (dy2.t() @ xb).to(ctx.wdt) if ctx.needs_input_grad[1] else None
        db = dy2.float().sum(0).to(ctx.wdt) if ctx.has_b and ctx.needs_input_grad[2] else None
        return dx, dW, db


def fp8_linear(x, W, b):
    """Rows must be a multiple of 16 is NOT required; K and N must be multiples of 16 (true for all BiomedCLIP GEMMs)."""
    with torch.autocast("cuda", enabled=False):
        return _FP8Linear.apply(x, W, b)

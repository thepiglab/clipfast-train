"""Transformer MLP (fc1 -> exact GELU -> fc2) as a custom op with GEMM-epilogue fusion in forward AND backward.

forward:  pre, act = fc1_gelu(x)                (Triton, MODE 1: bias + GELU epilogue, writes both)
          y = act @ W2^T + b2                   (cuBLAS addmm)
backward: dpre = (dy @ W2) * GELU'(pre)         (Triton, MODE 3: dGELU epilogue; no dact tensor, no gelu_bwd pass)
          dW2 = dy^T act, db2 = sum dy, dW1 = dpre^T x, db1 = sum dpre, dx = dpre @ W1   (cuBLAS)
Saved for backward: x, pre, act (bf16) -- the same tensors autograd saves for the unfused version.
Compute in bf16 with fp32 accumulation (autocast semantics: weights cast to bf16 per call, fp32 master grads).
"""
import torch

from .gemm_epi import dgelu_mm, fc1_gelu


@torch.library.custom_op("cftrain::mlp_fwd", mutates_args=())
def mlp_fwd(x: torch.Tensor, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor,
            b2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape[0], w1.shape[0]
    pre = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)   # separate buffers: outputs may not alias
    act = torch.empty_like(pre)
    fc1_gelu(x, w1, b1, pre=pre, act=act)
    y = torch.addmm(b2, act, w2.t())
    return y, pre, act


@mlp_fwd.register_fake
def _(x, w1, b1, w2, b2):
    M = x.shape[0]
    return x.new_empty(M, w2.shape[0]), x.new_empty(M, w1.shape[0]), x.new_empty(M, w1.shape[0])


@torch.library.custom_op("cftrain::mlp_bwd", mutates_args=())
def mlp_bwd(dy: torch.Tensor, x: torch.Tensor, pre: torch.Tensor, act: torch.Tensor, w1: torch.Tensor,
            w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dy = dy.contiguous()
    dpre = dgelu_mm(dy, w2, pre)
    dx = dpre @ w1
    dw1 = dpre.t() @ x
    db1 = dpre.sum(0, dtype=torch.float32).to(dpre.dtype)   # no fp32 copy of [M,3072]; bf16 out like autocast
    dw2 = dy.t() @ act
    db2 = dy.sum(0, dtype=torch.float32).to(dy.dtype)
    return dx, dw1, db1, dw2, db2


@mlp_bwd.register_fake
def _(dy, x, pre, act, w1, w2):
    return (x.new_empty(x.shape), w1.new_empty(w1.shape), w1.new_empty(w1.shape[0]),
            w2.new_empty(w2.shape), w2.new_empty(w2.shape[0]))


@torch.library.custom_op("cftrain::mlp_bwd_recompute", mutates_args=())
def mlp_bwd_recompute(dy: torch.Tensor, x: torch.Tensor, pre: torch.Tensor, w1: torch.Tensor,
                      w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    act = torch.nn.functional.gelu(pre)          # recomputed (bf16 out, fp32 math inside): not kept from forward
    return mlp_bwd(dy, x, pre, act, w1, w2)


@mlp_bwd_recompute.register_fake
def _(dy, x, pre, w1, w2):
    return (x.new_empty(x.shape), w1.new_empty(w1.shape), w1.new_empty(w1.shape[0]),
            w2.new_empty(w2.shape), w2.new_empty(w2.shape[0]))


SAVE_ACT = True          # False: keep only the fc1 pre-activation for backward (-6 KB/token/layer), recompute GELU


def _setup(ctx, inputs, output):
    x, w1, b1, w2, b2 = inputs
    ctx.save_act = SAVE_ACT
    if SAVE_ACT:
        ctx.save_for_backward(x, output[1], output[2], w1, w2)
    else:
        ctx.save_for_backward(x, output[1], w1, w2)


def _backward(ctx, dy, dpre_unused, dact_unused):
    if ctx.save_act:
        x, pre, act, w1, w2 = ctx.saved_tensors
        return mlp_bwd(dy, x, pre, act, w1, w2)
    x, pre, w1, w2 = ctx.saved_tensors
    return mlp_bwd_recompute(dy, x, pre, w1, w2)


mlp_fwd.register_autograd(_backward, setup_context=_setup)


MIN_ROWS = 16384          # kar: the Triton GEMM loses to cuBLAS below ~16k rows (m12608: -7..-12 %)


def fused_mlp(x, w1, b1, w2, b2):
    """x [..., 768] (fp32 or bf16), fp32 master params -> fc2(GELU(fc1(x))) in bf16 with fused epilogues."""
    shp = x.shape
    bf = torch.bfloat16
    x2 = x.reshape(-1, shp[-1]).to(bf).contiguous()
    with torch.autocast("cuda", enabled=False):
        y = mlp_fwd(x2, w1.to(bf), b1.to(bf), w2.to(bf), b2.to(bf))[0]
    return y.view(*shp[:-1], w2.shape[0])

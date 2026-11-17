"""dq, dk, dv of O = softmax(Q K^T / 8) V. reference: fp32 math, bf16 out; exact: fp64."""
import torch

SCALE = 0.125


def _bwd(q, k, v, do, dt):
    q, k, v, do = (t.to(dt) for t in (q, k, v, do))
    p = torch.softmax((q @ k.transpose(1, 2)) * SCALE, -1)
    o = p @ v
    dv = p.transpose(1, 2) @ do
    dp = do @ v.transpose(1, 2)
    delta = (do * o).sum(-1, keepdim=True)
    ds = p * (dp - delta) * SCALE
    return ds @ k, ds.transpose(1, 2) @ q, dv


def reference(q, k, v, o, lse, do):
    return torch.stack(_bwd(q, k, v, do, torch.float32)).to(torch.bfloat16)


def exact(q, k, v, o, lse, do):
    return torch.stack(_bwd(q, k, v, do, torch.float64))

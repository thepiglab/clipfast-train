import math

import torch


def _f(dy, w2, pre, dt):
    p = pre.to(dt)
    dg = 0.5 * (1 + torch.erf(p / math.sqrt(2))) + p * torch.exp(-0.5 * p * p) / math.sqrt(2 * math.pi)
    return (dy.to(dt) @ w2.to(dt)) * dg


def reference(dy, w2, pre):
    return _f(dy, w2, pre, torch.float32).to(torch.bfloat16)


def exact(dy, w2, pre):
    return _f(dy, w2, pre, torch.float64)

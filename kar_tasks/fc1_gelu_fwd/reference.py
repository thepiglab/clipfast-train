import torch


def _f(x, w, b, dt):
    pre = x.to(dt) @ w.to(dt).T + b.to(dt)
    return torch.stack((pre, torch.nn.functional.gelu(pre)))


def reference(x, w, b):
    return _f(x, w, b, torch.float32).to(torch.bfloat16)


def exact(x, w, b):
    return _f(x, w, b, torch.float64)

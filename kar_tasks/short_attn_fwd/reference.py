import torch

SCALE = 0.125


def _fwd(q, k, v, dt):
    q, k, v = (t.to(dt) for t in (q, k, v))
    return torch.softmax((q @ k.transpose(1, 2)) * SCALE, -1) @ v


def reference(q, k, v):
    return _fwd(q, k, v, torch.float32).to(torch.bfloat16)


def exact(q, k, v):
    return _fwd(q, k, v, torch.float64)

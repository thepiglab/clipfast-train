import torch

SCALE = 0.125


def make(seed, BH, N, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    q, k, v, do = (torch.randn(BH, N, 64, generator=g, device=device).to(torch.bfloat16) for _ in range(4))
    s = (q.float() @ k.float().transpose(1, 2)) * SCALE
    lse = torch.logsumexp(s, -1)
    o = (torch.softmax(s, -1) @ v.float()).to(torch.bfloat16)
    return q, k, v, o, lse.contiguous(), do

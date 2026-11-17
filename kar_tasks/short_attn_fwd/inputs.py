import torch


def make(seed, BH, N, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(BH, N, 64, generator=g, device=device).to(torch.bfloat16) for _ in range(3))

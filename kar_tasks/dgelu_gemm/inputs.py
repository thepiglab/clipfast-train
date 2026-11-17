import torch


def make(seed, M, K, N, device="cuda"):
    """K = 768 (fc2 out features), N = 3072 (fc1 out features)."""
    g = torch.Generator(device=device).manual_seed(seed)
    dy = (0.01 * torch.randn(M, K, generator=g, device=device)).to(torch.bfloat16)
    w2 = (torch.randn(K, N, generator=g, device=device) / N ** 0.5).to(torch.bfloat16)
    pre = torch.randn(M, N, generator=g, device=device).to(torch.bfloat16)
    return dy, w2, pre

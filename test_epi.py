"""Unit tests + timing: fused-epilogue MLP (cftrain.mlp_epi) and Triton ViT attention forward (cftrain.attn_vit).
Error of each bf16 path vs an fp32 reference, next to the error of the PyTorch autocast path (the bar)."""
import json, time
import torch, torch.nn.functional as F
import cftrain.attn_vit as av
from cftrain.mlp_epi import fused_mlp

out = open("results/epi_test.jsonl", "a")
torch.manual_seed(0)
dev = "cuda"


def err(a, b):
    return dict(mae=float((a.float() - b.float()).abs().mean()), cos=float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)))


def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return round((time.perf_counter() - t0) / n * 1e6, 1)


# ---------------- MLP ----------------
for M in (197 * 128, 197 * 256):
    w1 = (torch.randn(3072, 768, device=dev) / 768 ** .5).requires_grad_()
    b1 = (0.1 * torch.randn(3072, device=dev)).requires_grad_()
    w2 = (torch.randn(768, 3072, device=dev) / 3072 ** .5).requires_grad_()
    b2 = (0.1 * torch.randn(768, device=dev)).requires_grad_()
    x0 = torch.randn(M, 768, device=dev)
    g = torch.randn(M, 768, device=dev)
    res = {}
    for name in ("ref_fp32", "autocast", "fused", "fused_compiled"):
        ps = [w1, b1, w2, b2]
        for q in ps: q.grad = None
        x = x0.clone().requires_grad_()
        if name == "ref_fp32":
            y = F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
        elif name == "autocast":
            with torch.autocast("cuda", torch.bfloat16):
                y = F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)
        else:
            fn = fused_mlp if name == "fused" else torch.compile(fused_mlp)
            with torch.autocast("cuda", torch.bfloat16):
                y = fn(x, w1, b1, w2, b2)
        (y.float() * g).sum().backward()
        res[name] = [y.detach().float(), x.grad.float()] + [q.grad.float() for q in ps]
    r = dict(kind="mlp_epi_test", M=M)
    names = ["y", "dx", "dw1", "db1", "dw2", "db2"]
    for name in ("autocast", "fused", "fused_compiled"):
        r[name] = {n: err(a, b) for n, a, b in zip(names, res[name], res["ref_fp32"])}
    r["PASS"] = all(r["fused"][n]["mae"] <= 1.5 * r["autocast"][n]["mae"] + 1e-7 and r["fused_compiled"][n]["mae"] <= 1.5 * r["autocast"][n]["mae"] + 1e-7
                    and all(torch.isfinite(t).all() for t in res["fused"]) for n in names)
    xb = x0.to(torch.bfloat16).requires_grad_()
    def ac():
        with torch.autocast("cuda", torch.bfloat16):
            F.linear(F.gelu(F.linear(xb, w1, b1)), w2, b2).backward(g.to(torch.bfloat16))
    def fu():
        with torch.autocast("cuda", torch.bfloat16):
            fused_mlp(xb, w1, b1, w2, b2).backward(g.to(torch.bfloat16))
    r["autocast_fwdbwd_us"], r["fused_fwdbwd_us"] = timeit(ac), timeit(fu)
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")

# ---------------- attention forward ----------------
def sdpa(t):
    q, k, v = (t[:, :, i].transpose(1, 2) for i in range(3))
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
for B in ():
    qkv = torch.randn(B, 197, 3, 12, 64, device=dev, dtype=torch.bfloat16)
    g = torch.randn(B, 197, 12, 64, device=dev, dtype=torch.bfloat16)
    res = {}
    for name, ver in (("ref_fp32", None), ("fa2_fwd", "fa2"), ("triton_fwd", "triton")):
        t = qkv.float() if ver is None else qkv.clone()
        t.requires_grad_()
        if ver is None:
            o = sdpa(t)
        else:
            av.FWD_VERSION = ver
            o = av.vit_attention(t)
        (o.float() * g.float()).sum().backward()
        res[name] = (o.detach().float(), t.grad.float())
    r = dict(kind="attn_fwd_test", B=B, fwd_cfg=list(av.FWD_CFG))
    for name in ("fa2_fwd", "triton_fwd"):
        r[name] = dict(out=err(res[name][0], res["ref_fp32"][0]), dqkv=err(res[name][1], res["ref_fp32"][1]))
    r["PASS"] = bool(r["triton_fwd"]["out"]["mae"] <= 1.5 * r["fa2_fwd"]["out"]["mae"] and
                     r["triton_fwd"]["dqkv"]["mae"] <= 1.5 * r["fa2_fwd"]["dqkv"]["mae"])
    for ver in ("fa2", "triton"):
        av.FWD_VERSION = ver
        r[ver + "_fwd_us"] = timeit(lambda: av.vit_attn_fwd(qkv))
    av.FWD_VERSION = "fa2"
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")

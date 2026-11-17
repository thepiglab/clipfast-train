"""Unit test + timing for cftrain.attn_vit (ViT attention custom op, bespoke backward)."""
import json, os, time
import torch, torch.nn.functional as F
import sys
import cftrain.attn_vit as av
av.BWD_VERSION = int(sys.argv[1]) if len(sys.argv) > 1 else av.BWD_VERSION
from cftrain.attn_vit import vit_attention

out = open("results/attn_vit_test.jsonl", "a")
torch.manual_seed(0)
for B in (8, 256):
    qkv = torch.randn(B, 197, 3, 12, 64, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, 197, 12, 64, device="cuda", dtype=torch.bfloat16)

    def sdpa(t):
        q, k, v = (t[:, :, i].transpose(1, 2) for i in range(3))
        return F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    res = {}
    r0 = {"bwd_version": av.BWD_VERSION}
    for name, fn, dt in [("ref_fp32", sdpa, torch.float32), ("fa2_bf16", sdpa, torch.bfloat16),
                         ("ours", vit_attention, torch.bfloat16), ("ours_compiled", torch.compile(vit_attention), torch.bfloat16)]:
        t = qkv.to(dt).detach().requires_grad_(True)
        torch.cuda.empty_cache()
        junk = torch.full((int(8e9) // 4,), float("nan"), device="cuda"); del junk   # unwritten rows -> NaN
        o = fn(t)
        (o.float() * g.float()).sum().backward()
        res[name] = (o.detach().float(), t.grad.float())
    r = dict(kind="attn_vit_test", B=B, **r0)
    for name in ("fa2_bf16", "ours", "ours_compiled"):
        o, d = res[name]
        r[name] = dict(out_mae=float((o - res["ref_fp32"][0]).abs().mean()),
                       dqkv_mae=float((d - res["ref_fp32"][1]).abs().mean()),
                       dqkv_finite=bool(torch.isfinite(d).all()),
                       dqkv_cos=float(F.cosine_similarity(d.flatten(), res["ref_fp32"][1].flatten(), dim=0)))
    r["PASS"] = bool(r["ours"]["dqkv_finite"] and r["ours"]["dqkv_mae"] <= 1.5 * r["fa2_bf16"]["dqkv_mae"]
                     and r["ours_compiled"]["dqkv_mae"] <= 1.5 * r["fa2_bf16"]["dqkv_mae"])
    # timing fwd+bwd per layer
    for name, fn in [("fa2_sdpa", sdpa), ("ours", vit_attention)]:
        t = qkv.detach().requires_grad_(True)
        for _ in range(3):
            fn(t).backward(g)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20):
            fn(t).backward(g)
        torch.cuda.synchronize()
        r[name + "_fwdbwd_us"] = round((time.perf_counter() - t0) / 20 * 1e6, 1)
    print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n")

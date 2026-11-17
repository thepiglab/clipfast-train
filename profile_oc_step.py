"""Phase-0 profile of ONE open_clip BiomedCLIP training step (fwd+bwd+AdamW), broken down by phase and op class.

Modes: fp32 (TF32 off), amp_bf16 (open_clip --precision amp_bf16), amp_compile (amp + torch.compile(model)).
Text is padded to 256 exactly as open_clip's tokenizer does (that is the baseline); a second case trims to the
batch max so the text-trim ceiling is visible.
Phases are measured by profiling (fwd) , (fwd+bwd) and (fwd+bwd+opt) separately and differencing.
Output: one JSON line per (mode, B, case) with ms per phase, GPU-us per op class per phase, top kernels, peak memory.
"""
import argparse
import copy
import json
import os
import socket
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

from cftrain.ref import load_reference
from cftrain.ref import text_ids

CLASSES = [  # order matters: first match wins
    ("multi_tensor", "optimizer"), ("adam", "optimizer"), ("fused_adam", "optimizer"),
    ("flash_bwd", "attn_bwd"), ("flash_fwd", "attn_fwd"), ("fmha", "attention"), ("cudnn", "attention"),
    ("efficient_attention", "attention"), ("attention", "attention"),
    ("gemv", "gemm"), ("gemm", "gemm"), ("cutlass", "gemm"), ("nvjet", "gemm"), ("xmma", "gemm"), ("sm90_", "gemm"),
    ("sm100_", "gemm"), ("sm120_", "gemm"), ("splitk", "gemm"), ("cublas", "gemm"),
    ("layer_norm_grad", "layernorm_bwd"), ("gammabeta", "layernorm_bwd"), ("layernorm_bwd", "layernorm_bwd"),
    ("layer_norm", "layernorm"), ("layernorm", "layernorm"),
    ("gelubackward", "gelu_bwd"), ("gelu_backward", "gelu_bwd"), ("gelu", "gelu"),
    ("dropout", "dropout"), ("bernoulli", "dropout"), ("philox", "dropout"), ("softmax", "softmax/loss"), ("nll_loss", "softmax/loss"), ("cross_entropy", "softmax/loss"),
    ("conv", "conv"), ("embedding", "embed/index"), ("index", "embed/index"), ("scatter", "embed/index"),
    ("copy", "copy/cast"), ("cat", "copy/cat"), ("fill", "fill/zero"), ("reduce", "reduce"), ("sum", "reduce"),
    ("norm", "reduce"), ("elementwise", "elementwise"),
]


def classify(name):
    n = name.lower()
    pre = ""
    if n.startswith("triton"):
        pre = "triton:"
        if "mm" in n.split("_")[:3] or "_mm_" in n or "addmm" in n or "bmm" in n:
            return "triton:gemm"
    for key, tag in CLASSES:
        if key in n:
            return pre + tag
    return pre + "other"


def clip_loss(fi, ft, scale):
    fi, ft = F.normalize(fi, dim=-1), F.normalize(ft, dim=-1)
    logits = scale.exp() * fi @ ft.T
    lab = torch.arange(fi.shape[0], device=fi.device)
    return (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2


def prof(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(iters):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    wall_ms = ev0.elapsed_time(ev1) / iters
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    per, cls = defaultdict(float), defaultdict(float)
    n_kern = 0
    for e in p.key_averages():
        t = getattr(e, "device_time_total", 0) or 0
        if t <= 0:
            continue
        per[e.key] += t / iters
        cls[classify(e.key)] += t / iters
        n_kern += e.count
    return wall_ms, dict(cls), dict(per), n_kern / iters


def sub(a, b):
    return {k: round(a.get(k, 0) - b.get(k, 0), 1) for k in set(a) | set(b) if abs(a.get(k, 0) - b.get(k, 0)) > 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="fp32,amp_bf16,amp_compile")
    ap.add_argument("--batches", default="256")
    ap.add_argument("--cases", default="pad256,trim")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--stack", choices=["oc", "cf"], default="oc", help="cf = cftrain lean model, packed text")
    ap.add_argument("--dropout", type=float, default=0.1)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                job=os.environ.get("CFTRAIN_RUN_ID"))
    out = open(a.out, "a")
    base, _, _ = load_reference()
    base.train()
    vocab = base.text.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    nparam = sum(p.numel() for p in base.parameters())
    for B in [int(b) for b in a.batches.split(",")]:
        x = torch.randn(B, 3, 224, 224, device="cuda")
        ids256 = text_ids(B, "pmc", vocab, False, gen).cuda()
        lens = (ids256 != 0).sum(1)
        for case in a.cases.split(","):
            ids = ids256 if case == "pad256" else ids256[:, :int(lens.max())].contiguous()
            for mode in a.modes.split(","):
                amp = mode != "fp32"
                state = {}
                if a.stack == "oc":
                    m = copy.deepcopy(base)
                    fwd = torch.compile(m) if mode == "amp_compile" else m
                    call = lambda: fwd(x, ids)[:3]
                else:
                    from cftrain.model import LeanCLIP, pack_ids
                    m = LeanCLIP(base, text_dropout=a.dropout, attn_dropout=a.dropout).cuda().train()
                    fwd = torch.compile(m) if mode == "amp_compile" else m
                    pk = pack_ids(ids.cpu())
                    pk = [t.cuda() for t in pk[:5]] + [pk[5]]
                    call = lambda: fwd(images=x, packed=pk)
                opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.1,
                                        fused=True)

                def f_fwd():
                    state.clear()
                    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                        fi, ft, s = call()
                        loss = clip_loss(fi, ft, s)
                    state["loss"] = loss
                    return loss

                def f_fb():
                    f_fwd().backward()

                def f_step():
                    f_fb()
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                try:
                    torch.cuda.reset_peak_memory_stats()
                    f_step()                                    # init optimizer state
                    torch.cuda.synchronize()
                    base_mem = torch.cuda.memory_allocated()     # params + adam state (+ cached compile stuff)
                    torch.cuda.reset_peak_memory_stats()
                    f_fb()
                    torch.cuda.synchronize()
                    peak_fb = torch.cuda.max_memory_allocated()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats()
                    f_fwd()
                    torch.cuda.synchronize()
                    act = torch.cuda.memory_allocated() - base_mem   # saved activations held by the graph
                    state.clear()
                    w0, c0, p0, k0 = prof(lambda: f_fwd() and None, a.iters)
                    state.clear()
                    w1, c1, p1, k1 = prof(f_fb, a.iters)
                    opt.zero_grad(set_to_none=True)
                    w2, c2, p2, k2 = prof(f_step, a.iters)
                    top = sorted(p2.items(), key=lambda kv: -kv[1])[:25]
                    r = dict(kind=a.stack + "_profile", mode=mode, dropout=a.dropout, B=B, case=case, L=int(ids.shape[1]),
                             mean_len=float(lens.float().mean()),
                             ms=dict(fwd=round(w0, 2), bwd=round(w1 - w0, 2), opt=round(w2 - w1, 2), step=round(w2, 2)),
                             gpu_us=dict(fwd=round(sum(c0.values()), 1), fwd_bwd=round(sum(c1.values()), 1),
                                         step=round(sum(c2.values()), 1)),
                             kernels_per_step=round(k2), samples_per_s=round(B / w2 * 1e3, 1),
                             cls_fwd={k: round(v, 1) for k, v in sorted(c0.items(), key=lambda kv: -kv[1])},
                             cls_bwd=sub(c1, c0), cls_opt=sub(c2, c1),
                             mem_gb=dict(params_opt=round(base_mem / 1e9, 2), saved_act=round(act / 1e9, 2),
                                         peak_fwd_bwd=round(peak_fb / 1e9, 2)),
                             n_params=nparam, top=[(k[:120], round(v, 1)) for k, v in top])
                    r.update(meta)
                except torch.OutOfMemoryError:
                    r = dict(kind=a.stack + "_profile", mode=mode, dropout=a.dropout, B=B, case=case, oom=True, **meta)
                out.write(json.dumps(r) + "\n")
                out.flush()
                print(json.dumps(r)[:1500], flush=True)
                del m, opt, fwd, state
                torch._dynamo.reset()
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

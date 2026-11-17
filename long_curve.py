"""Long-run loss-curve parity: identical batches (same cache, sampler seed, LR schedule, AdamW, dropout 0) for
  oc_amp   open_clip amp_bf16, text padded to 256 (the baseline)
  oc_fp32  open_clip fp32 (reference)
  cf       cftrain defaults (packed text, compile, bespoke ViT attention bwd)
Writes one JSON line per step (loss) + a summary with samples/s; compare with compare_curves.py."""
import argparse
import json
import math
import os
import time

import torch

from cftrain.ref import load_reference
from cftrain.data import GPUImages, make_loader
from cftrain.engine import LN100, Stepper, build_optimizer, cosine_lr, param_groups, set_lr
from cftrain.loss import clip_loss
from cftrain.model import LeanCLIP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=["oc_amp", "oc_fp32", "oc_amp_compile", "oc_amp_norr", "cf", "cf_sdpa",
                                                       "cf_eager_sdpa"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ref, _, _ = load_reference()
    for m in ref.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    ref.train()
    cfi = a.impl.startswith("cf")
    loader, sampler = make_loader(a.data, a.B, img_path="cache", pack=cfi, workers=4, seed=0)
    gi = GPUImages(dev)
    if cfi:
        import cftrain.model as cm
        cm.VIT_ATTN = "sdpa" not in a.impl
        model = LeanCLIP(ref, text_dropout=0, attn_dropout=0).to(dev).train()
        opt = build_optimizer(model, a.lr, 0.2)
        st = Stepper(model, opt, compile="eager" not in a.impl)
    else:
        model = ref
        opt = torch.optim.AdamW(param_groups(model, 0.2), lr=a.lr, betas=(0.9, 0.98), eps=1e-6, fused=True)
        if a.impl == "oc_amp_norr":   # same model, different (equally valid) bf16 GEMM rounding
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        fwd = torch.compile(model) if a.impl == "oc_amp_compile" else model
    out = open(a.out, "w")
    step, epoch, t0, n = 0, 0, None, 0
    while step < a.steps:
        sampler.epoch = epoch
        for batch in loader:
            if step >= a.steps:
                break
            lr = cosine_lr(step, a.lr, a.warmup, a.steps)
            set_lr(opt, lr)
            x = gi(batch)
            if cfi:
                pk = batch["packed"]
                loss = st(dict(images=x, packed=[t.to(dev, non_blocking=True) for t in pk[:5]] + [pk[5]]))
            else:
                with torch.autocast("cuda", torch.bfloat16, enabled=a.impl != "oc_fp32"):
                    fi, ft, s = fwd(x, batch["ids"].to(dev, non_blocking=True))[:3]
                loss = clip_loss(fi, ft, model.logit_scale)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                with torch.no_grad():
                    model.logit_scale.clamp_(0, LN100)
            lv = float(loss)
            if not math.isfinite(lv):
                print("non-finite loss at step", step, flush=True)
            out.write(json.dumps(dict(step=step, loss=lv, lr=lr)) + "\n")
            step += 1
            if step == 20:
                torch.cuda.synchronize()
                t0, n = time.time(), 0
            elif t0 is not None:
                n += a.B
        epoch += 1
    torch.cuda.synchronize()
    r = dict(kind="long_curve_summary", impl=a.impl, steps=a.steps, B=a.B,
             samples_per_s=round(n / (time.time() - t0), 1), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
             job=os.environ.get("CFTRAIN_RUN_ID"))
    out.write(json.dumps(r) + "\n")
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()

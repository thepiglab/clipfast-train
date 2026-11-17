"""Parity of the cftrain stack against open_clip (BiomedCLIP), dropout off on both sides.

Gradient parity (one batch, B=32, caption-like text lengths, TF32 off):
  reference = open_clip fp32 grads, mapped into the lean layout (qkv concatenated, patch conv as GEMM).
  noise floor = open_clip amp_bf16 vs fp32.  Each cftrain variant must sit within that floor:
  global grad cosine >= floor_cos - 1e-4, and per-tensor min cosine / median rel-err reported next to the floor.
  Variants: lean fp32 padded / packed (should be ~exact), lean amp padded / packed / compiled,
  LoRA (reference: dA = s B^T dW_eff, dB = s dW_eff A^T from open_clip fp32 grads at merged weights),
  LiT (text grads only; image features from the locked tower, as cached).
Loss-curve parity: 40 AdamW steps cycling 4 fixed batches; open_clip fp32 / amp vs lean amp (packed, compiled).
"""
import argparse
import copy
import json
import os
import socket

import torch
import torch.nn.functional as F

from cftrain.ref import load_reference
from cftrain.ref import text_ids
from cftrain.engine import Stepper, build_model, build_optimizer, param_groups, LN100
from cftrain.loss import clip_loss
from cftrain.model import LeanCLIP, pack_ids

DEV = "cuda"


def no_dropout(ref):
    for m in ref.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    cfg = ref.text.transformer.config
    cfg.hidden_dropout_prob = cfg.attention_probs_dropout_prob = 0.0
    return ref


def ref_grads(ref, x, ids, amp):
    m = copy.deepcopy(ref)
    m.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
        fi, ft, s = m(x, ids)[:3]
    loss = clip_loss(fi, ft, m.logit_scale)
    loss.backward()
    return m, float(loss.detach())


def lean_layout_grads(m):
    """Grads of an open_clip model, re-expressed in LeanCLIP's parameter layout (dict name -> fp32 tensor)."""
    grads = {n: (p.grad if p.grad is not None else torch.zeros_like(p)) for n, p in m.named_parameters()}
    assert any(float(v.abs().sum()) > 0 for v in grads.values()), "no gradients"
    g = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in g.named_parameters():
            p.copy_(grads[n])
    lean = LeanCLIP(g, text_dropout=0, attn_dropout=0)
    return {n: p.detach().float() for n, p in lean.named_parameters()}


def compare(gl, gr, names=None):
    names = names or [n for n in gr if n in gl]
    a = torch.cat([gl[n].flatten() for n in names])
    b = torch.cat([gr[n].flatten() for n in names])
    cos = [float(F.cosine_similarity(gl[n].flatten(), gr[n].flatten(), dim=0)) for n in names
           if gr[n].norm() > 0]
    rel = torch.tensor([float((gl[n] - gr[n]).norm() / (gr[n].norm() + 1e-30)) for n in names])
    worst = sorted(zip(cos, names), key=lambda t: t[0])[:3]
    return dict(global_cos=float(F.cosine_similarity(a, b, dim=0)), rel_l2=float((a - b).norm() / b.norm()),
                min_tensor_cos=min(cos), median_rel=float(rel.median()), max_rel=float(rel.max()),
                worst=[(n, round(c, 6)) for c, n in worst], n_tensors=len(names))


def lean_grads(model, x, ids, amp, packed=False, compile=False):
    model.zero_grad(set_to_none=True)
    fwd = torch.compile(model, dynamic=False) if compile else model
    inp = dict(images=x)
    if packed:
        pk = pack_ids(ids.cpu())
        inp["packed"] = [t.to(DEV) for t in pk[:5]] + [pk[5]]
    else:
        L = int((ids != 0).sum(1).max())
        inp["ids"] = ids[:, :L]
    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
        fi, ft, s = fwd(**inp)
    loss = clip_loss(fi, ft, model.logit_scale)
    loss.backward()
    return {n: p.grad.detach().float() for n, p in model.named_parameters() if p.grad is not None}, float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    skip = set(a.skip.split(","))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                job=os.environ.get("CFTRAIN_RUN_ID"))
    out = open(a.out, "a")

    def rec(**kw):
        kw.update(meta)
        out.write(json.dumps(kw) + "\n")
        out.flush()
        print(json.dumps(kw), flush=True)

    ref, _, _ = load_reference()
    ref = no_dropout(ref).train()
    vocab = ref.text.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    x = torch.randn(a.B, 3, 224, 224, device=DEV)
    ids = text_ids(a.B, "pmc", vocab, False, gen).to(DEV)

    # ---------------- gradient parity: full fine-tuning ----------------
    m32, l32 = ref_grads(ref, x, ids, False)
    G = lean_layout_grads(m32)
    del m32
    mamp, lamp = ref_grads(ref, x, ids, True)
    Gamp = lean_layout_grads(mamp)
    floor = compare(Gamp, G)
    del mamp
    rec(kind="grad_parity", variant="open_clip_amp_bf16 (noise floor)", loss=lamp, loss_fp32=l32, **floor)
    variants = [("lean_fp32_padded", False, False, False), ("lean_fp32_packed", False, True, False),
                ("lean_amp_padded", True, False, False), ("lean_amp_packed", True, True, False),
                ("lean_amp_packed_compiled", True, True, True), ("lean_amp_packed_compiled_vattn", True, True, True),
                ("lean_amp_packed_compiled_mlpepi", True, True, True), ("lean_amp_packed_compiled_attnfwd", True, True, True),
                ("lean_amp_packed_compiled_allk", True, True, True), ("lean_amp_packed_compiled_recomp", True, True, True),
                ("lean_fp8_packed_compiled", True, True, True)]
    import cftrain.model as cm
    VIT_DEFAULT = cm.VIT_ATTN          # shipped default (bespoke ViT attention bwd); later sections use it
    MLP_DEFAULT = cm.MLP_EPI
    import cftrain.attn_vit as av
    FWD_DEFAULT = av.FWD_VERSION
    for name, amp, packed, comp in variants:
        if name in skip:
            continue
        cm.FP8 = "fp8" in name
        cm.VIT_ATTN = any(t in name for t in ("vattn", "attnfwd", "allk"))
        cm.MLP_EPI = "mlpepi" in name or "allk" in name or "recomp" in name
        import cftrain.mlp_epi as me
        me.SAVE_ACT = "recomp" not in name
        import cftrain.attn_vit as av
        av.FWD_VERSION = "triton" if ("attnfwd" in name or "allk" in name) else "fa2"
        lean = LeanCLIP(ref, text_dropout=0, attn_dropout=0).to(DEV).train()
        g, loss = lean_grads(lean, x, ids, amp, packed, comp)
        c = compare(g, G)
        ok = c["global_cos"] >= floor["global_cos"] - 1e-4 if amp else c["global_cos"] > 0.99999
        if cm.FP8:   # opt-in reduced precision: report against a looser bar (fp8 error is ~8x bf16's)
            ok = c["global_cos"] >= 0.999
        rec(kind="grad_parity", variant=name, loss=loss, loss_fp32=l32, PASS=bool(ok), **c)
        del lean, g
        cm.FP8, cm.VIT_ATTN, cm.MLP_EPI = False, VIT_DEFAULT, MLP_DEFAULT
        av.FWD_VERSION = FWD_DEFAULT
        torch._dynamo.reset()

    # ---------------- LoRA gradient parity ----------------
    if "lora" not in skip:
        r, alpha = 8, 16
        for amp in (False, True):
            torch.manual_seed(1)
            lean = LeanCLIP(ref, lora_r=r, lora_alpha=alpha, text_dropout=0, attn_dropout=0,
                            frozen_dtype=torch.bfloat16 if amp else torch.float32).to(DEV).train()
            with torch.no_grad():
                for mod in lean.modules():
                    if mod.__class__.__name__ == "LoRA":
                        mod.B.normal_(0, 0.02)            # non-zero B so dA is non-trivial
            merged = lean.export_to(copy.deepcopy(ref)).to(DEV)
            mm, _ = ref_grads(merged, x, ids, False)
            Gm = lean_layout_grads(mm)                      # dW_eff in lean layout
            del mm
            g, loss = lean_grads(lean, x, ids, amp, packed=True)
            gl, gr = {}, {}
            for n, mod in lean.named_modules():
                if mod.__class__.__name__ == "LoRA":
                    blk, nm = n.rsplit(".lora_", 1)
                    dW = Gm[f"{blk}.{nm}_w"]
                    gr[n + ".A"] = mod.scale * mod.B.detach().float().T @ dW
                    gr[n + ".B"] = mod.scale * dW @ mod.A.detach().float().T
                    gl[n + ".A"], gl[n + ".B"] = g[n + ".A"], g[n + ".B"]
            for n in ("v_proj", "t_p1", "t_p2", "logit_scale"):
                gl[n], gr[n] = g[n], Gm[n]
            c = compare(gl, gr)
            rec(kind="grad_parity", variant=f"lora_r{r}_{'amp' if amp else 'fp32'}_packed", loss=loss,
                PASS=bool(c["global_cos"] >= (floor["global_cos"] - 1e-4 if amp else 0.99999)), **c)
            del lean, merged, Gm

    # ---------------- LiT (locked image tower) ----------------
    if "lit" not in skip:
        lean = build_model(ref, "lit", text_dropout=0, attn_dropout=0).to(DEV).train()
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            fi_cache = lean.encode_image(x).to(torch.bfloat16)       # what the LiT cache stores
        lean.zero_grad(set_to_none=True)
        pk = pack_ids(ids.cpu())
        with torch.autocast("cuda", torch.bfloat16):
            fi, ft, s = lean(img_feats=fi_cache, packed=[t.to(DEV) for t in pk[:5]] + [pk[5]])
        clip_loss(fi, ft, lean.logit_scale).backward()
        g = {n: p.grad.float() for n, p in lean.named_parameters() if p.grad is not None}
        names = [n for n in g if n.startswith("t_") or n == "logit_scale"]
        c = compare(g, G, names)
        ft_floor = compare(Gamp, G, names)
        rec(kind="grad_parity", variant="lit_cached_image_amp_packed", n_trainable=sum(
            p.numel() for p in lean.parameters() if p.requires_grad),
            floor_text_cos=ft_floor["global_cos"], PASS=bool(c["global_cos"] >= ft_floor["global_cos"] - 1e-4), **c)
        del lean

    # ---------------- audit: trainer packing (dummy sequence) under a NaN-poisoned allocator ----------------
    # Registered after the NaN bug of 2026-11-16 (FlashAttention bwd leaves dK/dV rows outside every sequence
    # unwritten). Any kernel that leaves gradient rows unwritten shows up as non-finite grads here.
    if "poison" not in skip:
        lean = LeanCLIP(ref, text_dropout=0, attn_dropout=0).to(DEV).train()
        bad_total = 0
        for comp in (False, True):
            fwd = torch.compile(lean) if comp else lean
            for tm in (256, 1024):
                lean.zero_grad(set_to_none=True)
                pk = pack_ids(ids.cpu(), tok_multiple=tm, fixed_max_len=True)
                pk = [t.to(DEV) for t in pk[:5]] + [pk[5]]
                junk = torch.full((int(20e9) // 4,), float("nan"), device=DEV)
                del junk
                with torch.autocast("cuda", torch.bfloat16):
                    fi, ft, sc = fwd(images=x, packed=pk)
                loss = clip_loss(fi, ft, lean.logit_scale)
                junk = torch.full((int(20e9) // 4,), float("nan"), device=DEV)
                del junk
                loss.backward()
                bad_total += sum(1 for q in lean.parameters() if q.grad is not None and not torch.isfinite(q.grad).all())
        rec(kind="audit", variant="packed_dummy_nan_poisoned_allocator", nonfinite_grad_tensors=bad_total,
            PASS=bad_total == 0)
        del lean
        torch._dynamo.reset()

    # ---------------- loss-curve parity ----------------
    if "curve" in skip:
        print("DONE", flush=True)
        return
    batches = []
    for i in range(4):
        xb = torch.randn(a.B, 3, 224, 224, device=DEV, generator=torch.Generator(DEV).manual_seed(100 + i))
        batches.append((xb, text_ids(a.B, "pmc", vocab, False, gen).to(DEV)))
    curves = {}

    def oc_curve(amp):
        m = copy.deepcopy(ref).train()
        opt = torch.optim.AdamW(param_groups(m, 0.2), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, fused=True)
        ls = []
        for s in range(a.steps):
            xb, ib = batches[s % 4]
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                fi, ft, sc = m(xb, ib)[:3]
            loss = clip_loss(fi, ft, m.logit_scale)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                m.logit_scale.clamp_(0, LN100)
            ls.append(float(loss))
        return ls

    curves["open_clip_fp32"] = oc_curve(False)
    curves["open_clip_amp_bf16"] = oc_curve(True)
    import cftrain.model as cm
    for name, comp in [("cftrain_amp_packed", False), ("cftrain_amp_packed_compiled", True),
                       ("cftrain_fp8_packed_compiled", True)]:
        cm.FP8 = "fp8" in name
        lean = LeanCLIP(ref, text_dropout=0, attn_dropout=0).to(DEV).train()
        st = Stepper(lean, build_optimizer(lean, 1e-5, 0.2), compile=comp)
        ls = []
        for s in range(a.steps):
            xb, ib = batches[s % 4]
            pk = pack_ids(ib.cpu())
            ls.append(float(st(dict(images=xb, packed=[t.to(DEV) for t in pk[:5]] + [pk[5]]))))
        curves[name] = ls
        cm.FP8 = False
        del lean, st
        torch._dynamo.reset()
    r = curves["open_clip_fp32"]
    for k, v in curves.items():
        d = [abs(p - q) for p, q in zip(v, r)]
        rec(kind="curve_parity", variant=k, max_abs_vs_fp32=max(d), final=v[-1], first=v[0], curve=[round(z, 5) for z in v])
    floor_d = max(abs(p - q) for p, q in zip(curves["open_clip_amp_bf16"], r))
    for k in ("cftrain_amp_packed", "cftrain_amp_packed_compiled", "cftrain_fp8_packed_compiled"):
        d = max(abs(p - q) for p, q in zip(curves[k], r))
        rec(kind="curve_parity_verdict", variant=k, max_abs_vs_fp32=d, floor_amp_vs_fp32=floor_d,
            PASS=bool(d <= 2 * floor_d + 1e-3))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

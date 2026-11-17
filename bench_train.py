"""Training-step throughput + memory: cftrain variants vs open_clip amp_bf16 (the baseline), one GPU.

Each row: samples/s (median of 3 windows x N steps, after warm-up), peak allocated (torch), peak device use
(cudaMemGetInfo-based, includes allocator slack / workspaces / graph pools), warm-up seconds.
Text lengths: 'pmc' = caption-like length distribution (bench.text_ids), padded to 256 for open_clip as its
tokenizer does; cftrain packs (no padding).
Impls:
  oc_amp            open_clip, autocast bf16, fused AdamW, text padded 256          <- baseline
  oc_amp_compile    + torch.compile(model)
  cf_eager          lean model, packed text, fused AdamW, eager
  cf_compile        + torch.compile
  cf_autotune       cf_compile with mode=max-autotune-no-cudagraphs (Triton GEMM templates + epilogue fusion)
  cf_sdpa           cf_compile with PyTorch SDPA for ViT attention instead of the bespoke backward (default)
  cf_graph          + whole-step CUDA graph (compile + capture)
  cf_eager_graph    eager + whole-step CUDA graph
  cf_recompute      cf_compile + per-block activation recompute
  cf_lora_unfused   LoRA as plain modules (x A^T B^T + separate add)
  cf_lora           LoRA r=8 (qkv,o,fc1,fc2, both towers), frozen bf16 base, compile
  cf_lora_graph     + CUDA graph
  cf_budget50       cf_compile with activation_memory_budget=0.5 (selective recompute by the partitioner)
  cf_fp8            cf_compile + FP8 (e4m3 rowwise) forward GEMMs, bf16 backward (opt-in)
  cf_freeze6        cf_compile with the first 6 blocks of both towers frozen (no dW, no backward below block 6)
  cf_mlpepi         cf_compile + fused-epilogue MLP (fc1+GELU fwd, dGELU bwd)
  cf_attnfwd        cf_compile + Triton short-seq attention forward
  cf_allk           both
  cf_recomp         cf_mlpepi saving only the fc1 pre-activation (GELU recomputed in backward): memory
  cf_lit            image tower locked, cached image features (text tower trained), compile
"""
import argparse
import copy
import json
import os
import socket
import statistics
import time

import torch

from cftrain.ref import load_reference
from cftrain.ref import text_ids
from cftrain.engine import GraphStepper, Stepper, build_model, build_optimizer
from cftrain.loss import clip_loss
from cftrain.model import pack_ids

DEV = "cuda"


def dev_used():
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--impls", default="oc_amp,cf_eager,cf_compile,cf_graph,cf_recompute,cf_lora,cf_lora_graph,cf_lit")
    ap.add_argument("--batches", default="128,256,512")
    ap.add_argument("--lens", default="pmc")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.0)
    a = ap.parse_args()
    meta = dict(host=socket.gethostname(), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                job=os.environ.get("CFTRAIN_RUN_ID"))
    out = open(a.out, "a")
    ref, _, _ = load_reference()
    ref.train()
    if a.dropout == 0:
        for m in ref.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
    vocab = ref.text.config.vocab_size
    gen = torch.Generator().manual_seed(0)
    base_dev = dev_used()
    for B in [int(b) for b in a.batches.split(",")]:
        for L in [int(l) if l.isdigit() else l for l in a.lens.split(",")]:
            x = torch.randn(B, 3, 224, 224, device=DEV)
            ids = text_ids(B, L, vocab, L != "pmc", gen)
            for impl in a.impls.split(","):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                d0 = dev_used()
                peak_dev = [d0]
                try:
                    graph = impl.endswith("_graph")
                    if impl.startswith("oc"):
                        m = copy.deepcopy(ref)
                        opt = torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0.2, fused=True)
                        fwd = torch.compile(m) if impl == "oc_amp_compile" else m
                        idd = ids.to(DEV)

                        def step():
                            with torch.autocast("cuda", torch.bfloat16):
                                fi, ft, s = fwd(x, idd)[:3]
                            loss = clip_loss(fi, ft, m.logit_scale)
                            loss.backward()
                            opt.step()
                            opt.zero_grad(set_to_none=True)
                            return loss
                    else:
                        mode = "lora" if "lora" in impl else "lit" if "lit" in impl else "full"
                        import cftrain.model as cm
                        cm.FUSED_LORA = "unfused" not in impl
                        cm.FP8 = "fp8" in impl
                        cm.VIT_ATTN = "sdpa" not in impl
                        cm.MLP_EPI = "mlpepi" in impl or "allk" in impl
                        import cftrain.attn_vit as av
                        av.FWD_VERSION = "triton" if ("attnfwd" in impl or "allk" in impl) else "fa2"
                        import cftrain.mlp_epi as me
                        me.SAVE_ACT = "recomp" not in impl
                        if "recomp" in impl:
                            cm.MLP_EPI = True
                        fz = int(impl.split("freeze")[1]) if "freeze" in impl else 0
                        tg = ("qkv",) if "lora_qkv" in impl else ("qkv", "o", "fc1", "fc2")
                        m = build_model(ref, mode, text_dropout=a.dropout, attn_dropout=a.dropout, lora_targets=tg,
                                        freeze_image_blocks=fz, freeze_text_blocks=fz).to(DEV).train()
                        opt = build_optimizer(m, 1e-5, 0.2, capturable=graph)
                        st = Stepper(m, opt, compile=impl not in ("cf_eager", "cf_eager_graph", "cf_fp8_eager"),
                                     recompute=impl == "cf_recompute",
                                     act_budget=float(impl.split("budget")[1]) / 100 if "budget" in impl else 1.0,
                                     compile_mode="max-autotune-no-cudagraphs" if "autotune" in impl else None)
                        runner = GraphStepper(st) if graph else st
                        pk = pack_ids(ids, tok_multiple=256 if graph else 0, fixed_max_len=graph)
                        pk = [t.to(DEV) for t in pk[:5]] + [pk[5]]
                        inp = dict(packed=pk)
                        if mode == "lit":
                            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                                inp["img_feats"] = m.encode_image(x).to(torch.bfloat16)
                        else:
                            inp["images"] = x

                        def step():
                            return runner(inp)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        loss = step()
                    torch.cuda.synchronize()
                    warm = time.perf_counter() - t0
                    peak_dev.append(dev_used())
                    ts = []
                    for _ in range(3):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(a.steps):
                            loss = step()
                        torch.cuda.synchronize()
                        ts.append((time.perf_counter() - t0) / a.steps * 1e3)
                        peak_dev.append(dev_used())
                    ms = statistics.median(ts)
                    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    r = dict(kind="train_bench", impl=impl, B=B, L=L, ms=round(ms, 2),
                             samples_per_s=round(B / ms * 1e3, 1), peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                             dev_used_gb=round(max(peak_dev) - base_dev, 2), warm_s=round(warm, 1), loss=float(loss),
                             trainable_params=ntr, text_tokens=int((ids != 0).sum()), dropout=a.dropout)
                except torch.OutOfMemoryError:
                    r = dict(kind="train_bench", impl=impl, B=B, L=L, oom=True)
                except Exception as e:  # noqa: BLE001  (record and continue: one broken variant must not kill the matrix)
                    r = dict(kind="train_bench", impl=impl, B=B, L=L, error=f"{type(e).__name__}: {str(e)[:400]}")
                r.update(meta)
                out.write(json.dumps(r) + "\n")
                out.flush()
                print(json.dumps(r), flush=True)
                for n in ("m", "opt", "st", "runner", "fwd", "inp", "step"):
                    locals().pop(n, None)
                m = opt = st = runner = fwd = inp = step = loss = None  # noqa: F841  (loss may live in a graph pool)
                import gc
                gc.collect()
                torch._dynamo.reset()
                import torch._functorch.config as fc
                fc.activation_memory_budget = 1.0
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

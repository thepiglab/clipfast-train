"""cftrain: fast BiomedCLIP contrastive fine-tuning (full / LoRA / LiT), one GPU or torchrun DDP.

Examples
  # 1) one-time cache (CPU job): exact open_clip eval pixels as uint8 + token ids
  python3 -m cftrain.prep --data captions.csv --out /path/to/cache_mydata --workers 64
  # 2) train (1 GPU)
  python3 train.py --data /path/to/cache_mydata --out runs/full_b256 --batch-size 256 --epochs 5 --lr 1e-5
  # LoRA / LiT
  python3 train.py --data ... --mode lora --lora-r 16 --lr 2e-4
  python3 train.py --data ... --mode lit --lr 5e-5          # image tower locked + its features cached once
  # 8 GPUs
  torchrun --nproc_per_node 8 train.py --data ... --batch-size 256   # per-GPU batch; global = 8x
  # raw data without a cache: --img-path cpu (open_clip transform, PIL) or --img-path gpu (nvJPEG)

Outputs in --out: log.jsonl (loss, lr, samples/s, memory), checkpoint_*.pt (open_clip-compatible state_dict with
LoRA merged: load with open_clip.create_model_and_transforms(..., pretrained=<file>)), args.json.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist

from cftrain.ref import load_reference
from cftrain.data import GPUImages, Tok, make_loader
from cftrain.engine import GraphStepper, Stepper, build_model, build_optimizer, cosine_lr, encode_dataset, set_lr
from cftrain.loss import clip_loss, retrieval_acc


DEBUG = bool(int(os.environ.get("CFTRAIN_DEBUG", "0")))


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ap.add_argument_group("data")
    d.add_argument("--data", required=True, help="cache dir (from cftrain.prep), folder, .csv or .jsonl")
    d.add_argument("--val-data", default=None, help="optional held-out set (same formats); retrieval R@1 logged")
    d.add_argument("--img-path", choices=["auto", "cache", "cpu", "gpu"], default="auto")
    d.add_argument("--image-col", default="image")
    d.add_argument("--caption-col", default="caption")
    d.add_argument("--workers", type=int, default=-1, help="loader workers per GPU (default: NUMA-local cores - 2)")
    m = ap.add_argument_group("model / mode")
    m.add_argument("--mode", choices=["full", "lora", "lit", "lock-text"], default="full")
    m.add_argument("--lora-r", type=int, default=16)
    m.add_argument("--lora-alpha", type=float, default=32)
    m.add_argument("--lora-targets", default="qkv,o,fc1,fc2")
    m.add_argument("--lora-towers", default="image,text")
    m.add_argument("--freeze-image-blocks", type=int, default=0, help="freeze the first N ViT blocks (+ patch emb)")
    m.add_argument("--freeze-text-blocks", type=int, default=0, help="freeze the first N BERT layers (+ emb)")
    m.add_argument("--text-dropout", type=float, default=0.1, help="open_clip's HF BERT default is 0.1")
    m.add_argument("--attn-dropout", type=float, default=0.1)
    o = ap.add_argument_group("optimisation")
    o.add_argument("--batch-size", type=int, default=256, help="per GPU")
    o.add_argument("--epochs", type=float, default=1)
    o.add_argument("--max-steps", type=int, default=0)
    o.add_argument("--lr", type=float, default=1e-5)
    o.add_argument("--wd", type=float, default=0.2)
    o.add_argument("--beta2", type=float, default=0.98)
    o.add_argument("--eps", type=float, default=1e-6)
    o.add_argument("--warmup", type=int, default=100)
    o.add_argument("--seed", type=int, default=0)
    f = ap.add_argument_group("speed / memory")
    f.add_argument("--compile", type=int, default=1)
    f.add_argument("--compile-mode", default=None, help="e.g. max-autotune-no-cudagraphs")
    f.add_argument("--cuda-graph", type=int, default=0, help="whole-step CUDA graph (1 GPU; fixed token buckets)")
    f.add_argument("--recompute", type=int, default=0, help="per-block activation recompute (big batches)")
    f.add_argument("--act-budget", type=float, default=1.0,
                   help="<1: compile recomputes cheap ops (LN/GELU/casts) to cut activation memory (e.g. 0.5)")
    f.add_argument("--precision", choices=["amp_bf16", "fp32"], default="amp_bf16")
    f.add_argument("--ddp-bf16-grads", type=int, default=0, help="all-reduce gradients in bf16 (halves DDP traffic)")
    f.add_argument("--fp8", type=int, default=0, help="opt-in FP8 (e4m3 rowwise) forward GEMMs; bf16 backward")
    f.add_argument("--tok-multiple", type=int, default=256, help="packed-token bucket (shape reuse; graphs: 1 per bucket)")
    s = ap.add_argument_group("output")
    s.add_argument("--out", required=True)
    s.add_argument("--log-every", type=int, default=20)
    s.add_argument("--save-every-epochs", type=float, default=1)
    s.add_argument("--val-every-steps", type=int, default=0)
    return ap.parse_args(argv)


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local


def to_dev(batch, dev):
    out = {}
    if "packed" in batch:
        pk = batch["packed"]
        out["packed"] = [t.to(dev, non_blocking=True) for t in pk[:5]] + [pk[5]]
    else:
        ids = batch["ids"]
        L = int((ids != 0).sum(1).max())
        out["ids"] = ids[:, :L].to(dev, non_blocking=True)
    return out


@torch.no_grad()
def evaluate(model, loader, gpu_images, dev, amp):
    model.eval()
    fis, fts = [], []
    for batch in loader:
        inp = to_dev(batch, dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            fi = model.encode_image(gpu_images(batch))
            ft = model.encode_text_packed(*inp["packed"]) if "packed" in inp else model.encode_text(inp["ids"])
        fis.append(fi.float())
        fts.append(ft.float())
    model.train()
    fi, ft = torch.cat(fis), torch.cat(fts)
    i2t, t2i = retrieval_acc(fi, ft)
    return dict(val_i2t_r1=i2t, val_t2i_r1=t2i, val_n=len(fi),
                val_loss=float(clip_loss(fi, ft, model.logit_scale)))


def main(argv=None):
    a = get_args(argv)
    rank, world, local = setup_dist()
    dev = torch.device("cuda", local)
    torch.manual_seed(a.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True     # only affects fp32 GEMMs (amp path is bf16)
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    img_path = a.img_path
    if img_path == "auto":
        img_path = "cache" if os.path.exists(os.path.join(a.data, "images.u8")) else "cpu"
    ncpu = len(os.sched_getaffinity(0))
    workers = a.workers if a.workers >= 0 else max(2, min(16, ncpu // max(1, int(os.environ.get(
        "LOCAL_WORLD_SIZE", 1))) - 2))
    ref, _, octok = load_reference("cpu")
    import cftrain.model as cm
    cm.FP8 = bool(a.fp8)
    tok = Tok()
    model = build_model(ref, a.mode, lora_r=a.lora_r, lora_alpha=a.lora_alpha,
                        lora_targets=tuple(a.lora_targets.split(",")), lora_towers=tuple(a.lora_towers.split(",")),
                        freeze_image_blocks=a.freeze_image_blocks, freeze_text_blocks=a.freeze_text_blocks,
                        text_dropout=a.text_dropout, attn_dropout=a.attn_dropout,
                        frozen_dtype=torch.bfloat16 if a.precision == "amp_bf16" else torch.float32).to(dev).train()
    amp = a.precision == "amp_bf16"
    graph = bool(a.cuda_graph) and world == 1
    loader, sampler = make_loader(a.data, a.batch_size, img_path=img_path, pack=True, workers=workers,
                                  seed=a.seed, rank=rank, world=world, tok=tok, device_index=local,
                                  tok_multiple=a.tok_multiple, image_col=a.image_col, caption_col=a.caption_col)
    gpu_images = GPUImages(dev)
    feats_cache = None
    if a.mode in ("lit", "lock-text"):
        which = "image" if a.mode == "lit" else "text"
        t0 = time.time()
        enc_loader, _ = make_loader(a.data, a.batch_size, img_path=img_path, pack=True, workers=workers,
                                    tok=tok, device_index=local, tok_multiple=a.tok_multiple, train=False,
                                    image_col=a.image_col, caption_col=a.caption_col, sequential=True)
        n = enc_loader.n_samples
        feats_cache = encode_dataset(model, enc_loader, gpu_images, which, n, dev, amp)
        if rank == 0:
            print(f"cached {which} features for {n} samples in {time.time() - t0:.1f}s", flush=True)
        del enc_loader
    ddp = None
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], gradient_as_bucket_view=True,
                                                        static_graph=True)
        if a.ddp_bf16_grads:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            ddp.register_comm_hook(None, default_hooks.bf16_compress_hook)
    opt = build_optimizer(model, a.lr, a.wd, betas=(0.9, a.beta2), eps=a.eps, capturable=graph)
    st = Stepper(model, opt, amp=amp, compile=bool(a.compile), world=world, rank=rank, ddp_model=ddp,
                 recompute=bool(a.recompute), act_budget=a.act_budget, compile_mode=a.compile_mode)
    runner = GraphStepper(st) if graph else st
    steps_per_epoch = len(sampler)
    total = a.max_steps or int(math.ceil(a.epochs * steps_per_epoch))
    val_loader = None
    if a.val_data and rank == 0:
        vimg = "cache" if os.path.exists(os.path.join(a.val_data, "images.u8")) else (
            "cpu" if img_path == "cache" else img_path)
        val_loader, vs = make_loader(a.val_data, a.batch_size, img_path=vimg, pack=True, workers=min(workers, 4),
                                     tok=tok, device_index=local, train=False, numa=False, sequential=True)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(json.dumps(dict(event="start", mode=a.mode, img_path=img_path, world=world, per_gpu_batch=a.batch_size,
                              steps=total, steps_per_epoch=steps_per_epoch, trainable_params=ntr, workers=workers,
                              numa_cpus=loader.numa_cpus, graph=graph)), flush=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a") if rank == 0 else None
    step, epoch = 0, 0
    t_last, n_last = time.time(), 0
    torch.cuda.reset_peak_memory_stats()
    save_every = max(1, int(a.save_every_epochs * steps_per_epoch))
    while step < total:
        sampler.epoch = epoch
        for batch in loader:
            if step >= total:
                break
            set_lr(opt, cosine_lr(step, a.lr, a.warmup, total))
            if DEBUG and "packed" in batch:
                pk = batch["packed"]
                tok, pos, cu = pk[0], pk[1], pk[2]
                assert int(tok.min()) >= 0 and int(tok.max()) < 30522, ("tok range", int(tok.min()), int(tok.max()))
                assert int(pos.max()) < 512 and int(cu[-1]) == tok.shape[0], ("pos/cu", int(pos.max()), cu[-3:])
                assert pk[3].shape[0] == a.batch_size and pk[4].shape[0] == cu.shape[0], (pk[3].shape, pk[4].shape)
            inp = to_dev(batch, dev)
            if feats_cache is not None and a.mode == "lit":
                inp["img_feats"] = feats_cache[batch["idx"].to(dev, non_blocking=True)]
            else:
                inp["images"] = gpu_images(batch)
            if feats_cache is not None and a.mode == "lock-text":
                inp.pop("packed", None)
                inp.pop("ids", None)
                inp["txt_feats"] = feats_cache[batch["idx"].to(dev, non_blocking=True)]
            loss = runner(inp)
            step += 1
            n_last += a.batch_size * world
            if rank == 0 and (step % a.log_every == 0 or step == total):
                lv = float(loss)
                dt = time.time() - t_last
                r = dict(step=step, epoch=epoch, loss=lv, lr=cosine_lr(step - 1, a.lr, a.warmup, total),
                         logit_scale=float(model.logit_scale.exp()), samples_per_s=round(n_last / dt, 1),
                         peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
                t_last, n_last = time.time(), 0
                print(json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
                log.flush()
            if rank == 0 and val_loader is not None and a.val_every_steps and step % a.val_every_steps == 0:
                r = dict(step=step, **evaluate(model, val_loader, gpu_images, dev, amp))
                print(json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
            if rank == 0 and (step % save_every == 0 or step == total):
                save(model, ref, os.path.join(a.out, f"checkpoint_{step}.pt"))
        epoch += 1
    if rank == 0 and val_loader is not None:
        r = dict(step=step, final=True, **evaluate(model, val_loader, gpu_images, dev, amp))
        print(json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
    if world > 1:
        dist.destroy_process_group()


def save(model, ref, path):
    """open_clip-compatible state_dict (LoRA merged), loadable by open_clip as `pretrained=`."""
    sd = model.export_to(ref).state_dict()
    torch.save(sd, path)
    print(json.dumps(dict(event="saved", path=path)), flush=True)


if __name__ == "__main__":
    main()

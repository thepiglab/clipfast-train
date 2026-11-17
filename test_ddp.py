"""DDP parity: torchrun --nproc_per_node N test_ddp.py --out results/ddp.jsonl
Each rank takes its shard of one global batch; after one DDP backward (all-gathered features, local loss), the
averaged gradients must equal the single-GPU gradients of the global batch (computed on rank 0), to fp32 noise.
fp32, dropout 0, TF32 off, so the comparison is tight."""
import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from cftrain.ref import load_reference
from cftrain.ref import text_ids
from cftrain.loss import clip_loss
from cftrain.model import LeanCLIP, pack_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=16, help="per rank")
    ap.add_argument("--device", default="cuda", help="cuda (NCCL) or cpu (gloo; checks the math without GPUs)")
    a = ap.parse_args()
    cpu = a.device == "cpu"
    dist.init_process_group("gloo" if cpu else "nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if not cpu:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cpu") if cpu else torch.device("cuda", torch.cuda.current_device())
    ref, _, _ = load_reference("cpu" if cpu else "cuda")
    gen = torch.Generator().manual_seed(0)
    G = a.B * world
    x = torch.randn(G, 3, 224, 224, generator=gen).to(dev)
    ids = text_ids(G, "pmc", ref.text.config.vocab_size, False, gen)
    model = LeanCLIP(ref, text_dropout=0, attn_dropout=0, frozen_dtype=torch.float32).to(dev).train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=None if cpu else [dev.index])
    sl = slice(rank * a.B, (rank + 1) * a.B)
    pk = pack_ids(ids[sl])
    fi, ft, s = ddp(images=x[sl], packed=[t.to(dev) for t in pk[:5]] + [pk[5]])
    loss = clip_loss(fi, ft, model.logit_scale, world, rank)
    loss.backward()
    g_ddp = torch.cat([p.grad.flatten() for p in model.parameters()])
    lsum = loss.detach().clone()
    dist.all_reduce(lsum)
    if rank == 0:
        m1 = LeanCLIP(ref, text_dropout=0, attn_dropout=0, frozen_dtype=torch.float32).to(dev).train()
        pk = pack_ids(ids)
        fi, ft, s = m1(images=x, packed=[t.to(dev) for t in pk[:5]] + [pk[5]])
        l1 = clip_loss(fi, ft, m1.logit_scale)
        l1.backward()
        g1 = torch.cat([p.grad.flatten() for p in m1.parameters()])
        r = dict(kind="ddp_parity", device=a.device, world=world, per_rank_B=a.B, loss_ddp_mean=float(lsum / world),
                 loss_single=float(l1), grad_cos=float(F.cosine_similarity(g_ddp, g1, dim=0)),
                 grad_rel_l2=float((g_ddp - g1).norm() / g1.norm()))
        r["PASS"] = bool(abs(r["loss_ddp_mean"] - r["loss_single"]) < 1e-4 and r["grad_rel_l2"] < 1e-4)
        print(json.dumps(r), flush=True)
        open(a.out, "a").write(json.dumps(r) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

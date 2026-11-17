"""CLIP contrastive loss, single GPU and data-parallel (all-gathered features), same math as open_clip ClipLoss.

open_clip semantics reproduced:
- features L2-normalised, logits = exp(logit_scale) * img @ txt^T, symmetric CE / 2 (fp32 here);
- DDP: gather_with_grad=True + local_loss=True: each rank scores its B local rows against all world*B columns,
  gradients flow through the gather (all_gather with autograd = reduce-scatter in backward). Loss is mean over
  local rows; DDP's gradient averaging then gives the global mean, identical to the single-GPU global-batch loss.
- logit_scale is clamped to ln(100) after each optimizer step by the trainer (open_clip does this in train loop).
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F


def _gather_with_grad(x):
    import torch.distributed.nn.functional as dnf
    return torch.cat(dnf.all_gather(x), 0)


def clip_loss(fi, ft, logit_scale, world=1, rank=0):
    fi = F.normalize(fi.float(), dim=-1)
    ft = F.normalize(ft.float(), dim=-1)
    s = logit_scale.float().exp()
    B = fi.shape[0]
    if world > 1:
        ai, at = _gather_with_grad(fi), _gather_with_grad(ft)
        li = s * fi @ at.T                     # [B, world*B]
        lt = s * ft @ ai.T
        lab = torch.arange(B, device=fi.device) + rank * B
    else:
        li = s * fi @ ft.T
        lt = li.T
        lab = torch.arange(B, device=fi.device)
    return (F.cross_entropy(li, lab) + F.cross_entropy(lt, lab)) / 2


@torch.no_grad()
def retrieval_acc(fi, ft):
    fi, ft = F.normalize(fi.float(), dim=-1), F.normalize(ft.float(), dim=-1)
    sim = fi @ ft.T
    lab = torch.arange(fi.shape[0], device=fi.device)
    return float((sim.argmax(1) == lab).float().mean()), float((sim.argmax(0) == lab).float().mean())

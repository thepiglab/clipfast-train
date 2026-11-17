"""Model / optimizer / step construction shared by the CLI, the parity tests and the benchmarks."""
import math

import torch
import torch.nn.functional as F

from .loss import clip_loss
from .model import LeanCLIP

LN100 = math.log(100.0)


def build_model(ref, mode="full", lora_r=8, lora_alpha=16, lora_targets=("qkv", "o", "fc1", "fc2"),
                lora_towers=("image", "text"), freeze_image_blocks=0, freeze_text_blocks=0, text_dropout=0.1,
                attn_dropout=0.1, frozen_dtype=torch.bfloat16):
    """mode: full | lora | lit (image tower locked, text trained) | lock-text (text locked, image trained)."""
    kw = dict(freeze_image_blocks=freeze_image_blocks, freeze_text_blocks=freeze_text_blocks,
              text_dropout=text_dropout, attn_dropout=attn_dropout, frozen_dtype=frozen_dtype)
    if mode == "full":
        return LeanCLIP(ref, **kw)
    if mode == "lora":
        return LeanCLIP(ref, lora_r=lora_r, lora_alpha=lora_alpha, lora_targets=lora_targets,
                        lora_towers=lora_towers, **kw)
    if mode == "lit":
        return LeanCLIP(ref, train_image=False, **kw)
    if mode == "lock-text":
        return LeanCLIP(ref, train_text=False, **kw)
    raise ValueError(mode)


def param_groups(model, wd):
    """open_clip's rule: no weight decay for params with ndim < 2 (gains, biases) and logit_scale."""
    decay, no = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # t_type is row 0 of open_clip's 2-D token_type_embeddings, which open_clip decays: keep that
        (no if (p.ndim < 2 and n != "t_type") or "logit_scale" in n else decay).append(p)
    return [dict(params=decay, weight_decay=wd), dict(params=no, weight_decay=0.0)]


def build_optimizer(model, lr, wd=0.2, betas=(0.9, 0.98), eps=1e-6, capturable=False):
    """Fused multi-tensor AdamW on fp32 master params (open_clip defaults for ViT: beta2 .98, eps 1e-6).
    lr is a 0-d CUDA tensor so schedules work inside CUDA graphs."""
    lr_t = torch.tensor(lr, device="cuda", dtype=torch.float32)
    return torch.optim.AdamW(param_groups(model, wd), lr=lr_t, betas=betas, eps=eps, fused=True,
                             capturable=capturable)


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * (step + 1) / warmup
    e = step - warmup
    es = max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * e / es)) * base


def set_lr(opt, lr):
    for g in opt.param_groups:
        if torch.is_tensor(g["lr"]):
            g["lr"].fill_(lr)
        else:
            g["lr"] = lr


class Stepper:
    """One optimizer step: forward (autocast bf16), CLIP loss, backward, fused AdamW, logit_scale clamp.

    compile: torch.compile the two tower functions (fixed shapes per bucket; dynamic=False).
    Inputs are a dict: images [B,3,224,224] (or img_feats), and packed / ids (or txt_feats).
    """

    def __init__(self, model, opt, *, amp=True, compile=False, world=1, rank=0, ddp_model=None,
                 recompute=False, act_budget=1.0, compile_mode=None):
        self.model, self.opt, self.amp, self.world, self.rank = model, opt, amp, world, rank
        self.fwd = ddp_model if ddp_model is not None else model
        model.recompute = recompute
        self.compiled = compile
        if compile and act_budget < 1.0:
            # min-cut partitioner: recompute cheap pointwise/LN ops in backward instead of saving them
            import torch._functorch.config as fc
            fc.activation_memory_budget = act_budget
        if compile:
            self.fwd = torch.compile(self.fwd, mode=compile_mode)

    def loss(self, inp):
        if self.compiled and "packed" in inp:      # packed token count varies: keep it symbolic (one compile)
            torch._dynamo.mark_dynamic(inp["packed"][0], 0)
            torch._dynamo.mark_dynamic(inp["packed"][1], 0)
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            fi, ft, s = self.fwd(**inp)
        return clip_loss(fi, ft, self.model.logit_scale, self.world, self.rank), fi, ft

    def __call__(self, inp):
        loss, _, _ = self.loss(inp)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            self.model.logit_scale.clamp_(0, LN100)
        return loss.detach()


@torch.no_grad()
def encode_dataset(model, loader, gpu_images, which, n, device="cuda", amp=True):
    """Embeddings of the frozen tower for every sample index (LiT cache). Returns [n,512] bf16 on device."""
    out = torch.zeros(n, 512, dtype=torch.bfloat16, device=device)
    model.eval()
    for batch in loader:
        idx = batch["idx"].to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            if which == "image":
                f = model.encode_image(gpu_images(batch))
            else:
                pk = batch.get("packed")
                if pk is not None:
                    f = model.encode_text_packed(*[t.to(device, non_blocking=True) for t in pk[:5]], pk[5])
                else:
                    f = model.encode_text(batch["ids"].to(device))
        out[idx] = f.to(torch.bfloat16)
    model.train()
    return out


class GraphStepper:
    """Whole-step CUDA graph (forward + loss + backward + AdamW + clamp) per input-shape key.

    Requirements: fixed shapes per key (images [B,...]; packed text padded to a token multiple with a fixed
    max_len), capturable optimizer, no host syncs in the step. Graphs share one memory pool and replay in order.
    """

    def __init__(self, stepper, warmup=3):
        self.st, self.warmup = stepper, warmup
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle()

    @staticmethod
    def _key(inp):
        return tuple((k, tuple(v.shape)) if torch.is_tensor(v) else
                     (k, tuple(tuple(t.shape) if torch.is_tensor(t) else t for t in v)) for k, v in sorted(inp.items()))

    def _body(self, sinp):
        st = self.st
        loss, _, _ = st.loss(sinp)
        loss.backward()
        st.opt.step()
        st.opt.zero_grad(set_to_none=False)
        with torch.no_grad():
            st.model.logit_scale.clamp_(0, LN100)
        return loss.detach()

    def _capture(self, inp):
        clone = lambda v: v.clone() if torch.is_tensor(v) else [clone(t) for t in v] if isinstance(v, (list, tuple)) else v
        sinp = {k: clone(v) for k, v in inp.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):   # warm-up = real optimizer steps on this batch (as with any warm-up)
                self._body(sinp)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            sloss = self._body(sinp)
        return g, sinp, sloss

    def __call__(self, inp):
        k = self._key(inp)
        if k not in self.graphs:
            self.graphs[k] = self._capture(inp)
        g, sinp, sloss = self.graphs[k]

        def cp(dst, src):
            if torch.is_tensor(dst):
                dst.copy_(src, non_blocking=True)
            elif isinstance(dst, (list, tuple)):
                for d, s_ in zip(dst, src):
                    cp(d, s_)
        for kk, v in inp.items():
            cp(sinp[kk], v)
        g.replay()
        return sloss

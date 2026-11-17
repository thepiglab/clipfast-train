"""Trainable lean BiomedCLIP (open_clip ViT-B/16 + PubMedBERT-256), same math as open_clip, fewer wasted FLOPs.

Started from LeanBiomedCLIP in thepiglab/clipfast (clipfast/train_model.py); extended for training modes:
- full fine-tuning: fp32 master params, bf16 autocast compute (open_clip --precision amp_bf16 numerics);
- LoRA (r, alpha, targets in {qkv,o,fc1,fc2}) on either/both towers; frozen base weights stored in bf16, so autograd
  never computes their dW (only dX) and no fp32 copy / Adam state exists for them;
- tower freezing (LiT): a frozen tower can be skipped entirely by feeding cached embeddings;
- partial freezing: freeze the first N blocks of a tower (no dW, and no dX below the first trainable block).
Exactness tricks (no approximation):
- last block of each tower runs only the CLS query row + its MLP (the loss only reads CLS);
- text is packed (varlen, no padding) with FlashAttention varlen, or padded to the batch's bucket with a key mask.
Dropout matches open_clip's HF BERT defaults when enabled (hidden 0.1 / attn-probs 0.1); vision has none.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import varlen_attention

V_LIN = ("qkv", "o", "f1", "f2")
FUSED_LORA = True        # module-level switch (benchmarks / parity compare fused vs unfused)
VIT_ATTN = True          # bespoke ViT attention backward (cftrain.attn_vit); False = PyTorch SDPA
MLP_EPI = False          # fused-epilogue MLP (cftrain.mlp_epi): fc1+GELU epilogue fwd, dGELU epilogue bwd
FP8 = False              # opt-in FP8 forward GEMMs for dense trainable block linears (cftrain.fp8)
LORA_NAME = {"qkv": "qkv", "o": "o", "fc1": "f1", "fc2": "f2", "q": "qkv", "v": "qkv"}


def _P(t, trainable=True, dtype=torch.float32):
    return nn.Parameter(t.detach().clone().to(dtype), requires_grad=trainable)


class LoRA(nn.Module):
    """y += (x @ A^T) @ B^T * scale. A ~ kaiming-uniform, B = 0 (so step 0 == base model)."""

    def __init__(self, n_in, n_out, r, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(r, n_in))
        self.B = nn.Parameter(torch.zeros(n_out, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return F.linear(F.linear(x, self.A), self.B) * self.scale


class _LoRALinear(torch.autograd.Function):
    """Frozen base linear + LoRA in 3 GEMMs forward / 4 backward, no dW for the base weight, no M x N temporaries:
      fwd: xa = x A^T [M,r];  y = x W^T + b;  y += xa (sB)^T   (cuBLAS beta=1 accumulate, in place)
      bwd: g = dy (sB) [M,r]; dx = dy W; dx += g A;  dA = g^T x;  dB = s dy^T xa
    (Unsloth fast_lora style). Compute dtype = W's dtype (bf16 when frozen in bf16)."""

    @staticmethod
    def forward(ctx, x, W, b, A, B, scale):
        dt = W.dtype
        x = x.to(dt)
        Ac, sB = A.to(dt), (B * scale).to(dt)
        x2 = x.reshape(-1, x.shape[-1])
        xa = x2 @ Ac.t()
        y = torch.addmm(b.to(dt), x2, W.t()) if b is not None else x2 @ W.t()
        y.addmm_(xa, sB.t())
        ctx.save_for_backward(x2, xa, W, Ac, sB)
        ctx.scale, ctx.shape, ctx.adt, ctx.bdt = scale, x.shape, A.dtype, B.dtype
        return y.view(*x.shape[:-1], W.shape[0])

    @staticmethod
    def backward(ctx, dy):
        x2, xa, W, Ac, sB = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1]).to(W.dtype)
        g = dy2 @ sB                                   # [M, r]
        dx = None
        if ctx.needs_input_grad[0]:
            dx = dy2 @ W
            dx.addmm_(g, Ac)
            dx = dx.view(ctx.shape)
        dA = (g.t() @ x2).to(ctx.adt)
        dB = ((dy2.t() @ xa) * ctx.scale).to(ctx.bdt)
        return dx, None, None, dA, dB, None


def lora_linear(x, W, b, lo):
    """Fused path when the base weight is frozen and already in the compute dtype (bf16 under amp, or fp32
    without autocast); otherwise None (caller falls back to the unfused module)."""
    amp = torch.is_autocast_enabled()
    if W.requires_grad or (amp and W.dtype != torch.bfloat16) or (not amp and W.dtype != x.dtype):
        return None
    with torch.autocast("cuda", enabled=False):
        return _LoRALinear.apply(x.to(W.dtype), W, b, lo.A, lo.B, lo.scale)


class Block(nn.Module):
    pass


class LeanCLIP(nn.Module):
    def __init__(self, ref, *, train_image=True, train_text=True, lora_r=0, lora_alpha=16,
                 lora_targets=("qkv", "o", "fc1", "fc2"), lora_towers=("image", "text"),
                 freeze_image_blocks=0, freeze_text_blocks=0, text_dropout=0.1, attn_dropout=0.1,
                 frozen_dtype=torch.bfloat16):
        super().__init__()
        self.text_dropout, self.attn_dropout = text_dropout, attn_dropout
        tr, v, t = ref.visual.trunk, ref.visual, ref.text
        tgt = {LORA_NAME[x] for x in lora_targets}

        def tower_mode(train, name):
            return "frozen" if not train else ("lora" if lora_r > 0 and name in lora_towers else "full")

        vm, tm = tower_mode(train_image, "image"), tower_mode(train_text, "text")
        self.modes = dict(image=vm, text=tm)

        def P(src, trainable, gemm=False):
            # frozen GEMM weights are stored in the compute dtype (bf16: what autocast would cast them to anyway);
            # frozen LN gains/biases/embeddings stay fp32 so the frozen model is exactly open_clip's amp model
            return _P(src, True) if trainable else _P(src, False, frozen_dtype if gemm else torch.float32)

        # ---------------- image tower ----------------
        emb = vm == "full" and freeze_image_blocks == 0
        self.v_patch_w = P(tr.patch_embed.proj.weight.reshape(768, -1), emb, True)   # conv(16, stride 16) == GEMM
        self.v_patch_b, self.v_cls, self.v_pos = (P(tr.patch_embed.proj.bias, emb), P(tr.cls_token, emb),
                                                  P(tr.pos_embed, emb))
        self.v_blocks = nn.ModuleList()
        for i, b in enumerate(tr.blocks):
            m = Block()
            dense = vm == "full" and i >= freeze_image_blocks
            for name, src in dict(n1w=b.norm1.weight, n1b=b.norm1.bias, qkv_w=b.attn.qkv.weight, qkv_b=b.attn.qkv.bias,
                                  o_w=b.attn.proj.weight, o_b=b.attn.proj.bias, n2w=b.norm2.weight, n2b=b.norm2.bias,
                                  f1_w=b.mlp.fc1.weight, f1_b=b.mlp.fc1.bias, f2_w=b.mlp.fc2.weight,
                                  f2_b=b.mlp.fc2.bias).items():
                setattr(m, name, P(src, dense, name.endswith("_w") and src.ndim == 2))
            if vm == "lora" and i >= freeze_image_blocks:
                for nm in tgt:
                    w = getattr(m, nm + "_w")
                    setattr(m, "lora_" + nm, LoRA(w.shape[1], w.shape[0], lora_r, lora_alpha))
            self.v_blocks.append(m)
        self.v_nw, self.v_nb = P(tr.norm.weight, vm == "full"), P(tr.norm.bias, vm == "full")
        self.v_proj = P(v.head.proj.weight, vm != "frozen", True)          # LoRA mode trains the small projection too
        self.v_heads = tr.blocks[0].attn.num_heads

        # ---------------- text tower ----------------
        bert = t.transformer
        e = bert.embeddings
        emb = tm == "full" and freeze_text_blocks == 0
        self.t_word, self.t_pos = P(e.word_embeddings.weight, emb), P(e.position_embeddings.weight, emb)
        self.t_type = P(e.token_type_embeddings.weight[0], emb)
        self.t_lnw, self.t_lnb, self.t_eps = P(e.LayerNorm.weight, emb), P(e.LayerNorm.bias, emb), e.LayerNorm.eps
        self.t_blocks = nn.ModuleList()
        for i, L in enumerate(bert.encoder.layer):
            a = L.attention
            m = Block()
            dense = tm == "full" and i >= freeze_text_blocks
            for name, src in dict(
                    qkv_w=torch.cat([a.self.query.weight, a.self.key.weight, a.self.value.weight]),
                    qkv_b=torch.cat([a.self.query.bias, a.self.key.bias, a.self.value.bias]),
                    o_w=a.output.dense.weight, o_b=a.output.dense.bias,
                    n1w=a.output.LayerNorm.weight, n1b=a.output.LayerNorm.bias,
                    f1_w=L.intermediate.dense.weight, f1_b=L.intermediate.dense.bias,
                    f2_w=L.output.dense.weight, f2_b=L.output.dense.bias,
                    n2w=L.output.LayerNorm.weight, n2b=L.output.LayerNorm.bias).items():
                setattr(m, name, P(src, dense, name.endswith("_w") and src.ndim == 2))
            if tm == "lora" and i >= freeze_text_blocks:
                for nm in tgt:
                    w = getattr(m, nm + "_w")
                    setattr(m, "lora_" + nm, LoRA(w.shape[1], w.shape[0], lora_r, lora_alpha))
            self.t_blocks.append(m)
        self.t_heads = bert.config.num_attention_heads
        self.t_p1, self.t_p2 = P(t.proj[0].weight, tm != "frozen", True), P(t.proj[2].weight, tm != "frozen", True)
        self.logit_scale = _P(ref.logit_scale, True)
        self.pad_id = t.config.pad_token_id or 0
        self.train_image, self.train_text = train_image, train_text
        self.recompute = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _lin(x, p, nm):
        lo = getattr(p, "lora_" + nm, None)
        w, b = getattr(p, nm + "_w"), getattr(p, nm + "_b")
        if lo is not None and FUSED_LORA:
            y = lora_linear(x, w, b, lo)
            if y is not None:
                return y
        if FP8 and lo is None and w.requires_grad and torch.is_autocast_enabled():
            from .fp8 import fp8_linear
            return fp8_linear(x, w, b)
        y = F.linear(x, w, b)
        if lo is not None:
            y = y + lo(x)
        return y

    def _mlp(self, h, p):
        if (MLP_EPI and torch.is_autocast_enabled() and p.f1_w.requires_grad and p.f2_w.requires_grad
                and h.numel() // h.shape[-1] >= 16384
                and getattr(p, "lora_f1", None) is None and getattr(p, "lora_f2", None) is None and not FP8):
            from .mlp_epi import fused_mlp
            return fused_mlp(h, p.f1_w, p.f1_b, p.f2_w, p.f2_b)
        return self._lin(F.gelu(self._lin(h, p, "f1")), p, "f2")

    def _drop(self, x):
        return F.dropout(x, self.text_dropout, self.training) if self.text_dropout > 0 else x

    def _ck(self, fn, *args):
        if self.recompute and self.training and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            return cp.checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    # ------------------------------------------------------------------ image tower
    def _v_block(self, x, p, last):
        B, N, C = x.shape
        H, lin = self.v_heads, self._lin
        h = F.layer_norm(x, (C,), p.n1w, p.n1b, 1e-6)
        qkv = lin(h, p, "qkv").view(B, N, 3, H, C // H)
        if VIT_ATTN and not last and qkv.dtype == torch.bfloat16:
            from .attn_vit import vit_attention     # FA2 fwd + bespoke Triton bwd writing dqkv in place
            a = vit_attention(qkv).reshape(B, N, C)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if last:
                q, x = q[:, :, :1], x[:, :1]
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, q.shape[2], C)
        x = x + lin(a, p, "o")
        h = F.layer_norm(x, (C,), p.n2w, p.n2b, 1e-6)
        return x + self._mlp(h, p)

    def encode_image(self, x):
        """x: [B,3,224,224] normalized float. Returns [B,512] (unnormalized, like open_clip)."""
        B = x.shape[0]
        H, lin = self.v_heads, self._lin
        # patchify as a GEMM: [B,3,14,16,14,16] -> [B,196,3*16*16] (same (c,kh,kw) order as the conv weight)
        pt = x.reshape(B, 3, 14, 16, 14, 16).permute(0, 2, 4, 1, 3, 5).reshape(B, 196, 768)
        x = F.linear(pt, self.v_patch_w, self.v_patch_b)
        x = torch.cat([self.v_cls.expand(B, -1, -1), x], 1) + self.v_pos   # promotes to fp32 like timm
        N, C = x.shape[1], x.shape[2]
        nb = len(self.v_blocks)
        for i, p in enumerate(self.v_blocks):
            x = self._ck(self._v_block, x, p, i == nb - 1)
        x = F.layer_norm(x[:, 0], (C,), self.v_nw, self.v_nb, 1e-6)
        return F.linear(x, self.v_proj)

    # ------------------------------------------------------------------ text tower
    def _block_tail(self, x, a, p, C):
        x = F.layer_norm(x + self._drop(self._lin(a, p, "o")), (C,), p.n1w, p.n1b, self.t_eps)
        m = self._mlp(x, p)
        return F.layer_norm(x + self._drop(m), (C,), p.n2w, p.n2b, self.t_eps)

    def _t_block_pad(self, x, am, p, last, pa):
        B, L, C = x.shape
        H = self.t_heads
        qkv = self._lin(x, p, "qkv").view(B, L, 3, H, C // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if last:
            q, x = q[:, :, :1], x[:, :1]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=am, dropout_p=pa)
        return self._block_tail(x, a.transpose(1, 2).reshape(B, q.shape[2], C), p, C)

    def _t_block_packed(self, x, p, cu, cls_idx, cu_cls, max_len, last, pa):
        C = x.shape[-1]
        H = self.t_heads
        qkv = self._lin(x, p, "qkv").view(-1, 3, H, C // H)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        if last:
            q, x = q[cls_idx], x[cls_idx]
            # cu_cls has a zero-length query for the dummy sequence (when present), so every K/V row lies inside
            # cu: FlashAttention's backward leaves dK/dV rows outside all sequences unwritten (garbage -> NaN)
            a = varlen_attention(q, k, v, cu_cls, cu, 1, max_len, pa)
        else:
            a = varlen_attention(q, k, v, cu, cu, max_len, max_len, pa)
        return self._block_tail(x, a.reshape(-1, C), p, C)

    def _text_head(self, x):
        return F.linear(F.gelu(F.linear(x, self.t_p1)), self.t_p2)

    def encode_text(self, ids):
        """Padded path: ids [B,L] (L = the batch's bucket); key-padding mask when any row is padded."""
        B, L = ids.shape
        H = self.t_heads
        mask = ids != self.pad_id
        am = mask[:, None, None, :]
        x = self.t_word[ids] + self.t_pos[:L] + self.t_type
        C = x.shape[-1]
        x = self._drop(F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps))
        nb = len(self.t_blocks)
        pa = self.attn_dropout if self.training else 0.0
        for i, p in enumerate(self.t_blocks):
            x = self._ck(self._t_block_pad, x, am, p, i == nb - 1, pa)
        return self._text_head(x[:, 0])

    def encode_text_packed(self, tok, pos, cu, cls_idx, cu_cls, max_len):
        """Packed path: tok/pos [T] (no padding), cu [B+1] int32 cumulative lengths, cls_idx [B] (= cu[:-1]),
        cu_cls = arange(B+1) int32 (+ a repeated B for the dummy sequence), max_len python int."""
        H = self.t_heads
        x = self.t_word[tok] + self.t_pos[pos] + self.t_type
        C = x.shape[-1]
        D = C // H
        x = self._drop(F.layer_norm(x, (C,), self.t_lnw, self.t_lnb, self.t_eps))
        nb = len(self.t_blocks)
        pa = self.attn_dropout if self.training else 0.0
        for i, p in enumerate(self.t_blocks):
            x = self._ck(self._t_block_packed, x, p, cu, cls_idx, cu_cls, max_len, i == nb - 1, pa)
        return self._text_head(x)

    def forward(self, images=None, ids=None, packed=None, img_feats=None, txt_feats=None):
        fi = img_feats if img_feats is not None else self.encode_image(images)
        if txt_feats is not None:
            ft = txt_feats
        elif packed is not None:
            ft = self.encode_text_packed(*packed)
        else:
            ft = self.encode_text(ids)
        return fi, ft, self.logit_scale

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def merged(self, p, nm):
        w = getattr(p, nm + "_w").float()
        lo = getattr(p, "lora_" + nm, None)
        if lo is not None:
            w = w + (lo.B.float() @ lo.A.float()) * lo.scale
        return w

    @torch.no_grad()
    def export_to(self, ref):
        """Copy weights (LoRA merged) back into an open_clip model in place; returns it."""
        tr, v, t = ref.visual.trunk, ref.visual, ref.text
        cp = lambda dst, src: dst.copy_(src.reshape(dst.shape).to(dst.dtype))
        cp(tr.patch_embed.proj.weight, self.v_patch_w)
        cp(tr.patch_embed.proj.bias, self.v_patch_b)
        cp(tr.cls_token, self.v_cls)
        cp(tr.pos_embed, self.v_pos)
        for b, p in zip(tr.blocks, self.v_blocks):
            for dst, src in [(b.norm1.weight, p.n1w), (b.norm1.bias, p.n1b), (b.attn.qkv.weight, self.merged(p, "qkv")),
                             (b.attn.qkv.bias, p.qkv_b), (b.attn.proj.weight, self.merged(p, "o")),
                             (b.attn.proj.bias, p.o_b), (b.norm2.weight, p.n2w), (b.norm2.bias, p.n2b),
                             (b.mlp.fc1.weight, self.merged(p, "f1")), (b.mlp.fc1.bias, p.f1_b),
                             (b.mlp.fc2.weight, self.merged(p, "f2")), (b.mlp.fc2.bias, p.f2_b)]:
                cp(dst, src)
        cp(tr.norm.weight, self.v_nw)
        cp(tr.norm.bias, self.v_nb)
        cp(v.head.proj.weight, self.v_proj)
        bert = t.transformer
        e = bert.embeddings
        cp(e.word_embeddings.weight, self.t_word)
        cp(e.position_embeddings.weight, self.t_pos)
        e.token_type_embeddings.weight[0].copy_(self.t_type.float())
        cp(e.LayerNorm.weight, self.t_lnw)
        cp(e.LayerNorm.bias, self.t_lnb)
        for L, p in zip(bert.encoder.layer, self.t_blocks):
            a = L.attention
            qkv = self.merged(p, "qkv")
            for j, lin in enumerate([a.self.query, a.self.key, a.self.value]):
                cp(lin.weight, qkv[j * 768:(j + 1) * 768])
                cp(lin.bias, p.qkv_b[j * 768:(j + 1) * 768])
            for dst, src in [(a.output.dense.weight, self.merged(p, "o")), (a.output.dense.bias, p.o_b),
                             (a.output.LayerNorm.weight, p.n1w), (a.output.LayerNorm.bias, p.n1b),
                             (L.intermediate.dense.weight, self.merged(p, "f1")), (L.intermediate.dense.bias, p.f1_b),
                             (L.output.dense.weight, self.merged(p, "f2")), (L.output.dense.bias, p.f2_b),
                             (L.output.LayerNorm.weight, p.n2w), (L.output.LayerNorm.bias, p.n2b)]:
                cp(dst, src)
        cp(t.proj[0].weight, self.t_p1)
        cp(t.proj[2].weight, self.t_p2)
        ref.logit_scale.copy_(self.logit_scale)
        return ref


def pack_ids(ids, pad_id=0, buckets=(16, 32, 48, 64, 96, 128, 192, 256), tok_multiple=0, fixed_max_len=False):
    """CPU-side packing of padded ids [B,Lpad] -> (tok, pos, cu, cls_idx, cu_cls, max_len).
    tok_multiple > 0 pads the packed token count up to a multiple (one dummy sequence) so shapes repeat
    (for CUDA graphs / compile without recompiles); the dummy sequence is never read by the loss."""
    ids = ids.cpu()
    mask = ids != pad_id
    lens = mask.sum(1)
    tok = ids[mask]
    pos = torch.arange(ids.shape[1]).expand_as(ids)[mask]
    B = len(lens)
    cu = torch.zeros(B + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(lens, 0)
    ml = int(lens.max())
    max_len = next((b for b in buckets if ml <= b), ml)
    if tok_multiple:   # fixed shapes for CUDA graphs: always one dummy sequence (>= 1 token) up to a multiple
        T = int(cu[-1])
        Tp = -(-(T + 1) // tok_multiple) * tok_multiple
        tok = torch.cat([tok, torch.full((Tp - T,), pad_id, dtype=tok.dtype)])
        pos = torch.cat([pos, torch.zeros(Tp - T, dtype=pos.dtype)])
        cu = torch.cat([cu, torch.tensor([Tp], dtype=torch.int32)])
        max_len = max(buckets[-1], tok_multiple) if fixed_max_len else max(max_len, Tp - T)
    cls_idx = cu[:B].long()
    cu_cls = torch.arange(B + 1, dtype=torch.int32)
    if len(cu) == B + 2:                       # dummy sequence: zero queries in the CLS-only last layer
        cu_cls = torch.cat([cu_cls, cu_cls[-1:]])
    return tok, pos, cu, cls_idx, cu_cls, max_len

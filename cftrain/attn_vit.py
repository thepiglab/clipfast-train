"""ViT self-attention (N=197, head 64, no mask) as a torch custom op with a bespoke Triton backward.

Layouts are the ones the block already has, so no permute/contiguous copies:
  qkv  [B, N, 3, H, D]  (the qkv GEMM output viewed)       -> forward input
  out  [B, N, H, D]     (feeds the out-proj GEMM as [B*N, C]) -> forward output
  dqkv [B, N, 3, H, D]  written directly by the backward kernels (feeds the qkv GEMM's backward)
Forward: FlashAttention-2 (aten._scaled_dot_product_flash_attention) for O and LSE.
Backward (kar task kar_tasks/short_attn_bwd, all canaries rejected), generalised to strided layouts:
  v2 (default): one program per (b,h), S/P once, dQ in fp32 scratch: 2.07x FA2 bwd at B=256, 2.36x at B=1024;
  v1: two kernels without atomics: 1.47x / 1.59x.
"""
import torch
import triton
import triton.language as tl

SCALE = 0.125
FWD_VERSION = "fa2"       # "triton": bespoke short-seq forward (kar task short_attn_fwd); "fa2": PyTorch FA2
FWD_CFG = (64, 64, 4)    # BM, BN, warps (set from the kar A/B winner)
BWD_BM = 64              # query block of the v2 backward (kar r5 decides 32 vs 64)
BWD_VERSION = 2          # 2: one program per head (kar cand_triton_v2), 1: two kernels (cand_triton_v1)


@triton.jit
def _dkdv(QKV, O, DO, LSE, DQKV, N, H, scale,
          s_b, s_n, s_o_b, s_o_n, s_l_bh,
          BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    b = bh // H
    h = bh % H
    qb = QKV + b.to(tl.int64) * s_b + h * D
    kb = qb + H * D
    vb = qb + 2 * H * D
    ob = O + b.to(tl.int64) * s_o_b + h * D
    dob = DO + b.to(tl.int64) * s_o_b + h * D
    lb = LSE + bh.to(tl.int64) * s_l_bh
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    nmask = rn < N
    k = tl.load(kb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(vb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        q = tl.load(qb + rm[:, None] * s_n + rd[None, :], mask=mmask[:, None], other=0.0)
        do = tl.load(dob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0)
        o = tl.load(ob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0)
        lse = tl.load(lb + rm, mask=mmask, other=0.0)
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
    db = DQKV + b.to(tl.int64) * s_b + h * D
    tl.store(db + H * D + rn[:, None] * s_n + rd[None, :], (dk * scale).to(tl.bfloat16), mask=nmask[:, None])
    tl.store(db + 2 * H * D + rn[:, None] * s_n + rd[None, :], dv.to(tl.bfloat16), mask=nmask[:, None])


@triton.jit
def _dq(QKV, O, DO, LSE, DQKV, N, H, scale,
        s_b, s_n, s_o_b, s_o_n, s_l_bh,
        BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b = bh // H
    h = bh % H
    qb = QKV + b.to(tl.int64) * s_b + h * D
    kb = qb + H * D
    vb = qb + 2 * H * D
    ob = O + b.to(tl.int64) * s_o_b + h * D
    dob = DO + b.to(tl.int64) * s_o_b + h * D
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mmask = rm < N
    q = tl.load(qb + rm[:, None] * s_n + rd[None, :], mask=mmask[:, None], other=0.0)
    do = tl.load(dob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0)
    o = tl.load(ob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * s_l_bh + rm, mask=mmask, other=0.0)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    dq = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        k = tl.load(kb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(vb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), k)
    db = DQKV + b.to(tl.int64) * s_b + h * D
    tl.store(db + rm[:, None] * s_n + rd[None, :], (dq * scale).to(tl.bfloat16), mask=mmask[:, None])


@triton.jit
def _bwd_head(QKV, O, DO, LSE, DQKV, DQACC, DELTA, N, H, scale,
              s_b, s_n, s_o_b, s_o_n, s_l_bh,
              BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, NP: tl.constexpr):
    """kar cand_triton_v2 (one program per (b, h); S/P once; dQ in an fp32 scratch only this program touches),
    generalised to the qkv [B,N,3,H,D] / out [B,N,H,D] layouts."""
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    qb = QKV + b.to(tl.int64) * s_b + h * D
    kb = qb + H * D
    vb = qb + 2 * H * D
    ob = O + b.to(tl.int64) * s_o_b + h * D
    dob = DO + b.to(tl.int64) * s_o_b + h * D
    lb = LSE + bh.to(tl.int64) * s_l_bh
    dqb = DQACC + bh.to(tl.int64) * NP * D
    deb = DELTA + bh.to(tl.int64) * NP
    db = DQKV + b.to(tl.int64) * s_b + h * D
    rd = tl.arange(0, D)
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        do = tl.load(dob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        o = tl.load(ob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        tl.store(deb + rm, tl.sum(do * o, 1))
        tl.store(dqb + rm[:, None] * D + rd[None, :], tl.zeros((BM, D), tl.float32))
    tl.debug_barrier()
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        k = tl.load(kb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(vb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        dk = tl.zeros((BN, D), tl.float32)
        dv = tl.zeros((BN, D), tl.float32)
        for m0 in range(0, N, BM):
            rm = m0 + tl.arange(0, BM)
            mmask = rm < N
            q = tl.load(qb + rm[:, None] * s_n + rd[None, :], mask=mmask[:, None], other=0.0)
            do = tl.load(dob + rm[:, None] * s_o_n + rd[None, :], mask=mmask[:, None], other=0.0)
            lse = tl.load(lb + rm, mask=mmask, other=0.0)
            delta = tl.load(deb + rm)
            s = tl.dot(q, tl.trans(k)) * (scale * 1.4426950408889634)       # kar r4 (v3d): exp2, +5.8 %
            p = tl.exp2(s - lse[:, None] * 1.4426950408889634)
            p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
            dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
            dk += tl.dot(tl.trans(ds), q)
            dqp = dqb + rm[:, None] * D + rd[None, :]
            tl.store(dqp, tl.load(dqp) + tl.dot(ds, k))
            tl.debug_barrier()
        tl.store(db + H * D + rn[:, None] * s_n + rd[None, :], (dk * scale).to(tl.bfloat16), mask=nmask[:, None])
        tl.store(db + 2 * H * D + rn[:, None] * s_n + rd[None, :], dv.to(tl.bfloat16), mask=nmask[:, None])
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        dq = tl.load(dqb + rm[:, None] * D + rd[None, :])
        tl.store(db + rm[:, None] * s_n + rd[None, :], (dq * scale).to(tl.bfloat16), mask=mmask[:, None])


@triton.jit
def _fwd_strided(QKV, O, LSE, N, H, scale_log2, s_b, s_n, s_o_b, s_o_n,
                 BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    """kar task short_attn_fwd candidate, generalised to qkv [B,N,3,H,D] -> O [B,N,H,D], LSE [B,H,N] (natural log)."""
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    b = bh // H
    h = bh % H
    qb = QKV + b.to(tl.int64) * s_b + h * D
    kb = qb + H * D
    vb = qb + 2 * H * D
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mmask = rm < N
    q = tl.load(qb + rm[:, None] * s_n + rd[None, :], mask=mmask[:, None], other=0.0)
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        k = tl.load(kb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(vb + rn[:, None] * s_n + rd[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where(nmask[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    ob = O + b.to(tl.int64) * s_o_b + h * D
    tl.store(ob + rm[:, None] * s_o_n + rd[None, :], (acc / l_i[:, None]).to(tl.bfloat16), mask=mmask[:, None])
    tl.store(LSE + bh.to(tl.int64) * N + rm, (m_i + tl.log2(l_i)) * 0.6931471805599453, mask=mmask)


@torch.library.custom_op("cftrain::vit_attn_fwd", mutates_args=())
def vit_attn_fwd(qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    B, N, _, H, D = qkv.shape
    if FWD_VERSION == "triton":
        qkv = qkv.contiguous()
        out = torch.empty(B, N, H, D, device=qkv.device, dtype=torch.bfloat16)
        lse = torch.empty(B, H, N, device=qkv.device, dtype=torch.float32)
        BM, BN, w = FWD_CFG
        _fwd_strided[(triton.cdiv(N, BM), B * H)](qkv, out, lse, N, H, SCALE * 1.4426950408889634, qkv.stride(0),
                                                  qkv.stride(1), out.stride(0), out.stride(1), BM=BM, BN=BN, D=D,
                                                  num_warps=w, num_stages=2)
        return out, lse
    q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))          # [B,H,N,D] strided views
    r = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=SCALE)
    out = r[0].transpose(1, 2).contiguous()                              # [B,N,H,D] (free if already so)
    return out, r[1].contiguous()                                        # lse [B,H,N] fp32


@vit_attn_fwd.register_fake
def _(qkv):
    B, N, _, H, D = qkv.shape
    return qkv.new_empty(B, N, H, D), qkv.new_empty(B, H, N, dtype=torch.float32)


@torch.library.custom_op("cftrain::vit_attn_bwd", mutates_args=())
def vit_attn_bwd(qkv: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, dout: torch.Tensor) -> torch.Tensor:
    B, N, _, H, D = qkv.shape
    qkv, out, dout = qkv.contiguous(), out.contiguous(), dout.contiguous()
    dqkv = torch.empty_like(qkv)
    BM = BN = 64
    if BWD_VERSION == 1:
        args = (qkv, out, dout, lse, dqkv, N, H, SCALE, qkv.stride(0), qkv.stride(1), out.stride(0), out.stride(1),
                lse.stride(1))
        _dkdv[(triton.cdiv(N, BN), B * H)](*args, BM=BM, BN=BN, D=D, num_warps=4, num_stages=2)
        _dq[(triton.cdiv(N, BM), B * H)](*args, BM=BM, BN=BN, D=D, num_warps=4, num_stages=2)
    else:
        BM = BWD_BM
        NP = triton.cdiv(N, BM) * BM
        dqacc = torch.empty(B * H, NP, D, device=qkv.device, dtype=torch.float32)   # per-call scratch
        delta = torch.empty(B * H, NP, device=qkv.device, dtype=torch.float32)
        _bwd_head[(B * H,)](qkv, out, dout, lse, dqkv, dqacc, delta, N, H, SCALE, qkv.stride(0), qkv.stride(1),
                            out.stride(0), out.stride(1), lse.stride(1), BM=BM, BN=BN, D=D, NP=NP,
                            num_warps=4, num_stages=2)
    return dqkv


@vit_attn_bwd.register_fake
def _(qkv, out, lse, dout):
    return torch.empty_like(qkv)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], output[0], output[1])


def _backward(ctx, dout, dlse):
    qkv, out, lse = ctx.saved_tensors
    return vit_attn_bwd(qkv, out, lse, dout)


vit_attn_fwd.register_autograd(_backward, setup_context=_setup)


def vit_attention(qkv):
    """qkv [B,N,3,H,D] bf16 -> attention output [B,N,H,D] bf16."""
    return vit_attn_fwd(qkv)[0]

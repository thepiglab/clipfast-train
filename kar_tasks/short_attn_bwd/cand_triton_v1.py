"""Short-sequence attention backward, no atomics: two Triton kernels.
  dkdv: one program per (key block, head): loops over all query blocks -> dK, dV of its key block.
  dq:   one program per (query block, head): loops over all key blocks -> dQ of its query block.
S/P are recomputed from the saved LSE (natural log of sum exp(scale * qk)); Delta = rowsum(dO * O) inline.
P and dS are rounded to bf16 before their GEMMs (as FlashAttention-2 does); accumulation fp32.
"""
import torch
import triton
import triton.language as tl

SCALE = 0.125


@triton.jit
def _dkdv(Q, K, V, O, DO, LSE, DK, DV, N, scale,
          BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_n, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * N * D
    rn = pid_n * BN + tl.arange(0, BN)
    rd = tl.arange(0, D)
    nmask = rn < N
    k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
        do = tl.load(DO + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
        o = tl.load(O + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
        lse = tl.load(LSE + bh.to(tl.int64) * N + rm, mask=mmask, other=0.0)
        delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
        dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(tl.bfloat16)), q)
    tl.store(DK + base + rn[:, None] * D + rd[None, :], (dk * scale).to(tl.bfloat16), mask=nmask[:, None])
    tl.store(DV + base + rn[:, None] * D + rd[None, :], dv.to(tl.bfloat16), mask=nmask[:, None])


@triton.jit
def _dq(Q, K, V, O, DO, LSE, DQ, N, scale,
        BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * N * D
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mmask = rm < N
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
    do = tl.load(DO + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
    o = tl.load(O + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
    lse = tl.load(LSE + bh.to(tl.int64) * N + rm, mask=mmask, other=0.0)
    delta = tl.sum(do.to(tl.float32) * o.to(tl.float32), 1)
    dq = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        p = tl.exp(s - lse[:, None])
        p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), k)
    tl.store(DQ + base + rm[:, None] * D + rd[None, :], (dq * scale).to(tl.bfloat16), mask=mmask[:, None])


CFG = dict(BM=64, BN=64, warps=4, stages=2)


def kernel_fn(q, k, v, o, lse, do):
    BH, N, D = q.shape
    out = torch.empty(3, BH, N, D, device=q.device, dtype=torch.bfloat16)
    dq, dk, dv = out[0], out[1], out[2]
    BM, BN, w, st = CFG["BM"], CFG["BN"], CFG["warps"], CFG["stages"]
    _dkdv[(triton.cdiv(N, BN), BH)](q, k, v, o, do, lse, dk, dv, N, SCALE, BM=BM, BN=BN, D=D,
                                     num_warps=w, num_stages=st)
    _dq[(triton.cdiv(N, BM), BH)](q, k, v, o, do, lse, dq, N, SCALE, BM=BM, BN=BN, D=D,
                                   num_warps=w, num_stages=st)
    return out


def _warm():
    x = torch.randn(12, 197, 64, device="cuda", dtype=torch.bfloat16)
    kernel_fn(x, x, x, x, torch.zeros(12, 197, device="cuda"), x)
    torch.cuda.synchronize()


_warm()

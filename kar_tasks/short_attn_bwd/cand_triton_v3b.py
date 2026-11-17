"""Round 4 variant of cand_triton_v2: BM=32
Base: short-sequence attention backward, ONE kernel per (batch, head), S/P computed once.
Hypothesis (from v1 at 35 % of floor): v1 recomputes S and reloads Q/K/V/dO/O in two kernels (7 GEMM tiles per
(m,n) pair instead of 5). One program per head loops over key blocks (outer) and query blocks (inner); dK/dV of
the key block accumulate in registers; dQ accumulates in an fp32 scratch row block that only this program touches
(read-modify-write stays in L1/L2, no atomics), converted to bf16 at the end. Delta = rowsum(dO*O) is computed
once per head into a small scratch in a first pass.
Predicted: ~5/7 of v1's tensor work and ~half its loads -> 1.3-1.5x over v1. Kill: < 1.1x over v1 on b256.
"""
import torch
import triton
import triton.language as tl

SCALE = 0.125


@triton.jit
def _bwd_head(Q, K, V, O, DO, LSE, OUT, DQACC, DELTA, N, BH, scale,
              BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, NP: tl.constexpr):
    bh = tl.program_id(0)
    base = bh.to(tl.int64) * N * D
    rd = tl.arange(0, D)
    dqb = DQACC + bh.to(tl.int64) * NP * D
    deb = DELTA + bh.to(tl.int64) * NP
    # pass 0: delta, zero dq scratch
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        do = tl.load(DO + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        o = tl.load(O + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        tl.store(deb + rm, tl.sum(do * o, 1))
        tl.store(dqb + rm[:, None] * D + rd[None, :], tl.zeros((BM, D), tl.float32))
    tl.debug_barrier()   # scratch written by other threads of this CTA must be visible
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
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
            lse = tl.load(LSE + bh.to(tl.int64) * N + rm, mask=mmask, other=0.0)
            delta = tl.load(deb + rm)
            s = tl.dot(q, tl.trans(k)) * scale
            p = tl.exp(s - lse[:, None])
            p = tl.where(mmask[:, None] & nmask[None, :], p, 0.0)
            dv += tl.dot(tl.trans(p.to(tl.bfloat16)), do)
            dp = tl.dot(do, tl.trans(v))
            ds = (p * (dp - delta[:, None])).to(tl.bfloat16)
            dk += tl.dot(tl.trans(ds), q)
            dqp = dqb + rm[:, None] * D + rd[None, :]
            tl.store(dqp, tl.load(dqp) + tl.dot(ds, k))
            tl.debug_barrier()
        ob = OUT + base
        tl.store(ob + BH * N * D + rn[:, None] * D + rd[None, :], (dk * scale).to(tl.bfloat16), mask=nmask[:, None])
        tl.store(ob + 2 * BH * N * D + rn[:, None] * D + rd[None, :], dv.to(tl.bfloat16), mask=nmask[:, None])
    for m0 in range(0, N, BM):
        rm = m0 + tl.arange(0, BM)
        mmask = rm < N
        dq = tl.load(dqb + rm[:, None] * D + rd[None, :])
        tl.store(OUT + base + rm[:, None] * D + rd[None, :], (dq * scale).to(tl.bfloat16), mask=mmask[:, None])


CFG = dict(BM=32, BN=64, warps=4, stages=2)
_scratch = {}


def kernel_fn(q, k, v, o, lse, do):
    BH, N, D = q.shape
    BM, BN = CFG["BM"], CFG["BN"]
    NP = triton.cdiv(N, BM) * BM
    out = torch.empty(3, BH, N, D, device=q.device, dtype=torch.bfloat16)
    # per-call scratch (no state carried between calls)
    dqacc = torch.empty(BH, NP, D, device=q.device, dtype=torch.float32)
    delta = torch.empty(BH, NP, device=q.device, dtype=torch.float32)
    _bwd_head[(BH,)](q, k, v, o, do, lse, out, dqacc, delta, N, BH, SCALE, BM=BM, BN=BN, D=D, NP=NP,
                     num_warps=CFG["warps"], num_stages=CFG["stages"])
    return out


def _warm():
    x = torch.randn(12, 197, 64, device="cuda", dtype=torch.bfloat16)
    kernel_fn(x, x, x, x, torch.zeros(12, 197, device="cuda"), x)
    torch.cuda.synchronize()


_warm()

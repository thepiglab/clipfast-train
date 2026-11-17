"""Short-sequence attention forward: one program per (64-query block, head); 64-key blocks with online softmax
(exp2), fp32 accumulation, P rounded to bf16 for P@V (as FA2). Writes O and LSE (natural log)."""
import torch
import triton
import triton.language as tl

SCALE = 0.125


@triton.jit
def _fwd(Q, K, V, O, LSE, N, scale_log2,
         BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr):
    pid_m, bh = tl.program_id(0), tl.program_id(1)
    base = bh.to(tl.int64) * N * D
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mmask = rm < N
    q = tl.load(Q + base + rm[:, None] * D + rd[None, :], mask=mmask[:, None], other=0.0)
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    for n0 in range(0, N, BN):
        rn = n0 + tl.arange(0, BN)
        nmask = rn < N
        k = tl.load(K + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + base + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale_log2
        s = tl.where(nmask[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(O + base + rm[:, None] * D + rd[None, :], o.to(tl.bfloat16), mask=mmask[:, None])
    lse = (m_i + tl.log2(l_i)) * 0.6931471805599453
    tl.store(LSE + bh.to(tl.int64) * N + rm, lse, mask=mmask)


CFG = dict(BM=32, BN=64, warps=2, stages=2)


def attn_fwd(q, k, v):
    BH, N, D = q.shape
    o = torch.empty_like(q)
    lse = torch.empty(BH, N, device=q.device, dtype=torch.float32)
    _fwd[(triton.cdiv(N, CFG["BM"]), BH)](q, k, v, o, lse, N, SCALE * 1.4426950408889634, BM=CFG["BM"],
                                          BN=CFG["BN"], D=D, num_warps=CFG["warps"], num_stages=CFG["stages"])
    return o, lse


def kernel_fn(q, k, v):
    return attn_fwd(q, k, v)[0]


def _warm():
    x = torch.randn(12, 197, 64, device="cuda", dtype=torch.bfloat16)
    kernel_fn(x, x, x)
    torch.cuda.synchronize()


_warm()

"""Triton GEMMs with training epilogues for the MLP's exact-erf GELU (bf16 in/out, fp32 accumulate + epilogue).

GEMM core and tile order copied from the kar-validated `clipfast/kernels.py` of thepiglab/clipfast
(_fused_linear: grouped tile order, fixed per-shape configs); epilogues are new:
  MODE 1 (fc1 forward):  pre = x @ w^T + b -> written; act = GELU(pre) from the fp32 value -> written
  MODE 3 (fc1/fc2 bwd):  dpre = (dy @ w2) * GELU'(pre)   (pre loaded in the epilogue)
Removes the separate GELU forward pass (read pre, write act) and the separate gelu_backward pass
(read dact + pre, write dpre), and the bf16 rounding between GEMM and epilogue.
"""
import torch
import triton
import triton.language as tl

CONFIGS = {"wide": (128, 256, 32, 8, 8, 3), "b": (128, 128, 64, 8, 4, 3), "c": (256, 128, 32, 8, 8, 3),
           "small": (64, 128, 64, 8, 4, 3)}


@triton.jit
def _gemm_epi(x_ptr, w_ptr, b_ptr, aux_ptr, y_ptr, y2_ptr, M, N, K,
              sxm, sxk, swn, swk, sam, san, sym, syn,
              MODE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    per_group = GROUP_M * num_n
    gid = pid // per_group
    first_m = gid * GROUP_M
    gsize = min(num_m - first_m, GROUP_M)
    pm = first_m + (pid % per_group) % gsize
    pn = (pid % per_group) // gsize
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + rm[:, None].to(tl.int64) * sxm + rk[None, :] * sxk
    wp = w_ptr + rn[None, :].to(tl.int64) * swn + rk[:, None] * swk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = rk < K - k * BK
        a = tl.load(xp, mask=(rm[:, None] < M) & kmask[None, :], other=0.0)
        b = tl.load(wp, mask=(rn[None, :] < N) & kmask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        xp += BK * sxk
        wp += BK * swk
    omask = (rm[:, None] < M) & (rn[None, :] < N)
    yoff = rm[:, None].to(tl.int64) * sym + rn[None, :] * syn
    if MODE == 1:
        bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        pre = acc + bias[None, :]
        tl.store(y_ptr + yoff, pre.to(tl.bfloat16), mask=omask)
        act = 0.5 * pre * (1.0 + tl.math.erf(pre * 0.7071067811865476))
        tl.store(y2_ptr + yoff, act.to(tl.bfloat16), mask=omask)
    if MODE == 3:
        p = tl.load(aux_ptr + rm[:, None].to(tl.int64) * sam + rn[None, :] * san, mask=omask, other=0.0).to(tl.float32)
        dg = 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476)) + p * tl.exp(-0.5 * p * p) * 0.3989422804014327
        tl.store(y_ptr + yoff, (acc * dg).to(tl.bfloat16), mask=omask)


def _launch(x, w_nk_strides, w, N, b, aux, y, y2, mode, cfg):
    M, K = x.shape
    BM, BN, BK, G, nw, ns = CONFIGS["small"] if M < 2048 else CONFIGS[cfg]
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    swn, swk = w_nk_strides
    aux_ = aux if aux is not None else y
    _gemm_epi[grid](x, w, b if b is not None else y, aux_, y, y2 if y2 is not None else y, M, N, K,
                    x.stride(0), x.stride(1), swn, swk, aux_.stride(0), aux_.stride(1), y.stride(0), y.stride(1),
                    MODE=mode, BM=BM, BN=BN, BK=BK, GROUP_M=G, num_warps=nw, num_stages=ns)


def fc1_gelu(x, w, b, cfg="wide", out=None, pre=None, act=None):
    """x [M,K] bf16, w [N,K] bf16, b [N] -> pre, act [M,N] bf16. Writes into `out` [2,M,N] (returned) or into the
    separate tensors `pre`/`act` (returns (pre, act))."""
    M, N = x.shape[0], w.shape[0]
    if pre is not None:
        _launch(x, (w.stride(0), w.stride(1)), w, N, b, None, pre, act, 1, cfg)
        return pre, act
    if out is None:
        out = torch.empty(2, M, N, device=x.device, dtype=torch.bfloat16)
    _launch(x, (w.stride(0), w.stride(1)), w, N, b, None, out[0], out[1], 1, cfg)
    return out


def dgelu_mm(dy, w2, pre, cfg="wide"):
    """dy [M,K2] bf16 (K2 = fc2 out = 768), w2 [K2, N] (fc2 weight), pre [M,N] -> (dy @ w2) * GELU'(pre)."""
    M, N = dy.shape[0], w2.shape[1]
    y = torch.empty(M, N, device=dy.device, dtype=torch.bfloat16)
    # GEMM "w" viewed as [N, K2]: element (n, k) = w2[k, n]
    _launch(dy, (w2.stride(1), w2.stride(0)), w2, N, None, pre, y, None, 3, cfg)
    return y


CFG = "b"


def kernel_fn(x, w, b):
    return fc1_gelu(x, w, b, cfg=CFG)


_x = torch.randn(4096, 768, device="cuda", dtype=torch.bfloat16)
for _m in (4096, 100):
    kernel_fn(_x[:_m], torch.randn(3072, 768, device="cuda", dtype=torch.bfloat16), torch.zeros(3072, device="cuda", dtype=torch.bfloat16))
torch.cuda.synchronize()

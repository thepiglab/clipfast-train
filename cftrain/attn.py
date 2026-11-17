"""Varlen (packed) attention with autograd.

bf16/fp16: aten._flash_attention_forward (FlashAttention-2 inside PyTorch; its backward is registered in
derivatives.yaml, dropout supported via philox seed/offset). No host sync: max lengths are python ints.
fp32 (parity tests only): per-sequence SDPA math loop (slow, exact).
"""
import torch
import torch.nn.functional as F


def varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=0.0):
    """q [Tq,H,D], k/v [Tk,H,D]; cu_* int32 [B+1]. Returns [Tq,H,D]."""
    if q.dtype in (torch.bfloat16, torch.float16):
        return torch.ops.aten._flash_attention_forward(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p,
                                                       False, False)[0]
    assert dropout_p == 0.0, "fp32 varlen reference path has no dropout"
    cq, ck = cu_q.tolist(), cu_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].transpose(0, 1)[None]
        ki = k[ck[i]:ck[i + 1]].transpose(0, 1)[None]
        vi = v[ck[i]:ck[i + 1]].transpose(0, 1)[None]
        outs.append(F.scaled_dot_product_attention(qi, ki, vi)[0].transpose(0, 1))
    o = torch.cat(outs)
    if o.shape[0] < q.shape[0]:      # dummy padding tokens past cu_q[-1]
        o = torch.cat([o, o.new_zeros(q.shape[0] - o.shape[0], *o.shape[1:])])
    return o

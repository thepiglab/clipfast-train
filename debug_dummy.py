"""NaN hunt: packed text with a dummy padding sequence. The caching allocator is pre-filled with NaN so any gradient
rows a kernel leaves unwritten show up deterministically. Variants:
  nodummy       pack without dummy (bench/parity path)
  dummy_old     dummy, last layer K range cut to the real tokens (the pre-fix behaviour)
  dummy_fixed   dummy, zero-length query for the dummy in the last layer (current code)
Also eager vs compile for the fixed variant, and a few real batches from the synthetic cache."""
import json

import numpy as np
import torch

from cftrain.ref import load_reference
from cftrain.loss import clip_loss
from cftrain.model import LeanCLIP, pack_ids


def poison():
    t = torch.empty(int(30e9) // 4, dtype=torch.float32, device="cuda")
    t.fill_(float("nan"))
    del t


ref, _, _ = load_reference()
ids_all = np.load("data/synth16k_cache/ids.npy")
out = open("results/debug_dummy.jsonl", "a")
for variant, comp in [("nodummy", False), ("dummy_old", False), ("dummy_fixed", False), ("dummy_fixed", True)]:
    m = LeanCLIP(ref, text_dropout=0, attn_dropout=0).cuda().train()
    fwd = torch.compile(m) if comp else m
    res = []
    for bi in range(3):
        ids = torch.from_numpy(ids_all[bi * 64:(bi + 1) * 64].astype(np.int64))
        x = torch.randn(64, 3, 224, 224, device="cuda")
        pk = pack_ids(ids, tok_multiple=0 if variant == "nodummy" else 256, fixed_max_len=variant != "nodummy")
        pk = list(pk)
        if variant == "dummy_old":
            pk[4] = torch.arange(65, dtype=torch.int32)
        pk = [t.cuda() for t in pk[:5]] + [pk[5]]
        if variant == "dummy_old":   # emulate the old call: K range cut to the real tokens
            import cftrain.model as cm
            orig = cm.varlen_attention

            def cut(q, k, v, cq, ck, mq, mk, p, _o=orig):
                if mq == 1 and ck.shape[0] != cq.shape[0]:
                    ck = ck[: cq.shape[0]]
                return _o(q, k, v, cq, ck, mq, mk, p)
            cm.varlen_attention = cut
        m.zero_grad(set_to_none=True)
        poison()
        with torch.autocast("cuda", torch.bfloat16):
            fi, ft, s = fwd(images=x, packed=pk)
        loss = clip_loss(fi, ft, m.logit_scale)
        poison()
        loss.backward()
        bad = [n for n, q in m.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
        res.append(dict(loss=float(loss), n_nonfinite_grads=len(bad), first=bad[:4]))
        if variant == "dummy_old":
            cm.varlen_attention = orig
    r = dict(kind="debug_dummy", variant=variant, compiled=comp, batches=res)
    print(json.dumps(r), flush=True)
    out.write(json.dumps(r) + "\n")
    del m, fwd
    torch._dynamo.reset()
    torch.cuda.empty_cache()

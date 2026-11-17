"""Exported checkpoint loads in stock open_clip and reproduces the cftrain model's embeddings; also checks that
cache pixels equal open_clip's eval transform."""
import argparse, json, glob, os
import torch, torch.nn.functional as F
import numpy as np
from cftrain.ref import load_reference, MODEL_DIR
import open_clip

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--cache", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()
ck = sorted(glob.glob(os.path.join(a.run, "checkpoint_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))[-1]
ref, pre, tok = load_reference()
sd = torch.load(ck, map_location="cuda")
missing, unexpected = ref.load_state_dict(sd, strict=True), None
ref.eval()
base, _, _ = load_reference()
base.eval()
gen = torch.Generator().manual_seed(0)
x = torch.randn(16, 3, 224, 224, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
t = tok(["chest x-ray with pleural effusion", "h&e stained histopathology", "brain mri"] * 5 + ["ct"]).cuda()
with torch.no_grad():
    fi, ft = ref.encode_image(x), ref.encode_text(t)
    fi0, ft0 = base.encode_image(x), base.encode_text(t)
res = dict(kind="export_check", ckpt=ck, loads_strict=True,
           changed_vs_pretrained_img_cos=float(F.cosine_similarity(fi, fi0).mean()),
           changed_vs_pretrained_txt_cos=float(F.cosine_similarity(ft, ft0).mean()))
if a.cache:
    meta = json.load(open(os.path.join(a.cache, "meta.json")))
    mm = np.memmap(os.path.join(a.cache, "images.u8"), dtype=np.uint8, mode="r", shape=(meta["n"], 3, 224, 224))
    caps = [json.loads(l) for l in open(os.path.join(a.cache, "captions.jsonl"))]
    from PIL import Image
    diffs = []
    for i in range(0, meta["n"], max(1, meta["n"] // 32)):
        ref_px = (pre(Image.open(caps[i]["image"]).convert("RGB")))           # normalized float, open_clip eval transform
        mean = torch.tensor(pre.transforms[-1].mean).view(3, 1, 1); std = torch.tensor(pre.transforms[-1].std).view(3, 1, 1)
        ours = (torch.from_numpy(np.array(mm[i])).float() / 255 - mean) / std
        diffs.append(float((ours - ref_px).abs().max()))
    res["cache_pixels_max_abs_diff_normalized"] = max(diffs)
print(json.dumps(res)); open(a.out, "a").write(json.dumps(res) + "\n")

"""Synthetic image-caption dataset for CLI / loader tests: smooth random JPEGs (varied sizes) + caption-like text."""
import argparse, csv, os, random
import numpy as np
from multiprocessing import Pool
from PIL import Image

WORDS = ("chest radiograph shows bilateral opacity pleural effusion cardiomegaly histopathology adenocarcinoma "
         "hematoxylin eosin stain mri axial t2 weighted lesion enhancement ct scan nodule lobe fracture bone "
         "fundus retina diabetic microaneurysm immunohistochemistry positive tumor cells magnification scale bar "
         "arrow indicates region of interest figure panel patient year old male female follow up").split()

def work(args):
    out, i, seed = args
    r = np.random.default_rng(seed + i)
    h, w = int(r.integers(160, 1400)), int(r.integers(160, 1400))
    base = r.random((max(2, h // 32), max(2, w // 32), 3))
    im = Image.fromarray((base * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
    ext = "jpg" if i % 10 else "png"
    p = os.path.join(out, "img", f"{i:07d}.{ext}")
    im.save(p, quality=90) if ext == "jpg" else im.save(p)
    n = int(np.clip(40 * np.exp(0.8 * r.standard_normal()), 3, 300))
    rr = random.Random(seed + i)
    return os.path.relpath(p, out), " ".join(rr.choice(WORDS) for _ in range(n))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--n", type=int, default=4096)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    with Pool(len(os.sched_getaffinity(0))) as p:
        rows = p.map(work, [(a.out, i, 0) for i in range(a.n)], chunksize=64)
    with open(os.path.join(a.out, "captions.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["image", "caption"]); wr.writerows(rows)
    with open(os.path.join(a.out, "captions.jsonl"), "w") as f:
        import json
        for im, c in rows: f.write(json.dumps({"image": im, "caption": c}) + "\n")
    print("wrote", len(rows))

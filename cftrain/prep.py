"""Prepare a fast training cache from an image-caption dataset (run on CPUs, e.g. a burst-QOS job).

  python -m cftrain.prep --data <folder|csv|jsonl> --out <cache_dir> [--workers 64]

Writes cache_dir/{images.u8 (uint8 [N,3,224,224] memmap), ids.npy (uint16 [N,256] token ids), captions.jsonl,
meta.json}. Pixels are exactly open_clip's BiomedCLIP eval transform before ToTensor/Normalize
(Resize(224, bicubic) on the shortest side + CenterCrop(224) with PIL), so cache-mode training sees the same
pixels as open_clip inference. Unreadable images are recorded in meta.json["bad"] and replaced by black images
with their caption kept (they are rare; drop them from the source list if you prefer).
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

_STATE = {}


def _init(path, n, tf_repr):
    from cftrain.data import open_clip_transform
    tf = open_clip_transform(train=False)
    _STATE["tf"] = type(tf)(tf.transforms[:-2])          # drop ToTensor + Normalize -> PIL 224x224
    _STATE["mm"] = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n, 3, 224, 224))


def _work(chunk):
    from PIL import Image
    bad = []
    tf, mm = _STATE["tf"], _STATE["mm"]
    for i, p in chunk:
        try:
            im = tf(Image.open(p).convert("RGB"))
            mm[i] = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)
        except Exception as e:  # noqa: BLE001
            mm[i] = 0
            bad.append((i, p, str(e)[:200]))
    mm.flush()
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--caption-col", default="caption")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from cftrain.data import Tok, read_pairs
    t0 = time.time()
    pairs = read_pairs(a.data, a.image_col, a.caption_col)
    if a.limit:
        pairs = pairs[:a.limit]
    n = len(pairs)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "images.u8")
    np.memmap(path, dtype=np.uint8, mode="w+", shape=(n, 3, 224, 224)).flush()
    with open(os.path.join(a.out, "captions.jsonl"), "w") as f:
        for p, c in pairs:
            f.write(json.dumps({"image": p, "caption": c}) + "\n")
    tok = Tok()
    ids = np.zeros((n, 256), dtype=np.uint16)
    for s in range(0, n, 4096):
        ids[s:s + 4096] = tok([c for _, c in pairs[s:s + 4096]]).numpy().astype(np.uint16)
    np.save(os.path.join(a.out, "ids.npy"), ids)
    lens = (ids != 0).sum(1)
    chunks = [[(i, pairs[i][0]) for i in range(s, min(n, s + 256))] for s in range(0, n, 256)]
    bad = []
    with Pool(a.workers, initializer=_init, initargs=(path, n, None)) as pool:
        for k, b in enumerate(pool.imap_unordered(_work, chunks)):
            bad += b
            if k % 50 == 0:
                print(f"{min(n, (k + 1) * 256)}/{n} images  {time.time() - t0:.0f}s", flush=True)
    meta = dict(n=n, source=os.path.abspath(a.data), bad=bad, seconds=time.time() - t0,
                text_len_mean=float(lens.mean()), text_len_p50=float(np.median(lens)),
                text_len_p99=float(np.percentile(lens, 99)), text_truncated=int((lens >= 256).sum()))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "bad"} | {"bad": len(bad)}))


if __name__ == "__main__":
    main()

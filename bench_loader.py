"""Loader throughput (samples/s) per image path, with GPU finishing (decode/resize/normalize) included."""
import argparse, json, os, socket, time
import torch
from cftrain.data import GPUImages, make_loader

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--B", type=int, default=256)
ap.add_argument("--batches", type=int, default=40); ap.add_argument("--workers", default="4,8,12")
a = ap.parse_args()
gi = GPUImages(torch.device("cuda"))
out = open(a.out, "a")
for path, spec in [("cache", a.cache), ("gpu", a.data), ("cpu", a.data)]:
    for w in [int(x) for x in a.workers.split(",")]:
        dl, s = make_loader(spec, a.B, img_path=path, pack=True, workers=w)
        it = iter(dl); n = 0
        for _ in range(3): gi(next(it))
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(a.batches):
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            x = gi(b); n += x.shape[0]
        torch.cuda.synchronize(); dt = time.time() - t0
        r = dict(kind="loader", path=path, workers=w, B=a.B, samples_per_s=round(n / dt, 1), numa_cpus=dl.numa_cpus,
                 cpus=len(os.sched_getaffinity(0)), host=socket.gethostname(), job=os.environ.get("CFTRAIN_RUN_ID"))
        print(json.dumps(r), flush=True); out.write(json.dumps(r) + "\n"); out.flush()
        del dl, it

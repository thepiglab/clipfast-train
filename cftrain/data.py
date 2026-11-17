"""Image-caption data for BiomedCLIP training.

Dataset specs (--data):
  folder/            images with same-stem .txt captions (a.jpg + a.txt), or folder/ + a captions.csv/.jsonl inside
  file.csv           columns image,caption (names configurable); image paths relative to the CSV's folder
  file.jsonl         {"image": ..., "caption": ...} per line
  cache dir          output of `python -m cftrain.prep` (uint8 memmap images + captions): the fast path

Image paths (--img-path):
  cpu    PIL decode + open_clip's own train transform (exact open_clip parity; RandomResizedCrop(224,(0.9,1)) bicubic)
  gpu    workers read raw bytes; nvJPEG batched decode on the GPU (torchvision.io.decode_jpeg device='cuda'),
         GPU bicubic antialias resize + center crop (non-JPEG files fall back to PIL in the worker)
  cache  uint8 [N,3,224,224] memmap made once by prep.py (open_clip eval transform: resize 224 bicubic + center
         crop); the step only does H2D + normalize. No per-epoch augmentation.
Batches are collated inside workers (one IPC message per batch), pinned, and text is tokenized (and optionally
packed) in the worker so the training process never touches strings.
"""
import csv
import io
import json
import os
import random

import numpy as np
import torch

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------------------- dataset listing
def read_pairs(spec, image_col="image", caption_col="caption"):
    """Return (list[(abs_image_path, caption)])."""
    if os.path.isdir(spec):
        for nm in ("captions.csv", "captions.jsonl", "metadata.csv", "metadata.jsonl"):
            if os.path.exists(os.path.join(spec, nm)):
                return read_pairs(os.path.join(spec, nm), image_col, caption_col)
        out = []
        for root, _, files in os.walk(spec):
            for f in sorted(files):
                if f.lower().endswith(IMG_EXT):
                    t = os.path.join(root, os.path.splitext(f)[0] + ".txt")
                    if os.path.exists(t):
                        out.append((os.path.join(root, f), open(t).read().strip()))
        return out
    base = os.path.dirname(os.path.abspath(spec))
    fix = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    if spec.endswith(".jsonl"):
        out = []
        for line in open(spec):
            if line.strip():
                r = json.loads(line)
                out.append((fix(r[image_col]), r[caption_col]))
        return out
    with open(spec, newline="") as f:
        return [(fix(r[image_col]), r[caption_col]) for r in csv.DictReader(f)]


# ----------------------------------------------------------------------------- tokenizer
class Tok:
    """open_clip's HF tokenizer for BiomedCLIP (lowercased, [CLS] ... [SEP], pad 0, max 256)."""

    def __init__(self, tokenizer=None, context_length=256):
        if tokenizer is None:
            from cftrain.ref import MODEL_DIR
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.tok = tokenizer
        self.L = context_length

    def __call__(self, texts):
        if hasattr(self.tok, "tokenizer") and not hasattr(self.tok, "batch_encode_plus"):
            return self.tok(texts)                       # open_clip HFTokenizer object
        return self.tok(list(texts), return_tensors="pt", max_length=self.L, padding="max_length",
                        truncation=True).input_ids


# ----------------------------------------------------------------------------- datasets (batch-level)
class _BatchDS(torch.utils.data.Dataset):
    """Indexed by a list of sample indices (from the batch sampler); returns a collated batch."""

    def __init__(self, pairs, tok, img_path, pack=False, train=True, tok_multiple=256, fixed_max_len=True):
        self.pairs, self.tok, self.img_path, self.pack, self.train = pairs, tok, img_path, pack, train
        self.tok_multiple, self.fixed_max_len = tok_multiple, fixed_max_len
        self._tf = None

    def __len__(self):
        return len(self.pairs)

    def _text(self, caps):
        ids = self.tok(caps)
        out = {"ids": ids}
        if self.pack:
            from .model import pack_ids
            out["packed"] = pack_ids(ids, tok_multiple=self.tok_multiple, fixed_max_len=self.fixed_max_len)
        return out

    def __getitem__(self, idxs):
        caps = [self.pairs[i][1] for i in idxs]
        out = self._text(caps)
        out["idx"] = torch.as_tensor(list(idxs))
        if self.img_path == "cpu":
            if self._tf is None:
                self._tf = open_clip_transform(self.train)
            from PIL import Image
            out["images"] = torch.stack([self._tf(Image.open(self.pairs[i][0]).convert("RGB")) for i in idxs])
        else:  # gpu: raw bytes; decode happens in the main process on the GPU
            raw, pil = [], {}
            for j, i in enumerate(idxs):
                p = self.pairs[i][0]
                b = open(p, "rb").read()
                if p.lower().endswith((".jpg", ".jpeg")):
                    raw.append(torch.frombuffer(bytearray(b), dtype=torch.uint8))
                else:
                    from PIL import Image
                    im = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
                    raw.append(torch.from_numpy(im.copy()).permute(2, 0, 1).contiguous())  # decoded uint8 CHW
                    pil[j] = True
            out["raw"], out["decoded"] = raw, pil
        return out


class _CacheDS(_BatchDS):
    def __init__(self, cache_dir, tok, pack=False, tok_multiple=256, fixed_max_len=True):
        meta = json.load(open(os.path.join(cache_dir, "meta.json")))
        self.N = meta["n"]
        self.imgs = None
        self.path = os.path.join(cache_dir, "images.u8")
        caps = [json.loads(l)["caption"] for l in open(os.path.join(cache_dir, "captions.jsonl"))]
        super().__init__([(None, c) for c in caps], tok, "cache", pack, True, tok_multiple, fixed_max_len)
        ids_path = os.path.join(cache_dir, "ids.npy")
        self.ids = np.load(ids_path, mmap_mode="r") if os.path.exists(ids_path) else None

    def __getitem__(self, idxs):
        if self.imgs is None:
            self.imgs = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(self.N, 3, 224, 224))
        s = np.sort(np.asarray(idxs))
        if self.ids is not None:
            ids = torch.from_numpy(np.ascontiguousarray(self.ids[s]).astype(np.int64))
            out = {"ids": ids}
            if self.pack:
                from .model import pack_ids
                out["packed"] = pack_ids(ids, tok_multiple=self.tok_multiple, fixed_max_len=self.fixed_max_len)
        else:
            out = self._text([self.pairs[i][1] for i in s])
        out["u8"] = torch.from_numpy(np.ascontiguousarray(self.imgs[s]))
        out["idx"] = torch.from_numpy(s.astype(np.int64))
        return out


def open_clip_transform(train=True):
    """open_clip's BiomedCLIP transforms (image_transform with the model's mean/std)."""
    import open_clip
    return open_clip.image_transform(224, is_train=train, mean=MEAN, std=STD)


class _Sampler(torch.utils.data.Sampler):
    """Shuffled, sharded per rank, drop_last (fixed batch shape). sequential=True: every index in order,
    last partial batch kept (for feature caching / eval)."""

    def __init__(self, n, bs, seed, rank=0, world=1, sequential=False):
        self.n, self.bs, self.seed, self.rank, self.world, self.epoch = n, bs, seed, rank, world, 0
        self.sequential = sequential

    def __iter__(self):
        if self.sequential:
            for i in range(0, self.n, self.bs):
                yield list(range(i, min(self.n, i + self.bs)))
            return
        g = random.Random(self.seed + self.epoch)
        idx = list(range(self.n))
        g.shuffle(idx)
        per = self.n // self.world
        idx = idx[self.rank * per:(self.rank + 1) * per]
        for i in range(0, len(idx) - self.bs + 1, self.bs):
            yield idx[i:i + self.bs]

    def __len__(self):
        if self.sequential:
            return -(-self.n // self.bs)
        return (self.n // self.world) // self.bs


def _pin_worker(cpus):
    def init(_):
        if cpus:
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
    return init


def gpu_numa_cpus(device_index):
    """CPUs on the NUMA node of this GPU, intersected with the job's allowed CPUs (empty set if unknown)."""
    allowed = os.sched_getaffinity(0)
    try:
        bus = torch.cuda.get_device_properties(device_index).pci_bus_id if hasattr(
            torch.cuda.get_device_properties(device_index), "pci_bus_id") else None
        if bus is None:
            import subprocess
            bus = subprocess.check_output(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader",
                                           "-i", str(device_index)], text=True).strip()
        bus = bus.lower()[-12:] if len(bus) > 12 else bus.lower()
        node = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
        if node < 0:
            return set()
        cl = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
        cpus = set()
        for part in cl.split(","):
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
        return cpus & allowed
    except Exception:
        return set()


def make_loader(spec, batch_size, *, img_path="cache", pack=True, workers=8, seed=0, rank=0, world=1,
                tok=None, numa=True, device_index=0, train=True, tok_multiple=256, image_col="image",
                caption_col="caption", sequential=False, fixed_max_len=True):
    tok = tok or Tok()
    if img_path == "cache":
        ds = _CacheDS(spec, tok, pack, tok_multiple, fixed_max_len)
        n = ds.N
    else:
        pairs = read_pairs(spec, image_col, caption_col)
        ds = _BatchDS(pairs, tok, img_path, pack, train, tok_multiple, fixed_max_len)
        n = len(pairs)
    sampler = _Sampler(n, batch_size, seed, rank, world, sequential)
    cpus = gpu_numa_cpus(device_index) if numa else set()
    dl = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=None, num_workers=workers,
                                     pin_memory=img_path != "gpu", persistent_workers=workers > 0,
                                     prefetch_factor=4 if workers > 0 else None,
                                     worker_init_fn=_pin_worker(sorted(cpus)) if cpus else None)
    dl.n_samples, dl.numa_cpus = n, len(cpus)
    return dl, sampler


# ----------------------------------------------------------------------------- GPU-side image finishing
class GPUImages:
    """Turns a loader batch into normalized bf16/fp32 [B,3,224,224] on the GPU."""

    def __init__(self, device, dtype=torch.float32):
        self.dev, self.dtype = device, dtype
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255

    def __call__(self, batch):
        if "images" in batch:
            return batch["images"].to(self.dev, non_blocking=True).to(self.dtype)
        if "u8" in batch:
            x = batch["u8"].to(self.dev, non_blocking=True).float()
            return ((x - self.mean) / self.std).to(self.dtype)
        import torchvision.io as tio
        raw, dec = batch["raw"], batch["decoded"]
        # nvJPEG decode is asynchronous w.r.t. the host buffers and the training stream: without ordering, decoded
        # pixels (and later allocations) raced with the step (NaN / garbage token ids; fine under
        # CUDA_LAUNCH_BLOCKING=1). Decode + resize on a side stream, wait for that stream only, then hand off.
        if not hasattr(self, "_side"):
            self._side = torch.cuda.Stream(self.dev)
        cur = torch.cuda.current_stream(self.dev)
        with torch.cuda.stream(self._side):      # overlaps with the previous step still running on `cur`
            jpeg_idx = [i for i in range(len(raw)) if i not in dec]
            out = [None] * len(raw)
            if jpeg_idx:
                try:
                    ims = tio.decode_jpeg([raw[i] for i in jpeg_idx], mode=tio.ImageReadMode.RGB, device=self.dev)
                    for i, im in zip(jpeg_idx, ims):
                        out[i] = im
                except RuntimeError:          # one bad JPEG fails the whole nvJPEG batch: retry singly, then PIL
                    for i in jpeg_idx:
                        try:
                            out[i] = tio.decode_jpeg(raw[i], mode=tio.ImageReadMode.RGB, device=self.dev)
                        except RuntimeError:
                            from PIL import Image
                            im = np.asarray(Image.open(io.BytesIO(raw[i].numpy().tobytes())).convert("RGB"))
                            out[i] = torch.from_numpy(im.copy()).permute(2, 0, 1).to(self.dev)
                            self.n_fallback = getattr(self, "n_fallback", 0) + 1
            for i in dec:
                out[i] = raw[i].to(self.dev, non_blocking=False)
            x = torch.stack([resize_center_crop(im) for im in out]).float()
            x = ((x - self.mean) / self.std).to(self.dtype)
        self._side.synchronize()              # host waits for decode/resize only (not for the training step)
        cur.wait_stream(self._side)
        x.record_stream(cur)
        del raw, out
        return x


def resize_center_crop(im, size=224):
    """uint8 CHW on GPU -> float CHW 224x224: shortest side to 224 (bicubic, antialias), center crop."""
    import torch.nn.functional as F
    c, h, w = im.shape
    s = size / min(h, w)
    nh, nw = max(size, round(h * s)), max(size, round(w * s))
    x = F.interpolate(im[None].float(), size=(nh, nw), mode="bicubic", antialias=True, align_corners=False)[0]
    t, l = (nh - size) // 2, (nw - size) // 2
    return x[:, t:t + size, l:l + size].clamp_(0, 255).round_()

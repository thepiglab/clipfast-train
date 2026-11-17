"""Reference BiomedCLIP loader (open_clip, offline) and synthetic caption-like token ids.
Vendored from thepiglab/clipfast (clipfast/load.py, bench.py) so this repo is standalone.
"""
import json
import os
import tempfile

import torch

# A local snapshot of the Hugging Face repo microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# (open_clip_config.json + open_clip_pytorch_model.bin + tokenizer files).
MODEL_DIR = os.environ.get("BIOMEDCLIP_DIR", "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
NAME = "biomedclip_local"


def load_reference(device="cuda"):
    """Return (model fp32 eval, preprocess, tokenizer) exactly as open_clip builds them."""
    import open_clip
    cfg = json.load(open(os.path.join(MODEL_DIR, "open_clip_config.json")))
    tc = cfg["model_cfg"]["text_cfg"]          # open_clip >= 3 renamed the HF text-tower keys
    if "proj" in tc:
        tc["hf_proj_type"] = tc.pop("proj")
    if "pooler_type" in tc:
        tc["hf_pooler_type"] = tc.pop("pooler_type")
    tc["hf_model_pretrained"] = False          # weights come from the open_clip checkpoint below
    d = tempfile.mkdtemp(prefix="oc_cfg_")
    json.dump(cfg["model_cfg"], open(os.path.join(d, NAME + ".json"), "w"))
    open_clip.add_model_config(d)
    model, _, preprocess = open_clip.create_model_and_transforms(
        NAME, pretrained=os.path.join(MODEL_DIR, "open_clip_pytorch_model.bin"),
        image_mean=cfg["preprocess_cfg"]["mean"], image_std=cfg["preprocess_cfg"]["std"])
    tok = open_clip.get_tokenizer(NAME)
    return model.to(device).eval(), preprocess, tok


def probe_texts():
    labels = ["adenocarcinoma histopathology", "brain MRI", "covid line chart", "squamous cell carcinoma histopathology",
              "immunohistochemistry histopathology", "bone X-ray", "chest X-ray", "pie chart",
              "hematoxylin and eosin histopathology"]
    long = ("The chest radiograph demonstrates bilateral patchy airspace opacities predominantly in the lower lobes, "
            "with small bilateral pleural effusions and mild cardiomegaly; no pneumothorax is identified. ") * 6
    return ["this is a photo of " + l for l in labels] + ["a", long, "CT", "fundus photograph of a diabetic retina"]


def probe_images(preprocess, n_random=23, seed=0):
    """Example images shipped with the model + deterministic smooth random images, preprocessed (fp32 NCHW)."""
    from PIL import Image
    import numpy as np
    ex = os.path.join(MODEL_DIR, "example_data", "biomed_image_classification_example_data")
    ims = [Image.open(os.path.join(ex, f)).convert("RGB") for f in sorted(os.listdir(ex))]
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        h, w = int(rng.integers(64, 1024)), int(rng.integers(64, 1024))
        base = rng.random((max(2, h // 16), max(2, w // 16), 3))
        im = Image.fromarray((base * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
        ims.append(im)
    return torch.stack([preprocess(im) for im in ims]), ims


def load_preprocess_only():
    """open_clip's eval transform for BiomedCLIP (Resize 224 bicubic, CenterCrop, ToTensor, Normalize) without the
    model."""
    from open_clip.transform import image_transform
    cfg = json.load(open(os.path.join(MODEL_DIR, "open_clip_config.json")))
    return image_transform(224, is_train=False, mean=cfg["preprocess_cfg"]["mean"], std=cfg["preprocess_cfg"]["std"])


def load_tokenizer_only():
    """The HF fast tokenizer BiomedCLIP uses (model dir's tokenizer.json)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)


def pmc_lengths(B, gen):
    """Caption-like token lengths: lognormal, median ~40, clipped to [5, 256]."""
    z = torch.randn(B, generator=gen)
    return (40 * torch.exp(0.8 * z)).round().clamp(5, 256).long()


def text_ids(B, L, vocab, mixed=False, gen=None):
    """[CLS] tok... [SEP] padded with 0 to 256, like open_clip's HF tokenizer output. L="pmc": caption-like lengths."""
    ids = torch.zeros(B, 256, dtype=torch.long)
    pl = pmc_lengths(B, gen) if L == "pmc" else None
    for i in range(B):
        n = int(pl[i]) if pl is not None else (L if not mixed else int(torch.randint(max(3, L // 2), L + 1, (1,), generator=gen)))
        ids[i, 0] = 2
        ids[i, 1:n - 1] = torch.randint(1000, vocab, (n - 2,), generator=gen)
        ids[i, n - 1] = 3
    return ids

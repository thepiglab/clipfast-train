# clipfast-train

Fast, parity-tested fine-tuning for **BiomedCLIP** (open_clip ViT-B/16 + PubMedBERT-256) on a single NVIDIA RTX PRO
6000 Blackwell (sm_120) or a multi-GPU node: full fine-tuning, LoRA, partial freezing and locked-tower (LiT)
training. Same math as open_clip (gradients and loss curves within bf16 noise), about **2.5x faster at ~0.55x the
memory** for full fine-tuning, up to ~11x with a locked image tower.

The companion inference runtime (bf16/CUDA-graph forward, Rust runtime plan) is
[thepiglab/clipfast](https://github.com/thepiglab/clipfast). Design, measurements and dead ends:
[docs/TRAINING_STACK_DESIGN.md](docs/TRAINING_STACK_DESIGN.md).

## Requirements

- PyTorch >= 2.13 with CUDA (tested: torch 2.13 + CUDA 13.0), Triton, open_clip_torch 4.x, transformers, timm,
  torchvision (GPU JPEG path), numpy, Pillow. `kar` (kernel-autoresearch) only for `kar_tasks/`.
- BiomedCLIP weights: a local snapshot of the Hugging Face repo
  `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; point `BIOMEDCLIP_DIR` at it.
  No weights are included in this repository.

## Quick start

```bash
export BIOMEDCLIP_DIR=/path/to/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
# 1) one-time cache (CPU, parallel): exact open_clip eval pixels as uint8 + token ids (150 KB/sample).
#    Data: a folder (image + same-stem .txt, or captions.csv/.jsonl inside), a CSV (image,caption) or a JSONL.
python3 -m cftrain.prep --data captions.csv --out /path/to/cache --workers 64
# 2) train on one GPU
python3 train.py --data /path/to/cache --val-data /path/to/val_cache --out runs/full --batch-size 256 --epochs 5 --lr 1e-5
# 3) one node, N GPUs (global batch = N x per-GPU batch)
torchrun --standalone --nproc_per_node 8 train.py --data /path/to/cache --out runs/full8 --batch-size 256
```

Copy the cache to node-local disk for multi-epoch runs if your shared filesystem is slow (a cold read measured
~2.4 k samples/s; page-cached 27-57 k samples/s).

## Modes (measured: one RTX PRO 6000, B=256, caption-like text; open_clip amp_bf16 = 666 samples/s, 46.5 GB)

| goal | flags | samples/s | peak GB |
|---|---|---|---|
| full fine-tuning (default) | `--mode full` | 1686 | 26.5 |
| ... long run (+5 %, ~5 min of autotuning at start) | `--compile-mode max-autotune-no-cudagraphs` | 1763 | 26.5 |
| ... B=1024 on one GPU (open_clip runs out of memory) | `--batch-size 1024` / `+ --act-budget 0.5` | 1766 / 1645 | 86.5 / 51.6 |
| freeze the lower half of both towers | `--freeze-image-blocks 6 --freeze-text-blocks 6` | 2708 | 14.5 |
| freeze the lower 3/4 | `--freeze-image-blocks 9 --freeze-text-blocks 9` | 3691 | 7.6 |
| LoRA (memory-light; not faster than full FT at this width) | `--mode lora --lora-r 16 --lr 2e-4` | ~1420-1520 | 20-22 |
| LoRA on qkv only | `--mode lora --lora-targets qkv --lr 2e-4` | 2066 | 17.6 |
| LiT: image tower locked, its features cached once | `--mode lit --lr 5e-5` | 7342 | 7.6 |
| text tower locked | `--mode lock-text` | | |

Multi-GPU (DDP with all-gathered features and local loss, fixed batches, B=256 per GPU): 1 GPU 1630, 2 GPUs 3022
(1.85x), 3 GPUs 4497 (2.76x), 4 GPUs 5956 samples/s (3.65x; 3.76x with `--ddp-bf16-grads 1`).

Other flags: `--text-dropout/--attn-dropout` (open_clip's BERT default 0.1, kept), `--recompute 1` (per-block
checkpointing), `--precision fp32` (reference numerics), `--img-path cache|gpu|cpu` (auto = cache when `images.u8`
exists; `gpu` = nvJPEG decode + GPU bicubic resize from raw files; `cpu` = PIL + open_clip's train transform).

**Experimental, off by default** (each passes or is reported against the parity tests; see the design doc):
- `--fp8 1`: FP8 forward GEMMs. +5 % speed but **fails** gradient parity (cosine 0.992): not recommended.
- `--cuda-graph 1`: whole-step CUDA graphs; no gain at B >= 128.
- `cftrain.model.MLP_EPI = True`: Triton MLP with fused bias+GELU / dGELU GEMM epilogues (+1-2 %, parity passes);
  with `cftrain.mlp_epi.SAVE_ACT = False` it also recomputes GELU in backward (-18 % peak memory at B=512, -4 %).
- `cftrain.attn_vit.FWD_VERSION = "triton"`: Triton attention forward (slower than FlashAttention-2's forward).

Outputs: `log.jsonl` (loss, lr, samples/s, peak memory, validation retrieval R@1) and `checkpoint_<step>.pt`, an
**open_clip state_dict with LoRA merged**, loadable by stock open_clip (`check_export.py` loads it with strict=True).

## Parity (vs open_clip fp32, dropout 0, TF32 off)

- `parity.py`, gradient cosine: open_clip amp_bf16 (the noise floor) 0.99989; cftrain default 0.99986; LoRA
  (exact reference via dW_eff) 0.99981-0.99992; LiT 0.99988; fp32 paths match to ~1e-6.
- 40-step loss curves: max |loss - fp32 loss| cftrain 0.09-0.13 vs open_clip amp 0.17-0.25.
- 3000-step loss curves (`long_curve.py`): same final loss; largest windowed gap 0.027-0.076, inside open_clip
  amp's own spread under other valid bf16 roundings (0.019-0.073).
- NaN-poisoned-allocator audit (catches kernels that leave gradient rows unwritten): 0 non-finite tensors.
- DDP: 2/3/4-GPU NCCL runs equal the single-process global batch (gradient rel. L2 1.7e-6 to 5.1e-6).

## Layout

- `cftrain/model.py`: lean model (packed varlen text, CLS-only last layers, fused LoRA autograd, freezing, export).
- `cftrain/attn_vit.py`: ViT attention custom op: FlashAttention-2 forward + bespoke Triton backward
  (forward+backward per layer 1.45 vs 3.54 ms for PyTorch SDPA at B=256), reads q/k/v from the qkv GEMM output and writes dqkv in place.
- `cftrain/attn.py` varlen attention; `cftrain/loss.py` CLIP loss (+ gather_with_grad / local_loss);
  `cftrain/engine.py` optimizer, step, CUDA graphs; `cftrain/data.py` loaders + GPU image path;
  `cftrain/prep.py` cache builder; `cftrain/ref.py` reference loader (vendored from thepiglab/clipfast);
  `cftrain/fp8.py`, `cftrain/gemm_epi.py`, `cftrain/mlp_epi.py` experimental kernels.
- Tests and benchmarks: `parity.py`, `bench_train.py`, `profile_oc_step.py`, `long_curve.py`, `compare_curves.py`,
  `test_attn_vit.py`, `test_epi.py`, `test_ddp.py`, `ddp_bench.py`, `bench_loader.py`, `check_export.py`,
  `debug_dummy.py`, `make_synth.py` (synthetic image-caption set for smoke tests).
- `kar_tasks/`: frozen kernel tasks (reference, shapes, baseline, op model, candidates) for the kar method.

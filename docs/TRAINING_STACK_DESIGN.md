# BiomedCLIP training stack — design (Phase 0)

Status 2026-11-17. Target: one RTX PRO 6000 Blackwell Server Edition (sm_120, 96 GB,
188 SMs, 128 MB L2), scaling to 8 GPUs per node. Model: open_clip BiomedCLIP = timm ViT-B/16 (197 tokens,
d 768, 12 layers, MLP 3072, pre-LN, exact-erf GELU) + PubMedBERT-base (post-LN BERT, 12 layers, context 256,
HF dropout 0.1) + MLP text projection (768 -> 640 -> 512, GELU), CLIP InfoNCE loss with learnable logit scale.
195.9 M parameters.

Priorities: (1) parity with open_clip (grads and loss curves within bf16 noise), (2) speed, (3) memory.
Code: this repository (`cftrain/`, `train.py`, tests and benchmarks); precedent notes with sources:
`docs/precedents_research.md`. The inference runtime (kernels, Rust runtime plan) is
[thepiglab/clipfast](https://github.com/thepiglab/clipfast); this stack vendors its reference loader (`cftrain/ref.py`).

---------------------------------------------------------------------------------------------------------------------
## 1. Measured baseline: where one open_clip training step goes

`profile_oc_step.py`, B = 256, caption-like text lengths (lognormal, median 40, mean 53 tokens,
clipped to [5, 256]), text padded to 256 as open_clip's tokenizer does, fused AdamW, TF32 off.
Phases from profiling fwd, fwd+bwd and the full step separately. GPU ms per step:

| | fp32 | amp_bf16 (baseline) | amp_bf16 + torch.compile |
|---|---|---|---|
| step | 1222 ms (210 samples/s) | **404 ms (633 samples/s)** | 338 ms (757 samples/s) |
| fwd / bwd / opt | 469 / 751 / 1 | 125 / 276 / 2.7 | 103 / 234 / 1.5 |
| kernels per step | 1540 | 2129 | 1696 |
| saved activations | 71.3 GB | 43.5 GB | 40.7 GB |
| peak (fwd+bwd) | 75.6 GB | 47.8 GB | 44.4 GB |

amp_bf16 breakdown by op class (fwd + bwd GPU ms, % of 404 ms):

| class | ms | % | what it is |
|---|---|---|---|
| GEMM (cuBLAS, cutlass_80 bf16 kernels) | 158 | 39 | ~59.6 TFLOP -> **~377 TFLOP/s = 86 % of the measured 437 TFLOP/s bf16 peak** |
| attention | 57 | 14 | text: mem-efficient kernel (padding mask + dropout) 42 ms; image: flash 15.5 ms |
| elementwise (adds, residual, bias, scale) | 47 | 12 | |
| copy/cast (autocast fp32<->bf16) | 46 | 11 | LN outputs and residual stream are fp32 under autocast |
| LayerNorm fwd + bwd | 44 | 11 | |
| GELU fwd + bwd | 28 | 7 | |
| reduce (bias grads etc.) | 13 | 3 | |
| dropout | 4 | 1 | |
| AdamW (fused) | 3.7 | 1 | |

Ceilings this implies:
- GEMMs already run at 86 % of peak. Nothing will make the *same* GEMMs much faster; the levers are
  (a) doing fewer GEMM FLOPs (text padding, CLS-only last layer, frozen dW, frozen towers) and
  (b) deleting the 61 % of the step that is not GEMM (fusion), and (c) FP8 for the GEMMs themselves.
- **Text padding is the single biggest waste.** Text tokens per batch: 256 x 256 = 65.5 k padded vs 13.6 k real
  (mean 53). The text tower does 4.8x more work than needed. Trimming to the *batch* max does nothing here
  (the long tail puts a 256-token caption in almost every batch of 256; the "trim" rows equal "pad256"), so
  **varlen packing is required**, not bucketing.
- Optimizer: < 1 %. Kernel launch: 2129 launches x ~4 us = 8.5 ms CPU, fully hidden behind 404 ms of GPU work.

FLOP model per sample (fwd; training ~3x fwd for GEMMs, dense case):
- per token per layer: 12 d^2 = 7.08 M MAC = 14.2 MFLOP; 12 layers -> 170 MFLOP/token.
- image: 197 tokens -> 33.5 GFLOP + attention 1.4 GFLOP + patch 0.2 -> **35.2 GFLOP**; CLS-only last layer saves
  ~10/12 of a layer for 196 tokens -> -2.3 GFLOP (6.5 %).
- text padded 256: 43.5 + 2.4 (attn) = **46 GFLOP**; packed at mean 53 tokens: **9.0 GFLOP** (-80 %).
- step at B=256: dense padded 81 GFLOP x 3 x 256 = 62 TFLOP; packed + CLS-only = 42 GFLOP x 3 x 256 = 32 TFLOP.
  At the measured GEMM rate (377 TFLOP/s) the packed GEMM floor is **~85 ms/step = ~3000 samples/s**, vs 404 ms
  today. That is the bf16 speed-of-light target for full fine-tuning; FP8 GEMMs halve the GEMM part again.

Activation memory model (amp, saved for backward, per token per layer, current PyTorch graph): fp32 residual (3 KB)
x2, LN out fp32 + bf16 copy (4.5 KB) x2, qkv 4.5 KB, attn out 1.5 KB, fc1 out 6 KB, GELU out 6 KB ~= 34 KB.
Image 197 x 12 x 34 KB = 80 MB/sample; text padded 256 -> 104 MB/sample; packed 53 -> 22 MB/sample.
B=256: image 20 GB + text-padded 27 GB = 47 GB (matches the measured 43.5 GB). Packed: 20 + 5.5 = 26 GB.

---------------------------------------------------------------------------------------------------------------------
## 2. Techniques

Format per technique: mechanism -> expected gain (arithmetic, B = 128..1024) -> parity test -> effort -> deps.
"Phase" says when it ships. Speeds are per GPU; all gains are against open_clip amp_bf16 unless stated.

### 2.1 Text: varlen packing (+ CLS-only last layer)  — Phase 1
- Mechanism: tokenize to [CLS] ... [SEP] without padding; concatenate the batch into T = sum(len) tokens, carry
  `cu_seqlens`; all token-wise ops (embedding, GEMMs, LN, GELU) run on T rows; attention = FlashAttention-2 varlen
  (in PyTorch: `aten._flash_attention_forward` with cum_seq, which has a registered backward incl. dropout; also
  `torch.nn.attention.varlen.varlen_attn` exists in 2.13). The loss reads only the CLS rows, so the last BERT layer
  computes q/attention/MLP for the B CLS rows only (K/V still from all tokens); same for the last ViT block.
  Packed token count is bucketed (multiple of 256) so compile/graphs see few shapes.
- Gain: text GEMM FLOPs x(53/256) = -79 %; text attention -95 % (quadratic); text activations 27 GB -> 5.5 GB at
  B=256 (B=1024: 107 -> 22 GB). Step at B=256: text was ~55 % of padded work -> expect ~1.8-1.9x end-to-end with
  everything else unchanged (eager). Measured by the clipfast inference repo's train bench (B=128, caption-like lengths):
  open_clip amp 636/s -> lean packed eager 1205/s (1.9x), 27.4 -> 16.5 GB.
- Parity: exact math (mask semantics identical; position ids restart per sequence; token type 0). Test:
  `parity.py` lean_fp32_packed vs open_clip fp32 grads: global cosine > 0.99999; bf16 packed must be within the
  open_clip amp-vs-fp32 noise floor.
- Effort: done (model.py, attn.py, pack_ids). Deps: flash varlen in torch (present).

### 2.2 Fused LayerNorm / bias / GELU / residual, fwd and bwd, with recompute — Phase 1 (compile) then Phase 2 (bespoke)
- Mechanism (Phase 1): torch.compile/inductor fuses LN+residual+cast and bias+GELU into Triton kernels and removes
  most autocast copies (copy/cast 46 -> ~5 ms in the profile). Phase 2: our own kernels as autograd Functions:
  (a) GEMM epilogue = bias + exact-erf GELU writing both pre-act (for bwd) or nothing (recompute); cuBLASLt's GELU
  epilogue is tanh-approx (also llm.c's, Liger's, TE's) and would break parity, so the epilogue is ours (kar
  `fused_linear` from thepiglab/clipfast, plus a dGELU epilogue for backward);
  (b) residual-add + LayerNorm in one pass emitting bf16 LN output + fp32 residual + (mean, rstd);
  (c) LN backward in one kernel (dx, dgamma, dbeta with a two-stage column reduction, as llm.c layernorm_backward_kernel10);
  (d) recompute: save only fc1 pre-activation (bf16) and LN statistics; recompute GELU output and LN output in
  backward (llm.c `-r 2`), which removes the GELU-out (6 KB) and both LN outputs (9 KB) per token-layer.
- Gain: non-GEMM, non-attention time in the profile = 47 + 46 + 44 + 28 + 13 = 178 ms of 404 (44 %). Inductor
  gets ~66 ms of that back (404 -> 338). Bespoke fusion floor: these ops move ~O(10) bytes per element; at B=256
  packed (50 k image + 14 k text tokens x 24 layers, ~70 KB moved per token-layer fwd+bwd) ~= 110 GB -> ~75 ms at
  1.5 TB/s, much of which disappears into GEMM epilogues/prologues. Expect another 1.15-1.3x over compile.
  Memory: 34 -> ~19 KB per token-layer (-45 % activations): B=256 packed 26 -> ~14 GB; B=1024 104 -> ~57 GB.
- Parity: per-kernel gradient test vs fp64 reference (kar task: mean err <= 1.5x the bf16 reference's own error);
  end-to-end grad cosine within the noise floor; loss curve.
- Effort: compile = done. Bespoke: ~1 day for LN fwd/bwd + GELU epilogue fwd/bwd + autograd wiring (Phase 2).
- Deps: kar tooling; clipfast's `fused_linear` Triton kernel (fwd); torch.library custom ops.

### 2.3 FlashAttention fwd + bwd — Phase 1 (library), Phase 2 optional
- Mechanism: image tower (N = 197, head 64, no mask): SDPA already picks FA2 on sm_120 (dispatch order flash ->
  efficient -> math; flash allowed sm80..sm121). Text: FA2 varlen via 2.1 replaces the mem-efficient kernel with
  padding mask + dropout (42 ms). FA4-cute has sm_120 fwd/bwd paths (mma.sync, 99 KB smem) but is not in the
  container; PyTorch 2.13 ships `torch.nn.attention._fa4` hooks (to check).
- Gain: text attention 42 ms -> ~2 ms (packing). Image attention 15.5 ms at B=256 vs its FLOP floor
  (1.4 GFLOP fwd x 3.5 (fwd+bwd incl. recompute of P) x 256 = 1.3 TFLOP -> 3 ms at 437 TFLOP/s): a bespoke
  single-tile kernel (whole K/V of 197 x 64 fits in smem) could save ~8-10 ms/step (~5-8 % of an optimized step).
- Parity: SDPA math reference in fp32; grad cosine of q/k/v grads.
- Effort: library path done; bespoke short-seq FA bwd ~1-2 days (Phase 2, after 2.2). Deps: none.

### 2.4 Fused contrastive loss — not worth it at our sizes (Phase 2 only if multi-node)
- Mechanism: Inf-CL/Liger-style tiled LSE so the B x B (or B x world*B) logits never materialise; fwd+bwd in one
  pass.
- Gain arithmetic: logits are B_local x B_global fp32. 1 GPU B=1024: 4 MB; 8 GPUs x 1024: 1024 x 8192 x 4 B =
  32 MB, a few copies -> ~0.1 GB and < 1 ms. The loss is 0.003 % of FLOPs. **Expected gain ~0.** It matters only at
  >= 64 k global batch (Inf-CL's regime). We use open_clip's gather_with_grad + local_loss semantics instead.
- Parity: identical formula (fp32 logits); DDP loss equals single-GPU global-batch loss (test with 2 GPUs).
- Effort: 0 (skip). Deps: none.

### 2.5 Fused multi-tensor AdamW (bf16 params, fp32 master) — Phase 1 (torch fused), Phase 2 (master-weight variant)
- Mechanism: Phase 1: `torch.optim.AdamW(fused=True)` on fp32 params (open_clip semantics; one multi-tensor launch
  group; capturable, lr as a device tensor). llm.c-style variant: params stored bf16 (no per-step autocast weight
  cast), fp32 master + m + v in the optimizer, stochastic rounding when writing bf16.
- Gain: the optimizer is 3.7 ms (1 %) of the step; fused is already used in the baseline. The master-weight variant
  removes the per-forward weight cast (196 M x (4 R + 2 W) B = 1.2 GB -> ~1 ms) and frees nothing (bf16 0.39 +
  fp32 master 0.78 + m/v 1.57 + bf16 grads 0.39 = 3.1 GB, same as fp32 params + fp32 grads + m/v). **~1 % speed, 0
  memory: low priority.** What does save memory: no Adam state / no fp32 copy for frozen weights (2.7, 2.8).
- Parity: bit-compare one step vs torch AdamW (fp32); loss curve.
- Effort: done (torch fused). Master-weight variant ~0.5 day. Deps: none.

### 2.6 Whole-step CUDA graphs — Phase 1 (opt-in)
- Mechanism: capture fwd + loss + bwd + AdamW + logit-scale clamp into one graph per (B, packed-token bucket);
  static input buffers; capturable fused AdamW with lr as a device tensor; dropout RNG via graph-safe philox
  offsets; graphs share one memory pool.
- Gain: removes CPU launch cost (2129 launches ~ 8.5 ms eager; ~1700 compiled) and Python/dynamo overhead. At B=256
  full FT the GPU step (~100-150 ms optimized) hides it: ~0-3 %. It matters when the GPU step is short: LoRA/LiT at
  B <= 128 (step ~10-30 ms -> 10-30 % gain). Also frees CPU cores for data loading.
- Parity: identical kernels -> bitwise-identical loss vs the non-graph step on the same batch (test).
- Effort: done (engine.GraphStepper). Deps: fixed shapes (bucketed packing), no host syncs.

### 2.7 LoRA, with no dW for frozen weights — Phase 1 (PyTorch), Phase 2 (fused)
- Mechanism: y = x W^T + b + s (x A^T) B^T on qkv/o/fc1/fc2 of both towers; base W frozen and stored bf16 (no fp32
  copy, no grad, no Adam state). Autograd computes only dX for frozen W (its dW GEMM disappears) plus the rank-r
  A/B grads. Phase 2: Unsloth-style hand-written autograd that merges the LoRA up-projection into the base GEMM
  epilogue and computes dA/dB with addmm into preallocated buffers.
- Gain (FLOPs): dense step = 3 GEMMs per linear (fwd, dX, dW); LoRA = 2 + O(r/d). GEMM time -33 %; plus no weight
  grads/Adam for 196 M params. Memory: weights 0.39 GB bf16 + LoRA r=16 (4.7 M params x 16 B = 75 MB) vs 3.1 GB
  for full FT. Activations: inputs of LoRA'd linears are still saved (needed for dA), so activation memory is
  ~unchanged unless LoRA is restricted to qkv (then fc1/fc2 inputs need not be saved for dW: -12 KB/token-layer).
  Expected end-to-end vs full FT on the same stack: ~1.3-1.4x. **Measured: ~1.0x** (see section 4: at d=768 the
  memory-bound rank-r passes cost as much as the dW GEMMs saved; only a fused-in-GEMM LoRA could change that).
- Parity: exact reference: dA = s B^T dW_eff, dB = s dW_eff A^T from open_clip fp32 grads at the merged weights
  (`parity.py` lora rows); export merges LoRA into open_clip weights (checkpoint loads in stock open_clip).
- Effort: Phase 1 done; Phase 2 ~0.5-1 day. Deps: 2.2 kernels for the fused version.

### 2.8 Frozen-tower caching (LiT) and partial freezing — Phase 1
- Mechanism: `--mode lit` locks the image tower; its 512-d embeddings are computed once per dataset (no grad, bf16,
  1 KB/sample) and the training step runs only the text tower (packed) + loss. `--mode lock-text` is the mirror.
  `--freeze-image-blocks N` / `--freeze-text-blocks N` freeze the first N blocks (no dW, no dX below the first
  trainable block, frozen bf16 storage).
- Gain: per-sample FLOPs packed: image 35.2 GFLOP fwd (80 % of the packed model) disappears from every step after a
  one-off forward pass (~1/3 of one training epoch's image cost). Text-only step ~9 GFLOP x 3 -> **~5x samples/s
  vs packed full FT**, activations only ~22 MB/sample (B=1024: 22 GB), optimizer state 110 M params.
  Partial freezing of the first k of 12 blocks: removes k/12 of dW GEMMs and of activations below block k.
- Parity: LiT step's text grads == full-model text grads given cached bf16 image features (`parity.py`).
- Effort: done. Deps: deterministic preprocessing (cache path) so cached features are the ones the model sees.

### 2.9 FP8 GEMMs (opt-in) — Phase 2
- Mechanism: e4m3 forward GEMMs with per-tensor current scaling (TE's default recipe on sm_120; MXFP8 is not
  supported on sm 12.x by TE; TE is not in the container), bf16/e5m2 for dgrad/wgrad or keep bwd in bf16. In
  PyTorch: `torch._scaled_mm` with rowwise scales (the clipfast inference runtime already tested e4m3 rowwise fwd).
- Gain: measured FP8 tensor peak 825 vs bf16 437 TFLOP/s (1.89x). If GEMMs are ~65 % of an optimized step and FP8
  covers fwd only (1/3 of GEMM FLOPs): -16 % step; fwd + dgrad: -25 %. Memory: FP8 copies of activations for wgrad
  could halve saved GEMM inputs.
- Parity: not bitwise; needs a loss-curve test over a few hundred steps (allow drift ~ bf16-vs-fp32 drift x 2) and
  a zero-shot/retrieval check. Opt-in only.
- Effort: 1-2 days. Deps: 2.2 (fused quantize-in-LN), scaling recipe.

### 2.10 GPU data path, NUMA-pinned loaders — Phase 1
- Requirement: optimized full FT targets ~2000-3000 samples/s/GPU, LiT ~5-10 k/s/GPU, i.e. up to 24 k (full) or
  80 k (LiT) samples/s per 8-GPU node. PIL decode + bicubic resize of a ~512 px JPEG costs ~4-6 ms/core ->
  16 Zen 5 cores/GPU = 2.7-4 k img/s: borderline for full FT, insufficient for LiT/LoRA and for large radiographs
  (2-3 k px: ~30 ms/image -> ~500/s/GPU).
- Mechanism (three paths, `--img-path`):
  1. `cache` (default, fastest): one-off CPU job (`python -m cftrain.prep`, burst QOS, many cores) writes the exact
     open_clip eval pixels (Resize 224 bicubic + CenterCrop, PIL) as uint8 [N,3,224,224] memmap (150 KB/sample)
     + uint16 token ids. Training does H2D of uint8 (4x less PCIe than fp32) + normalize on GPU. 3000 samples/s =
     450 MB/s per GPU (page cache / Lustre). Trade-off: no per-epoch random crops (open_clip's default train
     aug is RandomResizedCrop scale (0.9, 1.0), mild); fine for few-epoch fine-tuning; exact parity with inference
     pixels.
  2. `gpu`: workers read raw bytes; nvJPEG batched decode on the GPU (`torchvision.io.decode_jpeg(list,
     device='cuda')`, torchvision 0.28 in the container); GPU antialiased bicubic (ported from Pillow's
     Resample.c, a = -0.5) + center crop. Not bit-identical to PIL (rounding); RTX PRO 6000 lists NVDEC but no
     NVJPG engine, so decode runs on SMs (~ a few % of the step).
  3. `cpu`: PIL + open_clip's own train transform (exact open_clip behaviour, slowest).
  Loaders collate whole batches in workers (one IPC message per batch), pinned memory, prefetch 4; workers are
  pinned to the GPU's NUMA node (PCI bus -> /sys numa_node -> cpulist, intersected with the job's CPU allocation),
  ~14 workers per GPU at 16 cores/GPU.
- Parity: cache pixels == open_clip eval transform before ToTensor (test: max abs diff 0 on probe images);
  gpu path: mean abs diff vs PIL in 0-255 units and embedding cosine.
- Effort: done (data.py, prep.py); benchmark of loader throughput pending.

### 2.11 8-GPU data parallel with all-gathered features — Phase 1
- Mechanism: torchrun; DDP (gradient_as_bucket_view, static_graph) overlapping gradient all-reduce with backward;
  loss = open_clip gather_with_grad + local_loss (each rank: B_local x B_global logits; autograd all_gather ->
  reduce-scatter of feature grads). Sampler shards a shared permutation, drop_last.
- Gain/cost arithmetic: fp32 grad all-reduce 784 MB per step; ring all-reduce moves 2 x 7/8 x 784 MB = 1.37 GB per
  GPU; on PCIe Gen5 (~40-50 GB/s effective, no NVLink on RTX PRO 6000) ~30 ms per step. Optimized full-FT step at
  B=256/GPU ~100 ms with ~65 ms of backward to overlap -> mostly hidden; bf16 gradient compression
  (`bf16_compress_hook`) halves it if exposed. LoRA: grads are 4.7 M params -> negligible. LiT: 110 M params
  -> ~20 ms, overlapped. Expected scaling 7-7.6x at B=256/GPU for full FT.
- Parity: 2-GPU run vs 1-GPU at the same global batch: identical loss to bf16 noise.
- Effort: done in train.py (needs a 2-GPU test). Deps: NCCL in the container (yes, torch build).

### 2.12 Activation recompute (memory)
- Per-block checkpointing (`--recompute 1`): keeps only block inputs (fp32 residual 3 KB/token-layer instead of
  ~34 KB) at +1 forward (~+33 % GEMM time). Use when B must exceed what fits: packed full FT fits B~1024 in 96 GB
  without it (26 GB at 256 -> ~100 GB at 1024: borderline) -> with recompute B=1024 needs ~15 GB of activations.
  Selective recompute (2.2(d)) gets ~-45 % for ~+5 % time and is the better default once bespoke kernels land.

---------------------------------------------------------------------------------------------------------------------
## 3. Rust/CUDA vs PyTorch + custom kernels — recommendation: **PyTorch + custom kernels**

- Where the time is: GEMMs already at 86 % of peak through cuBLAS; the wins are FLOP removal (packing, CLS-only,
  frozen dW, cached towers) and fusion of non-GEMM ops. Both are expressible as PyTorch modules + custom autograd
  ops (Triton or CUDA via `torch.library.custom_op` with explicit backward). A Rust rewrite would buy no extra
  FLOP removal.
- Training needs autograd, an optimizer, DDP/NCCL, a data loader, checkpoints loadable by open_clip, LoRA export,
  and quick iteration within a few days. In Rust all of that is new code with new failure modes; parity
  risk is high and the payoff (startup, memory of the runtime itself, no Python) matters for inference, not for a
  multi-hour fine-tune.
- clipfast's Rust runtime (clipfast-rs) remains the deployment target: our checkpoints export to
  open_clip format, which its converter consumes.
- So: Phase 2 = bespoke kernels (kar method) wrapped as torch custom ops, swapped in module by module with gradient
  parity tests. Revisit Rust only if Python overhead shows up in the profile after CUDA graphs (it should not).

---------------------------------------------------------------------------------------------------------------------
## 4. Results (measured; RTX PRO 6000, caption-like text, dropout 0 unless noted)

### Final, shipped defaults, `results/train_bench_v7_final.jsonl`, `results/parity_v6_final.jsonl`)

samples/s per GPU / torch peak GB; speed-up vs open_clip amp_bf16 at the same batch in brackets.

| variant (flags) | B=256 | B=512 | B=1024 |
|---|---|---|---|
| open_clip amp_bf16 (baseline) | 666 / 46.5 | 690 / 91.1 | OOM |
| cftrain full FT, compile, SDPA attention (Phase 1) | 1614 / 26.5 (2.42x) | 1672 / 47.3 (2.42x) | |
| **cftrain full FT default** (+ bespoke ViT attention bwd) | **1686 / 26.5 (2.53x)** | **1756 / 47.3 (2.54x)** | **1766 / 86.5** |
| + `--compile-mode max-autotune-no-cudagraphs` | 1763 / 26.5 (2.65x) | 1781 / 47.3 (2.58x) | |
| + `--act-budget 0.5` | | | 1645 / 51.6 |
| `--freeze-*-blocks 6` | 2708 / 14.5 (4.07x) | 2779 / 24.4 (4.03x) | |
| `--mode lora --lora-targets qkv` | 2066 / 17.6 (3.10x) | 2020 / 32.7 | |
| `--mode lit` (cached image features) | 7342 / 7.6 (11.0x) | 7405 / 12.5 (10.7x) | 7382 / 24.1 |

Parity of the shipped defaults: gradient cosine vs open_clip fp32 0.999862 (floor 0.999886), LoRA 0.99981,
LiT 0.999877; 40-step loss curve max deviation from fp32: cftrain eager 0.091, compiled 0.131 (open_clip amp
itself 0.248); NaN-poisoned-allocator audit 0 non-finite tensors; DDP math (2 gloo ranks) 8e-7 rel. L2.

### Multi-GPU and long-run parity (2026-11-16)

- **DDP, real NCCL, 2 GPUs** (`test_ddp.py`: one fp32 global batch split across ranks with
  all-gathered features vs one process: loss 7.442941 vs 7.442939, gradient rel. L2 2.2e-6 -> PASS.
- **Scaling** (synthetic 16 k set, B=256 per GPU; the two GPUs are on different sockets, `nvidia-smi topo` = SYS):
  1 GPU ~1620 samples/s; 2 GPUs 2620-2680 overall, **2936-2986 within an epoch (1.84x, 92 %)**: on this tiny set
  every 32 steps an epoch restart costs ~10 %. bf16 gradient all-reduce (`--ddp-bf16-grads 1`) +2 % only, so the
  all-reduce is overlapped, not the bottleneck.
- **Long-run loss curves**, 3000 AdamW steps, B=128, same batches/LR schedule (`long_curve.py`,
  `compare_curves.py`; windowed 100-step means; the model memorises the set, loss 5.75 -> 0.003):

  | run | max |window diff| vs open_clip fp32 | final window | samples/s | peak GB |
  |---|---|---|---|---|
  | open_clip fp32 | 0 | 0.0031 | 203 | 38.0 |
  | open_clip amp_bf16 | 0.019 | 0.0034 | 654 | 24.3 |
  | open_clip amp_bf16, cuBLAS reduced-precision reduction off | 0.035 | 0.0041 | 634 | 24.3 |
  | open_clip amp_bf16 + torch.compile | 0.073 | 0.0031 | 782 | 22.5 |
  | cftrain compiled, PyTorch SDPA attention | 0.027 | 0.0028 | 1372 | 14.9 |
  | cftrain eager, SDPA | 0.062 | 0.0034 | 1189 | 15.1 |
  | cftrain default (compiled + bespoke attention bwd) | 0.076 | 0.0030 | 1408 | 14.9 |
  | (before the t_type weight-decay fix) | 0.120 | 0.0029 | 1403 | 14.9 |

  All runs reach the same final loss; the gaps sit in the steepest part of the drop (loss ~0.9-1.0 around step
  1000), where small rounding differences shift the trajectory. One real mismatch was found and fixed:
  open_clip weight-decays the 2-D token_type_embeddings, the lean model held row 0 as a 1-D tensor and excluded
  it (0.120 -> 0.076). **Noise floor: open_clip itself, under equally valid bf16 roundings, deviates
  from its fp32 run by 0.019-0.073; cftrain's 0.027-0.076 is inside that spread -> long-run parity holds.**

### Phase 2, round 3 (2026-11-16): attention forward and MLP GEMM epilogues

- **Short-sequence attention forward** (kar task `kar_tasks/short_attn_fwd`, Triton, online softmax, 64x64 tiles, writes
  LSE for our backward): init passed, but every config is slower than PyTorch's FA2 forward (best -3.7 %; 218 vs
  206 us at B=256). FA2's forward is already efficient for N=197; only its backward was slow. **Dead end**; the op
  keeps FA2 forward + our backward (`cftrain.attn_vit.FWD_VERSION = "fa2"`; "triton" kept for reference, unit test
  passes).
- **MLP GEMM epilogues** (`cftrain/gemm_epi.py`, `cftrain/mlp_epi.py`; kar tasks `fc1_gelu_fwd`, `dgelu_gemm`; GEMM
  core from clipfast's kar-validated Triton kernel): fc1 + bias + exact-erf GELU writing pre and act
  (one rounding each), and dGELU fused into the input-grad GEMM. kar vs cuBLAS + separate op: fc1 +5.9 % / +12.7 %
  (M = 25 k / 50 k), dGELU +5.5 % / +12.5 %, both -7..-12 % at M = 12.6 k (used only for >= 16 k rows). Numerics
  better than autocast on every output and grad (single rounding). End-to-end: **+1.3 % at B=256
  (1612 -> 1634), +1.7 % at 512**, grad cosine 0.999912 (unfused 0.999915) -> within noise, kept **opt-in**
  (`cftrain.model.MLP_EPI`). Reason: the Triton GEMM core is ~10 % behind cuBLAS, which eats most of what the
  epilogue saves over inductor's already-fused GELU kernels. The real fix is a cuBLAS-speed GEMM with an erf-GELU
  epilogue (CUTLASS 2.x GemmUniversal 128x256x32 + LinearCombinationGELU, clipfast's plan for its Rust
  runtime) -- a C++-extension build, not attempted yet.

### DDP scaling to 4 GPUs (2026-11-17)

`ddp_bench.py` (fixed on-GPU batches, full fine-tuning, B=256 per GPU, GPUs spread over sockets): 1 GPU 1630 samples/s,
2 GPUs 3022 (1.85x, 93 %), 3 GPUs 4497 (2.76x, 92 %), 4 GPUs 5956 (3.65x, 91 %), 4 GPUs with bf16 gradient
all-reduce 6123 (3.76x, 94 %). `test_ddp.py` at 3 and 4 ranks: loss equal to the single-process global batch,
gradient rel. L2 5.1e-6 and 1.7e-6.

### Phase 2, rounds 4-6 (2026-11-17): attention backward tuning, MLP activation recompute

- **Attention backward, kar round 4** (control = v2): exp2 softmax recompute (v3d) **+5.75 %** at B=256 / +4.9 % at
  1024 -> KEEP; BM=32 (v3b) +2.3 % -> KEEP; 3 stages 0 %; BN=32 -28 %; fully unrolled block loops -84 % (register
  blow-up). **Round 5**: the stack v3d+v3b (v4) was **rejected** by the check (graph replay 2 max error 4.03x the
  reference's, limit 4.0) even though it would be ~2.5 % faster (697 us at B=256, 53 % of floor) -> not shipped;
  shipped v3d (exp2, BM 64). Round 6: unit test dqkv error = FA2's; fwd+bwd per layer 1446 us (FA2 3537 us,
  v2 1490 us); parity + NaN audit pass; end-to-end bespoke attention vs SDPA +4.3 % (B=256) / +6.3 % (B=512).
- **MLP activation recompute** (`cftrain.mlp_epi.SAVE_ACT = False`, with `MLP_EPI`): keeps only the fc1
  pre-activation and recomputes GELU in backward. Grad cosine 0.999912 (= unfused). Peak memory **-18 % at B=512
  (47.3 -> 38.7 GB)**, -13 % at 256, for -4 % speed. Use when memory-bound; `--act-budget 0.5` gives more (-45 % at
  B=1024) for -7 %.

### Development history

Benchmarks `bench_train.py` results in
`results/train_bench_v*.jsonl` (not committed). samples/s per GPU, peak = torch peak allocated.

| variant | B=128 | B=256 | B=512 | B=1024 |
|---|---|---|---|---|
| open_clip amp_bf16 (baseline) | 640 / 25.1 GB | 668 / 48.3 GB | 688 / 91.4 GB | OOM |
| open_clip amp_bf16, dropout 0.1 (its default) | | 627 / 47.8 GB | | |
| cftrain eager (packed, CLS-only, fused AdamW) | 1212 / 16.5 | 1251 / 26.4 | 1240 / 47.0 | 1237 / 87.1 |
| **cftrain compile** | **1497-1523 / 16.4** | **1574-1620 / 26.4** | **1685 / 47.0** | **1685 / 88.7** |
| cftrain compile, dropout 0.1 | | 1600 / 26.7 | | |
| + whole-step CUDA graph | 1411-1490 / 17.7 | 1573-1587 / 27.6 | 1672 / 48.3 | 1687 / 90.5 |
| + activation budget 0.5 (partitioner recompute) | 1406 / 12.8 | 1512 / 18.8 | | 1582 / 52.8 |
| + per-block recompute | 1249 / 9.1 | 1331 / 11.4 | 1368 / 15.7 | 1353 / 22.2 |
| LoRA r=8 all linears, unfused modules | 1536 / 13.2 | 1503 / 22.3 | | |
| LoRA r=8, fused autograd (Unsloth-style) | 1554 / 11.3 | 1521 / 20.3 | 1494 / 41.2 | 1413 / 78.3 |
| LiT (image tower locked, cached features) | 6362 / 7.5 | 7439 / 9.5 | 7771 / 12.7 | 7357 / 24.1 |
| freeze first 6 of 12 blocks (both towers) | | 2578 / 14.5 | | |
| freeze first 9 of 12 blocks | | 3691 / 7.6 | | |
| LoRA r=8 on qkv only | | 1910 / 17.2 | | |
| LoRA r=8 all linears + freeze first 6 | | 2348 / 11.5 | | |
| FP8 fwd GEMMs (opt-in, parity fails) | | 1632 / 24.2 | 1730 / 42.8 | |

Speed-up vs open_clip amp_bf16 at equal batch: **2.3-2.5x full fine-tuning, ~11x LiT**; memory 0.55x (full), and
B=1024 now fits on one GPU (open_clip OOMs at 1024; 0.6x memory with activation budget 0.5).

Findings that changed the plan:
- **CUDA graphs buy nothing for full FT at B >= 128** (GPU-bound, ~1400-1900 launches hidden behind 80-600 ms of
  GPU work) and cost graph-pool memory. Kept opt-in only.
- **LoRA is not faster than full fine-tuning for ViT-B / BERT-base.** The dW GEMM it removes costs 2KN FLOPs per
  token (fc1: 4.7 MFLOP = ~12 ns at 380 TFLOP/s), while the rank-r side path adds memory-bound passes over X and
  dY (~4(K+N) elements per token = ~30 KB = ~20 ns at 1.5 TB/s). The ratio K*N/(K+N) ~ d/2 is 4x larger for
  7B-class models, which is where Unsloth's speed-ups come from. Here LoRA's value is memory (no Adam state for 196 M
  params: 20 vs 26 GB at B=256) and regularisation. Speed for PEFT comes from **freezing** (no dW AND no backward
  below the first trainable block) and from **LiT caching**.
- Batch-max trimming of text is useless with long-tailed captions (a 256-token caption lands in almost every batch);
  packing is required.

Compiled-step profile, B=256, dropout 0.1, 159 ms): GEMM 85 ms (53 %, at the predicted packed floor),
LayerNorm fwd+bwd 24 ms (~85 % of its memory floor), attention 18 ms (FA2 bwd 7.4 + dq-convert 2.6 + fwd 2.7 for
the image tower; floor < 2 ms), GELU fwd+bwd 15 ms, casts 5 ms, AdamW 3.7 ms. Phase 2 ceilings from it:
FP8 forward GEMMs -13..-26 ms, short-sequence attention -10..-14 ms, GELU/residual GEMM epilogues -8..-12 ms,
LayerNorm -3..-5 ms -> ~105-115 ms/step = ~2200-2400 samples/s at B=256.

FP8 forward GEMMs e4m3, rowwise current scaling, bf16 backward): **+5 %** (1553 -> 1632 samples/s
at B=256, 1656 -> 1730 at 512) for a large parity cost: gradient cosine 0.9918 vs fp32 (bf16 floor 0.99989; median
per-tensor rel. error 15 % vs 1.7 %), 40-step loss-curve deviation 0.99 vs 0.22 for open_clip amp. The quantize
passes eat most of the FP8 GEMM gain at these sizes (K = 768). **Dead end for this model**: kept as `--fp8 1`
(opt-in, not recommended); revisit only with block-scaled FP8 fused into the LayerNorm kernel.

Bug found by the end-to-end smoke (fixed, registered as a parity audit): with the trainer's packing (a dummy
padding sequence makes token counts repeat), the CLS-only last text layer passed K/V rows outside every
`cu_seqlens` range; FlashAttention's backward leaves dK/dV for such rows **unwritten**, so they held whatever the
allocator returned -> intermittent NaN. Fix: give the dummy a zero-length query so every K/V row is in range.
Audit: allocator pre-filled with NaN; old call 139 non-finite grad tensors, fixed 0 (`results/debug_dummy.jsonl`).

Phase 2 kernel 1, ViT attention backward (`cftrain/attn_vit.py`, kar task `kar_tasks/short_attn_bwd`):
two Triton kernels without atomics (dK/dV per key block; dQ per query block), reading q/k/v straight from the qkv
GEMM output [B,N,3,H,D] and writing dqkv in place (no permute/cat copies). kar: init passed (all canaries rejected),
cand_triton_v1 1.47x FA2 backward at B=256, 1.59x at B=1024 (35-39 % of the memory floor). Unit test: dqkv error vs
fp32 identical to FA2's (MAE 2.0006e-4 both, cosine 0.999997), finite under a NaN-poisoned allocator, same under
compile; fwd+bwd per layer 1.78 vs 3.53 ms (eager). End-to-end parity unchanged (grad cosine 0.999861 = the
compiled SDPA path). **End-to-end +2.7 %** (1552 -> 1593 samples/s at B=256, 1660 -> 1704 at 512): inductor had
already hidden most of the layout copies. Default on (`cftrain.model.VIT_ATTN`).

Phase 2 plan (estimates, B=256): short-seq attention fwd/bwd ~1700; + GELU/residual epilogues ~1850;
+ bf16 residual stream (parity-gated) ~2000.

## 5. Parity protocol (every increment)

1. Gradient parity (`parity.py`, B=32, caption-like lengths, dropout 0, TF32 off): reference = open_clip
   fp32 grads mapped into the lean layout. Noise floor = open_clip amp_bf16 vs fp32 (global grad cosine, per-tensor
   min cosine, median relative error). Each variant (lean fp32 padded/packed: ~exact; bf16 padded/packed/compiled;
   LoRA; LiT) must have global cosine >= floor - 1e-4.
2. Loss-curve parity: 40 AdamW steps on 4 fixed batches: open_clip fp32 / amp vs cftrain amp (eager, compiled);
   max |loss - fp32 loss| <= 2 x (open_clip amp's) + 1e-3.
3. Kernels (Phase 2): kar tasks with fp64 reference, canaries, error budget 1.5x bf16 reference error; backward
   kernels checked on dx/dW/db separately.
4. Graph replay == eager step (same batch, same weights) bitwise; DDP 2-GPU == 1-GPU global batch.
5. Export: checkpoint loads in stock open_clip and reproduces the trained model's embeddings (cosine > 0.9999).

## 6. Ship plan

- Phase 1 (tonight): `train/train.py` CLI (cache/folder/CSV/JSONL; full / LoRA / LiT / lock-text; compile; graphs;
  DDP), `cftrain.prep` cache builder, parity + benchmark reports in `results/`
  and `README.md`.
- Phase 2 (2026-11-17..18): kar tasks for (1) residual+LN fwd/bwd, (2) GEMM + bias + erf-GELU epilogue fwd and
  dGELU bwd, (3) short-sequence FA bwd, (4) LoRA fused; each with gradient parity; then FP8 opt-in.

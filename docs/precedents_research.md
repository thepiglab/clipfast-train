# Precedents checked against primary sources (2026-11-16, research subagent; main branches)

Key consequences for BiomedCLIP (exact-erf GELU):
- cuBLASLt GELU epilogue, llm.c gelu, Liger GeGLU and TE gelu are all tanh-approx -> numerics mismatch with the
  pretrained weights. We need our own erf-GELU epilogue (or a separate fused bias+erf-GELU kernel).
- FA2 (setup.py archs 80;90;100;110;120) and FA4-cute (flash_fwd_sm120.py / flash_bwd_sm120.py, mma.sync, 99 KB smem)
  have sm_120 fwd+bwd. PyTorch SDPA order flash -> efficient -> math -> cudnn (cuDNN first only on sm9x/10x);
  flash allowed sm80..sm121; training head_dim <= 192 on sm12x -> head 64 fine.
- cuDNN SDPA bwd on sm_120 = experimental FROST CuTe-DSL engine (env var), no varlen, no dropout.
- TE: delayed + current-scaling FP8 OK on sm >= 8.9; MXFP8 "not supported on 12.0+ yet"; default recipe on sm_120 =
  Float8CurrentScaling; DelayedScaling not compatible with torch.compile. No contrastive loss in TE.

## llm.c (github.com/karpathy/llm.c)
- fused classifier (llmc/fused_classifier.cuh): forward CE + starts backward, never materialises normalised probs.
- fused residual+LN forward (layernorm.cuh fused_residual_forward_kernel5); single LN backward kernel
  (layernorm_backward_kernel10: dinp+dw+db, smem + fp32 scratch).
- GELU epilogue via CUBLASLT_EPILOGUE_GELU_AUX_BIAS / DGELU (tanh), OFF by default (gelu_fusion=0: "inefficient on
  Ada/Ampere").
- AdamW: bf16 params + fp32 master + stochastic rounding (adamw.cuh).
- recompute -r1 (GELU: L*B*T*4C -> B*T*4C), -r2 (+ln1/ln2: -2*L*B*T*C). ZeRO-1 reduce-scatter.
- README: ~7% faster than PyTorch nightly.

## Liger (github.com/linkedin/Liger-Kernel)
- FusedLinearCrossEntropy chunks BT (chunk = next_pow2(BT / cdiv(V,H))), grads accumulated during forward;
  README up to 80% memory for fused-linear losses, ~20% throughput.
- LayerNorm saves X,W,B,mean,rstd; GeGLU/SwiGLU recompute the activation in backward. No CLIP/InfoNCE loss.

## Unsloth (unsloth/kernels/fast_lora.py)
- hand-written autograd LoRA_MLP / LoRA_QKV / LoRA_W, addmm_ for A/B grads, no base dW.
- claims 2x / -70% VRAM vs HF (vs HF+FA2: 2.0-2.4x, -57..63%).

## Contrastive loss
- open_clip ClipLoss: gather_with_grad (autograd all_gather) + local_loss (B_local x B_global logits) -> linear
  memory, identical numerics. SigLipLoss has chunk_size.
- Inf-CL (arXiv 2410.17243): tile-wise LSE, recompute in backward, ring across GPUs. ViT-L/14 8xA800, loss memory at
  64k batch: CLIP 66 GB, OpenCLIP 8.6 GB, Inf-CL 0.36 GB.

## GPU images
- torchvision decode_jpeg(device='cuda') batched since 0.20 (nvJPEG, inputs on CPU; deprecated in 0.29 in favour of
  TorchCodec).
- nvJPEG HW decode lists Blackwell, but the RTX PRO 6000 spec lists NVDEC/NVENC, no NVJPG -> assume CUDA (hybrid)
  decode only.
- CUDA antialias bicubic is ported from Pillow Resample.c (a=-0.5) -> close to PIL, not bit-exact.

## Optimizer / compile
- torch fused AdamW: bf16 supported, state in param dtype (no fp32 master), step on device, capturable.
- Inductor cudagraph_trees default on. Compiled-optimizer recipe ~1.9x in its example.

## PEFT precedents
- CLIP-LoRA (Zanella & Ben Ayed, CVPRW'24): q,k,v both encoders all layers, r=2, alpha=1, dropout .25, lr 2e-4,
  500 it, bs 32, ViT-B/16.
- LiT (Zhai et al. CVPR'22): lock image tower, train text; Lu 70.1% vs Uu 57.2% ImageNet zero-shot.

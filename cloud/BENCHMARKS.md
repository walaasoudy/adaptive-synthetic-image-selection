# Pipeline optimization benchmarks (Modal, 2026-09-11)

Measured with `cloud/modal_bench.py` on Modal. Each benchmark runs the original and the optimized
behaviour through the same repository code, in the same container, on the same synthetic inputs
(768 px CXR-like JPEGs; synthetic latents for the Stage 1 step), so the comparison isolates the
change. No CheXpert data was used. Costs include the GPU plus the requested CPU (4 cores) and
memory (16 GiB) at Modal's published rates.

**Available hardware.** The Starter plan without a payment method may use only T4, L4 and A10;
L40S and A100 are refused at launch (`cloud/modal_gpu_probe.py`). All measurements are on A10 and
L4.

## Summary of decisions

| # | Change | Result | Decision |
|---|---|---|---|
| 1 | Classifier DataLoader workers + pinned memory + cudnn.benchmark | 2.2-4.9x faster; outputs identical | **Kept** (`7333044`) |
| 2 | Stage 2: several images per SDXL pipeline call | 1.07x faster; images change (PSNR 23-33 dB) and the original's bit-exact reproducibility is lost | **Rejected** |
| 3a | Stage 1: batched VAE latent caching | 1.05x faster at 3x VRAM; batch 16 OOMs | **Rejected** |
| 3b | Stage 1: free VAE + text encoders after caching | ~2 GB less host RAM for the whole run, no computation change | **Kept** (`33e53ad`) |
| 4 | Stage 1 gradient checkpointing | GC off does not fit in 24 GB (8x2 and 4x4 both OOM) | **GC stays on** (the only feasible setting) |
| - | Stage 1 container: 2 CPU cores instead of 4 | training is GPU-bound (99% util) | **Kept** (`d907986`) |

## 1. Classifier data loading (`scripts/utils/classifier.py`)

Used by the auxiliary classifier, the ASISM proxy runs (04b, 07b, 08b), Stage 4 and Stage 5.
Original: `num_workers=0`, so every 768 px JPEG is decoded and resized inside the training process
while the GPU waits.

| Phase | A10 original | A10 optimized | Speedup | L4 original | L4 optimized | Speedup |
|---|---|---|---|---|---|---|
| Train 48 steps (batch 32) | 30.6 s | 12.7 s | 2.4x | 50.5 s | 23.2 s | 2.2x |
| Eval pass, 1536 images | 23.0 s | 4.7 s | 4.9x | 34.5 s | 10.9 s | 3.2x |
| MC Dropout, 5 x 256 images | 19.1 s | 6.5 s | 3.0x | 29.5 s | 13.5 s | 2.2x |
| GPU utilization (eval) | 17% | 73% | | 17% | — | |
| Peak VRAM | 8.1 GiB | 8.2 GiB | same | 8.1 GiB | 8.2 GiB | same |

**Equivalence.** The first 12 shuffled training batches are bit-identical (images, targets, masks,
order) on both GPUs. On L4 the trained weights and predictions are bit-identical to the original
(max abs difference 0.0). On A10, where the original itself is not bit-reproducible between two
runs (weights differ by up to 0.125), the optimized-vs-original difference (0.090) is inside that
noise floor. `tests/run_all.py`: 194 passed.

**GPU choice for Stages 3-5.** Cost for the same work: A10 $0.0050 per 48 training steps vs L4
$0.0072 (+44%); A10 $0.0019 per 1536-image evaluation vs L4 $0.0034 (+83%). **A10 is both faster
and cheaper.** T4 was not measured (no TF32/bf16, expected to be slower per dollar).

**Projected effect on the pipeline (A10, estimates from the throughputs above):**

| Workload | Original | Optimized |
|---|---|---|
| Auxiliary classifier (8000 steps + 11 evals of gen_val) | ~2.3 h | ~0.8 h |
| 04b utility subsets (121 proxy runs) | ~15.0 h | ~4.4 h |
| 08b full-policy verification (15 runs) | ~1.9 h | ~0.6 h |
| Stage 4 (9 runs x 6000 steps + 13 evals) | ~17.8 h | ~5.7 h |
| **Total** (07b not included) | **~37 h (~$52)** | **~11.5 h (~$16)** |

The original 04b estimate (~7.4 min per proxy run) agrees with the config's own
`hours_per_proxy_run_estimate: 0.15` (9 min), which supports the projection.

## 2. Stage 2 batched generation — rejected

SDXL 768 px, 40 DPM-Solver steps, CFG 7, A10.

| | One image per call (original) | Four per call |
|---|---|---|
| Seconds per image | 6.96 | 6.47 (1.07x) |
| GPU utilization | 99.7% | 99.7% |
| Peak VRAM | 8.8 GiB | 15.2 GiB |

The original is fully deterministic: two runs with the same seed produce byte-identical images.
Batched images start from the same noise but diverge during denoising (PSNR 23-33 dB, mean
absolute pixel difference 2.3-10.5), so a resumed run would no longer reproduce an uninterrupted
one. 7% is not worth that; the generator is unchanged.

## 3. Stage 1 caches

VAE latent cache, 96 images at 768 px, fp32 VAE, A10:

| | Per image (original) | Batch 4 | Batch 8 | Batch 16 |
|---|---|---|---|---|
| Time | 15.9 s | 15.1 s | 15.6 s | 23.0 s (OOM, fell back to 8) |
| Peak VRAM | 1.7 GiB | 6.0 GiB | 11.6 GiB | 18.4 GiB |

The original loop already runs the GPU at 94%, so batching recovers ~5%. Rejected. Freeing the
VAE and both text encoders after the caches are built is kept.

## 4. Stage 1 training step and gradient checkpointing

Trainer's own `build_models()` (SDXL UNet bf16, LoRA r32, EMA), 768 px, synthetic latents. Every
variant averages the loss over 16 samples per optimizer step.

| Variant (micro-batch x accumulation, GC) | A10 s/step | A10 VRAM | L4 s/step | L4 VRAM |
|---|---|---|---|---|
| **8 x 2, GC on (frozen)** | **5.10** | 8.8 GiB | 6.73 | 8.8 GiB |
| 16 x 1, GC on | 4.82 | 11.7 GiB | 7.20 | 11.7 GiB |
| 8 x 2, GC off | OOM (21.3 GiB) | — | — | — |
| 4 x 4, GC off | OOM | — | OOM | — |

GPU utilization 98-99% in every variant that fits. Without gradient checkpointing the step does not
fit in 24 GB even at micro-batch 4, so GC stays on. 16 x 1 is 5.5% faster on A10 but changes the
frozen batch geometry, so it is not applied.

**Stage 1 projection (15,000 steps, A10):** 21.2 h of training plus a one-time latent cache of
~0.17 s per image (~5 h for ~110k gen_train + gen_val images) ≈ 26 h ≈ $36-38 with 2 cores and
24-32 GiB RAM. L4 costs about the same in total but takes ~28 h of training.

## Storage

The pinned SDXL files the pipeline actually loads are 12.9 GiB (`download_models` fetches only
those; the full repository snapshot is ~70 GB). Modal Volumes include 1 TiB/month free, so the
dataset, caches and checkpoints add no storage cost.

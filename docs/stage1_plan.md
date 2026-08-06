# Stage 1 Implementation Plan: SDXL + LoRA Fine-Tuning on CheXpert

## Context

This is Stage 1 of a 5-stage Master's thesis framework ("An Adaptive Quality-Aware Synthetic
Data Selection Framework for Medical Image Classification Using Diffusion Foundation Models").
The project directory is currently empty of code — no `.py`/`.ipynb` files, no dependency files,
no git repo, no dataset on disk (only `Proposal · MD`, `Literature review .md`, and a
`.claude/agents/` folder with three generic, unrelated subagent templates that specify no
diffusion/LoRA tooling or hardware conventions to build on).

Stage 1's job is to fine-tune SDXL with LoRA on CheXpert chest X-rays so that later stages can
generate controllable, label-conditioned synthetic images (Stage 2), score/select them (Stage 3 —
the thesis's novel ASISM contribution), train a classifier on selected synthetic data (Stage 4),
and run a real-vs-synthetic comparison (Stage 5). Because every later stage depends on what
Stage 1 produces and how it's structured, this plan optimizes not just for a working fine-tune,
but for **provenance and reproducibility** (so Stage 5's comparison isn't contaminated),
**resumability** (RunPod pods can be interrupted), and **clean handoff** (Stage 2 must reuse
Stage 1's exact captioning logic and weights deterministically).

Confirmed constraints from the user:
- **Hardware:** RunPod, PyTorch 2.8.0 template, 1× A40 (48GB VRAM), CUDA 12.8, 9 vCPU, 50GB RAM,
  80GB ephemeral container disk + a separate persistent volume. VS Code over SSH; Claude Code runs
  on the instance directly.
- **Dataset:** CheXpert-v1.0-small, not yet downloaded, to be obtained from Kaggle.
- **Captioning:** structured label-to-text templates (matching Rehman et al.'s tabular-to-text
  approach), not VLM-generated captions.
- Every design decision below states the choice, why, alternatives considered, why they were
  rejected, and trade-offs — per explicit user request.

Grounding papers (from `Literature review .md`): **Rehman et al.** (LoRA-tuned SD on
CheXpert/MIMIC-CXR, tabular-to-text prompts, 24GB→8GB memory reduction vs. full fine-tuning,
Grad-CAM-verified shift away from shortcut features) is the direct template, extended here from
SD to SDXL. **Niemeijer et al.** (TSynD) establishes that downstream stages need *controllable,
label-conditioned* generation, not just realism — so Stage 1 must preserve exact
label↔caption↔image provenance. The literature's repeated warning that **realism ≠ diagnostic
correctness** (Pozzi et al., Hosseini & Serag) is why Stage 1's validation is deliberately scoped
to training stability/plausibility only, deferring diagnostic fidelity to Stage 3/5.

No code is written in this planning turn. This plan is implemented only after explicit approval.

---

## 1. Papers and design choices Stage 1 is based on

- **Rehman et al.**: adopt the mechanism (LoRA on a diffusion UNet + structured label→text
  captions), swap base model SD→SDXL (the thesis's stated extension), swap available fields
  (age/sex/frontal-lateral + 14 CheXpert pathologies, since race isn't a CheXpert field).
- **Niemeijer et al. (TSynD)**: doesn't change Stage 1's architecture, but requires Stage 1 to
  preserve label↔caption↔image provenance so Stage 2 can do targeted/conditional generation and
  Stage 3 can later reason about which label combinations were under/over-represented.
- **Structural-conditioning theme** (edges/masks improve anatomical consistency): CheXpert has no
  segmentation masks, so ControlNet-style conditioning is explicitly **deferred**, not dropped —
  revisit if Stage 3's Grad-CAM/quality review later reveals systematic anatomical inconsistency.
- **"Realism ≠ diagnostic correctness"**: shapes §9 — Stage 1 validates training stability and
  qualitative plausibility only; diagnostic fidelity is Stage 3 (Grad-CAM/IQA) and Stage 5's job.

## 2. SDXL model variant

**Choice: `stabilityai/stable-diffusion-xl-base-1.0` UNet only (no refiner), pinned revision
hash, VAE upcast to fp32 during encode/decode.**

| Alternative | Why rejected |
|---|---|
| Base + refiner two-stage | Refiner specializes in natural-photo high-frequency detail touch-up; no evidence it helps grayscale radiographs, and doubles the LoRA-training surface. |
| SDXL-Turbo / SDXL-Lightning (distilled) | Distilled for few-step *inference* latency, which offline batch generation (Stage 2) doesn't need; training tooling for distilled variants is less mature; risks breaking the distillation guarantee. |
| SSD-1B / Vega (smaller distilled UNet) | Exist to fit smaller GPUs — irrelevant with 48GB headroom; reduced capacity risks losing fine pathology texture that Grad-CAM/classifier stages depend on. |
| SD 1.5 / SD 2.1 | Thesis title commits to SDXL; SDXL's dual text encoder and native 1024px training give more expressive conditioning than what Rehman et al. (SD) used — this thesis's stated extension. |
| fp16 checkpoint + `sdxl-vae-fp16-fix` | Valid alternative to the well-documented SDXL-VAE fp16 NaN bug; kept as a documented Stage-2 speed option, not Stage-1 default — fp32 VAE upcast is diffusers' standard, dependency-free mitigation. |

**Trade-off:** SDXL is heavier than SD1.5 and has two text encoders to manage; mitigated by the
A40's headroom and by caching text embeddings (§8).

## 3. LoRA configuration

**Choice: rank 32, alpha 32 (scale 1.0), target = UNet self+cross attention projections
(`to_q,to_k,to_v,to_out.0`) only, text encoders frozen, dropout 0.0, LR 1e-4 with warmup+cosine
decay, AdamW8bit (bitsandbytes) primary with plain AdamW fallback.**

| Decision | Why | Alternatives rejected |
|---|---|---|
| Rank 32 | Middle ground for 14-pathology, multi-view diversity — more capacity than single-subject DreamBooth (rank 4–8) needs, without eroding LoRA's memory-efficiency motivation (the reason Rehman et al. chose LoRA at all) via near-full-rank (64–128). | Rank 4–8 underfits label diversity; rank 64–128 defeats the memory-efficiency point and bloats per-checkpoint files. |
| Alpha = rank | Keeps alpha/rank scaling identity simple; tune convergence via LR only. | Alpha = 2×rank: no principled justification, conflates two knobs. |
| Self+cross attention only | Field-standard for SDXL LoRA; cross-attention binds prompt→content, self-attention governs spatial/anatomical layout — literature emphasizes anatomical consistency. | Cross-attention-only leaves spatial reasoning to natural-image-pretrained weights (poor match for CXR layout statistics). LoRA on conv/ResNet blocks: unnecessary complexity for unvalidated gain. |
| Text encoders frozen | Constrained template vocabulary doesn't need "medical understanding," just stable distinguishable embeddings; avoids catastrophic forgetting that would hurt Stage 2's generalization to unseen label combos; mirrors Rehman et al.'s efficiency framing. | Text-encoder LoRA (rank 8–16): adds VRAM/complexity to a second sensitive component with no evidence yet it's needed — **documented fallback** if v1 samples show weak conditioning on rare labels (e.g. Lung Lesion, Pleural Other). |
| Dropout 0.0 | Dataset scale (100k+ images) plus low-rank re-parameterization already regularizes. | Dropout 0.1: unnecessary at this scale, slows convergence; revisit only if val loss diverges from train loss. |
| LR 1e-4, warmup+cosine | Standard, well-tested starting point for rank-16–32 SDXL UNet LoRA; LoRA tolerates higher LR than full fine-tuning. | Prodigy (adaptive, LR-free): less battle-tested for SDXL LoRA, harder to debug; kept as a documented experiment, not default. |
| AdamW8bit primary | Extra memory headroom for batch/resolution/caching. | Plain AdamW: fine too since LoRA's trainable param count is small — kept as explicit fallback given real risk that bitsandbytes wheels lag brand-new CUDA 12.8/torch 2.8. Adafactor: not a validated choice in the SDXL-LoRA ecosystem. |

## 4. Folder structure

Rule: **everything that must survive a pod restart lives on the persistent volume** (conventionally
`/workspace`); container disk holds only OS/disposable scratch. Git repo is initialized on the
persistent volume (code/configs/small manifests tracked; raw/processed images, checkpoints, caches
`.gitignore`d — each checkpoint's own metadata JSON records the git commit hash for provenance
without committing binaries).

```
/workspace/chest-synth-thesis/
  data/chexpert/raw/                    # untouched Kaggle extract
  data/chexpert/processed/images_768/   # preprocessed images
  data/chexpert/processed/splits/       # gen_train.csv, gen_val.csv, classifier_heldout.csv, split_manifest.json
  data/chexpert/processed/captions/     # train_captions.jsonl, caption_template_version.json
  configs/stage1_lora_sdxl.yaml, accelerate_config.yaml, dataset_config.yaml
  scripts/data/       01_verify_download.py, 02_build_patient_splits.py, 03_preprocess_images.py, 04_generate_captions.py
  scripts/train/      train_lora_sdxl.py, launch_resumable.sh
  scripts/eval/       generate_probe_samples.py, compute_fid_clipscore.py
  scripts/utils/      caption_builder.py (importable — reused verbatim by Stage 2), seed.py, manifest.py
  checkpoints/stage1_lora_sdxl/<run_id>/checkpoint-<step>/, lora_weights/, latest.json, final/
  logs/stage1_lora_sdxl/<run_id>/       # tensorboard events
  outputs/stage1_samples/<run_id>/      # fixed-seed probe grids
  environment/requirements.txt, requirements-lock.txt
  docs/literature_review.md, proposal.md
  .cache/huggingface/                   # HF_HOME redirected here, not container disk
  .gitignore, README.md
```

`run_id` = timestamp + short git sha + config hash.

**Disk budget:** raw CheXpert-small ~11GB, processed/padded images ~10–20GB, SDXL base weights
~7–14GB, cached VAE latents+text embeddings for ~130k images ~15–20GB, checkpoints (LoRA-only
weights tiny; full resumable states capped at last 3) a few GB. Realistic total ~60–100GB —
**provision a ≥150–200GB persistent volume.**

## 5. Python libraries and versions

Torch 2.8.0+cu128 is already provided by the RunPod template — do not reinstall/downgrade.

| Package | Pin | Notes |
|---|---|---|
| diffusers | `>=0.31,<0.35` | Reference SDXL LoRA training script + PEFT-backed LoRA, reused by Stage 2 |
| transformers | `>=4.44,<4.50` | Two SDXL text encoders |
| accelerate | `>=0.34` | Mixed precision, grad accumulation, save/load_state for resume |
| peft | `>=0.13` | LoRA backend used by diffusers |
| bitsandbytes | `>=0.43` | **Verify CUDA 12.8 compat at setup**; fallback to `torch.optim.AdamW` if it fails |
| xformers | **not installed** | Use PyTorch 2.x SDPA (`AttnProcessor2_0`) instead — matches torch version by construction, avoids a cu128 wheel-availability risk |
| safetensors, pillow, numpy, pandas, opencv-python-headless, scipy | latest stable | Preprocessing |
| torchmetrics[image] | latest stable | FID, CLIP-score |
| torchxrayvision | latest stable | CXR-domain feature extractor for domain-appropriate FID (§9) |
| omegaconf | latest stable | Nested YAML config + CLI overrides |
| tensorboard | latest stable | Primary, offline-durable logging |
| wandb | latest stable (optional) | Supplementary remote dashboard |
| huggingface_hub, tqdm, python-dotenv | latest stable | Pinned model download, secrets, progress |

**Currency caveat:** exact compatible-version matrix for SDXL LoRA + torch 2.8/cu128 should be
verified with a Day-1 smoke test (few steps, few images), then frozen via
`pip freeze > requirements-lock.txt` — versions above are reasonable anchors, not guarantees.

## 6. Dataset preprocessing pipeline

- **Verification first:** since the exact Kaggle mirror isn't confirmed yet, `01_verify_download.py`
  checks structure rather than a checksum: row counts (224,316 train + 234 valid), column schema
  matches known CheXpert fields, spot-check image openability/dimensions, compare label prevalence
  against published CheXpert statistics.
- **View filtering:** **frontal-only default for v1.** Downstream classifier benchmarks
  predominantly use frontal images; mixing frontal+lateral without an explicit view tag would blur
  output distribution given their different pathology-appearance statistics. Lateral images are
  filtered (not deleted), flagged in the manifest — reversible. Rejected default alternative:
  include both with view as a caption tag (roughly doubles conditioning combinatorics for v1).
- **Resolution: 768, not 1024.** Source images are ~320–390px; upsampling to 1024 adds no real
  information and costs more compute. 512 considered but discards more available detail than
  necessary. First ablation candidate, not fixed.
- **Aspect handling: aspect-preserving resize + minimal letterbox padding, NOT non-uniform
  stretch-to-square and NOT center-crop.** Non-uniform stretch directly distorts the
  cardiothoracic ratio (central to Cardiomegaly assessment) — unacceptable given downstream stages
  depend on faithful pathology content. Center-crop risks losing peripheral pathology (e.g. a
  costophrenic-angle effusion, apical pneumothorax).
- **No horizontal flip augmentation** — flips invert left-right anatomy (heart position, aortic
  arch, gastric bubble); must be explicitly disabled since diffusers' reference script may default
  to flips.
- **Quality filtering:** verify openability, normalize grayscale→3-channel RGB, discard/flag
  images below a minimum resolution, flag likely-corrupt frames (near-uniform pixel statistics).
  Every filtering decision logged with image ID + reason — nothing silently dropped.
- **Normalization:** pixels to [-1, 1] before VAE encoding; VAE fp32 upcast (§2).
- **Patient-level split (most consequential decision in Stage 1):** split by patient ID, not image
  ID, into `gen_train` (~70%), `gen_val` (~10%, held out from training, used for val loss/qualitative
  monitoring), and `classifier_heldout` (~20%, **never touched by Stage 1 at all**). This quarantine
  is what keeps Stage 5's eventual real-vs-synthetic comparison uncontaminated — if the generator
  had ever seen a patient later used in Stage 5's real-only test set, that's a bias risk. The
  official CheXpert `valid.csv` (234 curated images) is similarly reserved untouched, for possible
  benchmark comparability later. Split seed, ratios, and code version persisted in
  `split_manifest.json`.

## 7. Caption generation strategy

Template: `"A {view} chest X-ray of a {age_bucket} {sex} patient. Findings: {finding_clause}. {device_clause}"`
- `age_bucket`: decade-binned (not raw integer) to reduce vocabulary sparsity while still encoding
  demographics, echoing Rehman et al.
- `finding_clause`: positive labels (see uncertainty policy below) sorted into CheXpert's canonical
  column order, joined with natural "X, Y, and Z" grammar (more stable CLIP embeddings than a comma
  token-dump); "no acute cardiopulmonary abnormality" when `No Finding == 1`, which is treated as
  authoritative over any spuriously co-positive fields (a known NLP-label-extraction artifact).
- `device_clause`: "Support devices are present." appended separately, not folded into pathology
  list, since it's not a disease finding.

**Uncertainty label (-1) policy — chosen: omit from the finding clause (treat as "not mentioned,"
neither asserted present nor absent).**
- **Why:** a -1 label means the original radiologist wasn't confident from the actual image —
  i.e. no reliable visual signature exists for that case. Asserting "possible X" in the caption
  binds a text token to a visually inconsistent signal, weakening the clean caption↔image
  association that Stage 2/3 need for controllable, trustworthy generation.
- **Rejected alternative 1** — hedged text ("possible X"): more literal, but pollutes the concept
  as above.
- **Rejected alternative 2** — blanket U-Ones/U-Zeros mapping: factually false for a real fraction
  of cases; worse for image-text pair training than for classifier label-noise tolerance, since it
  actively teaches a wrong association rather than just adding noise to a loss average.
- **Trade-off:** reduces effective positive-caption prevalence for labels with high -1 rates (e.g.
  Enlarged Cardiomediastinum, Lung Opacity) — accepted limitation. The raw ternary label is
  preserved unmodified in the manifest regardless, so this is purely a captioning-time policy,
  revisable later without redoing image preprocessing.
- **Template diversity:** 3–5 paraphrase variants randomly selected per image per epoch (label
  semantics fixed) — cheap, standard mitigation against memorizing prompt structure.

## 8. Training pipeline

- **Base implementation:** adapt diffusers' `examples/text_to_image/train_text_to_image_lora_sdxl.py`
  (not kohya-ss sd-scripts — kohya is oriented toward artistic/anime LoRA workflows, harder to
  integrate custom label-driven captioning and less citable/auditable for a thesis methodology
  section than the HF-maintained diffusers+accelerate+PEFT stack). Custom layer adds: CheXpert
  `Dataset` class, RunPod-aware resume logic, YAML-driven config, TensorBoard/W&B hooks, periodic
  qualitative + FID/CLIP-score callbacks.
- **Execution:** single A40 via HF Accelerate (single-process) — even without multi-GPU need,
  Accelerate provides the `save_state`/`load_state` resume primitives the RunPod-interruption
  requirement needs.
- **Precision: bf16** (not fp16) — A40 is Ampere, supports bf16 natively; avoids the numerical
  range issues behind the well-documented SDXL-VAE fp16 NaN bug, combined with fp32 VAE upcast.
- **Attention:** PyTorch 2.x SDPA, not xformers (per §5 rationale).
- **Batch size / grad accumulation for 48GB:** only LoRA params carry optimizer state (frozen
  base contributes activations only), so activation memory dominates, not parameter memory — SDXL
  LoRA is known to fit even 24GB cards at modest batch sizes. **Default: resolution 768, batch size
  8, grad accumulation 2 (effective 16)** with gradient checkpointing on. First lever to raise if
  headroom allows: batch size. First lever to drop on OOM (before resolution): batch size.
- **Latent/embedding caching:** precompute/cache VAE latents + text embeddings once per
  image/template-variant (captions are deterministic-from-labels) — removes VAE/text-encoder
  forward passes from the training hot loop, trading ~15–20GB disk for faster steps and lower peak
  memory.
- **EMA:** enabled on LoRA adapter weights only (decay ~0.9999) — cheap since LoRA weights are
  small; treated as an ablation toggle, not fixed.
- **Duration:** step-based, not epoch-based (gen_train is 100k+ images even after frontal-only
  filtering). Initial budget ~15,000–30,000 steps; actual stop point driven by val loss plateau +
  stable qualitative probes (§9), not a pre-decided epoch count.

## 9. Validation strategy

Deliberately scoped to *training convergence + controllability*, not clinical correctness
(explicitly deferred to Stage 3/5 per the literature's realism≠correctness warning):
1. **Training stability:** train + `gen_val` denoising MSE loss trends down, plateaus, no
   NaNs/divergence.
2. **Qualitative plausibility:** fixed probe-prompt set (one per pathology-positive, no-finding,
   a few multi-label combos), fixed seed, regenerated every checkpoint interval for direct
   checkpoint-to-checkpoint comparison — manually reviewed by the user, not formal radiologist
   review (explicitly out of scope for Stage 1).
3. **Informal directional conditioning check:** e.g. "cardiomegaly"-conditioned samples should
   visibly show larger cardiac silhouette more often than "no finding" samples — a sanity check,
   not a statistical claim.
4. **CLIP-score trend**, tracked relative to itself across checkpoints only — flagged as a weak
   signal since general-domain CLIP wasn't trained on medical images; useful for regressions, not
   an absolute bar.
5. **FID trend**, two variants: standard Inception-FID (field-standard, comparable to other work,
   but caveated as a poor fit for grayscale medical images), and a **domain FID using
   `torchxrayvision`'s DenseNet121 features** as the primary/more meaningful variant — a more
   literature-appropriate response to "FID doesn't reflect diagnostic content."

Explicitly NOT claimed by Stage 1 validation: diagnostic correctness, faithful pathology anatomy,
absence of shortcut learning — those are Stage 3/5 questions.

## 10. Checkpoints and logging

- **Cadence:** full resumable Accelerate state (optimizer state, LoRA weights, RNG, scheduler,
  step/epoch counters) every 500–1000 steps — affordable since only LoRA params carry optimizer
  state. **Retention:** last 3 full-resumable checkpoints; **all** lightweight inference-ready
  LoRA-weights-only snapshots kept (tiny, tens of MB) for possible Stage 2 multi-checkpoint sampling.
- **Naming:** `checkpoints/stage1_lora_sdxl/<run_id>/checkpoint-<step>/`. Each checkpoint's metadata
  JSON records step, epoch, wall-clock time, git commit hash, fully resolved config, dataset split
  manifest hash, seed, exact library versions. `latest.json` always points at the newest
  full-resumable checkpoint.
- **Resume on RunPod:** all paths point to the persistent volume by construction (§4), so nothing
  needed for resume lives only on ephemeral container disk. `launch_resumable.sh` checks
  `latest.json` on start, passes `--resume_from_checkpoint` if present. Training launched under
  `nohup`/`tmux` with output to a persistent-volume log file, so VS Code SSH disconnects don't kill
  training. **Recommend deliberately killing training early once and verifying resume actually
  works**, before committing to a long run.
- **Logging:** **TensorBoard required baseline** (fully offline, no network dependency, event files
  on persistent volume, inspectable via VS Code extension or port-forwarding). **W&B optional
  supplement** (via Accelerate's multi-tracker support) for cross-run comparison during rank/alpha/LR
  sweeps and remote progress checks — not a replacement for TensorBoard, since it needs network
  egress/API key and an external account-linked history.

## 11. Expected outputs (handoff to Stage 2)

- Final LoRA adapter weights (UNet-only safetensors) + manifest: base SDXL revision hash, LoRA
  config, step count, split manifest reference, library/version pins.
- `caption_builder.py` as an **importable module**, not a one-off script — Stage 2 must call the
  identical function to build generation prompts; any drift from train-time phrasing degrades
  conditioning fidelity.
- Patient-level split manifests, especially `classifier_heldout.csv`, intact and untouched by
  Stage 1.
- Training logs/metrics archive (loss curves, probe sample grids, FID/CLIP-score trends) as a
  durable record for the thesis write-up.
- `requirements-lock.txt` and the exact resolved config YAML for the final run.

## 12. Risks and alternatives

| Risk | Mitigation |
|---|---|
| OOM despite headroom planning | Priority order: reduce batch size → increase grad accumulation → drop resolution 768→512 → more aggressive gradient checkpointing. Latent caching already reduces peak memory. |
| Caption-template shortcut learning (e.g. burned-in L/R film markers) | Not Stage 1's job to solve (that's Rehman et al./Stage 3 Grad-CAM's role) — Stage 1 preserves enough provenance for Stage 3 to trace any suspicious correlation. |
| CheXpert label noise (NLP-extracted, not human-verified) | Propagates into caption noise by construction; accepted, documented limitation. Human-curated `valid.csv` preserved untouched for future use needing cleaner labels. |
| Disk exhaustion on RunPod | Budgeted ~60–100GB realistic use (§4); provision 150–200GB volume; checkpoint retention capped; latent caching is a disk-vs-compute toggle, disable if tight. |
| Pod interruption mid-training | Frequent cheap checkpoints, all paths on persistent volume, nohup/tmux survival, deliberate resume test before long runs (§10). |
| xformers/torch mismatch | Avoided by design — SDPA default instead of a cu128-fragile xformers wheel. |
| bitsandbytes/CUDA 12.8 compatibility | Smoke-test at setup; plain AdamW fallback is low-cost since LoRA's trainable param count is small. |
| diffusers/PEFT/Accelerate API churn | Pin exact versions after Day-1 smoke test, freeze via `requirements-lock.txt` rather than trusting memorized version numbers. |

**Higher-level alternatives rejected:**
- **Full UNet fine-tuning:** contradicts thesis title (LoRA specified) and Rehman et al.'s
  motivating memory-efficiency finding; higher overfitting risk on templated captions.
- **DreamBooth-style fine-tuning:** designed for few-shot single-concept personalization, not a
  large multi-label dataset needing fine-grained multi-attribute control.
- **ControlNet/structural conditioning:** valuable per the literature, but CheXpert has no
  segmentation masks (would need a separate lung-segmentation sub-project) — deferred, revisit if
  Stage 3 reveals systematic anatomical inconsistency LoRA-only can't fix.
- **CXR-domain-pretrained backbone** (e.g. RoentGen-style) instead of vanilla SDXL: real
  alternative, but rejected as primary since the thesis commits to SDXL and no widely validated
  SDXL-native CXR checkpoint is confidently known to exist; kept as a possible future ablation.

---

## Critical files to be created

- `scripts/data/02_build_patient_splits.py` — patient-level split logic guarding Stage 5's
  eventual real-vs-synthetic validity; must be built and verified first.
- `scripts/utils/caption_builder.py` — importable captioning module reused identically by Stage 1
  (training) and Stage 2 (generation prompts); highest-leverage correctness point for
  train/generation consistency.
- `scripts/train/train_lora_sdxl.py` — core LoRA training loop (diffusers/Accelerate-based),
  encoding the §3/§8 decisions.
- `configs/stage1_lora_sdxl.yaml` — single source of truth for every tunable decision (rank,
  alpha, LR, resolution, batch size, checkpoint cadence).
- `scripts/data/03_preprocess_images.py` — aspect-preserving resize/pad + quality filtering (§6),
  directly protecting diagnostic content (e.g. cardiothoracic ratio) later stages depend on.

## Verification (once implemented)

1. Run `01_verify_download.py` against the extracted Kaggle download; confirm row counts and label
   schema match expectations.
2. Run `02_build_patient_splits.py`; confirm no patient ID appears in more than one split, and
   `classifier_heldout` size matches the target ~20%.
3. Run `03_preprocess_images.py` and `04_generate_captions.py` on a small sample (~100 images);
   manually inspect a handful of processed images (no stretch distortion, no lost anatomy at
   letterbox edges) and captions (correct finding clause, correct -1 omission, correct device
   clause).
4. Day-1 smoke test: run `train_lora_sdxl.py` for a few dozen steps on a small subset; confirm no
   OOM, no NaNs, checkpoint save succeeds, and `pip freeze` gets locked to `requirements-lock.txt`.
5. Deliberately kill the smoke-test run and restart via `launch_resumable.sh`; confirm training
   resumes from `latest.json` with matching step/loss continuity — validates the RunPod-resume
   requirement before any long run is trusted.
6. After a longer run begins accumulating checkpoints, inspect the fixed-seed probe grids across
   checkpoints for the informal directional-conditioning sanity check (§9.3).

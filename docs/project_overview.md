# Project overview

A single reference for the whole pipeline as it currently stands, including the four
literature-driven changes (C1–C4). For the design rationale and citations see
`literature_review.md`; for the staged runtime plan see `stages2_to_5_plan.md`.

---

## 1. One-paragraph summary

Fine-tune a diffusion foundation model on chest X-rays, generate label-conditioned synthetic
images, then **select** which of those synthetic images to add to a real training set so that a
multi-label disease classifier improves — fairly across classes, and demonstrably because of
*which* images were chosen rather than *how many*. The selection stage (ASISM) scores every
synthetic image on six quality signals, learns a task-grounded utility from measured downstream
AUROC changes, ranks images, and applies a class-specific admission threshold chosen on
proxy evidence only. `final_eval_heldout` is opened once, for the final Stage 4/5 comparison.

---

## 2. Data splits (frozen)

Six patient-disjoint splits, built by `scripts/data/02b_build_sixway_splits.py`:

| Split | Used for |
|---|---|
| `gen_train`, `gen_val` | Stage 1 LoRA fine-tuning |
| `classifier_train` | real training data for every proxy and real classifier |
| `classifier_val` | model selection during classifier training |
| `asism_tuning_heldout` | **all** Stage 3 tuning evidence (proxy AUROC, threshold policy choice) |
| `final_eval_heldout` | opened once, Stage 4/5 only — never by ASISM |

**Frozen support rule:** every primary-endpoint label has ≥ 50 positive and ≥ 50 negative
patients in every split where it drives a decision. Patient IDs never cross splits.

---

## 3. Stages

### Stage 1 — generation model
`scripts/train/train_lora_sdxl.py`. SDXL fine-tuned with LoRA on CheXpert-derived captions
(`scripts/data/04_generate_captions.py`), bf16, gradient checkpointing, 512-resolution training.
Output: a frozen LoRA adapter.

### Stage 2 — synthetic generation
`scripts/generate/01_sample_label_recipes.py` → `02_generate_synthetic_images.py`. Label recipes
are sampled under **rarity-aware per-class quotas**, single- and multi-label compositions,
co-occurrence constraints, and a cap on positives per recipe. Generation uses DPM-Solver
multistep, ~40 steps, guidance ≈ 7. Output: a synthetic image pool (~5k in the production
namespace) plus a manifest with each image's `intended_label_vector`.

### Stage 3 — ASISM (this thesis)

`scripts/asism/01`–`09`.

#### 3.1 Six per-image signals (`01_compute_signals.py`)

| # | Signal | Method | Feature columns |
|---|---|---|---|
| 1 | **Similarity** | DINOv2 ViT-S/14 embeddings; k-NN (k = 15) vs. a per-label real reference pool; hierarchical fallback when exact-match references are scarce | `similarity_knn_mean`, `similarity_top1`, `similarity_topk_spread` |
| 2 | **Image quality** | blank / clipping / low-contrast / uniform-border / blur (Laplacian) checks | `iqa_composite`, `iqa_sharpness`, `iqa_contrast_std` |
| 3 | **Uncertainty** | auxiliary DenseNet121 + MC-Dropout, 20 passes; predictive std over the primary labels; moderate uncertainty is *not* penalised | `uncertainty_mean_std` |
| 4 | **Explainability** | Grad-CAM from the auxiliary classifier vs. empirically derived expected pathology regions | `explainability_region_overlap` |
| 5 | **Agreement** | mean P(intended positives) − penalty · mean P(confident *unintended* positives) | `agreement_score` |
| 6 | **Distinctiveness** | within-class k-NN redundancy (k = 10); `1 − mean top-k similarity` | *(design signal; its dedicated feature column is the piece still being wired into the learned feature set — the learned ranker currently consumes the 9 columns of signals 1–5)* |

A **near-duplicate / memorisation flag** fires when `similarity_top1 ≥ 0.95`; such images get no
novelty credit and are rejected (`reject_near_duplicates`). The 0.95 cut-off is grounded in
Dar et al., *Nature Biomedical Engineering* 2025 (**C3**).

#### 3.2 Go/No-Go gate (`02_gonogo.py`)

Each signal must pass checks on missingness, distinct-value count, a directionality probe,
redundancy correlation with other signals, and reproducibility. Verdict per signal:
`include` (model input), `ablation_only` (kept for ablation, excluded from the frozen selector),
or `exclude` (dropped). Frozen in `gonogo_report.json` / `asism_frozen_manifest.json`.

#### 3.3 Subset-utility measurement (`04` + `04b`, GPU)

The candidate pool is split into **train-role** and **val-role** image pools that are disjoint by
construction (`split_image_pool`). About 120 controlled subsets (random / single-signal / mixed,
sizes 100–250) are built, and each subset's **real downstream utility** is measured as
`augmented_macro_AUROC − real_only_macro_AUROC` — a proxy classifier trained on
`classifier_train` + the subset, evaluated on `asism_tuning_heldout` only. After an exposure
filter (each image in ≥ 3 subsets), ≈ 96 usable `(subset, utility)` pairs remain.

#### 3.4 Learned utility and ranking (`05_train_learned_asism.py`)

- **Set-Utility Network** — permutation-invariant Deep Sets, trained on the measured subset
  utilities (train-role subsets, early-stopped on val-role). Default: masked mean pooling.
  **C1 (ablation flag, default off):** masked mean *and* std pooling plus a fixed full-pool
  summary concatenated to the utility head (Xie et al., ICLR 2024).
- **Target generation** — per-image supervision for the ranker is the size-normalised
  leave-one-out marginal utility `(U(S) − U(S∖i))·(n−1)` by default. A **Data-Banzhaf MSR**
  estimator (`banzhaf_msr_targets`) — mean utility of subsets containing an image minus mean
  utility of subsets not containing it — is computed from the same measured subsets at no extra
  GPU cost; the LOO-vs-Banzhaf Spearman correlation is recorded in the training manifest, and
  Banzhaf stays opt-in until that diagnostic is reviewed.
  **C2 (default on):** the active target is transformed before the loss —
  `target_normalization ∈ {"standardize", "rank", "none"}` (default `"standardize"`), plus an
  optional `target_winsorize_quantile` — so one noisy proxy measurement cannot dominate training
  on ≈ 96 points. All transforms are order-preserving. `"none"` restores the prior behaviour.
- **Multi-Signal Utility Ranking Network** — small MLP, `9 → 128 → 64 → 32 → 1`, one scalar
  utility target, optimised by Smooth-L1 + 0.5 · pairwise-ranking loss. It is *not* a
  multi-objective / Pareto model.

#### 3.5 Adaptive class threshold (`06`, `07`/`07b`, `08`/`08b`, `09`)

- `06_learn_thresholds_select.py` — a fixed-target-ratio learned threshold; an internal ablation
  reference, **not** a Stage 4 condition.
- `07`/`07b` — per-class bootstrap **contexts** (resamples of one image pool, honestly tagged
  `independent_clinical_sample: False`), each a 10-dim vector (score-distribution stats + real
  patient prevalence + budget). A critic-guided hard-threshold grid search proposes candidates;
  `07b` (GPU) proxy-verifies a *diversified* subset of the grid (not the critic's top-k alone).
- `08_train_threshold_network.py` — **Adaptive Threshold Network**: 16-dim class embedding +
  context → 64 → 32 → 1 → sigmoid. Trained by supervised distillation on proxy-verified targets
  only. **Three-tier per-class governance** (`determine_per_class_official_method`):

  | Verified evidence for a class | Threshold source |
  |---|---|
  | ≥ 3 proxy-verified contexts on **both** train and image-disjoint held-out sides, and frozen acceptance criteria pass | learned Adaptive Threshold Network |
  | 1–2 verified contexts | median best proxy-verified threshold |
  | 0 verified contexts | fixed baseline threshold |

- `08b` (GPU) — for each candidate **threshold policy** it builds the final multi-label
  `selected_manifest`, trains real + selection, and measures utility on `asism_tuning_heldout`
  over several seeds. Policies: fixed baseline, hard-proxy-best, adaptive network,
  `literal_top_50_percent`, and `freematch_style_adaptive_percentile`.
  **C4:** the FreeMatch-style policy scales per-class leniency by *range-normalised* real
  prevalence (rarest present class → most lenient), and `enforce_per_class_selection_floor`
  lowers a class's threshold after thresholding if it would otherwise admit fewer than
  `min_selected_per_label` — the one-time-curation analogue of SST's class-fairness term
  (Zhao et al., IP&M 2025).
- `09_finalize_learned_selection.py` — applies the pre-registered tie rule
  (`tie_noise_band` + `simplicity_order`) to the `08b` measurements, commits to **one** winning
  policy, and writes the frozen `adaptive_selected_manifest.jsonl`. It never opens
  `final_eval_heldout`.

### Stages 4–5 — final comparison
`scripts/classify/01_train_conditions.py`, `scripts/eval/stage5_evaluate.py`. Real classifiers,
equal optimiser-step budget across all conditions and seeds, model selection on `classifier_val`,
final metric on `final_eval_heldout`:

| Condition | Training data |
|---|---|
| **A** | real only (`classifier_train`) |
| **B** | real + **all** Stage 2 synthetic images |
| **F** | real + ASISM-selected synthetic (`adaptive_selected_manifest`) |
| **D** | real + a random synthetic subset with **F's per-class size and composition** |

Primary comparison **C vs B**; **D** isolates selection quality from subset size. Confirmatory
tests use Holm–Bonferroni; exploratory use Benjamini–Hochberg.

---

## 4. The four literature-driven changes (C1–C4)

| ID | Change | Paper | State |
|---|---|---|---|
| **C1** | Set-Utility Network: optional mean+std pooling + full-pool summary | Xie et al., ICLR 2024 | flag `superset_conditioning`, **default off**, ablation only |
| **C2** | Ranking-target normalization (`standardize` / `rank` / `none`) + winsorize | 2D-OOB (NeurIPS 2024); Chi et al. (ICML 2026); Banzhaf robustness | **default `standardize`** — the one change to the frozen path; `"none"` reverts |
| **C3** | `near_duplicate_similarity = 0.95` grounded + pinned by a test | Dar et al., Nature Biomed. Eng. 2025 | documentation + test, **no logic change** |
| **C4** | FreeMatch-style policy: range-normalised leniency + per-class selection floor | SST, IP&M 2025 | active **inside the FreeMatch policy only** (`08b`/`09`) |

C5 (importance-weighted redundancy, InfoMax ICLR 2025) and C6 (inter-class similarity margin,
CosSIF CiBM 2024) are designed but not implemented; they would be ablation flags, default off.

---

## 5. Invariants the design enforces

- **Image-disjoint everywhere.** Train-role vs. val-role subsets share no images; threshold
  contexts come from one pool only.
- **Proxy-only tuning.** Every Stage 3 decision uses `asism_tuning_heldout`; `final_eval_heldout`
  is opened once, for Stage 4/5.
- **No pseudo-replication.** The ranker is never trained on a subset AUROC copied onto its member
  images; it is distilled from the Set-Utility Network's measured set-level utility.
- **Frozen manifests + hashes** at every step; a config change alters the config hash and forces
  a re-run from the affected stage forward.
- **Conservative governance.** The learned threshold network is used for a class only with
  sufficient verified, image-disjoint evidence *and* passing pre-registered acceptance criteria;
  otherwise the pipeline falls back.
- **One winning policy.** The fixed / FreeMatch / learned threshold policies do not each become a
  final-evaluation condition; proxy evidence selects one, which becomes Condition C.

---

## 6. Test and run status

- `python tests/run_all.py` — 186 passed / 0 failed (includes all C1–C4 tests).
- `python scripts/smoke/run_smoke_pipeline.py --phase local` — passes on CPU.
- Full smoke and the real `01`→`09` sequence need a GPU (Stage 1 SDXL, and the `--phase run`
  proxy-training steps `04b` / `07b` / `08b`).

# HAM10000 ASISM v2 — experimental contract

Written 2026-10-02, before any ASISM v2 component exists. This document is the single reference for
every parameter that can affect which synthetic images are selected, how many, and how the result is
judged. It records:

- each parameter's current value;
- where it lives in the repository;
- its status.

A value used by v2 that is not traceable to a line here is a contract violation.

**Statuses**

- **FROZEN**: inherited from v1 and unchanged. It is changed only by a dated amendment that gives
  the reason, written before the run it affects.
- **FIXED**: set by an amendment in `docs/ham10000_v2_signal_criteria.md`.
- **PENDING**: not decided. The owner and the step that needs the decision are named. Nothing that
  depends on a PENDING item may run until it is decided and written here.

**Rules that apply to everything below**

1. A threshold is written before the run that it judges. A failed criterion is never relaxed
   afterwards. A change is a new dated amendment, with its reason, before the next run.
2. No parameter is chosen because of a final-evaluation result. No parameter is chosen because of a
   `classifier_val` result except where this contract names `classifier_val` as the deciding split.
3. Historical artifacts (v1, aux v1/v2/V3a, P1, P1b) are never rewritten. New outputs go to new
   directories.
4. Each change has its own branch, with the code and its tests in separate commits.

---

## 1. Dataset

| Item | Value | Source | Status |
|---|---|---|---|
| Dataset | HAM10000: 10,015 images, 7,470 lesions, 7 classes | `configs/dataset_ham10000.yaml` | FROZEN |
| Classes, in order | nv, mel, bkl, bcc, akiec, vasc, df | `CLASSIFIER_TARGET_LABELS` (`scripts/utils/ham10000.py`) | FROZEN |
| Label model | single-label, softmax + cross-entropy | `configs/ham10000_stage4.yaml: loss` | FROZEN |

## 2. Splits and their permitted uses

Splits are **lesion-level** (HAM10000 has no patient id). They are stratified by diagnosis, use seed
42, and are named `ham-stratified-v1` (`configs/splits_ham10000.yaml: split_seed`). All 30 pairwise
leakage checks are zero, and the split files are md5-verified on the pod.

| Split | Images | Permitted uses in v2 | Status |
|---|---|---|---|
| `gen_train` | 3,586 | LoRA training; similarity reference; Grad-CAM reference; IQA calibration | FROZEN |
| `gen_val` | 388 | Stage 1 monitoring only | FROZEN |
| `classifier_train` | 1,641 | Real part of every classifier; auxiliary classifier training; agreement-judge training (see §6) | FROZEN, plus the judge use (PENDING, §6) |
| `classifier_val` | 1,401 | Auxiliary-classifier and judge acceptance; Stage 4 monitoring. **Never used to select or rank synthetic images, or to tune any ASISM parameter.** | FROZEN |
| `asism_tuning_heldout` | 1,377 | Utility labels only | FROZEN |
| `final_eval_heldout` | 1,622 | Final evaluation only, behind `assert_final_eval_access_allowed` (`scripts/utils/splits.py`) | FROZEN; read once already (§12) |

## 3. Seeds

| Use | Seed | Source | Status |
|---|---|---|---|
| Split construction | 42 | `splits_ham10000.yaml: split_seed` | FROZEN |
| Generation base seed | 20260917; per image `sha256(seed:recipe_id)` | `ham10000_stage2.yaml: seed` | FROZEN |
| Auxiliary classifier / CAM model | 42 | `ham10000_stage3.yaml: auxiliary_classifier_v3.seed` | FROZEN |
| V3a fold assignment | 42 | `auxiliary_classifier_v3.checkpoint_selection.fold_seed` | FROZEN |
| v1 utility subsets | 42 | `ham10000_stage3.yaml: subset_design.seed` | FROZEN for v1 |
| Stage 4 classifier | 42, 43, 44 | `ham10000_stage4.yaml: seeds` | FROZEN for v1. The v2 count is PENDING (§11). |
| Bootstrap | 42 | `scripts/eval/ham10000_compare_conditions.py: run(seed=42)` | FROZEN |
| v2 utility repeats | training seeds 42–46; role, design and fit seed 42 | `configs/ham10000_asism_v2_ranker.yaml` | APPROVED 2026-10-03 (§8) |

Training is not bit-deterministic: `cudnn.deterministic` is never set (`scripts/utils/classifier.py:136`).
Results are reproduced as distributions, not bit for bit. This is a recorded limitation, not
something to be fixed in v2. Fixing it would change the recipe that every existing result was
produced with.

## 4. Generation (Stages 1–2)

| Item | Value | Source | Status |
|---|---|---|---|
| Base model | SDXL base 1.0, revision `462165984030…` | `ham10000_stage1.yaml` | FROZEN |
| LoRA | rank 32, alpha 32, 8,000 steps, 768×576, LR 1e-4 cosine | `ham10000_stage1.yaml` | FROZEN |
| Checkpoint | `ham-lora-v1/final` | Stage 1 output | FROZEN |
| Sampling | 40 steps, guidance 7.0, DPMSolverMultistep, 768×576 | `ham10000_stage2.yaml` | FROZEN |
| Quotas | base 400, rarity exponent 0.5, [100, 1,500] per class | `ham10000_stage2.yaml` | FROZEN |
| Candidate pool | 3,168 images, `all_candidates.csv` sha256 `bf8047b5…` | `outputs/ham10000/stage2/ham-stratified-v1/` | FROZEN |

v2 generates no new images. Per class, the pool is the upper bound on any count that v2 selects:
nv 112, mel 276, bkl 273, bcc 400, akiec 486, vasc 736, df 885.

## 5. Preprocessing

| Item | Value | Status |
|---|---|---|
| Real images | RGB; letterbox to 512×512, grey pad (128,128,128); content box stored per image; JPEG quality 95 | FROZEN |
| Synthetic images | through the same function | FROZEN |
| Classifier input | 512 px, scaled to [−1, 1], no augmentation in Stage 4 | FROZEN |
| DINOv2 input | the timm transform resolved from the pinned model | FROZEN |

## 6. Signals

| Signal | Definition and fixed parameters | Source model | Status |
|---|---|---|---|
| Similarity | DINOv2 `vit_small_patch14_dinov2.lvd142m`, revision `936966a8…`; k-NN, k = 15, against `gen_train` of the same class; near-duplicate if similarity ≥ 0.95 | DINOv2, frozen | FROZEN |
| IQA | 5 defect flags plus continuous sharpness and contrast. Blur threshold 17.53, calibrated on `gen_train`; border flag measured on the content box at the inherited constant 0.30 (border calibration disabled, `ham10000_stage3.yaml`; the artifact records `inherited_constant_uncalibrated`). Corrected 2026-10-02: this row earlier read "border threshold 0.329, calibrated", which no artifact used | — | FROZEN. Role in v2: **ranking input and safety filter** (decision of Walaa, 2026-10-03, superseding the 2026-09-27 "IQA: keep, but as a safety filter"). As a ranking input, `iqa_composite` is one of the four features of the learned ranker, which learns its class-specific weight and direction. As a safety filter, `safety.reject_invalid_iqa` (§10) still removes undecodable images first; the quality flags are not safety criteria (E4 D2, approved 2026-10-02). Clarified 2026-10-02: this cell earlier read "used as a safety filter (S2 in the audit)"; no audit document defines an item S2 for IQA |
| Uncertainty | MC dropout, 20 passes; mutual information normalised by ln 7 | V3a (`6849c456…`) | FIXED (passed Q3) |
| Explainability | Grad-CAM for the intended class; two-sided conformal typicality against the `gen_train` reference of the same class | V3a, reference in `stage3_aux_v3/` | FIXED (passed Q4) |
| Agreement | `P(intended) − 0.5 · P(best rival)` if the rival's probability is ≥ 0.5, else `P(intended)` (`compute_agreement_scores`, `rival_confidence_threshold`, `penalty_weight`) | **the agreement judge** | Formula FROZEN; **judge PENDING** |

**Agreement judge (PENDING, owner: Walaa, needed by Step 3).** The V3a DenseNet fails Q1 and Q2.
The P1b DINOv2 probe fails Q1. Neither is accepted.

- Step 3 is an amendment written before any new judge is scored. It names **one** candidate, its
  training data and its acceptance criteria.
- If that candidate fails, the agreement work stops (audit stop condition SC1). No second candidate
  is tried against the same synthetic pool.
- Uncertainty and explainability stay with V3a whatever judge is chosen. Under Amendment 5, the
  V3a agreement against V3a uncertainty within class is |ρ| 0.795. A new judge changes that pair,
  so the Go/No-Go is re-run after Step 4.

**Go/No-Go** (`scripts/asism/ham10000_02_gonogo.py`): five hard checks and two diagnostic checks,
with the redundancy threshold at |ρ| < 0.90, computed within class and weighted by class size (the
|ρ| fix, commit `89de88b`). FIXED.

**Signal audit Q1–Q5** (`docs/ham10000_v2_signal_criteria.md`): thresholds as written, applied to
every new judge without change. For how Q5 is to be read, see Amendment 5. FIXED.

## 7. ASISM training and validation data

| Item | Value | Status |
|---|---|---|
| Real part of every utility proxy run | `classifier_train` | FROZEN |
| Utility measured on | `asism_tuning_heldout` | FROZEN |
| Utility train/validation subsets | disjoint subset sets drawn from the candidate pool. v1 used 64/16 with `val_pool_fraction` 0.20 | Design PENDING (§8) |
| Never used for ASISM | `classifier_val`, `final_eval_heldout` | FROZEN |

## 8. Utility target (APPROVED 2026-10-03; no measurement run yet)

Fixed now, as requirements on the v2 design:

- a multi-seed real-only baseline (the v1 single run, 0.5136, against a 4-seed mean of 0.4624);
- repeated measurement of each subset;
- subset size an explicit input;
- sizes that cover the range the quantity mechanism has to decide over, not only 60–180;
- every class's candidates exposed to measurement;
- before any ranking model is trained, report:
  - the within-subset variance and the between-subset variance;
  - the ICC;
  - the reliability at the chosen number of repeats.

**APPROVED (Walaa, 2026-10-03)**, replacing the PENDING list that stood here. The values are in
`configs/ham10000_asism_v2_ranker.yaml`, frozen in `scripts/asism_v2/prereg.py`, and described in
`docs/asism_v2_quantity_design_check_2026-10-03.md` §8 and §9:

- metric: macro AUROC (one-vs-rest) on `asism_tuning_heldout`, absolute;
- sizes: 125, 250, 500, 1,000 (train subsets); 125, 250, 500 (validation and test subsets);
- 120 train, 40 validation and 40 test subsets, image-disjoint by role, class fractions varied, half
  random and half tilted on one signal;
- 5 repeats per subset (training seeds 42 to 46): 1,000 proxy runs, about 4 GPU-hours;
- reliability target (gate G1): reliability of the 5-seed subset means ≥ 0.80, before any fit;
- the multi-seed real-only baseline exists as E4's size-0 cell (10 seeds, Stage 4 recipe); it is reported and does not
  enter the label.

The cheap proxy (224 px, 300 steps) failed its reliability test on random same-size subsets
(ICC 0.108). Written reason for re-using it: those subsets barely differ, the designed ones are built
to differ in size, class mix and signal; its direction agreed with the Stage 4 recipe (Spearman
0.755); and G1 re-tests its reliability on the designed subsets before anything is fitted.

No GPU run is approved by this section. The measurement needs its own approval; until then
`scripts/followup/ham10000_asism_v2_utility.py` refuses its measure phase (`MEASUREMENT_APPROVED`).
Walaa, 2026-10-03: G1 is the check that the labels are repeatable; acceptance criterion (a) in §9 is
the strict gate for which images matter.

**MEASUREMENT APPROVED (Walaa, 2026-10-04).** Asked to write "موافقة على قياس الـ GPU" for the
utility measurement of this section (1,000 proxy runs, about 4 GPU-hours, on RunPod), Walaa answered:
"موافقتى الصريحة." This approves the measure phase and the CPU phases around it (plan, G1, accept,
fit, select, stability, the threshold report). It does not approve Stage 4 of this design or any
Stage 5 run; each needs its own approval. The configuration measured is the one frozen in
`scripts/asism_v2/prereg.py` on this date, with the ranking network of §9 after its one documented
fix. Nothing was measured before this entry. `MEASUREMENT_APPROVED` is set in the commit after this
one. The GPU is an RTX PRO 4500 (the timings in the pod commands were recorded on an RTX 5090); the
whole measurement runs on that one GPU type.

## 9. Ranking network (APPROVED 2026-10-03; not trained on real labels yet)

Name: **Multi-Signal Utility Ranking Network**.

Fixed requirements:

- subset size is represented (for example mean, std and log n, not mean pooling alone);
- a capacity test on a known, noise-free utility function passes before the network is trained on
  real labels;
- validation is on held-out subsets, against a correlation threshold written before training;
- no architecture search. One documented fix is allowed after a failure (audit SC3).

**Set-utility form (Walaa approved the change 2026-10-03; implemented in
`scripts/asism_v2/pipeline.py`, CPU only, no real label used).** The ranker is supervised by measured
subset utility through

    U(S) = b + λ · log(1 + Σ_{i∈S} w_i),   w_i = 1 + (class-specific linear score of the 4 signals) + class term.

`w_i` is an image's effective count. The earlier mean-pooled form (mean score + λ·log(1+n)) is
retired: there an image below the current set mean was predicted to hurt, so a class could be dropped
at the first step and the selected count came from dilution. A CPU check on toy utilities that are not
of the fitted form (all images helpful, some harmful, no benefit, all harmful, classes of different
quality, useless-but-harmless images, a truly mean-pooled utility) is recorded in
`docs/asism_v2_quantity_design_check_2026-10-03.md`, with the failure modes it found.

**APPROVED (Walaa, 2026-10-03).** Loss: mean squared error on the subset means. Fit: Adam, learning
rate 0.03, weight decay 1e-5, full batch, up to 600 epochs, patience 60, starting λ 0.02, seed 42,
200 bootstrap models. Acceptance on the 40 test subsets, read once: (a) Spearman within size ≥ 0.50
with one-sided permutation p ≤ 0.05, and (b) lower test error than the same model with no signals.
If either fails, the learned ranking is not used and the failure is the reported result. No other
architecture or stopping formulation is tried at this point.

**AMENDMENT (Walaa, 2026-10-03, before any utility measurement).** Fit settings only: targets
standardised on the train subsets (mean and SD), weight decay 0, up to 5,000 epochs, patience 200.
Loss, model, Adam, learning rate 0.03, starting λ 0.02, seed 42, the 200 bootstrap models and both
acceptance criteria are unchanged. Reason and evidence (planted data on CPU, no HAM10000 outcome):
`docs/asism_v2_quantity_design_check_2026-10-03.md` §10 and §11.

**AMENDMENT (Walaa, 2026-10-04, before any utility measurement).** The per-image score only. It is
a neural network:

    w_i = 1 + network(x_i, class_i) + class term

- input: the image's four signals (standardised on the train images) and its class, one-hot;
- one hidden layer of 8 tanh units; one output;
- the output layer starts at zero, so every image starts at weight 1, as before.

Before: a class-specific linear score of the four signals. Unchanged: the set-utility form
`U(S) = b + λ·log(1 + Σ w_i)`, the loss, every fit setting of the amendment above, the 200 bootstrap
models, both acceptance criteria and their thresholds, and all of §10. In acceptance criterion (b),
"the same model with no signals" is this network with its four signal inputs held at zero.

Reason: the thesis framework names this component a ranking network, and a linear score has no
hidden layer. This is the one architecture; none other is tried (the "no architecture search"
requirement above). The capacity requirement above is met again for the network before any real
label exists: planted utilities on CPU, including one in which the signals have no effect, recorded
in `docs/asism_v2_quantity_design_check_2026-10-03.md` §14.

The linear score stays in the code and is fitted on the same subsets as a reported baseline
(`linear_additive`), evaluated on the test subsets in the same single read. It does not gate and is
never used to select. **If the network fails acceptance, the learned ranking is not used and that is
the reported result, whatever the linear baseline did (Walaa, 2026-10-04).**

**THE ONE DOCUMENTED FIX (Walaa, 2026-10-04, before any utility measurement).** How the class
enters the network, and nothing else:

    score(x_i, class_i) = output_{class_i}(tanh(hidden(x_i)))

- input: the image's four standardised signals only;
- the same one hidden layer of 8 tanh units;
- one output unit per class; an image's score is the output unit of its own class. Every output
  starts at zero.

Before (the amendment above, same day): the class was a one-hot input next to the signals and there
was one output. Reason: that network failed the capacity requirement on planted data. On the frozen
pool and the 200 designed subsets, with a planted utility in which each class has its own signal
weights and little noise, it left a training error about 50 times the linear score's and reached a
within-size Spearman of 0.770 on the test subsets where the linear score reached 0.990. The cause is
capacity, not overfitting: eight units that receive the class as an added input cannot give seven
classes seven different signal directions. A longer fit or 32 units did not repair it; a class-specific
output did (0.949 with the same hidden layer and about the same number of parameters). What was run,
on planted data only, is recorded in `docs/asism_v2_quantity_design_check_2026-10-03.md` §14.

This is the fix that the "no architecture search" requirement allows after a failure. No further
change to the architecture is made, whatever the planted or the real results are. Everything the
amendment above left unchanged is still unchanged, including what a failed acceptance means.

Documented limitation of this form (design check §4): it is conservative. Helpful images whose weight
is not confidently above 0 are left out, a mostly harmful class can be nearly excluded, harmless
images that add nothing can be kept, and there is no stop for diminishing returns.

## 10. Selection and quantity (LOCKED 2026-10-03)

**Removed for v2** (they stay in the v1 config and code, for reproducing v1):

- `selection.quality_floor_percentile: 25`;
- `selection.target_synthetic_to_real_ratio: 1.0`;
- `selection.min_accepted_per_class: 50` and `max_accepted_per_class: 2000`;
- the v2 follow-up `fill_to_total: 300`.

**Kept:** the safety filters (`safety.reject_invalid_iqa`, `safety.reject_near_duplicates`). They remove a candidate whose image cannot be decoded and scored (`iqa_valid` false) or that is a near-duplicate of a real image (`novelty_is_near_duplicate`). The IQA defect flags (blur, low contrast, border) are not safety criteria (E4 D2, approved 2026-10-02). On the frozen pool they remove 0 of 3,168.

**Requirements:**

- rank within class;
- add images while a pre-set criterion on the estimated marginal utility holds, then stop;
- counts may differ by class;
- record the counts and the stopping trajectory;
- no access to `classifier_val` or `final_eval_heldout`.

**LOCKED (Walaa, 2026-10-03).** Stop a class when the lower 95% confidence bound of the estimated
marginal utility of its next-ranked image reaches 0 or below. ASISM decides both which images and how
many; E4's q* is an independent external check and is not the ASISM count. Implemented in
`scripts/asism_v2/stopping.py`: one-sided 95% bound = 5th percentile over a bootstrap ensemble of the
ranker; the size term must be identifiable, so the supervision has to span several subset sizes.
200 bootstrap models; no minimum-gain threshold; the selection is repeated at fit seeds 42 to 46 and
the per-class counts are reported for each, with the selection at seed 42 the one that is used.

**AMENDMENT (Walaa, 2026-10-03, before any utility measurement).** Offer order only: within a class
the "next-ranked image" is the next by the image's own lower bound (5th percentile across the
bootstrap models of λ·w(x), highest first, ties by image_id), not by the point model's score. The
stopping condition, the quantile, the 200 models and the seeds are unchanged. Reason and evidence
(planted data on CPU, no HAM10000 outcome): `docs/asism_v2_quantity_design_check_2026-10-03.md` §11
and §12.

## 11. Downstream classifier (Stage 4)

| Item | Value | Source | Status |
|---|---|---|---|
| Architecture | DenseNet-121, ImageNet weights, dropout 0.2, 512 px | `ham10000_stage4.yaml: model` | FROZEN |
| Training | 3,000 steps, batch 32, LR 1e-4 constant, weight decay 1e-4, unweighted CE, no augmentation, last checkpoint | `ham10000_stage4.yaml: training` | FROZEN |
| Budget rule | equal optimiser steps across conditions | `ham10000_stage4.yaml` | FROZEN |
| Conditions | A real; B + all 3,168; C + ASISM v2; **D + size-matched random draws of the same per-class counts as C** | `scripts/asism_v2/selection_files.py: draw_matched_random` | DECIDED (Walaa, 2026-10-04): one D draw per Stage 4 seed, from the same safe pool as C |
| Seeds | 20 per condition, 42 to 61 (v1 used 3) | `ham10000_asism_v2_learned_stage4.yaml: seeds`; the number is the power calculation of `docs/ham10000_asism_v2_final_protocol.md` §4 | DECIDED (Walaa, 2026-10-04), before any Stage 4 run of the learned selection |

## 12. Evaluation (Stage 5)

| Item | Value | Source | Status |
|---|---|---|---|
| Primary metric | balanced accuracy | `PRIMARY_METRIC` in `scripts/eval/ham10000_compare_conditions.py` | FROZEN |
| Secondary metrics | macro-F1, macro AUROC (one-vs-rest), accuracy, per-class recall | same file | FROZEN |
| Added after v1 | macro average precision (mean of the one-vs-rest APs); multi-class Brier score (squared distance to the one-hot truth, summed over classes, 0 to 2); top-label ECE, 15 equal-width bins. Reported per condition with the same intervals, and their differences are in the exploratory family. v1 keeps its four metrics, so its recorded comparison re-generates unchanged | `added_metrics` in `scripts/utils/ham10000_conditions.py`; `scripts/utils/ham10000_metrics.py` | DECIDED (Walaa, 2026-10-04: top-label, 15 bins, every protocol after v1), before any v2 Stage 5 result; implemented |
| Intervals | lesion-level bootstrap, 2,000 resamples, seed 42, α 0.05 | `run(n_resamples=2000, seed=42, alpha=0.05)` | FROZEN |
| Multiplicity | Holm on the confirmatory family, Benjamini–Hochberg on the exploratory one | same file | FROZEN |
| Confirmatory comparison | v1: C against B. **v2 proposal: C against D** (selection at matched size) | — | PENDING |
| Test of the quantity | C against D tests which images only, since both have the same counts. How many is read from C against A and C against B (same metrics and intervals as above), and from the count of C set beside E4's quantity curve as an independent check. No new threshold is introduced by this row; whether these comparisons are confirmatory, and with what statistic, is the supervisor's decision | — | DECIDED as the reported comparisons (Walaa, 2026-10-04); statistic PENDING (supervisor) |
| **Test-set policy** | `final_eval_heldout` was read once, for v1 (2026-09-21), and every v2 design choice comes after it. Either (a) re-use it and state that history, or (b) set aside part of an unused split as the v2 test set before any v2 result exists | — | **PENDING (owner: supervisor)** |
| Final manifest | code commit, config hashes, checkpoints (judge, ASISM, Stage 4), selected counts per class, seeds | — | Written before the single final read |

## 13. Open decisions, in the order they are needed

| # | Decision | Needed by | Owner |
|---|---|---|---|
| 1 | Scope: full v2 or a narrowed claim | before E4 (GPU) | supervisor |
| 2 | Agreement judge: one candidate and its criteria | Step 3 | Walaa |
| 3 | E4 design and GPU budget | before E4 | Walaa |
| 4 | Utility proxy, metric, repeats, reliability target | Step 6 | Walaa — decided 2026-10-03 (§8) |
| 5 | Ranking correlation threshold | Step 8 | Walaa — decided 2026-10-03 (§9) |
| 6 | Stopping criterion | Step 9 | Walaa — decided 2026-10-03 (§10) |
| 7 | D draws, number of Stage 4 seeds | Step 11 | Walaa — decided 2026-10-04 (§11): 20 seeds, one D draw per seed |
| 8 | Test-set policy | before Stage 5 | supervisor |
| 9 | ECE bins | Step 10 | Walaa — decided 2026-10-04 (§12): 15 equal-width bins |
| 10 | How the quantity chosen by ASISM is tested | before Stage 5 | Walaa — decided 2026-10-04 (§12); the statistic stays with the supervisor |
| 11 | Ranker architecture: per-image network, one hidden layer of 8; and what a failed acceptance means | before the utility measurement | Walaa — decided 2026-10-04 (§9 amendment); the one documented fix (one output per class) approved the same day |

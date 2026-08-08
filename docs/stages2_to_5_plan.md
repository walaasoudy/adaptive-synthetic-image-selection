# Stages 2–5 Specification: Splits, Generation, ASISM, Classifier Experiments, Evaluation

This document is the **frozen methodology specification** implemented by the Stage 2–5 code in this
repository. Decisions here are fixed, not open. Where a value is configurable, the config key is
named and its **frozen v1 default** is stated.

Style follows `docs/stage1_plan.md`: decision → rationale → alternatives rejected → trade-off.

ASISM (Stage 3) is a **proposed adaptive multi-signal framework designed to investigate whether the
identified synthetic-selection gap can be addressed** (`docs/literature_review.md`: Niemeijer et
al./TSynD leave synthetic-real selection and balancing unresolved). Whether it does is an empirical
question, answered by the pre-specified primary comparison in §8, not asserted here.

---

## 1. Split architecture — FROZEN

### 1.1 Why the previous architecture was replaced

The superseded split (`split_manifest.json`, schema v1) allocated `gen_train` 0.70 + `gen_val` 0.10
+ `classifier_heldout` 0.20 = 100% of patients, with `asism_tuning_heldout` and `final_eval_heldout`
derived as children of `classifier_heldout`. That partition leaves no population for a classifier
development split, and deriving one from `gen_train` would mean the A–E classifiers' early-stopping
and threshold decisions were made on patients the generator itself trained on.

**Replacement: six direct, mutually exclusive, patient-level top-level partitions**, drawn from one
deterministic patient list with one recorded seed. **No split is derived from another.**

| Alternative rejected | Why |
|---|---|
| Carve `classifier_train`/`classifier_val` from `gen_train` | Classifier development data would be patients the generator trained on — generator/classifier development contamination. |
| Reuse `asism_tuning_heldout` as `classifier_val` | Collapses ASISM tuning and classifier model selection onto one population. |
| Reuse `gen_val` as `classifier_val` | `gen_val` is the generator's monitoring split; same contamination argument. |
| Keep v1 and forgo a classifier development split | Forces A–E model selection onto training data (invalid) or a heldout split (leakage). |

### 1.2 Frozen v1 allocation

| Split | Fraction | Role |
|---|---|---|
| `gen_train` | **0.60** | Stage 1 LoRA training; Stage 3 real-reference pool (§4.1); auxiliary classifier training (§2) |
| `gen_val` | **0.10** | Stage 1 monitoring; auxiliary classifier validation (§2) |
| `classifier_train` | **0.15** | Real component of A–E training (§7); real component of ASISM proxy training (§4.7) |
| `classifier_val` | **0.05** | A–E early stopping, checkpoint selection, threshold selection, hyperparameter/model selection — identical rules across all conditions |
| `asism_tuning_heldout` | **0.05** | ASISM Go/No-Go evidence (§4.6) and proxy-search evaluation (§4.7) |
| `final_eval_heldout` | **0.05** | Stage 5 final evaluation only (§8) |

Config: `configs/splits.yaml` → `fractions.*`. Sum asserted == 1.0 at build time.

### 1.3 Support-feasibility rule — FROZEN

Before production splits are frozen, each label in `primary_endpoint_label_set` (§5.2) must satisfy,
**in every split where that label drives a decision or metric** (`classifier_train`,
`classifier_val`, `asism_tuning_heldout`, `final_eval_heldout`):

- **≥ 50 positive patients**, and
- **≥ 50 negative patients**.

**Patient counts, not image counts, determine eligibility.** Image counts are reported as
supplementary information only. Positivity is at patient level: a patient is positive for a label if
any of their studies carries a positive (`1`) value for it.

If the verified full dataset fails this rule under the frozen fractions, the split-freeze command
**stops at the freeze boundary**, emits the complete support report, and proposes the smallest
justified fraction adjustment. Frozen fractions are never silently modified.

### 1.4 Development vs. production namespaces — FROZEN

Split artifacts are namespaced by `split_namespace`:

- **`production`** — requires the source CSV to pass the dataset-integrity contract for the complete
  intended CheXpert-v1.0-small training cohort (`configs/dataset_config.yaml` →
  `source.expected_train_rows`, within tolerance). Written to `splits/production/`. Only these are
  thesis splits.
- **`dev`** — the same 60/10/15/5/5/5 policy applied to the development subset, for pipeline smoke
  testing. Written to `splits/dev/`. Rare-label support insufficiency is **marked in the manifest**
  (`support_check.passed = false`, per-label failures listed). Never called production-frozen.

The two namespaces never overwrite each other. Downstream stages record which namespace they
consumed and refuse to mix.

### 1.5 Superseded artifacts

Schema-v1 artifacts (`splits/*.csv`, `splits/split_manifest.json`) are **retained, not deleted**, and
marked superseded by `splits/SUPERSEDED_v1.md`. No Stage 2–5 code reads them. New manifests carry
`manifest_version: 2` and a `split_namespace` field; a mismatch between an artifact's recorded split
hash and the current frozen manifest is a **hard error** via `validate_upstream_artifact()`.

### 1.6 `final_eval_heldout` access rule — FROZEN

`final_eval_heldout` **is** accessed during the one-time split-construction procedure, for exactly:
deterministic patient assignment; disjointness and exact-partition assertions; the §1.3
positive/negative patient-support check; schema/integrity validation; hashing and manifest creation.

**After the manifest is frozen**, no code may load its images, load its labels for outcome
computation, compute outcome statistics, or make any model/method decision from it before Stage 5.
Before Stage 5, only non-outcome-bearing existence/schema/hash verification is permitted.

This is enforced programmatically by `scripts/utils/splits.py::assert_final_eval_access_allowed()`,
which requires an explicit access-purpose token and refuses outcome-bearing purposes unless a Stage 5
final-evaluation run ID is supplied. The claim "the file is never accessed" would be false and is not
made.

---

## 2. Auxiliary real-only reference classifier — ASISM prerequisite

Stage 3's uncertainty (§4.3), explainability (§4.4), and agreement (§4.5) signals require a
real-data classifier. This is that dependency: trained once, frozen, reused. It is **not** condition
A; condition A is trained separately, from scratch, on `classifier_train` under §7's frozen protocol.

- **Data:** `gen_train` train / `gen_val` validate. Never `classifier_train`, `classifier_val`, or
  any heldout split. Rationale: the generator's own splits are the correct home for a scorer of the
  generator's output, and this leaves the entire classifier development pool unspent for A–E.
- **Initialization:** default **ImageNet-pretrained** DenseNet121. A CXR-pretrained checkpoint is
  permitted only with documented training-set provenance proving no overlap with our evaluation
  patients. CheXpert-pretrained checkpoints are **rejected by default** — most publish no
  patient-level provenance, so overlap with `final_eval_heldout` cannot be excluded. Provenance
  (source, hash, datasets, license) is recorded in the run manifest.
- **Architecture:** DenseNet121 with dropout **activatable at inference**, so MC Dropout (§4.3) is
  genuinely available rather than nominally specified.
- **Label semantics:** identical to §5/§6 — the scorer speaks the same label language as the
  experiments it feeds.

---

## 3. Stage 2 — synthetic generation

**Terminology:** conditioning labels are `intended_label_vector`, never "ground truth." §4.5
measures intent-vs-content agreement; it does not establish clinical truth.

**Recipe validity — auditable, every decision recorded with a reason:**
1. **Empirical co-occurrence** from `gen_train` at or above `min_support_patients`.
2. **Medical-rule overrides**, both directions: an allow-list for clinically plausible combinations
   rare in the data, and a block-list for combinations judged likely CheXpert NLP-extraction noise.
3. **`No Finding` recipes are the all-zero intended vector over the 12 primary disease labels**, and
   are mutually exclusive with any positive pathology.
4. **`Support Devices`** may be carried as a conditioning/context attribute. It does **not** affect
   the primary disease-agreement score (§7) unless `secondary_agreement.enabled` is turned on.
5. No `-1` intent is ever encoded in a recipe.
6. Frontal-only, matching Stage 1's `view_filter`.
7. Age/sex coverage and per-recipe quotas are config-driven, oversampling rare classes subject to
   the support rule.

**Pilot gate — FROZEN:** full production generation **refuses to start** unless a versioned
`pilot_approval_manifest.json` exists, records a completed pilot, records passing automatic artifact
checks, and carries an explicit approval record. Enforced at runtime in
`02_generate_synthetic_images.py`.

**Captions** are built by `scripts/utils/caption_builder.py` verbatim — the same module Stage 1
trained with. Any drift between train-time and generation-time phrasing degrades conditioning
fidelity (`docs/stage1_plan.md` §11).

---

## 4. Stage 3 — ASISM

### 4.0 Artifacts — FROZEN

Five independent versioned parquet artifacts under `data/chexpert/synthetic/scores/`:
`similarity_scores.parquet`, `iqa_scores.parquet`, `uncertainty_scores.parquet`,
**`explainability_scores.parquet`**, `agreement_scores.parquet`. Each carries `image_id`, its
scores, and a schema-version + provenance block. Merged by `image_id` at §4.7 over the signals that
cleared §4.6. Artifacts of excluded-but-valid signals are retained for audit.

### 4.1 Similarity
Hierarchical reference retrieval: exact `intended_label_vector` match → nearest multilabel
neighborhood → pathology prototypes → class-agnostic CXR-realism fallback, so no image goes
unscored. **k-NN aggregation**, not single-nearest-neighbor. A **separate** near-duplicate/
memorization score (high single-NN similarity with low rank-k spread) keeps "realistic" and
"memorized" distinguishable.

### 4.2 IQA
Generic no-reference IQA is a baseline component, not the main signal. Domain sub-scores:
blur/sharpness, clipping, contrast, histogram abnormality, border/padding artifact, anatomical
completeness, grayscale consistency. Whether the generic component adds anything beyond the domain
checks is an ablation, not an assumption.

### 4.3 Uncertainty
**High uncertainty is not equated with low quality.** Bands: very-low (possibly redundant), moderate
(possibly most informative — TSynD's premise), extreme **in combination with** poor
similarity/quality/agreement (the actual OOD flag). Mechanism: MC Dropout on the frozen auxiliary
classifier. Deep Ensembles is a documented upgrade path, not the default.

### 4.4 Explainability
Grad-CAM against the auxiliary classifier, scored by overlap with an expected region. The fixed
central-thorax box is an explicit **baseline**; pathology-specific expected regions are configured
per label. **Region overlap is an explainability-plausibility signal, not proof of diagnostic
correctness** — stated wherever the score is reported.

### 4.5 Intended-label agreement
```
agreement = mean P(intended positive disease labels)
          − penalty × mean P(confidently predicted unintended disease labels)
```
For a **`No Finding` recipe**, agreement is high when predicted probability is low across all 12
primary disease labels. Computed over the 12 primary labels only; `Support Devices` is excluded
unless `secondary_agreement.enabled`. Kept **separate** from uncertainty, similarity, IQA, and
explainability — a confident correct reading and an uncertain correct reading are different facts.

**A proxy label-consistency signal, never clinical proof.** A generator and a classifier can share a
bias and agree while both are wrong.

### 4.6 Go/No-Go gate — FROZEN, applies to all five signals

Each signal is checked for: technical validity; score directionality; numerical stability;
reproducibility; missing/invalid-output rate; redundancy/correlation with other signals; downstream
usefulness. Evidence comes from `asism_tuning_heldout` only.

- **Fails validity/stability/reproducibility** → excluded from the frozen selector; exclusion and
  reason recorded; artifact retained if technically valid.
- **Passes but weak/redundant** → retained as **ablation-only**, not necessarily weighted into the
  selector.
- **A signal is never forced into the selector merely because it was implemented.**
- If a component is excluded, the manifest explicitly records whether the resulting ASISM variant
  remains the primary method or becomes an ablation/alternative.

### 4.7 Tuning — FROZEN data flow and bounded search

**Proxy data flow (frozen):**
- Proxy training data: **`classifier_train` + the actually-constructed candidate synthetic subset.**
  `gen_train` is **not** the real component of proxy training.
- Proxy evaluation data: **`asism_tuning_heldout` only.**
- Fixed optimizer-step budget; **no candidate-specific early stopping.**
- `classifier_val` is **not** used to rank ASISM candidates.
- `final_eval_heldout` is never used.

**Two-stage bounded search:**
- **Stage A, coarse screening** — fixed small trial count, one fixed proxy seed, small step budget.
- **Stage B, shortlist validation** — 3–5 retained configurations, stronger fixed budget,
  pre-declared folds/seeds, simplicity-preferring tie rule.

**Before execution:**
```
total proxy runs = coarse_trials × coarse_folds × coarse_seeds
                 + shortlist_size × shortlist_folds × shortlist_seeds
estimated GPU-hours = total proxy runs × hours_per_proxy_run
```
A configurable compute-budget gate blocks execution if the estimate exceeds budget; the search space
is reduced **before** any results are seen, never after. Search execution resumes at trial
granularity.

### 4.8 Adaptive selection
Ordered policy: (1) absolute **minimum quality floor** over surviving signals, applied to everything
first; (2) per-label **class-specific quota** informed by real prevalence and target synthetic:real
ratio; (3) **min/max accepted count** per class; (4) uncertainty-band awareness. **A poor image is
never accepted merely because its pathology is rare** — rarity changes how many images compete for a
quota, never whether the floor applies.

---

## 5. Label policy — FROZEN

### 5.1 `classifier_target_label_set`
All **14** CheXpert observations. This is the classifier's output space; all 14 are predicted and
reported.

### 5.2 `primary_endpoint_label_set`
The **12** disease labels: the 14 minus `No Finding` and minus `Support Devices`.

`Enlarged Cardiomediastinum`, `Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`,
`Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Pleural Other`,
`Fracture`.

- **`No Finding`** — an absence-of-disease meta-label, not a pathology. Secondary outcome only.
- **`Support Devices`** — a device-presence label, not a disease, and the highest-prevalence, easiest
  label in CheXpert; including it would flatter the macro-average without measuring diagnostic
  performance. Secondary outcome only.

Both remain in the output space; both are excluded from the **primary endpoint**.

### 5.3 Minimum support for reporting
The §1.3 rule (≥50 positive and ≥50 negative **patients**) also governs metric eligibility. A label
below threshold in the evaluating split is excluded from the primary macro-average **by this
pre-specified rule** and reported separately with its support counts. Support is a property of the
split, computed at split-build time — so the rule never requires inspecting model performance.

---

## 6. Real-label uncertainty policy — FROZEN

CheXpert labels are `1` / `0` / `-1` (uncertain) / blank (not mentioned).

**Policy: preserve raw labels; mask `-1` and blank in training loss and in evaluation.**

Three fields stored separately, never collapsed:

| Field | Meaning |
|---|---|
| `raw_label` | Original value, unmodified: `1`, `0`, `-1`, blank |
| `training_target` | Value fed to the loss under the frozen mapping |
| `loss_mask` / `eval_mask` | Per-label boolean: does this label participate |

`-1` is **never** silently converted to `0` or `1`. U-Ones/U-Zeros is a documented alternative, not
the default: it asserts a fact the original radiologist declined to assert. Masking applies
identically to training, validation, threshold selection, final evaluation, and metric computation.
**Effective patient N is reported for every metric.**

---

## 7. Stage 4 — conditions A–E — FROZEN

Architecture, initialization, label policy (§5), uncertainty policy (§6), optimizer family, batch
size, augmentation, checkpoint-selection rule, validation rule, and threshold-selection rule are
**identical across all conditions**. Real component is `classifier_train`; all model selection is on
`classifier_val`.

| Condition | Real | Synthetic |
|---|---|---|
| **A** | `classifier_train` | none |
| **B** | `classifier_train` | all Stage 2 images |
| **C** | `classifier_train` | frozen ASISM selection |
| **D** | `classifier_train` | **exactly 5** independent deterministic matched-random draws |
| **E** | none | all Stage 2 images |

**Condition D matching (frozen):** every draw matches C on exact total synthetic sample count;
per-label marginal positive counts within a frozen tolerance; single-label vs. multi-label
proportion; and relevant sampling/interleaving constraints. Joint label-vector matching is attempted
only where support permits and is never allowed to make matching infeasible. **Residual imbalance is
recorded per draw.**

**Seed policy (frozen):** A, B, C, E use **3 fixed model-training seeds** each; **every** D draw uses
the same 3 seeds. Total runs = (4 conditions × 3 seeds) + (5 draws × 3 seeds) = **27**. Run count and
GPU cost are computed before production execution. Seeds and draws are never reduced after seeing
results.

**Fairness protocol (frozen): equal optimizer steps** across A–E and all D draws — not equal epochs.
At fixed epochs, a larger dataset receives more gradient updates, conflating "more data" with "more
training." Equal steps isolates the data-composition effect that C vs. D exists to measure. Recorded
per run: optimizer steps, epochs, effective dataset size, real sample exposures, synthetic sample
exposures, interleaving/sampling policy, model seed, dataset/draw ID, config hash, checkpoint hash.

---

## 8. Stage 5 — final evaluation — FROZEN

**Primary endpoint:** macro-AUROC over `primary_endpoint_label_set` (§5.2) on `final_eval_heldout`.
**Primary comparison:** **Condition C vs. Condition D** (D as its across-draw distribution).
All other metrics and comparisons are secondary or exploratory.

**Statistics:** patient-level paired bootstrap for effect sizes and 95% CIs. **Holm–Bonferroni** for
the pre-specified confirmatory family; **Benjamini–Hochberg FDR** for exploratory analyses, labelled
as such. **Effect sizes are reported alongside every p-value/CI.** The confirmatory family and
correction configuration are frozen before Stage 4 and are not revised after seeing results.

**Reporting:** per-label AUROC; macro-AUROC (primary); micro-AUROC; AUPRC; sensitivity/specificity;
F1; calibration (Brier, ECE); patient-level bootstrap CIs; corrected paired comparisons with effect
sizes; **effective patient N after masking for every metric**; `No Finding` and `Support Devices`
reported separately; condition D's full across-draw distribution; leave-one-signal-out ASISM
component-contribution analysis.

**Execution guards (frozen).** Stage 5 refuses to run unless: the production split manifest is
frozen; the final split hash matches; the frozen experiment-protocol manifest exists; the ASISM
freeze manifest exists; all required A–E/D checkpoint hashes exist; the classification-threshold
policy is frozen; and an explicit final-evaluation run ID is supplied.

**Technical resume vs. methodological re-evaluation.** Predictions are written incrementally and
resumably; a technical interruption may resume from the same frozen artifacts and complete the same
partial prediction files without altering the protocol. Aggregate results must never be used to
modify ASISM, classifier selection, thresholds, or the protocol. Once any final-evaluation result
has been inspected, the frozen methodology cannot be changed and the same evaluation cannot be re-run
for methodological reasons. A genuine methodological error requires returning to the relevant
development stage, documenting the change, re-freezing affected artifacts, and treating the result as
a **new pre-specified evaluation run** — reported alongside the original, never silently replacing it.

---

## 9. Execution order

1. Build six-way splits; run the §1.3 support check.
2. Validate and freeze the versioned split manifest.
3. Stage 1, using `gen_train`/`gen_val` only.
4. Stage 2 pilot generation → pilot approval manifest.
5. Stage 2 full generation (gated on §3's pilot gate).
6. Auxiliary classifier (§2), on `gen_train`/`gen_val`.
7. Compute the five ASISM signals (§4.1–§4.5).
8. Go/No-Go gate (§4.6).
9. Bounded proxy search (§4.7) on `classifier_train` + candidates, evaluated on
   `asism_tuning_heldout`.
10. Freeze ASISM configuration.
11. Freeze the experiment protocol (§5, §6, §7, §8).
12. Train A–E (27 runs).
13. Stage 5 evaluation on `final_eval_heldout`.
14. Patient-level statistics, corrections, tables, figures.

---

## 10. Leakage rules (consolidated)

- `final_eval_heldout` — split construction (§1.6) and Stage 5 only. Never ASISM tuning, Go/No-Go,
  proxy search, hyperparameter tuning, early stopping, checkpoint selection, threshold selection,
  model selection, label-exclusion decisions, or search-space reduction.
- `asism_tuning_heldout` — ASISM Go/No-Go and proxy-search evaluation only.
- `classifier_train` — A–E real component and ASISM proxy real component.
- `classifier_val` — A–E model selection only; never ASISM candidate ranking; never trained on.
- `gen_train` / `gen_val` — Stage 1, auxiliary classifier, ASISM reference pool. Never A–E model
  selection, never evaluation.
- All six splits are patient-disjoint by construction from one deterministic list and one seed.
- Stage 5 bootstrap resampling is patient-level.

---

## 11. Validation status vocabulary

A stage carries one or more of: **implemented**; **locally smoke-tested**; **GPU-ready**;
**production-executed**. Fixtures are not scientific validation. Where the LoRA checkpoint,
generated images, trained classifiers, or a GPU are absent, downstream contracts and executable code
are still implemented and tested with explicitly labelled fixtures, and fail cleanly at real upstream
gates. No fake production artifacts are created and no scientific results are claimed.

---

## 12. Engineering vs. proposed contribution

**Engineering, not claimed as research contribution:** config-driven pipelines; atomic, resumable,
idempotent scripts; versioned manifests and hash-checked lineage; patient-level split discipline;
parquet score artifacts; programmatic leakage guards.

**Proposed research contribution:** an adaptive multi-signal framework — similarity with explicit
memorization awareness (§4.1), multi-faceted technical quality (§4.2), uncertainty treated as
potential informativeness rather than a pure quality penalty (§4.3), explainability plausibility
(§4.4), and intended-label agreement as a proxy consistency check (§4.5) — admitted to the selector
only through the Go/No-Go gate (§4.6), tuned under a bounded compute-accounted search (§4.7), and
applied through a class-aware quota-and-floor policy (§4.8). Whether this constitutes a genuine
contribution is answered by the §8 primary comparison and the component-contribution analysis, not
asserted here.

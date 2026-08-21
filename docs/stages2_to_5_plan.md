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

### 1.2 Frozen v2 allocation

| Split | Fraction | Role |
|---|---|---|
| `gen_train` | **0.48** | Stage 1 LoRA training; Stage 3 real-reference pool (§4.1); auxiliary classifier training (§2) |
| `gen_val` | **0.10** | Stage 1 monitoring; auxiliary classifier validation (§2) |
| `classifier_train` | **0.15** | Real component of A–E training (§7); real component of ASISM proxy training (§4.7) |
| `classifier_val` | **0.09** | A–E early stopping, checkpoint selection, threshold selection, hyperparameter/model selection — identical rules across all conditions |
| `asism_tuning_heldout` | **0.09** | ASISM Go/No-Go evidence (§4.6) and proxy-search evaluation (§4.7) |
| `final_eval_heldout` | **0.09** | Stage 5 final evaluation only (§8) |

Config: `configs/splits.yaml` → `fractions.*`. Sum asserted == 1.0 at build time.

**v2 revision note (production run against the full CheXpert-v1.0-small cohort, 2026-08-08):**
the original v1 allocation (60/10/15/5/5/5) failed the §1.3 support-feasibility rule for
`Lung Lesion` and `Atelectasis` — their confident-negative patient population is real but thin
enough (≈660 and ≈580 patients dataset-wide, respectively) that a 5% decision-bearing split fell
short of the required 50. Raising `classifier_val`/`asism_tuning_heldout`/`final_eval_heldout` from
5% to 9% each (funded by lowering `gen_train` from 60% to 48%) clears both with margin. A third
failing label, `Pleural Other`, could not be fixed this way — its dataset-wide confident-negative
population is only ≈100 patients, so no split-fraction choice gives four disjoint decision-bearing
splits 50 each. `Pleural Other` is excluded from `primary_endpoint_label_set` instead (§5.2) rather
than distorting the split policy further for a label no fraction choice can support.

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
3. **`No Finding` recipes are the all-zero intended vector over `GENERATION_TARGET_LABELS`**, and
   are mutually exclusive with any positive pathology.
4. **`Support Devices`** may be carried as a conditioning/context attribute. It does **not** affect
   the primary disease-agreement score (§7) unless `secondary_agreement.enabled` is turned on.
5. No `-1` intent is ever encoded in a recipe.
6. Frontal-only, matching Stage 1's `view_filter`.
7. Age/sex coverage and per-recipe quotas are config-driven, oversampling rare classes subject to
   the support rule.

**v3 revision note (2026-08-08):** recipe eligibility (co-occurrence mining, quotas, the
single-label guarantee, and the caption row) is keyed to `GENERATION_TARGET_LABELS`
(`scripts/utils/labels.py`) — `PRIMARY_ENDPOINT_LABELS` plus `INSUFFICIENT_SUPPORT_LABELS` — not
`PRIMARY_ENDPOINT_LABELS` alone. `Pleural Other`'s exclusion from the primary endpoint (§5.2)
reflects real-cohort support scarcity for *evaluation*; it is not a reason to also stop generating
or training on synthetic examples of it. §4.5 agreement scoring remains scoped to
`PRIMARY_ENDPOINT_LABELS` only, unchanged — as does the matched-random marginal-matching logic, if
the §7.1 control is re-enabled.

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
For a **`No Finding` recipe**, agreement is high when predicted probability is low across all 11
primary disease labels. Computed over the 11 primary labels only; `Support Devices` is excluded
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

### 4.9 Learned ASISM extension (v4 revision note, 2026-08-21) — resolves `docs/novelty_target_decision.md`

**This is ASISM as the thesis defines it** — the **five** scoring signals (§4.1 similarity, §4.2
IQA, §4.3 uncertainty, §4.4 explainability, §4.5 intended-label agreement) feeding the two novel
learned components, "Multi-Objective Ranking Network (Novel)" and "Adaptive Threshold Learning
(Novel)". It implements **Option 1** of `docs/novelty_target_decision.md` (pre-registered weakly
supervised set-utility learning).

**v3 revision note (2026-08-21):** §4.7's `weighted_score_baseline` is **no longer part of the
thesis pipeline**. It predates the learned components and was never specified by the thesis, which
defines ASISM as the full module including both learned networks. `03_tune_freeze_select.py` is
retained for reference; no Stage 4 condition consumes its output, and the learned stages no longer
import it (they load candidates via `scripts/asism/candidate_pool.py`).
Pipeline (`scripts/asism/04` through `09`, config: `configs/stage3_asism.yaml` →
`signals.ranking_network` / `signals.threshold_network`):

1. **`04_build_utility_subsets.py` / `04b_evaluate_utility_subsets.py`** — build controlled
   candidate subsets (random, single-signal, mixed compositions) from an image pool split
   train/val-role and disjoint by construction (`split_image_pool`), then measure each subset's real
   downstream utility: `augmented_macro_auroc − real_only_macro_auroc` on the proxy protocol already
   frozen by §4.7 (same `classifier_train`-based proxy training, `asism_tuning_heldout`-only
   evaluation — no new leakage surface).
2. **`05_train_learned_asism.py`** — trains `SetUtilityNetwork` (permutation-invariant Deep Sets) on
   *measured* subset-level utility only, then distills per-image ranking scores
   (`MultiObjectiveRankingNetwork`, trained with a pairwise ranking loss against the distilled
   scores) — never by copying a subset's AUROC onto its member images. This is the mechanism that
   avoids the pseudo-replication problem `novelty_target_decision.md` raised.
3. **`06_learn_thresholds_select.py`** — the fixed-ratio learned threshold. An intermediate stage of
   the pipeline and an internal ablation reference; it is **not** a Stage 4 condition (the former
   condition G was removed — §7 v3 revision note).
4. **`07`/`07b`/`08`/`08b`** — builds bootstrap class contexts (honestly tagged
   `independent_clinical_sample: False` — they are resamples of one candidate pool, not new clinical
   evidence), a critic-guided hard-threshold grid search, proxy-verifies a diversified (not
   critic-only) subset of that grid, and trains `AdaptiveThresholdNetwork` (class-aware, via class
   embedding + context vector) against verified-only targets.
5. **`09_finalize_learned_selection.py`** — per class, `determine_per_class_official_method` picks
   exactly one of three outcomes, never a blend: `fixed_target_ratio_threshold_distillation_baseline_v1`
   (zero verified train contexts for that class), `hard_proxy_best_among_verified` (some verified
   train evidence but below `min_verified_contexts_per_class` on either side), or
   `adaptive_threshold_network` (enough verified evidence on both train and held-out sides **and**
   the frozen acceptance criteria in `configs/stage3_asism.yaml` → `acceptance_criteria` pass).

**Governance status:** implemented but **not yet supervisor-approved** — see the resolution note at
the top of `docs/novelty_target_decision.md`. Not yet run on production data (§11).

**Interaction with §7/§8:** `09`'s output (`adaptive_selected_manifest`) is the selection consumed by
Stage 4 **condition F** — the thesis's "real + selected synthetic" arm, and the only synthetic-
selection condition that survives the §7 v3 revision note. `06`'s `learned_selected_manifest` is
retained on disk as an ablation reference but feeds no Stage 4 condition.

---

## 5. Label policy — FROZEN

### 5.1 `classifier_target_label_set`
All **14** CheXpert observations. This is the classifier's output space; all 14 are predicted and
reported.

### 5.2 `primary_endpoint_label_set`
The **11** disease labels: the 14 minus `No Finding`, `Support Devices`, and `Pleural Other`.

`Enlarged Cardiomediastinum`, `Cardiomegaly`, `Lung Opacity`, `Lung Lesion`, `Edema`,
`Consolidation`, `Pneumonia`, `Atelectasis`, `Pneumothorax`, `Pleural Effusion`, `Fracture`.

- **`No Finding`** — an absence-of-disease meta-label, not a pathology. Secondary outcome only.
- **`Support Devices`** — a device-presence label, not a disease, and the highest-prevalence, easiest
  label in CheXpert; including it would flatter the macro-average without measuring diagnostic
  performance. Secondary outcome only.
- **`Pleural Other`** — **v2 revision note (2026-08-08):** excluded for insufficient support, not for
  a clinical-relevance reason like the two above. On the full production CheXpert cohort only ≈100
  patients dataset-wide carry a confident negative label for it, so no §1.2 fraction choice can give
  every decision-bearing split the §1.3 minimum of 50 negative patients. Reported as a secondary
  outcome with its support counts, per §5.3.

All three remain in the output space; all three are excluded from the **primary endpoint**.

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

## 7. Stage 4 — conditions A/B/F — FROZEN (v3 revision note 2026-08-21)

Architecture, initialization, label policy (§5), uncertainty policy (§6), optimizer family, batch
size, augmentation, checkpoint-selection rule, validation rule, and threshold-selection rule are
**identical across all conditions**. Real component is `classifier_train`; all model selection is on
`classifier_val`.

| Condition | Real | Synthetic |
|---|---|---|
| **A** | `classifier_train` | none |
| **B** | `classifier_train` | all Stage 2 images |
| **F** | `classifier_train` | Learned ASISM selection (§4.9 `adaptive_selected_manifest`) |

**v3 revision note (2026-08-21) — supersedes the v2 note.** The condition set was reduced from A–G
to **A/B/F** to match the three arms the thesis specifies for Stage 5 (real only / real + all
synthetic / real + selected synthetic). Conditions removed and why:

| Removed | Was | Why removed |
|---|---|---|
| **C** | real + weighted-baseline ASISM | The thesis does not specify a weighted ASISM selector; ASISM's Stage-3 definition is the full module including the two learned components. C was an implementation-history artifact, not a thesis arm. |
| **G** | real + fixed-ratio learned baseline | An internal ablation of F's threshold, not a thesis arm. |
| **E** | synthetic only | Not among the thesis's three arms. |
| **D** | matched-random control | Removed as a consequence of removing C (it was matched to C). **See the limitation below — this removal has a scientific cost the other three do not.** |

`configs/stage4_classifier.yaml` → `conditions: [A, B, F]`. Total runs = 3 conditions × 3 seeds =
**9** (superseding the 33 and, before it, 27 stated in earlier revisions).

### 7.1 Known limitation — no matched-random control

**F is a strict subset of B.** An F-over-B improvement therefore admits two explanations this design
cannot separate:

1. **ASISM selected good images** — the thesis claim, and
2. **using fewer synthetic images is simply better**, whichever ones they are, because unselected
   synthetic data is noisy enough that dilution alone helps.

Explanation 2 would produce an F-over-B win even with a *random* selector of the same size. F vs. B
is therefore evidence that *selection helps*, but **not** evidence that *ASISM's ranking is good* —
which is the actual novel contribution. The thesis's three-arm summary is the headline comparison,
not a complete experimental design, and a committee is likely to raise exactly this point.

**Resolving it** requires one added condition: matched-random draws of |F| images sharing F's
per-label profile, trained under the identical protocol. The machinery already exists and is
retained in working order (`01_train_conditions.py`: `build_matched_random_draw`, `profile_of`,
`configs/stage4_classifier.yaml` → `condition_d.*`); it needs re-pointing from the deleted C to F,
plus `n_draws × 3` additional runs (5 draws → 15 runs, total 24). **This is a supervisor decision
and must be settled before Stage 4 executes**, not after results are seen.

**Fairness protocol (frozen): equal optimizer steps** across every enabled condition — not equal
epochs. At fixed epochs, a larger dataset silently receives more gradient updates, conflating "more
data" with "more training." Equal steps isolates the data-composition effect that F vs. B exists to
measure. Recorded per run: optimizer steps, epochs, effective dataset size, real sample exposures,
synthetic sample exposures, interleaving/sampling policy, model seed, dataset ID, config hash,
checkpoint hash.

---

## 8. Stage 5 — final evaluation — FROZEN

**Primary endpoint:** macro-AUROC over `primary_endpoint_label_set` (§5.2) on `final_eval_heldout`.

**Primary comparison: F vs. B** — Learned-ASISM-selected synthetic images vs. all synthetic images
unselected. **Supporting comparisons:** A vs. B and A vs. F. Implemented in
`scripts/eval/compare_conditions.py`:

```
CONFIRMATORY_COMPARISONS = [("F", "B")]
SECONDARY_COMPARISONS    = [("A", "B"), ("A", "F")]
```

Document and code agree. With a single confirmatory comparison, Holm–Bonferroni reduces to the
uncorrected p-value for F vs. B; the correction machinery stays in place so that adding a
confirmatory comparison later (e.g. the matched-random control in §7.1) is a configuration change,
not a reanalysis.

**v3 revision note (2026-08-21) — supersedes the v2 note.** Earlier revisions of this section froze
**C vs. D** as primary and then recorded a conflict with a code-side **F vs. C**. Both are obsolete:
conditions C, D, E and G no longer exist (§7 v3 revision note), so neither comparison is defined.
The conflict is resolved by deletion, not by adjudication.

**What F vs. B can and cannot establish.** It tests whether selecting synthetic images beats using
them all. It does **not** isolate the quality of ASISM's ranking from the effect of simply using
fewer synthetic images — see §7.1. Any Stage 5 write-up must state this limitation explicitly
alongside the F vs. B result, or add the matched-random control described there.

Whether the learned selector (F) is accepted as a thesis method at all remains open in
`docs/novelty_target_decision.md`; this correction family assumes it is.

**Statistics:** patient-level paired bootstrap for effect sizes and 95% CIs. **Holm–Bonferroni** for
the pre-specified confirmatory family; **Benjamini–Hochberg FDR** for exploratory analyses, labelled
as such. **Effect sizes are reported alongside every p-value/CI.** The confirmatory family and
correction configuration are frozen before Stage 4 and are not revised after seeing results.

**Reporting:** per-label AUROC; macro-AUROC (primary); micro-AUROC; AUPRC; sensitivity/specificity;
F1; calibration (Brier, ECE); patient-level bootstrap CIs; corrected paired comparisons with effect
sizes; **effective patient N after masking for every metric**; `No Finding` and `Support Devices`
reported separately; the §7.1 limitation stated alongside the F vs. B result (or, if the
matched-random control is enabled, its full across-draw distribution); leave-one-signal-out ASISM
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
9. Learned ASISM (§4.9): `04` subset design → `04b` measured subset utility → `05` set-utility and
   ranking networks → `06`–`08b` threshold learning and proxy verification → `09` finalize. All
   proxy evaluation is on `asism_tuning_heldout` only.
10. Freeze ASISM configuration (§4.9 learned selector, pending the supervisor sign-off tracked in
    `docs/novelty_target_decision.md`).
11. Freeze the experiment protocol (§5, §6, §7, §8), including §7.1's matched-random-control
    decision, which must be settled before any Stage 4 run.
12. Train A/B/F (9 runs; 24 if the §7.1 matched-random control is added).
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

# HAM10000 ASISM v2: final selection, Stage 4 and evaluation protocol

Written and committed 2026-10-02, **before any E4 result was read**. At the time of writing, 1 of
the 60 E4 runs had finished; its metrics were not opened. Nothing in this document changes after an
E4, Stage 4 or Stage 5 result is seen. This fills the PENDING items of
`docs/ham10000_v2_experimental_contract.md` that the final pipeline needs. For each item it gives
the source in an existing approved document or in the code.

## 1. What is decided elsewhere and only applied here

| Item | Source |
|---|---|
| E4 verdict, q\*, per-class split | `docs/ham10000_e4_quantity_design.md` §4; `docs/ham10000_e4_verdict_consequences.md` V1 and V3 (approved 2026-10-02) |
| Signal set: similarity, IQA, uncertainty, explainability | `docs/ham10000_v2_gonogo_four_signals.md`; roles in §2 |
| Safety filters | contract §10 "Kept"; E4 D2 (`iqa_valid`, near-duplicates) |
| Stage 4 recipe | contract §11 (FROZEN), copied unchanged into `configs/ham10000_asism_v2_stage4.yaml` |
| Primary metric, bootstrap, Holm/BH | contract §12 (FROZEN) |
| Confirmatory pair C vs D | contract §11–12 (v2 proposal), `docs/ham10000_e4_verdict_consequences.md` §5 |

**Per verdict:**

- **COARSE.** Protocol `asism_v2`, with conditions A, B, C and D.
- **NO.** Protocol `asism_v2_none`, with conditions A and B. C = A, and D is not built.
- **GO.** Nothing is selected. E4b is designed first, as its own approved step (V2).

## 2. Within-class ranking (contract §9 / audit decision D-A)

> **Status correction, 2026-10-02 (audit, before any E4 result was read): NOT APPROVED.** The
> composite below is the methodology's named *baseline* (`ham10000_stage3.yaml`: "the weighted
> composite remains as the baseline"; methodology §3.4.1 names a hand-weighted sum as what the
> method avoids). `docs/ham10000_results_and_limitations.md` §12.0.1 reserves C for a
> learned-utility selection and forbids substituting another score under C's name. The record does
> not show option (a) of the CPU audit (SC2 for selection) as chosen: option (c) was. Whether
> "which" uses a repaired learned ranker (a new supervision instrument), SC2 with this composite
> run only as a named baseline, or no C at all is a decision for Walaa and the supervisor. Until it
> is made, C and D are not built from this section.

The learned ranking network (contract §9) is not used. The CPU audit
(`docs/ham10000_e4_cpu_audit.md`) measured the same-size selection ICC at 0.0–0.04. On that basis,
E4 was reduced to quantity only. Measured subset utility therefore cannot supervise a ranking
network: the audit estimated about 387 repeats per subset would be needed. This is audit stop
condition SC2 for learned utility, and it is recorded as a limitation.

The ranking used is the one the methodology already defines and the gate already uses: the
**equal-weight reference composite** (`scripts/asism/ham10000_02_gonogo.py`, `PRIMARY_SCORE_COLUMN`
and `check_usefulness`; `configs/ham10000_stage3.yaml`: "the weighted composite remains as the
baseline"). It is applied to the signals whose v2 role is ranking.

**Role of each admitted signal.** Admission by the gate makes a signal available; it does not give
it a role in the ranking (`docs/ham10000_v2_gonogo_four_signals.md`: the gate "does not start
utility or ranking").

| Signal | v2 role | Source of the role |
|---|---|---|
| similarity | scored, `similarity_knn_mean`, higher is better | `PRIMARY_SCORE_COLUMN`; v2 decision of 2026-09-27 ("keep") |
| explainability | scored, `explainability_calibrated_typicality`, higher is better | `PRIMARY_SCORE_COLUMN`; Q4 passed |
| IQA | **safety filter only** (`safety.reject_invalid_iqa`); not a ranking input | v2 decision of 2026-09-27: "IQA: keep, but as a safety filter", replacing "which image is best?" with "is the image bad enough to reject?"; contract §6 (FROZEN) "used as a safety filter"; the filter's content fixed by E4 D2 (approved 2026-10-02): `iqa_valid` only, the quality flags are not safety criteria |
| uncertainty | not scored | no a-priori direction (`PRIMARY_SCORE_COLUMN`: `None`, "never scored as good or bad"; methodology §3.3); the final pipeline introduces none |

How IQA's history reads, in order:

1. v1 (2026-09-16, `14f196e`) built `iqa_composite` as a ranking feature: defect tiers 0.1 apart
   plus a continuous sharpness/contrast term at weight 0.06, added so the score would clear the
   gate's `min_unique_values`.
2. The v2 decision (2026-09-27) moved IQA to the safety role, because the scores sat at about
   0.96–0.99 and would add little information as a ranking signal.
3. The contract (2026-10-02) froze IQA "used as a safety filter".
4. E4 D2 (2026-10-02) fixed what that filter removes: invalid IQA and near-duplicates, 0 of 3,168
   on the frozen pool. Any stricter quality filter would change the pool E4 measured.
5. The four-signal gate (2026-10-02) admitted IQA as a valid, non-redundant signal. That keeps its
   artifact usable for the safety filter and for reporting; it assigns no ranking role.

No later approved document gives IQA a ranking role. The verdict-consequences document (approved
2026-10-02) assigns the "which" job to "the safety filter, then a ranking within each class" over the
four gated signals, and IQA fills the first of those two parts.

**The ranking.**

- `similarity_knn_mean` and `explainability_calibrated_typicality` are each min-max normalised
  within the class, over the safe pool. The safety filter runs before the ranking.
- The composite is their unweighted mean. A missing composite takes the class minimum. On the frozen
  artifacts no value is missing and no two composites tie within a class.
- Ties are broken by `image_id`, ascending.
- C takes the top n_c of each class, where n_c is the V3 count.
- The selector refuses any other scored set (`SCORED_SIGNALS` in
  `scripts/followup/ham10000_asism_v2_select.py`). The manifest records the scored, safety-only
  and unscored signals.

No weight, column, direction or normalisation is new. No alternative ranking is computed or
compared.

**Limitations, stated with the result.**

- IQA's defect flags are not used anywhere in the v2 selection. A flagged image (blur, low contrast
  or border: 229 of 3,168, mostly df 109 and vasc 62) can be selected when its similarity and
  explainability rank it high. On these images the gate measured IQA's within-class rank
  correlation with similarity at 0.472 (the largest of any pair).
- The gate's usefulness check measured each signal's effect inside a composite that included IQA.
  It is a check for admission, so it is not rerun.
- The 2026-09-27 signal review behind the decision describes IQA as "flags nothing". That is
  inaccurate: the flags fire on 229 images. The role does not depend on it, because the decision
  rests on the score's narrow range, not on the flag count.

**IQA implementation audit (2026-10-02, CPU, no E4 or Stage 4 result used).**

- The frozen artifact (`iqa_scores.parquet`, sha256 `c2288166…`, commit `b6187d5`) matches its
  sidecar. Blur threshold 17.5326 from the `gen_train` calibration (sha256 `f18b7beb…`). Border on
  the content box at the inherited 0.30. No IQA code or config line has changed since `b6187d5`, so
  the artifact is current.
- Every flag and every `iqa_composite` value was rebuilt from the artifact's own measurements:
  0 mismatches in 3,168 rows, maximum composite difference 0.0. Results:
  - blurry 198, low-contrast 4, border 29, near-uniform 0, clipping 0;
  - range 0.688–0.996, higher is better;
  - `iqa_valid` true for all rows, no missing value.
  - A unit test now enforces this reconstruction
    (`test_iqa_composite_is_reproducible_from_its_own_reported_fields`).
- The composite has no 0.25 coefficient. Its fixed terms are:
  - flag penalties 0.40, 0.20, 0.10, 0.10, 0.10;
  - `CONTINUOUS_WEIGHT` 0.06;
  - tanh scales at twice the frozen blur and contrast thresholds.
- Within the clean tier the composite follows sharpness (Spearman 0.948) more than contrast (0.702).
  Sharpness's tanh term spans 0.52–1.00 there, contrast's 0.56–0.79. This is the 2026-09-16 design
  as written. It is not a coding error, and IQA does not rank in v2.
- `quality_floor_percentile: 25` (the P25 floor) is a v1 selection parameter. It applies to the
  ranking score (`ham10000_thresholds.quality_floor`, `ham10000_05_adaptive_thresholds.py`,
  `ham10000_v2_select.py`), never to IQA. It was inherited from the CheXpert `stage3_asism.yaml`.
  Contract §10 removed it for v2. No v2 module or config reads it, and a test guards this.

## 3. Condition D

- For each Stage 4 seed s, D is an independent draw, uniform and without replacement, from each
  class's safe pool.
- It uses C's per-class counts.
- The generator is `numpy.random.default_rng([20261002, s])`.
- One draw per seed, so D's spread includes the variation from random selection as well as from
  training.
- The selection manifest records each draw's sha256 and its overlap with C.

## 4. Number of Stage 4 seeds (contract §11)

**k = 20 per condition, seeds 42–61.** This is fixed from a power calculation on noise estimates
that already exist, before any E4 or Stage 4 result:

- At the full recipe, the real-only balanced-accuracy SD across seeds is 0.066 (4 seeds;
  `docs/ham10000_e4_cpu_audit.md`).
- At the cheaper recipe, the SD is 0.033–0.048.
- The test is a two-sided Welch test at α 0.05 with 80% power on the per-seed BA, so the minimum
  detectable difference ≈ 2.88 · σ · √(2/k).

| k | MDE at σ = 0.04 | MDE at σ = 0.066 | Stage 4 GPU (A/B/C/D, ≈ 18 min per run) |
|---|---|---|---|
| 10 | 0.053 | 0.087 | ≈ 12 h |
| **20** | **0.036** | **0.060** | **≈ 24 h** |

- k = 20 is the largest k whose total GPU cost, with E4, stays well inside the $65 cap.
- An effect smaller than these MDEs can go undetected. This is stated next to every result.
- No seed is added, dropped or replaced after a result.

## 5. Statistics

**Confirmatory (one test, Holm over a family of one).**

- The test is C − D on balanced accuracy.
- It is the frozen lesion-level paired bootstrap: 2,000 resamples, seed 42, two-sided 95%
  (`ham10000_compare_conditions.compare`, unchanged).

**Predeclared seed-robustness analyses, reported next to the confirmatory test and never in its
place** (`scripts/eval/ham10000_seed_robustness.py`). The frozen interval resamples lesions only, so
it does not include training-seed variation. Two analyses add it:

1. a Welch two-sided test on the 20 + 20 per-seed BA values;
2. a seed × lesion bootstrap: lesions are paired and resampled, seeds are resampled per condition,
   and the result is a percentile 95% interval.

If the frozen interval and these analyses disagree, all of them are reported. The confirmatory
statement is then qualified, and is not replaced.

**Exploratory, BH-corrected.**

- B − A, C − A, D − A, C − B and D − B, on every frozen metric.
- Per-class recall for C − D.

**Sources.**

- **Monitoring results on `classifier_val`.** Each Stage 4 run writes its classifier_val
  predictions. The same analysis runs on them (`scripts/followup/ham10000_asism_v2_compare.py
  --source classifier_val`). They are labelled MONITORING and decide nothing. The contract (§2)
  allows classifier_val for Stage 4 monitoring and forbids it only for selecting, ranking or
  tuning. None of those happens here.
- **Final results on `final_eval_heldout`.** These are produced only after the test-set policy
  (contract §12, owner: the supervisor) is signed. Prediction is
  `scripts/eval/ham10000_stage5_evaluate.py --protocol <asism_v2|asism_v2_none>` with a new run id,
  followed by `ham10000_asism_v2_compare --source final`.

AP, Brier score and ECE (contract §12, "to add") are not added. They stay out of scope, so no new
bin count is introduced.

## 6. Order of execution

1. E4 measure, then E4 analyze.
2. `ham10000_e4_consequences`, which writes `e4_consequences.json`.
3. `ham10000_asism_v2_select`. It writes the C and D files and the selection manifest, all hashed.
4. `ham10000_asism_v2_stage4_grid --protocol <...>`. This is resumable, and each run is checked
   against the selection hashes.
5. `ham10000_asism_v2_compare --source classifier_val`, the monitoring results.
6. **Stop.** The test-set policy is signed. Then Stage 5 runs once.

## 7. Integrity

- This document and its code are committed before the E4 metrics are read. The commit time and the
  E4 run timestamps show the order.
- A technically failed run is rerun at the same cell or seed. No run is discarded.

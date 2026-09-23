# HAM10000 — Results and Limitations (run `ham-final-v1`)

Namespace `ham-stratified-v1` · Stage 3 selection at commit `b6187d5` (code identity `9b7f6ba3…`) ·
Stage 2 images generated at `a1ddb26` (code identity `2132f199…`) · Stage 5 read once, on 2026-09-21.

All numbers below come from the saved artifacts. They were recorded as produced. No signal,
threshold, selection rule, configuration or hyperparameter was changed after any result was seen.

---

## 1. Primary result

| Condition | Training data | Balanced accuracy on final_eval_heldout (95% CI) |
|---|---|---|
| A | real `classifier_train` (1,641) | 0.566 [0.523, 0.608] |
| B | real + all 3,168 synthetic | **0.645** [0.595, 0.696] |
| C | real + 616 ASISM-selected synthetic | 0.612 [0.566, 0.658] |

Test set: 1,622 images from 1,194 lesions. Each value is the mean of 3 seeds, with a
lesion-level bootstrap of 2,000 resamples.

**Confirmatory test (pre-registered: C vs B, balanced accuracy):** difference −0.033,
95% CI [−0.063, −0.001], p = 0.047 (Holm, one test).

- The hypothesis that ASISM selection outperforms using all synthetic candidates is **not
  supported**.
- In this configuration, selection performed slightly but significantly *worse* than using
  every candidate.
- The effect is small, and its interval nearly reaches zero.

**Exploratory results (FDR-corrected):**

| Comparison | Difference | Adjusted p |
|---|---|---|
| B vs A, balanced accuracy | +0.079 | < 0.001 |
| C vs A, balanced accuracy | +0.046 | 0.036 |
| C vs A, macro-F1 | +0.052 | 0.009 |
| C vs B, macro-F1 | +0.002 | 0.89 (no difference) |

No per-class C-vs-B recall difference survived correction. The smallest adjusted p-values
were for akiec (−0.093) and df (−0.133), both at 0.081.

---

## 2. Limitations of the comparison design

**2.1 C vs B confounds selection with quantity *and* class composition.**
The two conditions differ in more than which images were chosen. Training-set composition:

| Class | A (real) | B (real + all) | C (real + selected) |
|---|---|---|---|
| nv | 1,085 | 1,197 | 1,135 |
| mel | 186 | 462 | 326 |
| bkl | 188 | 461 | 377 |
| bcc | 87 | 487 | 174 |
| akiec | 49 | 535 | 99 |
| vasc | 21 | 757 | 71 |
| df | 25 | 910 | 75 |
| **Total** | 1,641 | 4,809 | 2,257 |
| nv share | 66% | 25% | 50% |
| akiec + vasc + df share | 6% | 46% | 11% |

- B is a strongly **rebalanced** training set. C is only mildly rebalanced.
- The classes where C falls furthest behind B (akiec, vasc, df) are exactly the classes B
  oversamples most.
- The C-vs-B deficit is therefore at least partly a rebalancing effect, not only a
  selection-quality effect. The present design cannot separate the two.

**2.2 No random-selection control.**
There is no condition that trains on 616 randomly chosen candidates with C's per-class
counts. Without it, the design cannot show whether ASISM chooses *better* images than chance
at equal quantity. This is the most important missing control.

**2.3 The target policy limits augmentation of the rarest classes.**
- Per-class targets were pre-registered as real count × 1.0, clipped to [50, 2,000].
- The rarest classes therefore received targets of only 50: akiec 49, vasc 21 and df 25 were
  all raised to 50.
- The synthetic budget follows the real class distribution instead of correcting it.
- For nv the reverse holds: the target (1,085) far exceeds the available pool (112), so the
  target was unreachable.

---

## 3. Limitations of the learned Stage 3 components

**3.1 The utility signal is weak relative to its noise.**
- Utility labels came from 80 short proxy trainings (224 px, 300 steps), measured on
  `asism_tuning_heldout`.
- Only 32 of the 80 subsets scored above the real-only baseline (balanced accuracy 0.514).
- The differences between subsets are likely comparable to the run-to-run noise of a single
  proxy training. The noise floor was not measured before training the set model.

**3.2 The set model did not generalise.**
The Deep Sets model reached a validation Spearman of **−0.36** on the 16 held-out subsets.
It ranked unseen subsets no better than chance, and in the wrong direction.

**3.3 Sparse exposure.**
- Only 1,438 of the 3,168 candidates appeared in any utility subset.
- 951 of those 1,438 appeared fewer than 3 times, the configured minimum.
- The remaining 1,730 candidates were scored purely by extrapolation of the ranking network.

**3.4 The distillation is faithful, but to an unreliable teacher.**
- The ranking network reproduces the set model's leave-one-out marginals well: pairwise
  accuracy 0.91 on train and held-out images.
- As the manifest itself states, this measures fidelity to the set model, not downstream
  utility.
- The independent Banzhaf cross-check agreed only weakly (Spearman **0.08**).

---

## 4. Limitations of the adaptive thresholds

**4.1 The pooled quality floor removed large shares of some classes.**
The floor was set at p25 = 0.325 of the pooled normalised score, and 2,376 of 3,168 candidates
remained. Removed fractions per class:

| Class | Removed |
|---|---|
| mel | 49% |
| df | 38% |
| bcc | 34% |
| akiec | 18% |
| nv | 15% |
| vasc | 9% |
| bkl | 3% |

A single pooled floor treats score distributions that differ by class as if they were
comparable.

**4.2 Four classes were set by the floor, not the thresholds.**
- nv, akiec, vasc and df all ended at the class floor of 50.
- For akiec, df and vasc, the searched thresholds (0.94, 0.95, 0.95) kept only 22, 45 and 32
  images, and the floor raised each to 50.
- mel ended short of its target (140 of 186), because only 140 mel candidates survived the
  quality floor.
- In practice, the class floor and the target policy, rather than the learned thresholds,
  decided most of the final counts.

**4.3 The AdaptiveThresholdNetwork memorises rather than generalises.**
- In-sample residual to the frozen grid search: 0.010.
- Leave-one-class-out residual: **0.321** (mel 0.63, vasc 0.48).
- The thresholds are explained by the class embedding, not by the context features.
- With seven classes, the network cannot be expected to transfer to a new class.

---

## 5. Limitations of the generator (Stages 1–2)

- **Similar-looking classes.** The pigmented classes (mel, bkl, akiec, df) are visually alike
  in the generated images. vasc and bcc are the most distinct. On the 35-image pilot, a
  real-only auxiliary classifier agreed with the intended label for 14 of 35 images.
- **Model and snapshot choice.**
  - One LoRA (`ham-lora-v1`, SDXL, 8,000 steps) was trained.
  - Its validation loss was flat after about 6,500 steps.
  - The `final` snapshot was chosen by the author's visual review of four snapshot pilots,
    alongside an auxiliary-classifier agreement score. It was not chosen by downstream
    utility.
- **Uneven pool.** The candidate pool is uneven across classes: nv 112, df 885. This follows
  from the pre-registered stratified recipes, not from generation failures (3,168 of 3,168
  accepted).

---

## 6. Limitations of the classifier protocol (Stage 4)

- **Over-fitting.**
  - Every run trains for a fixed 3,000 steps at batch size 32.
  - For condition A that is roughly 58 epochs over 1,641 images, and training loss reaches
    about 10⁻³.
  - The final checkpoint is evaluated. By design there is no early stopping or checkpoint
    selection, to avoid tuning on validation data.
- **High seed variance.**
  - Condition A balanced accuracy spans 0.369–0.496 on `classifier_val` (SD 0.068) and
    0.497–0.605 on the test split (SD 0.060).
  - With only 3 seeds per condition, differences of a few points between conditions are
    close to seed variability.
  - The seed variance is included in the bootstrap intervals.
- **Fixed recipe.** A single architecture (DenseNet-121, ImageNet-pretrained, 512 px) with
  unweighted cross-entropy was used. The results may not transfer to other backbones, to
  class-weighted losses or to balanced sampling.

---

## 7. Limitations of the evaluation (Stage 5)

- **One dataset, one split.** There is one dataset (HAM10000) and one lesion-level split. No
  external test set was used.
- **Small rare-class test counts.**
  - vasc and df have 20 test images each, so one image changes recall by 0.05.
  - akiec has 50 images.
  - The per-class conclusions for these classes are imprecise, as the wide intervals show
    (df recall under C: 0.20, 95% CI [0.03, 0.39]).
- **Borderline confirmatory result.** The p-value is 0.047 against α = 0.05, and the CI
  upper bound is −0.001. It should be reported as a small, borderline effect, not as a
  decisive failure.
- **Exploratory analyses are not confirmatory.** Per-class results and the stability
  observation (seed SD on test: C 0.009, B 0.025, A 0.060) are exploratory. None was
  pre-registered.
- **Everything after this run is post-test.** `final_eval_heldout` has now been read. Any
  later changes (a v2 pipeline, extra conditions, noise-floor diagnostics) were designed after
  seeing the test results. They must be reported as post-hoc or follow-up analyses.

---

## 8. Process and provenance notes

- **Mid-pipeline code change.**
  - Stage 3 ran under a different code identity (`9b7f6ba3…`) from Stage 2 (`2132f199…`).
  - The reason was a reader-side fix to the content-box contract between Stage 2 and Stage 3:
    boxes are now read from Stage 2's `content_boxes.csv`.
  - The Stage 2 images were unchanged (`all_candidates.csv` sha `bf8047b5…`).
  - This is documented in `stage2_to_stage3_provenance_note.json`.
- **Optimizer fallback in Stage 1.** bitsandbytes 0.45.2 has no kernels for the RTX 5090
  (sm_120), so Stage 1 used torch AdamW instead of the 8-bit optimizer. This is recorded in
  the Stage 1 metadata.
- **Split usage.**
  - `gen_train`: IQA calibration and CAM references.
  - `classifier_train`: auxiliary classifier, proxies and targets.
  - `asism_tuning_heldout`: utility labels only.
  - `classifier_val`: Stage 4 measurement only, with no selection.
  - `final_eval_heldout`: read once, in Stage 5.

---

## 9. What the results do support

1. **Synthetic data from the LoRA-adapted SDXL generator improves the classifier.** Both
   augmented conditions beat real-only on balanced accuracy (B +0.079, C +0.046, both
   significant after correction).
2. **A subset one-fifth the size recovers much of the gain.**
   - C used 19% of the synthetic images (616 of 3,168).
   - It recovered about 58% of B's balanced-accuracy gain over A.
   - It matched B on macro-F1 (difference +0.002, not significant) and on accuracy.
3. **C was the most stable condition across seeds** (exploratory).

---

## 10. Suggested follow-up (post-hoc, to be labelled as such)

1. **Condition D:** 616 randomly selected candidates with C's per-class counts. This is the
   missing equal-quantity control (§2.2).
2. **Proxy noise-floor diagnostic:** repeat identical utility subsets under different seeds,
   and compare the between-subset spread with the within-subset spread (§3.1).
3. **More seeds** (e.g. 5) for Stage 4.
4. **A class-rebalancing target policy** and a per-class quality floor (§2.3, §4.1). This
   would test whether selection helps once B's rebalancing advantage is removed.

---

## 11. Post-hoc follow-up: proxy noise-floor diagnostic

*Added 2026-09-22, after `final_eval_heldout` was read. This is a post-hoc diagnostic. Nothing in
sections 1–10 was changed because of it, and no v1 artifact was re-selected or re-ranked.*

**Question.** Were the 80 v1 utility labels (§3.1) mostly seed noise? If training the *same* subset
twice moves balanced accuracy as much as swapping one subset for another, the set model had nothing
to learn, and the −0.36 validation Spearman (§3.2) is what one would expect.

**Design** (`scripts/followup/ham10000_proxy_noise_floor.py`, commit `1357407`):
- 12 of the 80 v1 subsets, one per utility-rank bin, each re-trained under 4 seeds (42–45), plus the
  real-only baseline under the same 4 seeds: 52 proxy runs per round.
- The plan (subsets, seeds, metrics) was frozen before the first run. Each round writes to its own
  directory and refuses to re-pick subsets.
- ICC = between-subset variance / (between + within). `repeats_needed` = the smallest number of
  averaged runs per label that reaches reliability 0.80 (Spearman–Brown).

**Decision rule, fixed before each round:**
- `repeats_needed` ≤ 5 on some metric → Stage 3 v2 uses that proxy with that many repeats.
- Otherwise → lengthen the proxy: round 2 at 1500 steps, then a last round at the Stage 4 recipe
  (512 px, 3000 steps).
- If the last round also fails, the proxy utility is reported as unreliable on this dataset at all
  three budgets, and v2 selects without a learned utility. There is no fourth round.

**Results** (v1-style label; ICC with 95% bootstrap CI over subsets; repeats estimated as
4 × within / between):

| Metric | Round 1: 224 px, 300 steps | Round 2: 224 px, 1500 steps |
|---|---|---|
| balanced accuracy | ICC 0.14 [0.00, 0.40], ~25 repeats | ICC 0.00 [0.00, 0.20], no detectable signal |
| macro AUROC | ICC 0.18 [0.00, 0.39], ~19 repeats | ICC 0.16 [0.00, 0.40], ~21 repeats |
| macro-F1 | ICC 0.13 [0.00, 0.43], ~28 repeats | ICC 0.08 [0.00, 0.26], ~49 repeats |
| accuracy | ICC 0.19 [0.00, 0.38], ~17 repeats | ICC 0.00 [0.00, 0.20], no detectable signal |
| **Verdict** | every metric: lengthen the proxy | no metric within 5 repeats |

Balanced accuracy in detail:

| | Round 1 | Round 2 |
|---|---|---|
| Within-subset SD (seed noise) | 0.033 | 0.032 |
| Between-subset SD (signal) | 0.013 | 0.000 |
| Baseline mean over 4 seeds | 0.462 | 0.508 |
| Mean of the 12 subsets | 0.511 | 0.541 |

Further observations:
- The paired label v1 used (subset minus the same seed's baseline) is noisier still: ICC 0.07 in
  round 1 and 0.00 in round 2, for every metric.
- Re-running v1's own seed 42 on the same GPU type moved the 12 labels by up to 0.031 (mean 0.015):
  a single v1 label is not reproducible to better than a few points.
- Restricted to subsets of equal size, the round-1 ICC was 0.003. This is a sensitivity check only,
  with 3–5 subsets per size. It suggests that the little between-subset spread there was came from
  subset size, not from which images were chosen.

**Interpretation.**
1. At the v1 proxy budget, one utility label was about 14% signal and 86% seed noise. The set model
   was trained on labels it could not have ranked reliably, which is consistent with §3.1–§3.4.
2. A five-times longer proxy did not help: the between-subset signal in balanced accuracy vanished
   entirely.
3. Adding synthetic images does help the proxy (+3 to +5 points of balanced accuracy over the
   baseline in both rounds). What the proxy cannot detect is *which* synthetic images were added.

**Round 3** (Stage 4 recipe: 512 px, 3000 steps; commit `ac813a4`): *running at the time of
writing. Its result and the branch of the decision rule it triggers will be added here unchanged.*

---

## 12. Stage 3 v2 design (post-hoc, pre-registered before any v2 result)

*Decided on 2026-09-22, after `final_eval_heldout` was read and before any v2 run. v2 is a
follow-up; `ham-final-v1` remains the primary result.*

v2 changes only the rules that §2.1, §2.3 and §4.1 identified. It is implemented in
`scripts/followup/ham10000_v2_select.py` and `configs/ham10000_v2_selection.yaml`, commit `07f9120`.

- **Target policy (5b).** Each class is filled to N = 300 images in total (real + synthetic):
  target = max(0, 300 − real count). nv therefore receives no synthetic images. A class with too few
  candidates takes all it has; the shortfall is reported and N is never changed.
- **Quality floor (6a).** p25 as in v1, but computed within each class instead of over the pooled
  pool. This addresses §4.1, where the pooled floor removed 49% of mel.
- **Within-class rule.** The highest-scoring candidates up to the target (top-k). The
  AdaptiveThresholdNetwork is not used in v2: under 5b the target alone fixes each class's count,
  and the network's failure to generalise (§4.3) stays recorded here as a limitation of v1.
- **Condition D2.** A random-selection control with C2's exact per-class counts. It is drawn with
  seed 42 from the safe pool (after the safety gate, before the quality floor), so C2 vs D2 measures
  the ranking and the floor together against no ordering at all. This is the control §2.2 found
  missing.
- **Which score orders the candidates** is decided by round 3's rule (§11), not chosen here: a
  retrained ranking network if the proxy proves reliable, a selection without a learned utility if
  not.
- **Planned Stage 4 v2 conditions:** C2, D2, B and A. Any v2 result is post-hoc and will be
  reported as such.

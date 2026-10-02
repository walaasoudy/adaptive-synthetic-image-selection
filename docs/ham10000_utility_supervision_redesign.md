# HAM10000 — Utility supervision redesign (design only, nothing implemented)

*Written 2026-09-24, after round 3 of the noise-floor diagnostic failed its pre-registered rule
(§11 of `ham10000_results_and_limitations.md`) and after the supervision audit that produced §14.
This document is a DESIGN. No code in this repository is changed by it, no model is trained, and no
selection is produced. It is written now, before any new measurement exists, so that every choice
below is on record ahead of the number that would justify it.*

**What this redesign does not touch.** The ranking network and `SetUtilityNetwork` are unchanged.
C2, D2, Stage 4 v2 and Stage 5 v2 are not produced. No IQA arm is substituted for a learned utility.
N = 300 and D2's draw are unchanged. `final_eval_heldout` is not read at any point in this plan, and
`ham-final-v1` stays the primary result and stays frozen.

---

## 1. What the audit established

The diagnostic measured reliability and found it absent. The audit found why, and the chain is
short:

1. Utility is measured as balanced accuracy on `asism_tuning_heldout`. That split holds **13 df
   images** and **20 vasc images**. Balanced accuracy averages recall over seven classes, so one df
   image changing its prediction moves the label by 1/(7×13) = **0.011**.
2. The between-subset spread the diagnostic is trying to resolve is **0.0134** (round 1). The
   instrument's smallest step is the size of the quantity being measured.
3. Each label is **one run** at seed 42. Its run-to-run SD is **0.0332**, about three df images.
4. A single real-only baseline is subtracted from all 80 labels. That run returned 0.5136 where four
   runs of the same recipe give 0.4529–0.4737 (§14), so every label was shifted about five points.
5. What structure the labels do carry is largely **subset size**, not subset composition: grouped by
   construction strategy the labels are indistinguishable (matched_random −0.0139, mixed −0.0166,
   single_signal −0.0151), while grouped by size they separate cleanly.
6. `SetUtilityNetwork` pools with a masked **mean**, so subset size is invisible to it by
   construction. The strongest pattern in the labels is the one the model cannot represent.
7. The train/val split is unbalanced in exactly that variable: 11 of 16 validation subsets are size
   180, against 24 of 64 in training.
8. Only 1,438 of 3,168 candidates ever appeared in a subset, and 951 of those appeared fewer than
   the configured minimum of three times.

Two facts about the labels are worth keeping in view, because they say the instrument is not
worthless:

- Against a re-measurement of the **same** recipe at four seeds, the v1 labels rank the 12 re-run
  subsets at Spearman **+0.73**. Single-run labels are noisy, not inverted.
- Against the **Stage 4** recipe (round 3) the same comparison is **−0.60**. The short proxy and the
  real training recipe disagree about which subsets are good.

Those two numbers separate the two problems this design has to solve, and they are not the same
problem:

> **Reliability** — does the instrument repeat itself? Addressed by §3–§8.
> **Validity** — does it measure what Stage 4 will do? Addressed by §9, and it is the one the
> diagnostic never tested.

---

## 2. The cost fact that reopens the design

Run time per proxy training, measured over 52 runs each:

| Recipe | Per run | 52 runs |
|---|---|---|
| 224 px, 300 steps (the v1 recipe) | **14.0 s** | 0.20 h |
| 224 px, 1500 steps | 63.7 s | 0.92 h |
| 512 px, 3000 steps (Stage 4 recipe) | 578.3 s | 8.35 h |

`MAX_AFFORDABLE_REPEATS = 5` was set before round 1 and is correct for the Stage 4 recipe: 80
subsets × 5 repeats at 578 s is 64 GPU-hours. At **14 s** the same 400 runs take **1.6 hours**.

The ceiling was therefore a statement about the expensive recipe, and §11 requires that any revision
of it be argued on cost and written down before the number it would change is looked at again. This
section is that argument, and it is written before any new label exists:

> **Revised ceiling, for the 224 px / 300 step proxy only.** Repeated measurement of one subset may
> use up to **20 repeats** at this recipe.

Three things this revision is **not**, stated so they cannot be read into it later:

- **It does not change round 3's rule, and it does not change round 3's verdict.**
  `MAX_AFFORDABLE_REPEATS = 5` remains in force for the 512 px / 3000 step recipe, which is the
  recipe round 3 measured and the recipe its rule was about. Round 3 needed 8 repeats at that
  recipe, that is still above 5, and it is still a failure. Nothing here re-runs it, re-analyses it
  or re-judges it. The two ceilings belong to two different instruments and are recorded separately.
- **It is not a plan to run 20 repeats of everything.** 20 is a ceiling on what one subset may cost,
  not a number of runs to schedule. Exactly two pieces of work are costed in this document — §8 and
  §9 — and neither of them is 132 × 20.
- **It is not justified by any new result.** The argument is the cost table above, which was
  measured during the three completed rounds, and it is written here before a single new label
  exists.

---

## 3. Problem 1 — metric resolution

**Decision: the label is no longer balanced accuracy.**

Balanced accuracy is a mean of seven thresholded recalls, three of which are computed over fewer
than 50 images. It cannot take a step smaller than 0.011. The label becomes a continuous score
computed from the predicted probabilities, with no thresholding:

- **Primary: macro one-vs-rest AUROC.** Chosen on measured evidence, not preference: across the
  twelve subset-level ICCs of the three rounds it produced the highest (0.35, round 3) and was the
  only metric ever to come within sight of the reliability target. It is rank-based over roughly
  13 × 1,364 pairs for df rather than over 13 recall steps.
- **Secondary, recorded but not the label: class-balanced log loss** — the mean over the seven
  classes of the mean negative log-likelihood of that class's images. Fully continuous and sensitive
  to probability mass, not only to ordering, so it registers a subset that makes the model more
  confident without flipping a prediction.
- **Balanced accuracy is still computed and stored** for every run, so the new labels can be
  compared with the old ones. It is never the training target.

**The risk this creates, stated before it can be discovered.** The thesis's confirmatory comparison
is balanced accuracy (§1, §12.1). A utility label built on AUROC optimises a different quantity, and
a selection that improves AUROC need not improve balanced accuracy. This is a validity risk, it is
not resolved by argument, and §9 is the experiment that tests it.

---

## 4. Problem 2 — rare-class support

**Decision: the evaluation split does not change. `classifier_val` is NOT pooled in.**

The obvious fix is to enlarge the rare classes by evaluating utility on
`asism_tuning_heldout` + `classifier_val`, which would take df from 13 to 28 and vasc from 20 to 40.
It is rejected, on evidence from the code rather than on caution:

- `scripts/classify/01_train_conditions.py` validates Stage 4 on `classifier_val`, and
  `scripts/eval/compare_conditions.py` records that each run is scored with **its own
  classifier_val-selected threshold**.
- Tuning the selection on a split that later chooses the operating thresholds would make Stage 5's
  thresholds partly a function of the selection they are meant to evaluate.

So the rare-class problem is addressed **through the metric** (§3) rather than through more images.
Two further requirements follow:

- **Per-class utility is recorded alongside the scalar label** — per-class AUROC for all seven
  classes in every run record. A subset that helps only df is currently invisible; it must not be.
- **The limit is stated, not solved.** With 13 df images, no metric makes `asism_tuning_heldout` a
  precise instrument for df. A larger rare-class evaluation pool would require re-splitting the
  dataset, which would invalidate v1, so it is not available. This is recorded as a bound on how
  good the supervision can get, not as a step.

---

## 5. Problem 3 — the baseline

**Decision: no baseline is subtracted. The label is the absolute metric of the augmented run.**

Two independent pieces of evidence:

- §11 measured the paired label directly (subset minus the same seed's baseline). Its ICC was
  **0.00–0.07** in every round — worse than the absolute label every time. Subtracting a baseline
  adds that run's noise rather than removing shared noise, because the two runs share no randomness.
- §14 showed the single subtracted constant was an outlier, which inverted the sign of the whole
  label set.

Every subset is measured under identical conditions, so absolute values rank them without any
reference point. A real-only baseline is still trained, **R times like every other condition**, and
reported next to the labels for interpretation. It is never part of the label.

---

## 6. Problem 4 — subset-size confounding

**Decision: every subset has the same size. Composition becomes the only thing that varies.**

With sizes 60/120/180 mixed, "which subset is better" and "how big is it" cannot be separated, and
the audit shows size wins. Fixing the size makes the question the one the ranking network is
actually asked.

**Size = 96 images**, chosen so that two arithmetic properties hold at once:

- 3,168 candidates ÷ 96 = **33 subsets cover the pool exactly once**, with no remainder and no
  image left out. §8 depends on this.
- 1,641 real + 96 = 1,737 images = 54 batches of 32 **plus a trailing batch of 9** — the same
  trailing batch as the real-only baseline (1,641 = 51×32 + 9). Under the current loader
  (`drop_last=False`, `scripts/utils/ham10000_classifier.py:143`) the augmented run and the baseline
  then have identically shaped epochs. At size 120 the trailing batch holds exactly **one** image
  (1,761 = 55×32 + 1), and the size-120 subsets are precisely the ones whose v1 labels were all
  positive. **This is a precaution, not a diagnosis:** no experiment here shows the trailing batch
  caused anything, and the size-120 effect did replicate at four seeds in round 1. The design simply
  removes a difference it does not need.

**The honest risk.** §11's same-size sensitivity analysis gave ICC ≈ 0.00 at round 1. Holding size
fixed may show that composition, on its own, moves the proxy very little. That would be a finding
about this dataset and this pool, and it must be reported as one rather than treated as another
failure to be engineered around.

---

## 7. Problem 5 — train/val balance

**Decision: the validation subsets are stratified and there are more of them.**

- With one size, the remaining strata are construction strategy (matched_random / single_signal /
  mixed) and the class composition of the subset. The train/val split is stratified on both, so the
  validation subsets are drawn from the same label distribution as the training ones.
- **At least 30 validation subsets**, against v1's 16. A Spearman over 16 points has a 95% interval
  roughly ±0.5 wide; the reported −0.36 cannot be distinguished from 0 at that size.
- The split is frozen and written to disk **before** any subset is measured, exactly as v1 did.

---

## 8. Problem 6 — image coverage — **Experiment 1: coverage construction**

**Decision: exact k-fold coverage by construction, not by random draw.**

v1 drew subsets independently, so coverage was whatever the draws produced: 1,730 candidates never
appeared at all, and the distillation target for 951 more rested on one or two measurements.

With 96-image subsets, **33 subsets partition the 3,168 candidates exactly once**. Repeating the
partition with different random pairings gives every candidate exactly one appearance per cover:

> **Experiment 1 — coverage construction.**
> **132 subsets = 33 unique subsets × 4 coverage cycles.** Every candidate appears in exactly 4
> subsets; none is absent and none falls below the minimum of three that v1 configured and never
> reached.

**This is a construction step. It is CPU work that writes subset definitions to disk, and it costs
no GPU time at all.** It produces 132 records in a `utility_subsets.jsonl`-style file and nothing
else. In particular:

> **132 subsets does NOT imply 132 × 20 = 2,640 training runs.** No training budget follows from
> this section. How many times each of the 132 subsets is measured is decided by Experiment 2 (§9),
> which measures `repeats_needed` on real data. Until that number exists, the training cost of the
> full supervision is undetermined and is not scheduled, not estimated as a commitment, and not
> approved.

The construction strategies of §7 are applied **within** each cover, so a cover is a partition into
96-image blocks stratified by strategy rather than a uniform shuffle. Every image still appears
exactly once per cover, and the train/val split of §7 is applied to the 132 subsets before any of
them is measured.

For orientation only, and binding on nothing, the GPU cost of the full supervision at 14 s per run
would be 132 × R runs: 5.1 hours at R = 10, 10.3 hours at R = 20. Which of those, if either, is ever
paid for is a separate decision taken after §9 reports.

---

## 9. **Experiment 2: reliability and validity** — the run that gates everything else

This is the only GPU work approved by this document, and it is cheap because the expensive half of
it has already been paid for.

**The target already on disk.** Round 3 measured **12 subsets × 4 seeds at the Stage 4 recipe**
(512 px, 3000 steps), including macro one-vs-rest AUROC for every one of those 48 runs. That is the
best available estimate of what those subsets do to a real training run, and it cost 8.35 GPU-hours
that are already spent.

**The run.** Build the new label — §3's metric, §5's no-baseline rule — for those **same 12
subsets**, at the cheap recipe.

> **Experiment 2 — reliability and validity.**
> **12 pre-existing round-3 subsets × 20 cheap-proxy repeats = 240 runs ≈ 56 minutes.**
> No new subsets are built for this. No new Stage-4-recipe run is paid for. The 132 subsets of §8
> are not measured here.

### 9.1 The two criteria, fixed numerically before the experiment exists

Both are evaluated on **macro one-vs-rest AUROC**, on both sides of the comparison.

**Criterion R — reliability.**

> Using the ICC and Spearman–Brown machinery of §11, computed over the 12 subsets × 20 repeats:
> **`repeats_needed` for reliability 0.80 must be ≤ 20.**

**Criterion A — agreement with the Stage 4 recipe.**

> **Statistic.** Spearman rank correlation ρ̂ between
> (a) the new label — the mean over the 20 cheap repeats — and
> (b) round 3's mean over its 4 seeds at 512 px / 3000 steps,
> across the same 12 subsets.
>
> **Test.** One-sided, alternative ρ > 0. The p-value is a Monte-Carlo permutation test over the 12
> subset labels: 200,000 permutations, `numpy.random.default_rng(42)`.
>
> **Pass requires both:** **ρ̂ ≥ 0.60** *and* **p ≤ 0.05**.

**Why 0.60, argued now rather than after the number is seen:**

- At n = 12 the one-sided 5% critical value is ρ ≈ 0.50, so 0.60 is deliberately **stricter than
  bare significance**. A correlation that merely clears a significance test must not licence the
  GPU-hours of a full supervision run.
- 0.60 sits below **+0.73**, which is what two measurements of the *same* recipe achieve against
  each other here (v1's single-run labels against round 1's 4-seed means, §1). That is the practical
  ceiling for agreement between any two proxy measurements on this data, so 0.60 is demanding
  without being unreachable.
- The current label's agreement with the Stage 4 recipe is **−0.60**. The bar therefore requires a
  genuine reversal, not a small improvement on a broken instrument.
- At n = 12, ρ̂ = 0.60 already implies p ≈ 0.02, so the two conditions will rarely disagree. The
  p-value is kept as an explicit guard, not as an independent hurdle, and neither condition may be
  dropped if the other passes.

**The decision rule.**

> The cheap proxy is accepted for utility supervision **only if it satisfies both the predefined
> reliability criterion (R) and the predefined agreement criterion (A) with the Stage-4-size
> measurements.** Either one failing is a failure. Reliability alone is explicitly not sufficient: a
> label that repeats itself perfectly while disagreeing with the real training recipe is a precise
> measurement of the wrong thing.

### 9.2 One attempt

The experiment is run **once**, at R = 20, with macro AUROC as the label and these 12 subsets. The
two criteria are evaluated once, and the result is written down whichever way it falls. If it fails,
R is not raised, the metric is not swapped for the secondary one, the subset sample is not changed,
and the thresholds above are not revisited in the light of the number that missed them. Any later
attempt is a different instrument and needs its own pre-registration, written before it runs.

**Recorded but not part of the decision** — reported alongside, so nothing is hidden, and with no
power to overturn R or A: ρ̂ against round 3's balanced accuracy; ρ̂ using the class-balanced log
loss of §3; per-class AUROC agreement; and the same quantities against rounds 1 and 2.

### 9.3 If it fails

The conclusion is that no 224 px / 300 step proxy can supervise this task, however many times it is
repeated. The redesign then moves to the next candidate instrument — a longer proxy priced against
§2's table, or a target that is not a full retraining at all. That branch is **not** designed here,
because designing it now would mean guessing at a number that does not yet exist.

### 9.4 The sample is small, and that is stated in advance

A Spearman over 12 points is a **screening test**, not a confirmation: its 95% interval is roughly
±0.4 wide even at ρ̂ = 0.7. It is strong enough to rule the cheap proxy **out**, and a pass is a
licence to proceed to §10 — not evidence that the instrument is good. Nothing in this document
treats a passed Experiment 2 as a result about ASISM.

---

## 10. Order of work, and what gates what

Nothing below starts until the step above it has produced a written result.

1. **§14 correction** — done (commit `656c906`).
2. **This design** — written before any new measurement.
3. **Experiment 2 (§9)** — 12 subsets × 20 repeats = **240 runs, about an hour of GPU**. Criteria R
   and A are evaluated once and the result is written down whichever way it falls. This is the only
   GPU work this document approves, and everything below it is gated on both criteria passing.
4. **Tests** — written only after Experiment 2 passes, and written against the design in §3–§8:
   exact coverage, fixed size, stratified split, no baseline in the label, per-class metrics
   recorded, `final_eval_heldout` unreachable.
5. **Experiment 1 (§8), then the full supervision** — build the 132 subsets (CPU, no GPU), then
   measure them at R repeats where **R is whatever Experiment 2 measured**, not 20 by default. The
   GPU cost of this step is not approved by this document and is decided when R is known.
6. **Retrain and re-evaluate the ranking network** on the new labels. The network architecture is
   not modified in this plan; if it fails on labels that are now reliable and valid, that is when
   the architecture becomes the question, and §6's mean pooling is the first thing to look at.
7. **C2 and D2** — only if step 6 produces a ranking that holds up on the stratified validation
   subsets. Until then C2 is not produced and nothing is substituted under its name (§12.0.1).

`final_eval_heldout` is not read in any step of this list. It is read once more in this project's
lifetime, and only for a v2 Stage 5 that does not currently exist.

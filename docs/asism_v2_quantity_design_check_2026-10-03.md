# ASISM V2 quantity logic: CPU design check (2026-10-03)

Status: CPU only, toy data only. No HAM10000 label, no GPU run, no Probe P, no real ranker fit.
Sections 1 to 6 report what was checked. Sections 7 and 8 were proposals; **Walaa approved both on
2026-10-03**, together with the sum form as the V2 direction (no fixed K, no ratio to the real data,
no other architecture or stopping formulation at this point, the conservative behaviour kept as a
documented limitation). Section 9 records how they were implemented on CPU.

## 1. The question

V2 must decide both which synthetic images are used and how many. No count, ratio or K is an input.
The count has to follow from each image's estimated contribution to classifier utility.

The first V2 draft modelled a set's utility as `b + mean(score) + λ·log(1+n)`. Under that form an
image is predicted to help only if its score is above the current set mean minus about λ. On toy
utilities this dropped a whole class at the first step (all images helpful: 120 of 240 selected, one
class 0) and selected one image when 169 of 240 were helpful.

The form checked here keeps the same four signals, the same class-specific linear score, the same
bootstrap and the same contract §10 rule, and changes only how images are pooled:

    U(S) = b + λ · log(1 + Σ_{i∈S} w_i),    w_i = 1 + score(x_i, class_i) + class term

An image's marginal utility is `λ·[log(1 + W + w_i) − log(1 + W)]`, whose sign is the sign of
`λ·w_i`. The image is judged on its own weight. A class keeps adding its next-ranked image while the
5th percentile of that marginal utility over the bootstrap ensemble is above 0.

## 2. What was run

Two classes, four standard-normal signals per image, 300 / 120 / 120 images per class for the
train / validation / pool roles, subsets of 10 to 80 images with class fractions varied, three
repeats per subset with Gaussian noise of SD 0.002 (0.01 in the noise rows), 20 bootstrap models
unless stated. The true utilities are deliberately not of the fitted form: a saturating
`0.80 + 0.15·(1 − exp(−Σq/60))` over a per-image quality `q`, and one truly mean-pooled case.

| Case | Truth | Selected (a, b) | Helpful (a, b) | Harmful selected | True utility: selected / all / best |
|---|---|---|---|---|---|
| all_help | every image helps | 120, 119 | 120, 120 | 0 | 0.9358 / 0.9359 / 0.9359 |
| some_harm | 30% harmful | 56, 37 | 82, 87 | 0 | 0.9092 / 0.8327 / 0.9301 |
| no_benefit | images do nothing | 0, 0 | 0, 0 | 0 | 0.80 / 0.80 / 0.80 |
| all_harm | every image hurts | 0, 0 | 0, 0 | 0 | 0.80 / 0.6049 / 0.80 |
| class_diff | a all helpful, b 68% harmful | 120, 1 | 120, 38 | 0 | 0.9052 / 0.7452 / 0.9236 |
| useless_half | half the images add exactly 0 | 110, 97 | 63, 57 | 0 (87 zero-value) | 0.9204 / 0.9204 / 0.9204 |
| meanpool | utility truly is a mean | 82, 88 | n/a | 50 below-mean | 0.9247 / 0.9079 / 0.9513 |

Sensitivity, all on some_harm (169 helpful of 240):

| Change | Selected | Harmful selected |
|---|---|---|
| base (λ start 0.02, seed 42, 20 bootstrap models) | 93 | 0 |
| λ start 0.005 | 93 | 0 |
| λ start 0.1 | 98 | 1 |
| noise SD 0.01 | 96 | 0 |
| 50 bootstrap models | 106 | 0 |
| fit seed 7, 20 bootstrap models | 136 | 1 |
| 100 bootstrap models, seeds 42 / 7 / 11 | 110 / 117 / 106 | 0 / 0 / 0 |

With all_help and noise 0.01: 228 of 240. With no_benefit and noise 0.01: 0.

## 3. Answers to the six questions

1. **Does the sum form remove the problem found in the mean form?** Yes. No class was dropped because
   another class held a better image, and helpful images below the selection's own mean quality were
   kept (119 in all_help, 37 in some_harm).
2. **Can the sum form favour larger sets for their size?** Partly. It selected nothing when the
   images did nothing or hurt, so size alone is not rewarded. But in useless_half it kept 87 images
   that add exactly nothing: a linear score cannot represent a step, so those images received small
   positive weights. They did not lower the true utility. The rule also never stops for diminishing
   returns: an image with a positive weight is kept however little it adds.
3. **Is there a parameter that could turn into tuning?** No free threshold, K, ratio or budget exists.
   The constant 1 in `w_i` is a scale convention. What can move the count is the fitting
   configuration: the number of bootstrap models, the fit seed, the starting λ and the optimiser
   settings. At 20 bootstrap models the count moved from 93 to 136 between two seeds; at 100 it moved
   from 106 to 117. These have to be fixed before any real fit (section 7).
4. **Does the rule isolate low-utility images rather than merely below-average ones?** Yes for
   harmful images: at most 1 was selected in any run. It is conservative: it left out 30% to 45% of
   the helpful images in some_harm, because the lower bound has to be above 0.
5. **Can it produce the four outcomes?** All images when all help (239 of 240). A subset when some
   harm (93 to 136 of 169 helpful, 0 or 1 harmful). Zero when there is no benefit or only harm.
   Different counts per class (120 and 1 in class_diff), although there it kept only 1 of the 38
   helpful images of the mostly harmful class.
6. **Is the decision made on utility contribution, with no fixed K or hidden ratio?** Yes. The same
   code and configuration returned 239, 93, 0, 0, 121 and 207 images on six truths. Nothing in the
   code reads a target count.

## 4. Failure modes

- **Conservative.** Helpful images whose weight is not confidently above 0 are left out. More noise
  or fewer measured subsets moves the count down, to zero in the limit. A small count from a real fit
  may reflect weak supervision rather than bad images, and has to be reported with the bootstrap
  spread of λ and of the weights.
- **Mostly harmful class.** A linear score places the class's zero crossing imprecisely; the class
  can be nearly excluded although some of its images help.
- **Harmless, useless images are kept.** The rule separates "not confidently helpful" from "helpful".
  It does not separate "adds nothing" from "adds a little" when the score is linear.
- **No stop for diminishing returns.** If "enough benefit" is meant to exclude images that help only
  slightly, that needs a minimum-gain value, which would be a new parameter and would need its own
  pre-registration. None is introduced here.
- **If utility truly is a mean** (an image below the set average dilutes it), the sum form
  over-selects: 170 of 240, true utility 0.9247 against a best of 0.9513, still above using all.
  Which form the real classifier follows is an empirical question; the held-out test in section 8
  measures it.
- **λ of the wrong sign.** If more synthetic data lowers the measured utility, λ is negative and
  nothing is selected. That is a result, not an error.
- **Count depends on the bootstrap.** See question 3.

## 5. What was implemented (CPU, untested on real labels)

- `scripts/asism_v2/pipeline.py`: the sum-pooled set term; `image_weight`; with one train size λ is
  held at 1 and recorded as frozen; identifiability of the size and class terms is judged on train
  subsets only, on class fractions.
- `scripts/asism_v2/stopping.py`: marginal utility of the sum form; refuses a ranker whose size term
  or class term was frozen.
- Tests: `tests/test_asism_v2_stopping.py`, `tests/test_asism_v2_pipeline.py`. Full suite: 930 passed
  before the sum change; the five V2 suites pass after it (35 tests).

## 6. What is still missing before V2 can run

A subset designer, a measurement runner, saving and loading a fitted ranker, a selector that writes
condition C and the matched random D for Stage 4, and the held-out acceptance check. The Stage 4
aggregator also rejects two valid adaptive outcomes (C equal to all candidates, C empty).

## 7. Pre-registration of the fit and the stopping rule (APPROVED 2026-10-03)

| Item | Proposed value | Reason |
|---|---|---|
| Bootstrap models | 200 | 20 gave 93 to 136 across seeds on the toy; 100 gave 106 to 117 |
| Lower-bound quantile | 0.05 | contract §10, locked |
| Fit seed | 42 | repository convention |
| Starting λ | 0.02 | default; 0.005 to 0.1 changed the toy count by 5 |
| Optimiser | Adam, lr 0.03, weight decay 1e-5, full batch, up to 600 epochs, patience 60 | as coded |
| Stability report | the selection repeated at fit seeds 42 to 46; per-class counts reported for each | shows how much of the count is the bootstrap |
| Minimum-gain threshold | none | not introduced |

## 8. Utility measurement and ranker acceptance (contract §8 and §9; APPROVED 2026-10-03)

**Why earlier labels failed.** They were measured on random subsets of one size. Random subsets
barely differ from one another, so the difference between them was below the training noise
(ICC 0.01 to 0.11). The proposal measures subsets that are built to differ.

| Item | Proposal |
|---|---|
| Metric | macro AUROC (one-vs-rest) on `asism_tuning_heldout`, absolute, no baseline subtraction |
| Recipe | the 224 px / 300 step proxy. Written reason for re-use: it was rejected for random same-size subsets; its direction agreed with the Stage 4 recipe (Spearman 0.755). Its reliability is re-tested on the designed subsets (gate G1) before anything is fitted |
| Image roles | each candidate image is assigned once, within class, to train / validation / test (60 / 20 / 20%). A subset uses images of one role only |
| Sizes | train: 125, 250, 500, 1,000. Validation and test: 125, 250, 500 |
| Class fractions | drawn per subset around the pool proportions, so the class term is identifiable |
| Composition | half of the subsets random within class; half tilted: within each class drawn from the upper or lower half of one signal (4 signals × 2 directions) |
| Count | 120 train, 40 validation, 40 test subsets |
| Repeats | 5 training seeds per subset |
| Runs | 200 × 5 = 1,000 proxy runs, about 14 s each: about 4 GPU-hours |
| G1, reliability | reliability of the 5-seed subset means ≥ 0.80, computed before any fit. If it fails, no ranker is trained on these labels |
| Ranker acceptance (test subsets, read once) | (a) Spearman between predicted and measured utility within size ≥ 0.50, one-sided permutation p ≤ 0.05; (b) lower test error than a size-and-class-only model with no signals. If (a) or (b) fails, the learned ranking is not used and the failure is the reported result |
| Reported with it | similarity-only and equal-weight baselines on the same test subsets; the E4 curve (sizes 0 to 3,168) as the external check of the size term, since selection can go beyond the largest measured size |

Not decided here: the supervisor's test-set policy, and how the count itself is tested at Stage 5
(C against D tests only which images).

## 9. Implementation of sections 7 and 8 (CPU only, 2026-10-03)

| Piece | Where |
|---|---|
| The approved values, in YAML and frozen in code; a differing YAML is refused | `configs/ham10000_asism_v2_ranker.yaml`, `scripts/asism_v2/prereg.py` |
| Safe-pool signal table, roles, the 200 designed subsets, the frozen plan | `scripts/asism_v2/supervision.py` |
| G1 reliability, ranker acceptance (test subsets read once), stability across the 5 fit seeds | `scripts/asism_v2/gates.py` |
| Explicit fit configuration; a signal mask for the no-signal and similarity-only models | `scripts/asism_v2/pipeline.py` |

Values that section 8 left open and the implementation had to fix. They are part of the frozen
configuration and are stated here so that they are not chosen later:

- Roles are assigned with seed 42, within class, by largest remainder (60 / 20 / 20%).
- Class fractions of a subset are drawn from Dirichlet(20 × the role's class proportions), then
  capped by what the role holds of each class. Design seed 42.
- Size cycles through the role's sizes; blocks of subsets alternate random and tilted, so every size
  has both kinds. The eight tilts (4 signals × upper / lower) are used in turn.
- A tilted class draw takes its images at random from the upper (or lower) half of the signal within
  that class and role. When the class needs more than half of what the role holds, it takes the top
  (or bottom) images by that signal.
- Training seeds 42 to 46.
- G1 is computed on the train and validation subsets only; the test subsets stay unread until the
  acceptance check. The gated number is the reliability of the subset means over all of them, as
  approved. Because subsets differ in size, that number includes the effect of size, and labels that
  depend on size alone can pass it. The same reliability within each size is therefore reported next
  to it. It does not gate; the within-size acceptance criterion (a) is what tests which images.
- Acceptance: 10,000 permutations of the measured values within size, seed 42. Criterion (b) compares
  test mean squared error with the same model fitted with every signal weight held at 0.
- The reported selection is the one at fit seed 42. The other four seeds are reported, never chosen
  from.

Dry run of the design on the real signal table (3,168 candidates, no label, nothing frozen): roles
hold 1,904 / 632 / 632 images; 1,000 runs planned; every image is in at least 5 subsets (median 26);
both the size term and the class term are identifiable. One 200-bootstrap fit at this size takes
roughly 1.5 CPU-hours, so the five-seed stability report is roughly 8 CPU-hours.

The measurement runner, saving and loading a fitted ranker, the selector for C and D, and the
Stage 4 change were built afterwards; see section 10.

## 10. Runner, saved ranker, selector, Stage 4 (CPU only, 2026-10-03)

Walaa confirmed on 2026-10-03 the values listed in section 9 (Dirichlet concentration 20, the tilt
rule, 10,000 permutations) and the reading of G1: G1 stays a check that the labels are repeatable
before anything is fitted, and acceptance criterion (a), the within-size Spearman of at least 0.50
on the test subsets, is the strict gate that shows the ranker learned which images matter and not
only how many. No threshold changed.

| Piece | Where |
|---|---|
| plan / measure / G1. The measure phase is the only GPU step; it refuses to run while `MEASUREMENT_APPROVED` is `False`, and then also without `--i-understand-this-trains-real-models` | `scripts/followup/ham10000_asism_v2_utility.py` |
| Test-subset outcomes kept in their own file, which G1 never opens | `utility_runs_test.jsonl` |
| Saved ranker (point model, bootstrap models, normaliser, provenance, content hash) | `scripts/asism_v2/persist.py` |
| accept / fit / select / stability phases | `scripts/followup/ham10000_asism_v2_learned_select.py` |
| C, the matched random D per Stage 4 seed, the selection manifest | `scripts/asism_v2/selection_files.py` |
| Stage 4 protocols `asism_v2_learned` and `asism_v2_learned_all_or_none`, the outcome check in the aggregator | `scripts/utils/ham10000_conditions.py`, `scripts/classify/ham10000_aggregate_conditions.py`, `configs/ham10000_asism_v2_learned_stage4.yaml` |
| CPU dry run of the whole path | `scripts/smoke/asism_v2_ranker_dry_run.py` |

Order, each step refusing unless the one before it succeeded: plan, measure, G1, accept (test subsets
read once), fit, select, stability. A failed G1 or a ranker that is not accepted ends the path, and
that failure is the result.

The count is never configured. The selection manifest records one of three outcomes, all of them
valid results:

- `subset`: C and D are written and Stage 4 trains A, B, C, D (`asism_v2_learned`).
- `all`: C is every candidate, which is B's training set, and D would be the same set. No C or D is
  built; Stage 4 trains A and B (`asism_v2_learned_all_or_none`) and reports the outcome.
- `none`: C is empty, which is A's training set. Same protocol, same reporting.

The aggregator checks that the protocol matches the recorded outcome and that C trained on the
recorded images. It no longer treats "C has as many images as B" as a fault for this selector. The
earlier protocols (`v1`, `v2`, `asism_v2`, `asism_v2_none`) are unchanged.

Still open, and not decided by this code:

- Contract §13 decision 7 (D draws, number of Stage 4 seeds). The new Stage 4 config copies the
  earlier design's values, 20 seeds and one independent D draw per seed, and D's draw seed base is
  20261003. They are placeholders until that decision is made; Stage 4 is a GPU step and is not
  approved.
- How HOW MANY is tested at Stage 5. C against D holds the count fixed.
- Test-set policy and the confirmatory statistic (supervisor).

### Dry run (CPU, no training, not evidence)

`scripts/smoke/asism_v2_ranker_dry_run.py` runs plan, measure, G1, accept, fit, select and stability
on the real signal table (3,168 candidates, the frozen 200 subsets, 1,000 cells, training seeds 42 to
46) with the classifier replaced by a planted formula of the model's own form,
`base + lam * log(1 + sum of planted image weights)` plus seed noise. Nothing is trained, no image or
split is read, and the outputs are stamped `scientific_evidence: false`. The fit used 20 bootstrap
models instead of 200 to keep the run near one hour. Outputs are under
`C:\Users\walaa\ham10000_work\asism_v2_dry_run_*`. The planted numbers are arbitrary: they say how
the code behaves, not what HAM10000 will show.

| Scenario | planted lam / noise SD | G1 | G1 within size 125 / 250 / 500 / 1000 (not gating) | Noise-free truth, within-size Spearman | Ranker, within-size Spearman | Accepted | Path |
|---|---|---|---|---|---|---|---|
| 1 | 0.012 / 0.003 | 0.980, passed | 0.82 / 0.48 / 0.37 / 0.09 | 0.74 | 0.39 (p 0.008) | no | stopped at acceptance; fit refused |
| 2 | 0.012 / 0.001 | 0.998, passed | 0.97 / 0.94 / 0.74 / 0.70 | 0.88 | 0.40 (p 0.007) | no | stopped at acceptance; fit refused |
| 3 | 0.06 / 0.001 | 0.9999, passed | 1.00 / 1.00 / 0.99 / 0.98 | 1.00 | 0.66 (p 0.0001) | yes | complete |

What the dry run showed about the code:

- Every phase ran and wrote its files; 800 fit rows and 200 test rows, five seeds each.
- A ranker that is not accepted stops the path: the fit phase refused, nothing was selected.
- Scenario 3 produced the saved ranker, C (2,513 of 3,168 images), 20 D files at C's per-class counts,
  the trajectory, the manifest (`selection_outcome: subset`, protocol `asism_v2_learned`) and the
  stability report for fit seeds 42 to 46: totals 2,368 to 2,718, Jaccard with the seed-42 selection
  0.89 to 0.95. The class with the widest range was vasc, 299 to 583.
- Planted values above 1 were refused by the measurement validator (an earlier attempt with
  base 0.86 and lam 0.06).

What it showed about the approved design, before any real measurement:

1. **G1 passes on size alone.** In scenario 1 the gated number is 0.98 while the same number within
   size 1,000 is 0.09. This is the behaviour section 9 warned about and the reason acceptance
   criterion (a) is the strict gate.
2. **The approved fit settings underfit a truth of the model's own form.** In scenario 2 the noise-free
   truth ranks the test subsets at 0.88 within size and has a training error of 2.1e-7; the fitted
   ranker reaches 0.40 and a training error of 2.5e-6, and its best epoch is the last one (599 of
   600). In scenario 3 the truth is at 1.00 and the ranker at 0.66, again stopped by the epoch limit,
   and 234 of the 2,513 selected images have a non-positive planted weight. Diagnosis on the planted
   data of scenario 2, outside the frozen configuration:

   | Fit | Validation error | Within-size Spearman against the noise-free truth |
   |---|---|---|
   | approved (lr 0.03, weight decay 1e-5, 600 epochs) | 3.0e-6 | 0.47 |
   | 5,000 epochs | 2.5e-6 | 0.52 |
   | lr 0.003, 5,000 epochs | 2.8e-6 | 0.54 |
   | weight decay 0, 600 epochs | 2.1e-6 | 0.38 |
   | weight decay 0, 3,000 epochs | 9.1e-7 | 0.63 (still improving at the last epoch) |

   The cause is scale. The target moves by about 0.001 within a size, so the loss gradient on a signal
   weight is of the order of 1e-6, the same order as the weight-decay term, and the intercept and the
   size coefficient are nearly collinear, which makes a first-order optimiser slow. Neither more epochs
   alone nor a smaller learning rate alone repairs it.

   Consequence: with the approved fit settings a real ranker could fail the 0.50 criterion because of
   the optimiser, not because the signals carry no utility. The settings in section 7 are approved and
   are NOT changed here. A change is a dated amendment that Walaa approves before any measurement.
   The amendment to consider: fit on standardised targets (train mean and SD), no weight decay or one
   set on that scale, and an epoch limit that the fit does not reach, then confirm on planted data that
   the fitted ranker comes close to the noise-free truth.
3. **The 0.50 criterion has a ceiling set by the noise.** With the planted noise of scenario 1 even the
   noise-free truth reaches only 0.74. How much room the real labels leave is known only after they are
   measured; the within-size reliability that G1 reports is the number to read then.

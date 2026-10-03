> **Superseded in part by Walaa's locked decisions of 2026-10-03.**
> - The policy-regression threshold (`scripts/asism_v2/thresholds.py`) is REMOVED; quantity follows
>   contract §10 (`scripts/asism_v2/stopping.py`).
> - The frozen-embedding (DINOv2 linear probe) instrument and the G0/G1 stages below are NOT ADOPTED.
> - The one-fixed-size requirement is lifted: §10 stopping needs the size term, so supervision spans
>   several sizes (contract §8).
> - The text below is kept as the record of what was proposed.

# ASISM V2 supervision audit and next experiment gate

This is a proposal for review, not authorization for GPU execution. Historical
V1 code, outputs and conclusions remain fixed. E4 is an independent quantity
experiment and is not used to tune the V2 utility instrument.

## What the completed supervision attempts establish

V1 measured 80 subsets (64 train, 16 validation) with one 224 px / 300 step
classifier run each. Its label was balanced-accuracy difference from a reused
real-only baseline. The set model's validation Spearman was -0.362. Image
coverage and same-size signal were poor. These measurements remain V1 evidence.

The later predeclared `utility_instrument_check` ran 12 historical subsets at
20 repeats each, 240 runs. Its label was absolute macro AUROC, without baseline
subtraction. Agreement with four-seed Stage-4-size results was rho=0.755,
p=0.0032, but reliability reached only 0.708 at 20 repeats against the frozen
0.80 criterion. **The instrument was rejected.** The report is at
`C:/Users/walaa/ham10000_work/followup/utility_instrument_check/instrument_check_report.json`.
No V2 training may relabel those 240 runs as accepted supervision, increase
the repeat count after seeing the result, or reuse the old 12 subsets as if
they constituted an independent confirmatory study.

The same-size Stage 4 recipe had ICC about 0.037 in the existing diagnostic;
its between-subset variation was much smaller than training variation. Even a
larger number of subsets does not repair noisy measurements of each subset.
The proposed 120-240 subset range is therefore **not justified as a GPU
measurement commitment**. It needs an instrument that first passes independent
reliability and validity checks at a fixed quantity.

## CPU architecture implemented

`scripts/asism_v2/features.py`: exactly four candidate features, DINO
similarity, IQA composite, MC-dropout mutual information and calibrated
Grad-CAM typicality. Train-only normalization. Missing, duplicated or
nonfinite features fail explicitly. The signs and sizes of contributions are
learned, including IQA; ablation against similarity-only remains required.

`scripts/asism_v2/pipeline.py`: additive class-specific image scores, trained
directly from measured subset utility. The set prediction combines the mean
image score, log set count, and class fractions. At fixed size and class mix,
same-class replacement utility is proportional to the image-score difference.
The architecture stores 4 slopes per class, centered class-mix terms, size and
intercept. *Correction (Claude, 2026-10-03):* at one fixed size AND one fixed per-class
allocation, which is the planned design, neither log(count) nor class-mix is identifiable; both
are frozen by `fit_ranker` and recorded in `history['frozen_unidentifiable']`. What is identified
is 28 class-specific slopes plus the intercept (29, not 35).
The size coefficient is frozen at zero for the planned fixed-size supervision:
that design cannot identify a quantity effect, which belongs to E4. It cannot
represent pairwise redundancy or
context-dependent marginal utility; that is an explicit model limitation and
requires a predeclared richer-model comparison if real labels support it.
Only train-role subsets update weights, validation selects the epoch, and
test-role outcomes are rejected by the fitting function. Images in those roles must be
disjoint. No image-level teacher labels are generated; therefore OOF teacher
distillation is not required for this architecture. If the architecture is
changed to distillation, OOF targets become mandatory.

`scripts/asism_v2/thresholds.py`: separate policy-utility regression predicts
the performance of predeclared per-class top-k choices, then chooses a count
and converts it to an exact score threshold. It requires measured classifier
outcomes for each policy, complete repeated grids, a fixed background policy,
image-disjoint policy contexts and held-out or OOF ranker scores. E4 supplies
the upper count; this policy may select fewer. The random comparator must match
the **realized** count per class. No real policy measurements are yet accepted,
so this learned threshold has no real fitted artifact or claim.

The model is **multi-signal, single objective**. Calling it a multi-objective
network would be inaccurate. A true multi-objective model requires separate,
reliable clinical/quality objectives and a frozen aggregation rule; current
rare-class outcomes do not establish those targets. This naming decision is
scientific, not a code formatting choice.

## Dataset split and sample count rationale for a future accepted instrument

At one fixed subset size of 96 from 3,168 candidates, a role-partitioned
five-cover design would have 165 subsets and five exposures per image: train
1,920 images = 20 subsets/cover = 100 subsets; validation 576 images = 6 per
cover = 30; test 672 images = 7 per cover = 35. Each role's subsets use only
its own image IDs; class-stratified assignment and composition variation must
be verified before measurement. This choice addresses coverage and independent
validation/test sample counts. It does **not** prove 100 train subsets are
statistically sufficient for 35 identifiable parameters, nor that any repeat count is
affordable. Do not generate or measure this design until a fixed-size
instrument has passed its own noise and transfer gate. The learning curve at
40/60/80/100 train subsets would be prespecified; the test 35 are reported once.

## Next genuinely new instrument: staged proposal

Scientific question: can a deterministic frozen-feature classifier rank
same-size synthetic subsets in agreement with the fixed Stage 4 recipe,
without the 224 px / 300 step proxy's training-seed noise?

Candidate instrument: extract one frozen, pinned DINOv2 embedding for each
`classifier_train` real image, each synthetic candidate and each
`asism_tuning_heldout` image. Fit a deterministic regularized multinomial
linear probe to real + subset embeddings; score absolute macro OVR AUROC on
`asism_tuning_heldout`. The CPU probe is implemented in
`scripts/asism_v2/instrument.py`: per-feature mean and population SD fitted on
training records only, one-hot ridge least squares with lambda=1 and an
unpenalized intercept, followed by softmax. No stochastic training or class
weighting is used. These values are
fixed for the G0 screen; no search is permitted against the 12 target outcomes. No
`classifier_val` or `final_eval_heldout` outcome is used. V3a remains the
separate uncertainty/CAM judge; this instrument does not replace it.

Stage G0 (exploratory screen): score the 12 historical subsets on the frozen
embedding instrument and compare with their existing four-seed full-recipe
means. Because these subsets have sizes 60/120/180, test **within-size**
agreement only: residualize ranks by size and use exact stratified permutations
(the observed sizes contain 3, 4 and 5 subsets, so 3! x 4! x 5! = 17,280
permutations). Predeclare passage as rho >= 0.60 and one-sided p <= 0.05.
This licenses a fixed-size check, not the full supervision run. A failed G0
rules out this proposed instrument without retuning regularization to those
12 outcomes.

Expected G0 GPU cost: one embedding pass over about 6,200 images. No measured
throughput for this exact task is recorded locally, so set a hard **2 GPU-hour
cap**, with a 100-image timing pilot included in that cap. A dollar estimate
needs the actual RunPod GPU type/rate. All linear-probe runs are CPU after
embeddings are frozen. Before a G0 GPU run, an extraction runner and provenance
writer must bind every image ID and file hash to the pinned DINO checkpoint;
these are not yet implemented. This document is not a launch command.

Stage G1 (separate later approval only if G0 passes): 24 independently frozen,
class-stratified **96-image** subsets, each measured by the full 512 px / 3,000
step Stage 4 recipe at seeds 42-45. Absolute macro OVR AUROC on
`asism_tuning_heldout`; per-class AUROC, BA and log loss secondary. The frozen
embedding instrument scores the identical 24 subsets. Before looking at the
results, require full-recipe label reliability at four seeds >= 0.50 and
cross-instrument Spearman rho >= 0.60 with one-sided exact/Monte-Carlo
permutation p <= 0.05. These are screening gates, not proof of ASISM benefit.
At historical 578.3 s per full-recipe run, 24 x 4 = 96 runs cost about 15.42
GPU-hours; four real-only controls would add 0.64 h if required for descriptive
context, for about 16.1 h. The controls do not enter the absolute label.
These costs and thresholds are proposal values for approval **before** any run.

If G1 passes, repeated measurements and the number of train subsets for a
full ranker dataset must be priced from the observed fixed-size variance;
do not default to 20 repeats or 165 measured subsets. Threshold policy
measurements are a separate intervention and budget, likewise unapproved.

The final endpoint remains ASISM vs class-count-matched random at the same
realized quantity, using a supervisor-approved protected-test policy. The
previously viewed `final_eval_heldout` cannot support an untouched new
confirmatory claim without the supervisor's decision.

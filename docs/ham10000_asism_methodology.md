# Stage 3 — Adaptive Synthetic Image Selection Module (ASISM)

*Methodology section, HAM10000 instantiation. Every claim below corresponds to code in
`scripts/asism/ham10000_*.py` and to settings frozen in `configs/ham10000_stage3.yaml`.*

---

## 3.1 Motivation and scope

Stages 1 and 2 produce a pool of synthetic dermoscopy images. They do not produce a guarantee that
those images are useful. A generator fitted to a class with 115 real examples can memorise them, can
drift toward a plausible-looking but diagnostically wrong appearance, and can produce technically
defective frames; adding such images indiscriminately to a training set is as likely to hurt as to
help. ASISM is the module that decides **which** synthetic candidates enter the training set, and it
is the contribution this work evaluates.

ASISM is a **curation** step, not a training-time mechanism. It runs once, after generation and
before classifier training, and emits a single artifact — `asism_selected.csv` — which Stage 4
condition C consumes. Nothing downstream of that file knows how selection was made, which is what
allows conditions A, B and C to be trained under an identical protocol.

**Data discipline.** The split namespace `ham-stratified-v1` partitions HAM10000 at the *lesion*
level into `gen_train`, `gen_val`, `classifier_train`, `classifier_val`, `asism_tuning_heldout` and
`final_eval_heldout`. Each ASISM component is permitted exactly one reference split and refuses the
others by name and by image-id overlap:

| Purpose | Split | Enforced in |
|---|---|---|
| Generator fitting | `gen_train` | Stage 1 |
| Similarity / explainability reference, IQA calibration | `gen_train` only | `ham10000_signals.FORBIDDEN_REFERENCE_SPLITS` |
| Auxiliary (uncertainty, agreement, CAM) classifier training | `classifier_train` | `FORBIDDEN_AUXILIARY_SPLITS` |
| Subset-utility measurement | `asism_tuning_heldout` | `ham10000_03_build_utility_subsets.TUNING_SPLIT` |
| Final evaluation | `final_eval_heldout` | Stage 5 only, behind `assert_final_eval_access_allowed` |

`final_eval_heldout` is refused by name in every earlier stage, and any signal artifact whose
provenance record mentions it is rejected outright by the Go/No-Go gate. Calibration routines
additionally check image ids, because disjointness "by construction" is an assumption rather than a
measurement.

**Preprocessing.** HAM10000 images are 600×450; the classifier input is 512×512. Non-uniform
rescaling would alter lesion asymmetry and border regularity — two of the four ABCD diagnostic
criteria — and centre-cropping would truncate lesions reaching the frame edge. All images are
therefore aspect-preserving-resized and letterboxed, and the exact **content box** (for a 600×450
source, `(0, 0.125, 1, 0.875)`) is persisted per image. Real and synthetic images pass through the
same function, which is asserted by test rather than by convention. Every subsequent measurement
that depends on where the image ends and the padding begins reads this box.

---

## 3.2 The five signals

Each candidate receives five independent scores. They are computed once, written as Parquet tables
with provenance sidecars, and never recomputed downstream.

**(a) Similarity — DINOv2.** Candidates are embedded with DINOv2 (pinned revision) and compared by
k-NN against real `gen_train` images *of the same class*. DINOv2 rather than CLIP: it is trained
self-supervised on natural images with no medical labels and no paired text, so it carries no
diagnostic prior into what is meant to be an appearance measurement. It is deliberately not
fine-tuned on HAM10000, since an encoder tuned on the reference images would measure closeness to
its own training set. Reported columns include `similarity_knn_mean`, `similarity_top1` and
`similarity_topk_spread`, plus a `novelty_is_near_duplicate` flag used as a safety check.

**(b) Image Quality Assessment.** Five defect flags (near-uniform, blurry, low-contrast, border
artifact, clipped) combine into `iqa_composite ∈ [0,1]`, refined by a continuous sharpness/contrast
term weighted below the gap between defect tiers, so the continuous part orders images *within* a
defect tier but can never let a defective image outrank a sound one.

Two thresholds required recalibration rather than inheritance. The blur threshold carried over from
the chest-radiograph configuration flagged 31.9% of clinically accepted real images as blurry: it
was measuring how smooth dermoscopy naturally is. It is now the 2nd percentile of Laplacian-variance
sharpness over real `gen_train` images — one pooled threshold, so candidates of every class meet the
same technical bar — and the code refuses to run on an uncalibrated value rather than falling back
to a constant. The border-artifact score is measured on the **content box**, not the full canvas: on
the canvas, 60% of the scored band is padding this pipeline added, and the flag fired when an
image's edge brightness happened to match the padding grey, which darker mel and bkl images do far
more often (11.5% and 11.8% flagged, against 4.6% for nv). Measuring inside the content box scores
the image's own edge, which is what the flag claims to measure.

**(c) Uncertainty — Monte-Carlo dropout.** An auxiliary DenseNet121 trained on `classifier_train`
(unweighted, one seed) is run with dropout active for *n* stochastic passes. The softmax predictive
distribution is decomposed into predictive entropy `H[E q]`, expected entropy `E H[q]` and their
difference, the mutual information, which is the **epistemic** component — model uncertainty as
distinct from inherent class ambiguity. All three are normalised by `log 7`. Deep ensembles would
estimate the same quantity with lower variance at the cost of training several full models, which
the compute budget did not permit; the choice is recorded rather than presented as equivalent.

**(d) Explainability verification — Grad-CAM.** Grad-CAM attributions are computed for each
candidate under the class it was generated as (its intended class: `class_index` from the
candidate's `dx`, `scripts/asism/ham10000_01_compute_signals.py`; corrected 2026-10-03, this read
"under its predicted class", which the code never did). Chest-radiograph anatomical priors do not transfer, so instead
of asking whether attention falls in an expected region, the method asks whether the attention
*pattern* is **typical** of real images of that class: a per-class reference distribution of CAM
statistics (focus area, peripheral mass) is built from real `gen_train` images, and each candidate
receives a conformal two-sided typicality score. Peripheral mass uses the exact content box, so
"outside the lesion content" is known rather than approximated by a fixed band; a candidate without
a persisted box stops the run rather than silently falling back. The reference model is identified
in every artifact by `cam_model_id = "ham10000-classifier:<sha256>"`, so a CAM computed under a
different model cannot be mixed with these.

**(e) Agreement.** Whether the auxiliary classifier's argmax matches the class the image was
generated as, with a confidence margin. On a single-label 7-class problem this is a well-defined
contrast that has no counterpart in the multi-label setting.

---

## 3.3 The Go/No-Go gate

A signal being implemented is not the same as a signal being fit to influence selection. Before any
weighting or learning, `ham10000_02_gonogo.py` subjects each signal to five **hard** checks
(technical validity, selection eligibility, numerical stability, missing rate, provenance) and two
**diagnostic** ones (redundancy, usefulness). Thresholds are set in the configuration *before* the
artifacts are inspected and are not adjusted to admit a signal that failed. Only surviving signals
contribute feature columns to the ranking network — the gate binds the learned selector exactly as
it binds a weighted composite.

Two design points are specific to this dataset:

**Redundancy is measured within class.** Computed on the pooled candidate set, the correlation
between any two signals is dominated by class structure rather than signal behaviour, because nv is
roughly two-thirds of the data. On identical data the pooled Spearman correlation reaches ≈0.97
while the class-size-weighted within-class correlation is below 0.5; the gate uses the latter, and a
test quantifies the difference so the choice is visible rather than asserted.

**Uncertainty is not given an a-priori direction.** It is not obvious whether a more or a less
uncertain synthetic image is preferable, so instead of asserting a direction the gate verifies the
decomposition identity `MI = H[E q] − E H[q]` to within 1e-9 and that all entropies lie in [0,1]. A
broken identity is a computational error; a direction would have been an assumption.

---

## 3.4 Multi-Objective Ranking Network

### 3.4.1 What has to be avoided

The five signals must be combined into one ordering. The conventional approach is a hand-weighted
sum, whose weights are either guessed or tuned against validation performance. The alternative of
training a network to predict "image quality" from the signals is circular: the signals are the
model's own inputs, so it would learn to restate them.

The method therefore trains on **measured** utility, defined at the level of a *set* of images.

### 3.4.2 Measuring subset utility

`ham10000_03_build_utility_subsets.py` builds 80 subsets of the safe candidate pool at three sizes
(60/120/180), mixing random, single-signal-stratified and mixed-stratified designs across quantile
bands, split into disjoint train-role and val-role pools. Each subset is added to the real
`classifier_train` set, a proxy classifier is trained to a fixed optimiser-step budget, and
**balanced accuracy** is measured on `asism_tuning_heldout`. Utility is the delta against a
real-only baseline trained under identical settings, one per seed and reused across subsets.

Balanced accuracy rather than accuracy: on a seven-class problem that is two-thirds nv, plain
accuracy is maximised by never predicting a rare class — the exact failure the synthetic data exists
to address, so a subset that helped df and akiec would be scored as useless.

The proxy is deliberately cheaper than Stage 4 (224 px / 300 steps against 512 px / 3000). Its job
is to **rank** subsets, not to estimate final performance; every subset is measured under identical
settings and one fixed budget, so the comparison between them is fair. No number measured here is
reported as a result, and this is recorded in the artifact.

A feasibility phase runs first and refuses the build unless the design is actually drawable from the
real pool — in particular that the largest subset size fits inside the smallest val-side quantile
band, the binding constraint, since a single-signal subset draws entirely from one band of one
feature within one role pool. No size or ratio is ever adjusted automatically to make a failing
report pass.

### 3.4.3 From set utility to image utility

Subset utility cannot be copied onto member images: a poor image inside a good subset would inherit
a high score. A **Deep Sets** network (`SetUtilityNetwork`, masked mean pooling, permutation
invariant) is fitted to predict a subset's measured utility from its members' features. Each image's
target is then its **size-normalised leave-one-out marginal**: the predicted utility of a subset with
the image, minus without it. Targets are distilled from train-role subsets only; an image appearing
in fewer than three subsets receives no target, because a marginal averaged over one or two
measurements is noise.

An independent **Banzhaf (MSR)** estimator is computed over the same measurements at no extra GPU
cost, and its Spearman correlation with the distilled targets is recorded. Agreement is evidence the
ordering is not an artifact of the Deep Sets model; disagreement is a finding for the write-up, not
something to average away.

The ranking network (`MultiSignalUtilityRankingNetwork`) is then fitted on the admitted feature
columns to predict the standardised target, and scores every candidate — including those no subset
contained. Generalisation is measured, not assumed: pairwise ranking accuracy is reported on images
appearing **only** in val-role subsets, which the set model never trained on and which contributed
no target.

### 3.4.4 On the name

The network is *multi-objective* in that it **fuses five independent quality objectives into a
single ranking decision, learning their relative contribution from measured utility rather than from
hand-set weights**. It is **not** multi-objective optimisation in the Pareto sense. It optimises a
single target — an image's predicted marginal contribution — using two complementary loss terms: a
Smooth-L1 term for the value and a pairwise term for the order. Both score the same quantity; no
Pareto front is produced and no trade-off surface is exposed. The manifest records this as
`"optimisation": "single_objective_two_loss_terms(smooth_l1 + pairwise_ranking)"`.

This is a deliberate choice rather than a limitation: a Pareto front would still require selecting
one operating point, and that selection is the original problem restated.

---

## 3.5 Adaptive Threshold Learning

A single global cut cannot serve this dataset. nv has thousands of real images and its candidates
compete against a rich real distribution; df and vasc have around a hundred, and a cut set on the
pooled score distribution — two-thirds of which is nv — would admit almost nothing for them.

**Order is part of the method** and is fixed in code, not configuration:

```
safety → quality floor → class threshold → per-class floor → budget cap
```

The quality floor is **absolute and pooled** (25th percentile of the normalised ranking score) and
runs first, so rarity never buys admission for a technically poor image. A per-class floor would
define "bad" relative to each class's own candidates, letting the weakest class's mediocre images
clear a standard its own mediocrity had set. The per-class floor then runs *after*, so the most it
can admit is a starved class's **best remaining** candidates. A class with nothing above the pooled
floor selects nothing, and that outcome is reported explicitly — it is a finding about the generator
for that class, and Stage 4 must be read knowing it. Per-class attrition at the floor is reported
alongside, since a pooled floor need not remove the same share of every class.

Per-class budgets are proportional to each class's real training count, clipped to
[50, 2000]: a flat quota would hand df as many synthetic images as nv and change the class balance
far more than augmentation intends.

Two policies are computed on every run; a **pre-registered** configuration key decides which one
writes the selection, so the choice cannot be made after seeing the outcome.

**`percentile` (baseline, training-free).** Each class's admission percentile is scaled down — more
lenient — the rarer the class is in the real population, adapted from FreeMatch's self-adaptive
thresholding (Wang et al., ICLR 2023) and SST's class-fairness term (Zhao et al., IP&M 2025), but
applied as a one-time curation decision rather than a per-step recomputation. On HAM10000's real
distribution the effective behaviour is *nv strict (50th percentile), every other class lenient
(10th–16th)*; range normalisation and division by the largest prevalence agree to within 0.7
percentile points, so the normalisation choice is not consequential and is not claimed to be.

**`network` (proposed).** A frozen one-dimensional grid search finds each class's threshold under an
explicit objective — maximise the mean score of what is kept, penalised by the relative deviation
from the class's budget — with the grid, objective and tie rule fixed in advance (ties resolve to the
more selective cut). `AdaptiveThresholdNetwork` is then distilled from those thresholds given a
**class context vector**: real prevalence, log candidate count, the shape of the class's score
distribution (mean, sd, quartiles) and its target fraction. The context deliberately excludes the
candidates themselves, which the ranking network has already judged.

> **Stated limitation, reported in the manifest.** The distillation residual against the searched
> thresholds is a **fidelity** measure, not evidence of a learned rule. The network receives a
> per-class embedding and is fitted on one point per class, so it can drive the in-sample residual to
> near zero through the embedding alone. A leave-one-class-out residual is therefore computed and
> reported alongside it: each class is predicted by a network that never saw it, so only the context
> can supply the answer. Where the leave-one-class-out residual greatly exceeds the in-sample one,
> the thresholds in use are effectively the frozen search's own, and the network should be described
> as a smoothing of that search rather than as a learned threshold rule. With seven classes this is
> an inherent limit of the setting, not a defect of the fit, and it is reported either way. Neither
> residual is ever used to change the thresholds applied.

The safety gate — invalid IQA, memorised near-duplicates — is re-run against the **current**
artifacts at selection time and is independent of the Go/No-Go verdict: a memorised near-copy of a
real patient's lesion must not become selectable because the similarity signal happened to be
redundant with another one. It fails **closed**: a candidate with no row in a safety artifact is
treated as unsafe, and the manifest records which checks ran, so a removal count of zero cannot be
misread as "nothing failed" when the truth is "nothing was checked".

The module writes `asism_selected.csv` plus a provenance sidecar naming the namespace it came from.
An empty selection is refused rather than written, since it would make condition C a second copy of
condition A under a different name.

---

## 3.6 What ASISM does not claim

- It does not claim the selected images are clinically valid; it claims they are the subset whose
  members had the highest estimated marginal contribution under a measured proxy objective.
- It does not claim the signals are independent, only that no surviving signal is a near-restatement
  of another within class.
- It does not claim the proxy classifier's balanced accuracy approximates Stage 4 performance; it is
  a ranking instrument measured under one fixed budget.
- It does not claim multi-objective optimisation in the Pareto sense (§3.4.4).
- It does not claim the threshold network learned a transferable rule unless its leave-one-class-out
  residual supports that (§3.5).

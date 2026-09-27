# HAM10000 v2 — decision criteria for the three classifier-dependent signals

Status: **fixed before the analysis runs.** Agreed with Walaa on 2026-09-27. After the analysis
runs, the thresholds and rules below are applied as written. They are not changed to fit the result.
If a rule turns out to be wrong, the change is written here as a dated amendment. It says what the
result was when the change was made, and both verdicts are reported.

## What is being judged

The v2 auxiliary classifier (`auxiliary_classifier_v2`,
`cam_model_id = ham10000-classifier:db8f5001086c4507825f644134f91d4a62f7c51eb48945f8df769cf19f8f82e6`)
recomputed three signals on the 3,168 Stage 2 candidates of `ham-stratified-v1`:

- uncertainty
- agreement
- explainability

The artifacts are in `outputs/ham10000/stage3_aux_v2/ham-stratified-v1/signals/`. The v1 artifacts,
in `outputs/ham10000/stage3/ham-stratified-v1/signals/`, are the comparison. Their 36 stage3 files
were hash-identical before and after the v2 runs.

The question is not whether the classifier scores well. It is whether the classifier is good enough
for these three signals to be useful to ASISM. The answer decides one thing: keep v2, or build a v3
classifier.

## Disclosure: what was seen before these criteria were written

The signal driver prints two summary lines into its log, and both were seen before this document
existed:

- every one of the 3,168 candidates fell in the uncertainty band `low`;
- the argmax matched the intended diagnosis for 33.0% of candidates, pooled.

Nothing else from the v2 signal artifacts was looked at before this document was written. The files
were downloaded, their hashes checked, and their column names read. The criteria for Q3 (and the
pooled part of Q1) are therefore not blind, and are reported as such.

## General rules

- Both v1 and v2 are computed on the same 3,168 candidates, joined on `image_id`. Every quantity is
  reported pooled and per class. The class is the intended diagnosis, `dx` in `all_candidates.csv`.
- These are descriptive quantities, not hypothesis tests. Each proportion carries a 95% Wilson
  interval.
- The real per-class recall used in Q1 is the v2 classifier's recall on `classifier_val`, as written
  by the v2 training run. It was recorded before this document was written:

  | Class | nv | mel | bkl | bcc | akiec | vasc | df |
  |---|---|---|---|---|---|---|---|
  | Recall | 0.928 | 0.463 | 0.570 | 0.721 | 0.467 | 0.65 | 0.40 (6/15) |

  For the v1 column of Q1, v1's own `classifier_val` recall is used.
- Configuration is not changed for this analysis. That includes the uncertainty bands (0.05 and
  0.20), the agreement rival threshold and penalty weight, and the explainability border and mass
  fractions.

## Q1 — Is agreement sensible?

Measured, per class:

- the match rate: the mean of `agreement_is_argmax_match`;
- the normalised match rate: the match rate divided by the real `classifier_val` recall of that
  class.

The normalised rate is needed because the classifier does not recognise every real image of a class
either. Holding synthetic images to 100% would be unfair.

**Sensible** if both of these hold:

1. at most **1** class has a match rate at or below chance (1/7 = 0.143, point estimate);
2. across the 7 classes, the Spearman correlation between the synthetic match rate and the real
   recall is **≥ 0.5**.

## Q2 — Is mel→nv still influential?

Measured:

- the share of `mel` candidates predicted `nv` (`agreement_predicted_diagnosis`), for v1 and v2;
- among all mismatched candidates in all classes, the share predicted `nv`. This says whether `nv`
  has become the sink for errors.

**Still influential** if either of these holds:

1. the share of `mel` candidates predicted `nv` is **≥ 30%**;
2. `nv` receives **≥ 50%** of all mismatches.

## Q3 — Is uncertainty still near zero?

Measured: the normalised mutual information, `uncertainty_mutual_information / ln 7`. Its median,
P90, P99 and IQR are reported, with `uncertainty_mean_std` beside them. The script first checks that
this normalisation reproduces `uncertainty_band` for every row, against the unchanged bands 0.05 and
0.20. If it does not, the script stops rather than guessing the unit.

**Degenerate** (does not separate candidates) if both of these hold, pooled:

1. P90 **< 0.01**;
2. IQR **< 0.005**.

## Q4 — Is explainability more discriminative?

Measured, per class:

- the IQR of `explainability_calibrated_typicality`;
- the tie share: the fraction of candidates whose `explainability_peripheral_mass` value is shared,
  exactly, with at least one other candidate of the same class.

A class is **discriminative** if its IQR is **≥ 0.15** and its tie share is **< 20%**. The number
of the 7 classes that meet this is reported for v1 and for v2.

The `df` verdict is marked fragile: its v2 Grad-CAM reference has only 38 images.

## Q5 — Is there redundancy among the three?

Measured: the Spearman correlation between each pair of `agreement_score`, normalised mutual
information and `explainability_calibrated_typicality`. It is computed pooled and within each
class.

A pair is **redundant** if |ρ| is **≥ 0.7** pooled, and also within **≥ 4** of the 7 classes.

## Decision rule

| Outcome | Decision |
|---|---|
| Q1 sensible, Q2 not influential, and Q4 discriminative in ≥ 4 classes | **v2 is sufficient.** Move on to U1 (utility). |
| Q1 not sensible, or Q2 influential | **Build v3**, aimed at the classifier itself. |
| Q3 degenerate | **Not on its own a reason for v3.** The cause may be the MC Dropout method, not the classifier. Replacing it (for example with TTA or an ensemble) is a separate design decision for Walaa, fixed in writing before it is tried. |
| Q5 redundant | Does not change the v3 decision. Recorded as an input to the later ranking. |
| Q1 sensible and Q2 not influential, but Q4 discriminative in < 4 classes | **v2 is sufficient.** Explainability is recorded as a weak signal, and its fate is decided in the ranking. The weakness may lie in Grad-CAM itself rather than in the classifier. (Decided by Walaa on 2026-09-27, before the analysis ran.) |

## The v3 design, fixed now and built only if the rule above calls for it

v3 is written down before the analysis runs so that, if it is needed, it is not designed after the
result is seen. Nothing below is trained until the decision rule says "build v3".

**One change per step.** Each step is judged before the next one starts.

### Step v3a — checkpoint selection

Everything else is the same as v2, including the following:

- architecture;
- loss and inverse-frequency class weights;
- flips;
- `max_steps` 3000, batch size, learning rate and seed.

The only change is which checkpoint is kept:

- a checkpoint is saved every **250 steps**, which gives 12 candidates;
- the checkpoint with the highest **balanced accuracy on the selection data** is kept. Two
  checkpoints are tied if their balanced accuracies differ by **< 0.005**. The tie goes to the
  lower **NLL** on the same data;
- `classifier_val` is **not** used to choose anything. It stays the acceptance split, and the
  acceptance is measured once, on the chosen checkpoint, with criteria A and B unchanged.

The source of the selection data is **OPEN — to be decided by Walaa before v3a is trained.** It may
not be `gen_train`, which is the Grad-CAM reference, or `final_eval_heldout`, which is protected. The
counts that make this hard, from `splits/ham-stratified-v1`:

| Split | Images | vasc | df | Role today |
|---|---|---|---|---|
| classifier_train | 1,641 | 21 | 25 | trains the classifier |
| asism_tuning_heldout | 1,377 | 20 | 13 | ASISM tuning (utility / set model) |
| gen_val | 388 | 6 | 4 | LoRA validation (finished) |

Options:

1. **A held-out part of `classifier_train`.** Split by `lesion_id` and stratified, about 15%. This
   leaves about 3 vasc and 4 df images to select on, and takes them out of training.
2. **`asism_tuning_heldout`.** It is larger, but it mixes the classifier's model selection into the
   split that U1 (utility) will tune on.
3. **Cross-validated step selection on `classifier_train`.** Use 5 folds, split by `lesion_id` and
   stratified. For each fold, train on 4 folds and score the 12 checkpoints on the 5th. Pick the
   step with the best mean balanced accuracy across the folds, with the same NLL tie rule. Then
   retrain on all of `classifier_train` and keep that step. No other split is touched and no
   training data is lost. It costs about 6 trainings instead of 1.

### Step v3b — calibration (only if v3a is kept and a confidence problem remains)

Temperature scaling, with one scalar fitted by NLL on the same selection data as v3a.
`classifier_val` is not used to fit it.

Temperature scaling does not change the argmax. It cannot move the match rate in Q1 or the mel→nv
share in Q2. It can only change `agreement_score`, the probabilities, and the uncertainty
quantities. So v3b runs only if, after v3a:

- Q1 is sensible,
- Q2 is not influential,
- and Q3 is still degenerate.

If Q1 or Q2 still fail after v3a, v3b is not the fix. The next change is chosen from the evidence
and written here before it is tried.

## Out of scope for this analysis

The analysis does not touch the following:

- similarity or IQA;
- the ranking network;
- thresholds;
- selection;
- Stage 4;
- the final evaluation.

It reads the downloaded artifacts only and runs on the CPU.

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

Measured: the normalised mutual information, `uncertainty_mutual_information`. The column is
already divided by ln 7 when it is written (see Amendment 1). Its median, P90, P99 and IQR are
reported, with `uncertainty_mean_std` beside them. The script first checks that the column
reproduces `uncertainty_band` for every row, against the unchanged bands 0.05 and 0.20. If it does
not, the script stops rather than guessing the unit.

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

The source of the selection data was chosen by Walaa on 2026-09-27, before the analysis ran:
**option 3, cross-validated step selection on `classifier_train`.** The choice is conditional: it
is carried out only if the decision rule calls for v3. The data may not come from `gen_train`, which
is the Grad-CAM reference, or from `final_eval_heldout`, which is protected. The counts that make
this hard, from `splits/ham-stratified-v1`:

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

Options 1 and 2 were rejected:

- Option 1 leaves too few vasc and df images to choose a checkpoint on.
- Option 2 mixes classifier design into the split U1 needs later.

How option 3 is carried out, fixed now:

- **Folds.** 5 folds of `classifier_train`, grouped by `lesion_id` so that no lesion is on both
  sides. They are stratified by `dx`, with seed 42.
- **Training.** Each fold model is a full v2-recipe training to 3000 steps, saving the 12
  checkpoints (every 250 steps). All 12 come from one run per fold, and no checkpoint needs a run of
  its own. The total is 5 fold runs plus 1 final run, about 6 times the compute of v2.
- **Scoring.** Balanced accuracy is computed on the **pooled out-of-fold predictions**, not as the
  mean of 5 per-fold values. Every one of the 1,641 images is scored exactly once, by the model that
  did not train on it, and one balanced accuracy is computed per step. A single fold holds only about
  5 df and 4 vasc images, so per-fold values would be dominated by one or two images. Pooling puts
  all 25 df and 21 vasc images into the decision.
- **Ties.** A step within 0.005 of the best balanced accuracy is tied with it. The tie goes to the
  lower pooled out-of-fold NLL.
- **The final model.** It trains on all of `classifier_train` and stops at the chosen step, **with no
  rescaling.** Limitation: each fold model saw 80% of the images, so the same step count means
  slightly more passes over the data than in the final run. This is recorded, not corrected.
- **What stays out.** `classifier_val`, `asism_tuning_heldout` and `final_eval_heldout` take no
  part in choosing the step.

### Step v3b — calibration (only if v3a is kept and a confidence problem remains)

Temperature scaling: one scalar, fitted by NLL on the pooled out-of-fold logits at the chosen step,
the same data v3a selected on. It is then applied to the final model. `classifier_val` is not used
to fit it.

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

## Amendments

### Amendment 1 — 2026-09-27, before the analysis ran: the unit of Q3

The first version of Q3 said to divide `uncertainty_mutual_information` by ln 7. That column is
already divided by ln 7 when it is written (`compute_uncertainty_scores` in
`scripts/asism/ham10000_signals.py`, `scale = log(n_classes)`), so dividing again would have
normalised it twice. This was found while writing the analysis script, from the code, before any
value of the column had been read.

- **Corrected:** Q3 now reads the column as it is.
- **Unchanged:** the thresholds (P90 < 0.01, IQR < 0.005), the bands, and the check that the column
  reproduces `uncertainty_band`.
- **No second verdict:** no result existed under the old wording, so there is none to report beside
  the new one.

### Amendment 2 — 2026-10-02, after v3a was trained and before any v3 signal exists: applying the rule to v3

v3a finished on 2026-10-02 and passed acceptance on `classifier_val`. Cross-validation chose step 1500,
with a pooled out-of-fold balanced accuracy of 0.618 and no tie within 0.005. The final model scored
balanced accuracy 0.582 and macro-F1 0.569, and no class had zero recall. Its identity is
`ham10000-classifier:6849c456a9581c598a423b93b9c2cdbdbb79118fd6d31c72fe68eefcbe5f8bfd`.

This amendment was written before the v3 Grad-CAM reference or any v3 signal was computed. No
synthetic image had been scored by v3, so no Q1–Q5 number for v3 existed.

**Which classifier is frozen for the recompute (decided by Walaa, 2026-10-02).** v3a is the frozen
auxiliary classifier for the three classifier-dependent signals, because it passed acceptance criteria
A and B unchanged. Its balanced accuracy on `classifier_val` is 0.018 below v2's 0.600, and that gap
decides nothing:

- it is smaller than the seed-to-seed spread measured on this recipe, which is about 0.04;
- the criteria judge a classifier by how it reads the synthetic pool (Q1, Q2), not by its real
  balanced accuracy;
- v2 has already failed Q1 and Q2, so it is not a fallback.

Both classifiers are **not** scored on the synthetic pool so that the better-looking one can be kept.
That would choose the instrument after seeing its output.

**The rule, applied to v3 as written.** Q1–Q5, every threshold in this document, and the decision
table are applied to v3 unchanged. v1 and v2 are reported beside it for comparison only. The rows of
the decision table are read with "v3" in place of "v2":

| v3 outcome | Next step |
|---|---|
| Q1 sensible, Q2 not influential, Q4 discriminative in ≥ 4 classes | v3 is sufficient. Move on to U1 (utility). |
| Q1 sensible, Q2 not influential, Q4 discriminative in < 4 classes | v3 is sufficient, and explainability is recorded as a weak signal. |
| Q1 and Q2 pass, Q3 degenerate | Step v3b (temperature scaling), as written above. |
| Q1 not sensible, or Q2 influential | Training length was not the cause. The next change is chosen from the evidence and written here before it is tried. It is not chosen now. |

**v3's real recall, for Q1's normalisation.** These are the correct predictions and the support per
class, from the diagonal and row sums of the v3 acceptance confusion matrix
(`selection_metrics.json`, `classifier_val`, 1,401 images):

| nv | mel | bkl | bcc | akiec | vasc | df |
|---|---|---|---|---|---|---|
| 786/955 | 92/149 | 98/149 | 50/68 | 19/45 | 11/20 | 4/15 |

**A caution recorded before the result.** v3 predicts mel far more often than v2 does:

- 100 of the 955 real nv images in `classifier_val` were predicted mel;
- the precision for mel is 0.387.

So a higher mel match rate for synthetic mel is not on its own evidence that v3 recognises synthetic
mel. It could come from a general shift towards predicting mel. Q1's normalisation by real recall
covers part of this. The v3 mel match rate is to be read together with three things:

- its normalised value;
- how often synthetic nv is predicted mel;
- the mel share of all mismatches.

This caution changes no threshold and adds no criterion. It says how a passing number is to be read,
and it is written now so that it cannot be written to suit the result.

**Unchanged:**

- similarity and IQA;
- the v1 and v2 artifacts, which are not rewritten;
- every threshold;
- the out-of-scope list above.

### Amendment 3 — 2026-10-02, after the v3 verdict and before any probe exists: the next step, chosen from the evidence

**The v3 outcome, under Amendment 2 as written.** The v3 signals were computed on 2026-10-02 from
commit `357f98b`. The v1, v2 and Stage 2 files were hash-identical before and after the run. The
diagnostic gave:

- Q1 **not sensible**:
  - one class at or below chance (bkl, 1.8%);
  - Spearman 0.321, below 0.5.
- Q2 **influential**:
  - mel predicted nv 84.8% (234/276);
  - nv receives 89.4% of all mismatches.
- Q3 not degenerate, Q4 discriminative in 6 classes, Q5 no redundant pair.

The row "Q1 not sensible, or Q2 influential" applies: **training length was not the cause.** The
reading aids of Amendment 2 show that the rise in the mel match rate (0.4% to 15.2%) is not a
general shift towards mel. Only 1 of 112 synthetic nv images was predicted mel, and mel takes 3.1%
of all mismatches.

**What the evidence does and does not say.**

- The v3 classifier recognises real mel (recall 0.617) but reads 84.8% of synthetic mel as nv.
- v2 and v3 were both trained with inverse-frequency class weights. The nv share of the training
  data is therefore already compensated in the loss.
- The DINOv2 check (`docs/ham10000_synthetic_separability.md`, S3) found that synthetic mel and bkl
  lean towards nv **less** than real ones do.

Two explanations remain, and the evidence so far does not separate them:

1. **Classifier-specific.** The DenseNet classifier responds to the synthetic images differently
   from real ones, through domain shift or reliance on cues the synthetic images lack.
2. **Class fidelity.** The synthetic images do not carry the features by which real mel is
   recognised. They may still be separable by a cue of the generator's own, which is what S1's
   higher synthetic separability would also allow.

**The next step: an independent probe (P1).** A second classifier is trained on real images only,
from features that are not DenseNet's, and is read on the synthetic pool with the same Q2.

- If it does not read synthetic mel as nv, the behaviour is specific to the DenseNet classifier.
- If it does, real-trained classifiers of two kinds agree, and the images become the likelier
  cause.

**P1, fixed now:**

- **Features:**
  - the DINOv2 embeddings of the separability check;
  - `ham10000_work/separability/embeddings.npz`, sha256 `a82b9311…`;
  - encoder `vit_small_patch14_dinov2`, 384 dimensions;
  - 3,586 real `gen_train` images and 3,168 synthetic candidates.
  - Each vector is L2-normalised.
- **Model:** multinomial logistic regression with a weight matrix and a bias. The loss is the
  class-weighted mean cross-entropy plus 1e-4 × the squared L2 norm of the weights (not the bias).
  - The weights are inverse-frequency, normalised to mean 1 (`class_weights_from_records`), as for
    v2 and v3.
  - 1e-4 is the weight decay of the v2 and v3 recipe. It is a single value and is not tuned.
  - Optimiser: full-batch L-BFGS from zeros, run until the loss changes by less than 1e-9 or 1,000
    iterations.
- **Real-data check (cross-validation):**
  - five lesion-grouped folds on the real `gen_train` images, `lesion_grouped_folds` with seed 42;
  - lesion ids come from `HAM10000_metadata.csv`;
  - out-of-fold predictions give the probe's real recall per class.
- **Validity gate:** the out-of-fold predictions must pass the auxiliary classifier's own acceptance
  criteria A and B:
  - balanced accuracy > 0.478;
  - no class with zero recall.
  - A probe that fails the gate says nothing about the synthetic images.
- **Synthetic scoring:**
  - the probe is refitted on all 3,586 real images;
  - it predicts the 3,168 synthetic candidates;
  - Q1 and Q2 are computed exactly as defined above;
  - Q1 is normalised by the probe's own out-of-fold real recall.

**Decision table for P1.** It is read on Q2 alone, because Q2 is the question that failed and the
one the two explanations disagree on. Q1 for the probe is reported beside it.

| P1 outcome | Reading | Next step |
|---|---|---|
| Gate fails | P1 is not informative | Recorded as it is. Walaa decides; nothing is retried with other settings. |
| Gate passes, Q2 not influential | The nv reading is specific to the DenseNet classifier | A classifier-side change, chosen and written here before it is tried |
| Gate passes, Q2 influential | Two kinds of real-trained classifier read synthetic mel as nv; class fidelity is the likelier cause | A generator-side step, chosen and written here before it is tried |

Q2 is influential if either of its two conditions holds, as above. Which condition held is reported,
and a result that rests on one condition only is said to be so.

**Descriptive only.** These decide nothing.

- **D1, near or far misses.** These are the 234 synthetic mel images that v3 predicted nv.
  - Read from `agreement_intended_prob` (mel), `agreement_best_rival_prob` (nv) and
    `agreement_margin`.
  - Reported: the median and quartiles of p(mel).
  - Reported: the share where mel is certainly the runner-up, that is
    p(mel) > 1 − p(nv) − p(mel).
  - The full probability vector was not stored, so the exact rank of mel is not known when this
    condition fails.
- **D2, does DenseNet agree with DINOv2?**
  - For every synthetic mel image, take its cosine similarity to the real mel centroid minus its
    similarity to the real nv centroid. The centroids are built from all real `gen_train` images.
  - Compare the 42 images v3 read as mel with the 234 it read as nv.
  - Reported: the medians and the difference, with a 95% bootstrap interval (1,000 resamples,
    seed 42).

**Dropped, with the reason, before any value was read:**

- **A prior correction of the v3 logits.** v3 was trained with inverse-frequency weights, so its
  outputs do not carry the training prior in a form that can be subtracted once. Adjusting them
  would correct twice, and the full logits were not stored.
- **A comparison across generation settings.** `all_candidates.csv` records only a seed, distinct
  for every image. No prompt, guidance or other setting varies, so there is nothing to group by.

**Deferred.** A blind visual audit of synthetic mel, predicted mel against predicted nv:

- the full candidate images are on the network volume only;
- it is exploratory, not a clinical reading;
- it is done only if Walaa asks for it.

**Out of scope:** the generator, the v3 classifier and its signals, similarity and IQA, the ranking
network, thresholds, selection, Stage 4 and the final evaluation. P1 reads existing files and runs on
the CPU.

### Amendment 4 — 2026-10-02, after P1 and before P1b exists: the same probe on the classifier's own images

**The P1 outcome, under Amendment 3 as written.** P1 ran on 2026-10-02 from commit `eca89a9`, on the
CPU. The output is in `ham10000_work/v3_signals/probe_p1/`.

- The gate passed: out-of-fold balanced accuracy 0.643, and no class with zero recall.
- Q2 **not influential**:
  - mel predicted nv 12.0% (33/276);
  - nv receives 25.8% of all mismatches.
- Synthetic mel was read as mel in 76.8% (212/276) of cases. For v3 the figure was 15.2%.
- Q1, reported only: Spearman 0.821, one class at or below chance.
- **Decision: classifier_specific.**

Descriptive, deciding nothing:

- synthetic bkl is read as bkl in 1.5% of cases by the probe and 1.8% by v3, so both classifiers
  fail on it;
- v3's mel misses are far misses: median p(nv) 0.956, median p(mel) 0.030;
- the mel images that v3 read as nv are, in DINOv2, slightly more mel-like than the ones it read as
  mel (difference −0.0115, interval [−0.0258, −0.0003]).

**A caveat Amendment 3 did not state.** P1 was trained on `gen_train`, the real images the LoRA was
trained on. v3 was trained on `classifier_train`. The P1 result may therefore come from **which real
images** the probe learned from, and not only from **which features** it uses.

- A probe that knows the generator's own training images may read the generator's output more
  easily.
- This does not change P1's decision, which was read as written. It limits what that decision
  establishes.

**The next step: P1b.** P1b is the same probe on the images the v3 classifier saw.

- **Unchanged from P1:**
  - the model, the class weighting, the L2 penalty of 1e-4, and the L-BFGS settings;
  - the encoder and weights: `vit_small_patch14_dinov2`, `timm/vit_small_patch14_dinov2.lvd142m`
    at revision `936966a8…`;
  - the 3,168 synthetic embeddings, sha256 `a82b9311…`;
  - Q1, Q2 and the P1 decision table.
- **Changed:**
  - **Training:** all of `classifier_train`, 1,641 images, as for v3. There is no cross-validation.
  - **Gate:** the auxiliary classifier's acceptance criteria A and B on `classifier_val` (1,401
    images), as for v3. That means balanced accuracy > 0.478 and no class with zero recall.
  - **Q1's normalisation:** the probe's recall on `classifier_val`.
- **New embeddings:**
  - `classifier_train` and `classifier_val` only;
  - same encoder and weights as the separability embeddings, same `embed_images` code as the
    similarity signal, and the same preprocessed images that v3 was trained and accepted on;
  - written to a new directory, `outputs/ham10000/diagnostics/independent_probe/`. The separability
    embeddings are not rewritten.
  - A `final_eval_heldout` image is refused.
  - The two files must record the same encoder, weights revision and weights sha256, or P1b
    refuses to run.

**Reading P1 and P1b together, fixed now.**

| P1b outcome | Reading | Next step |
|---|---|---|
| classifier_specific | The caveat is closed. The nv reading belongs to the DenseNet classifier. | A classifier-side change, chosen and written in an Amendment 5 before it is tried |
| class_fidelity_likelier | P1's result came from training on the generator's own images. The reading is **mixed**. | Written as it is; Walaa decides |
| not_informative (gate fails) | P1b says nothing. P1 stands, with the caveat above. | Recorded as it is; nothing is retried with other settings |

D1 and D2 are not repeated, because they do not depend on the probe.

**Out of scope:** the same list as Amendment 3. P1b embeds real images on a GPU pod, then trains and
reads the probe on the CPU.

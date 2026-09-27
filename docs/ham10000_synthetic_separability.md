# HAM10000 — are the synthetic classes as distinct as the real ones?

Status: **fixed before the check runs.** Agreed with Walaa on 2026-09-27. It is an explanatory
diagnostic. It does not change `docs/ham10000_v2_signal_criteria.md`, whose verdict (build_v3)
stands as recorded.

## Why this check exists

The pre-registered v2 signal diagnostic returned **build_v3**: Q1 was not sensible and Q2 was
influential. The v2 classifier recognises real mel and real bkl at 46% and 57% (`classifier_val`
recall), but it matched only 0.4% of the synthetic mel and 0.4% of the synthetic bkl. 99.6% of
synthetic mel was read as nv. v3a, checkpoint selection, fixes the classifier. It cannot close a gap
that lies in the synthetic images. Before v3a's cost (about 6 trainings) is spent, this check asks
which of the two the gap comes from.

## Exploratory visual inspection (recorded, not evidence)

On 2026-09-27, 8 images per cell were drawn with seed 42 and viewed side by side:

- classes mel, bkl and nv;
- real images from `gen_train`, synthetic images from the scored pool of 3,168;
- the list is in `ham10000_work/v2_signals/visual_sample.json`.

Claude's reading:

- the real images vary widely in colour, size, background and lighting;
- the synthetic images of all three classes share one look: a pink-lilac speckled background and
  a centred red-brown lesion;
- synthetic mel keeps some mel features, such as irregular borders and a few blue-black dots, but
  its overall colour and shape sit closer to synthetic nv than to real mel;
- synthetic bkl looks like a light nevus.

Limitations:

- the reader is not a dermatologist;
- there are 8 images per cell;
- the reading was not blind, because the 99.6% mel→nv figure was already known;
- Walaa opened the predictions page before recording her own impression.

This reading is a hypothesis, and the check below tests it.

## Method

- **The encoder** is DINOv2, exactly as pinned for the similarity signal
  (`configs/ham10000_stage3.yaml`, `signals.similarity`: `vit_small_patch14_dinov2`, revision
  `936966a8…`). It is self-supervised with no medical labels, so it compares appearance without a
  diagnostic prior. It is independent of the auxiliary classifier under suspicion.
- **The images** are all 3,168 synthetic candidates of `ham-stratified-v1` and all 3,586 real
  `gen_train` images, which are the images the generator imitates. Each embedding is L2-normalised.
- **No new thresholds.** Every question compares synthetic with real. Its verdict rests only on
  whether a 95% interval of the difference excludes 0.
- **Seed** 42 throughout.

### S1 — Are the classes separable?

Per draw:

- take **38 images per class**, the smallest real class (df in `gen_train`), from real and from
  synthetic alike, so both sides have the same size and balance;
- in each set, score a nearest-centroid classifier (cosine to the class means of the training
  folds) with stratified 5-fold cross-validation, and record its balanced accuracy.

There are **200 draws**. The difference is synthetic − real balanced accuracy. The 2.5th and 97.5th
percentiles over the draws are reported.

**Confirms the hypothesis** if the 97.5th percentile is **< 0**, meaning the synthetic classes are
less separable. The pooled out-of-fold confusion of the synthetic set is reported with it.

### S2 — Is the variety within a class lower?

Per draw and per class, take 38 images from each side and compute the mean pairwise cosine
distance. There are 200 draws.

- **Per class:** the difference is synthetic − real, with the percentile interval.
- **Pooled:** the mean of the 7 per-class differences, with its percentile interval.

**Confirms the hypothesis** if the pooled 97.5th percentile is **< 0**, meaning the synthetic images
vary less. Per-class results are reported beside it and do not decide the verdict.

### S3 — Do synthetic mel and bkl lean towards nv?

- Split every real class into two halves, with seed 42.
- The first half builds the class centroids.
- The second half is the real query set. It never touches a centroid.

For class c in {mel, bkl}, the **nv-leaning share** is the fraction of queries that are closer, by
cosine, to the real nv centroid than to the real c centroid.

- It is computed for the synthetic c images, all of them, and for the real query half of c.
- The difference is synthetic − real, with a 95% bootstrap interval: 1,000 resamples of the queries
  on both sides.

**Confirms the hypothesis** if the lower 2.5th percentile is **> 0 for both mel and bkl**. The other
classes are reported for context only.

### Colour (descriptive only)

For each image, the mean CIE L\*a\*b\* (D65) of the central 50% crop is taken. The median per class,
for real and for synthetic, is reported. It explains what the embeddings see and decides nothing.

## Reading rule

| Outcome | Reading |
|---|---|
| S1, S2 and S3 all confirm | The gap lies in the synthetic distribution: the classes lack their own identity. The generator is discussed before v3a. |
| None confirms | The visual sample was misleading. Go back to v3a, as the criteria document says. |
| Mixed | Written down as it is, and decided with Walaa. |

Whichever reading holds, a departure from the build_v3 verdict is made only by a dated amendment to
`docs/ham10000_v2_signal_criteria.md` that gives the reason.

## What this check does not do

It does not:

- train anything;
- touch any signal, threshold, ranking, selection or v1 artifact;
- read `final_eval_heldout`.

It writes to `outputs/ham10000/diagnostics/synthetic_separability/` only.

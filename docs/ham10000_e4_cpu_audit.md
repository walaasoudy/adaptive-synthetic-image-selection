# E4 CPU audit: what the existing utility experiments already show

Date 2026-10-02. CPU only. Every number below was recomputed from the raw per-run files in
`ham10000_work/followup/`, and each one matches the report it came from.

## What exists

| Experiment | Recipe | Sizes | Subsets x seeds | Metric |
|---|---|---|---|---|
| Round 1 `proxy_noise_floor` | 224 px / 300 steps | 60, 120, 180 | 12 x 4 | BA, AUROC |
| Round 2 `proxy_noise_floor_stage4size` | 512 px / 3000 steps (the Stage 4 recipe) | 60, 120, 180 | 12 x 4 | BA, AUROC |
| Round 3 `proxy_noise_floor_steps1500` | 224 px / 1500 steps | 60, 120, 180 | 12 x 4 | BA, AUROC |
| `utility_instrument_check` | 224 / 300 | 60 to 180 | 12 x 20 | AUROC |
| `selection_headroom` | 224 / 300 | 0, 616 (15 random draws + C), 3168 | 54 seeds A/B, 10 per 616 subset | BA |
| `noise_decomposition` | 224 / 300 | 150, 616 | 12 seeds | AUROC vs evaluation-set size |

## Selection: which images, at a fixed size

| Run | Metric | Within-subset SD | Between-subset SD | ICC | Reliability at 5 repeats |
|---|---|---|---|---|---|
| Round 1 | BA | 0.033 | 0.013 | 0.140 | 0.45 |
| Round 1 | AUROC | 0.009 | 0.004 | 0.175 | 0.52 |
| Round 2 (full recipe) | BA | 0.040 | 0.010 | 0.056 | 0.23 |
| Round 2 (full recipe) | AUROC | 0.011 | 0.008 | 0.346 | 0.73 |
| Round 3 | BA | 0.032 | 0.000 | 0.000 | 0.00 |
| Round 3 | AUROC | 0.009 | 0.004 | 0.161 | 0.49 |
| Instrument check (20 repeats) | AUROC | 0.008 | 0.003 | 0.108 | 0.71 at 20 |
| Headroom, size 616 | BA | 0.041 | 0.006 | 0.022 | 179 repeats needed |
| Headroom, size 616 | AUROC | 0.010 | 0.001 | 0.010 | 387 repeats needed |

- The Round 2 AUROC ICC of 0.346 is the best result. Most of it comes from differences between the
  60/120/180 size groups. The same-size-only sensitivity reading in the report gives ICC 0.037 for
  AUROC and 0.0 for BA.
- In `noise_decomposition`, the noise floor does not shrink when the evaluation set grows. The fitted
  floor σ is about 0.009 AUROC, so the noise comes from training seeds, not from the size of the
  evaluation set.
- ASISM's selection C (616 images) sits inside the random-616 distribution: 0.5454 BA against a
  random mean of 0.5494, z = −0.28.

## Quantity: how many synthetic images

| Arm | n synthetic | BA mean (SD) | AUROC mean (SD) |
|---|---|---|---|
| A real only (54 seeds) | 0 | 0.473 (0.033) | 0.911 (0.009) |
| random 616 (150 runs) | 616 | 0.549 | 0.928 |
| C ASISM (10 seeds) | 616 | 0.545 (0.039) | 0.931 (0.006) |
| B all synthetic (54 seeds) | 3168 | 0.549 (0.048) | 0.938 (0.007) |

- The positive control (A against B) is +0.076 BA, t = 9.6. The proxy can see the effect of adding
  synthetic data.
- Going from 616 to 3168 gives +0.000 BA and +0.010 AUROC.

## Answers to the audit questions

1. **Selection is not measurable.** That holds for three recipes, including the full recipe, at
   sizes 60–180, and for the cheap recipe at 616. It would take 179–387 repeats per subset. The
   E4 decision rule (reliability 0.8 within 5 seeds) would very likely fail for selection. At
   larger sizes, subsets of a 3,168-image pool overlap more, so the between-subset spread can only
   shrink.
2. **Quantity is measurable** at the coarse level of 0, 616 and 3168 images. It has not been
   measured at the full recipe, at intermediate sizes, or per class. That is the only real gap that
   remains.
3. **E4 as designed** (5 sizes x 8 subsets x 5 seeds, ICC per size) mostly measures selection
   again. It adds little that is new for about $23.
4. **The real-only baseline is unstable at the full recipe.** Across 4 seeds the BA SD is 0.066
   (0.449–0.606). The v1 C-vs-B gap of 0.033 is smaller than this seed noise.

## The decision is hers and the supervisor's (contract decisions #1 and #3)

- (a) Invoke SC2 for selection now, from the existing evidence, and write up the narrowed claim.
- (b) Run E4 as written.
- (c) Narrow E4 to a quantity curve only, at the full recipe. This must be pre-registered before
  any run.

No GPU run was started.

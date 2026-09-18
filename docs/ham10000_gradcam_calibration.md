# HAM10000 Grad-CAM calibration — deferred until a trained classifier exists

Status: **blocked on a trained HAM10000 classifier.** No calibration values exist, and none may be
invented. This document records what is in place, what artifact is required, and the exact future
step.

## What is already enforced (code + tests)

| Guard | Where | Test |
|---|---|---|
| A reference must name its CAM model and say whether it was trained | `build_reference_distribution(..., cam_model_id=, cam_model_trained=)` (keyword-only, required) | `test_build_reference_requires_an_explicit_cam_model_provenance` |
| An untrained (e.g. random-init) CAM reference is marked non-scientific | `ReferenceDistribution.scientific` → `explainability_reference_scientific` | `test_a_reference_from_an_untrained_cam_model_is_marked_non_scientific` |
| Selection refuses non-scientific, uncalibrated, or non-permitted-split references | `selection_explainability_column`, `assert_selection_features_allowed` | `test_selection_column_requires_calibrated_rows_from_a_permitted_split`, `test_selection_guard_rejects_every_raw_or_diagnostic_explainability_feature` |
| Reference split is gen_train only; final_eval refused by name and by id | `build_reference_distribution` | `test_calibration_FAILS_*` |
| The CAM model is real-only, CrossEntropy, and not trained/selected on the reference split | `cam_model_identity(run_manifest, model_path)` | `test_cam_model_identity_accepts_only_a_real_only_classifier_not_trained_on_the_reference_split` |

The smoke test builds references from a random-init DenseNet only to exercise the plumbing; those
references carry `scientific=False` and can never reach selection.

## Required artifact

A HAM10000 classifier checkpoint directory produced by
`scripts/classify/ham10000_train_conditions.py` (or an equivalent script writing the same manifest):

- `model.pt` — DenseNet121 state dict, softmax + CrossEntropy, 7 classes in `CLASSIFIER_TARGET_LABELS` order
- `run_manifest.json` with `dataset=ham10000`, `loss=cross_entropy`,
  `data.synthetic_images=0`, and `data.real_train_split` / `data.selection_split` **not equal to the
  reference split (gen_train)** and never `final_eval_heldout`.

The identity used everywhere downstream is `cam_model_id = "ham10000-classifier:<sha256(model.pt)>"`.

## The open design decision (experimenter's)

The CheXpert pipeline trains its ASISM auxiliary classifier on **gen_train** and keeps
classifier_train unspent for A/B/C. For HAM10000 the explainability reference is also **gen_train**
(the images the generator imitates). The two conventions collide: a CAM model trained on gen_train
has memorised those images, so its attention on them is not a fair reference. `cam_model_identity`
refuses that combination. Options:

1. **Condition-A seed model as CAM model** — trained on classifier_train, selected on classifier_val,
   real-only. Passes the guard. Cost: the ASISM signal is computed with a model trained on the same
   real split that A/B/C train on.
2. **Dedicated auxiliary classifier on classifier_train** — same guard result as (1) without reusing
   an A/B/C run; costs one extra training run.
3. **Cross-fitted auxiliary on gen_train** — k models, each scoring the gen_train fold it did not see;
   keeps classifier_train unspent but costs k runs and needs a new reference builder.

## Future calibration step (after the chosen classifier is trained)

1. `identity = cam_model_identity(run_manifest, model_path, reference_split="gen_train")`
2. For every preprocessed gen_train image: Grad-CAM for its TRUE class with that model
   (`ham10000_explainability.gradcam`), rows via `explainability_rows(records, cam_fn, content_boxes)`
   using the persisted `gen_train_content_boxes.csv` — never the generic band.
3. `tie_report` on `explainability_peripheral_mass` per class. Keep the two-sided conformal p-value
   unless the ties demonstrably make it unusable; any change must be justified from that report.
4. Per class, `build_reference_distribution(values, image_ids, "gen_train", dx, statistic,
   final_eval_ids, cam_model_id=identity["cam_model_id"], cam_model_trained=True)`; classes with
   fewer than `min_reference_size` finite values are reported, not padded.
5. Persist the reference values, image ids, `cam_model_id`, tie reports and split hashes as an
   artifact; synthetic candidates are then calibrated with `calibrate_explainability`.

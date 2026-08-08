# Superseded schema-v1 split artifacts

The files in this directory's parent (`gen_train.csv`, `gen_val.csv`, `classifier_heldout.csv`,
`asism_tuning_heldout.csv`, `final_eval_heldout.csv`, `split_manifest.json`) were produced by
`scripts/data/02_build_patient_splits.py` under the schema-v1 three-way split
(gen_train 0.70 / gen_val 0.10 / classifier_heldout 0.20, with the ASISM-tuning and final-eval
splits derived as children of classifier_heldout).

That architecture is superseded by the six-way partition in
`scripts/data/02b_build_sixway_splits.py` (docs/stages2_to_5_plan.md §1), because it left no
patient population for a classifier development split, and deriving one from `gen_train` would
have meant the A-E classifiers' early-stopping and threshold decisions were made on patients the
SDXL generator had itself trained on.

These files are RETAINED for provenance and reproducibility of anything already built against
them. They are NOT read by any Stage 2-5 code. Current splits live under
`splits/production/` and `splits/dev/`.

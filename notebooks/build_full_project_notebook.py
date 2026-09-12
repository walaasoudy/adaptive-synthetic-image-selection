"""Build ONE Jupyter notebook that contains the entire project: every source file as its own
cell, each preceded by an explanation of what that file does and where it sits in the pipeline.

Run from the repository root:

    python notebooks/build_full_project_notebook.py

The repository stays the source of truth. This notebook is a reading/teaching copy: open it top
to bottom and you walk the pipeline in execution order, not in alphabetical order.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "thesis_full_project_code.ipynb"

# --------------------------------------------------------------------------------------------
# Per-file explanations. One entry per file that gets a cell.
# --------------------------------------------------------------------------------------------

EXPLANATIONS: dict[str, str] = {}

# ---------------------------------------------------------------- configuration layer
EXPLANATIONS["configs/dataset_config.yaml"] = """
**What it is:** a description of the CheXpert dataset itself - not of the pipeline.

Holds the Kaggle slug used to download it, the expected row counts used as an integrity check,
the column names (`Path`, `Sex`, `Age`, `Frontal/Lateral`, `AP/PA`), and the canonical order of
the **14 pathology columns**. That order is the serialization order used everywhere else.

**Why it is a separate file:** it should stay stable even when training hyperparameters change.
A change here means "the dataset changed", which is a much bigger event than "I tuned the LR".

**Read by:** `scripts/utils/config.load_dataset_config()`, so every script sees the same schema.
"""

EXPLANATIONS["configs/splits.yaml"] = """
**What it is:** the frozen six-way patient-level split policy - the most safety-critical config
in the project.

Six partitions are cut **directly** from one deterministic patient list with one seed; no split
is derived from another, and the fractions must sum to exactly 1.0 (asserted at build time):

| Split | Fraction | Used for |
|---|---:|---|
| `gen_train` | 0.48 | Stage 1 LoRA training; ASISM real-reference pool; auxiliary classifier |
| `gen_val` | 0.10 | Stage 1 monitoring; auxiliary classifier validation |
| `classifier_train` | 0.15 | real training data for every proxy and every headline classifier |
| `classifier_val` | 0.09 | model selection only (early stop, checkpoint, threshold, hparams) |
| `asism_tuning_heldout` | 0.09 | **all** Stage 3 tuning evidence |
| `final_eval_heldout` | 0.09 | Stage 5 only, opened once |

**The support rule** (`min_positive_patients: 50`, `min_negative_patients: 50`) is the reason the
fractions look odd. The original 5% decision-bearing splits left *Lung Lesion* and *Atelectasis*
short of 50 negative patients; raising those three splits to 9% fixed both. *Pleural Other* could
not be fixed by any fraction (only ~100 patients dataset-wide carry a confident negative for it),
so it was removed from the primary endpoint instead - see `scripts/utils/labels.py`.

**Namespacing** (`splits_root/<namespace>/`) is what makes it structurally impossible for a dev
subset split to overwrite a production split.
"""

EXPLANATIONS["configs/stage1_lora_sdxl.yaml"] = """
**What it is:** every tunable for Stage 1 (SDXL + LoRA fine-tuning), plus the `paths:` block that
the whole repository resolves against `PROJECT_ROOT`.

Notable sections: `paths` (everything that must survive a pod restart lives under `PROJECT_ROOT`),
`lora` (rank/alpha, UNet-only targets), `training` (precision, gradient accumulation/checkpointing,
latent+embedding caching, EMA), `optimizer`, `checkpointing`, `validation`, `captions`, `data`
(resolution, letterbox padding, quality thresholds), `dev_subset`, and the legacy `split:` block.

**One thing to know:** the `split:` block here is the **schema-v1 three-way** policy
(gen_train / gen_val / classifier_heldout). It is superseded by `configs/splits.yaml`. Stage 2-5
code never reads it; only `scripts/data/02_build_patient_splits.py` (itself superseded) does.
"""

EXPLANATIONS["configs/stage2_generation.yaml"] = """
**What it is:** the Stage 2 synthetic-generation contract - label recipes and the sampler.

`recipes`: rarity-aware per-class quotas (`base_quota`, `min_per_label`, `max_per_label`),
co-occurrence support thresholds, medical block rules (combinations that must never be generated),
and a cap on positives per recipe. `generation`: the diffusion sampler (DPM-Solver multistep,
~40 steps, guidance ~ 7, resolution, batch size, seed). `checkpoint.lora_weights_dir`: the
**explicitly pinned** Stage 1 adapter - there is no "latest checkpoint" magic, because a silently
moving generator would invalidate every downstream artifact hash. `pilot`: the human-review gate
that must be approved before a full generation run is permitted.
"""

EXPLANATIONS["configs/stage3_asism.yaml"] = """
**What it is:** the largest config in the project - all of ASISM. Worth reading slowly, because
almost every number in Stage 3 is here rather than in code.

- `auxiliary_classifier`: architecture/resolution/dropout of the reference classifier that the
  uncertainty, explainability and agreement signals query.
- `signals`: per-signal parameters - k for the similarity k-NN, the reference sampling policy and
  hierarchical fallback threshold, the `near_duplicate_similarity: 0.95` memorisation cut-off,
  IQA thresholds, MC-Dropout pass count, Grad-CAM settings, the agreement penalty.
- `gonogo`: the thresholds each signal must clear to be admitted (missingness, distinct values,
  directionality, redundancy correlation, reproducibility).
- `tuning` / `selection`: the legacy weighted-score selector's search space (used only by
  `03_tune_freeze_select.py`, which is **not** a pipeline step).
- `learned_asism`: the actual thesis contribution - `feature_columns` (the 9 admitted feature
  columns), `subset_design` (sizes, counts, feasibility thresholds), `set_utility_network`,
  `ranking_network` (including `target_source` and the C2 target-normalization options), and
  `threshold_network` (bootstrap contexts, verification budgets, acceptance criteria, the
  full-policy variant list and tie rule).

Every compute-spending step has its own `compute_budget` with `max_gpu_hours` - a gate that must
pass **before** any results are seen, so a search space can never be widened after looking at
performance.
"""

EXPLANATIONS["configs/stage4_classifier.yaml"] = """
**What it is:** the frozen Stage 4 protocol. The point of this file is that architecture,
initialization, label policy, uncertainty policy, optimizer, batch size, augmentation, checkpoint
selection, validation rule and threshold rule are **identical across conditions**. Only the
training data differs.

- `conditions: [A, B, C]` - A = real only, B = real + all synthetic, C = real + ASISM-selected.
- `training.max_steps` - the budget is in **optimizer steps, not epochs**. At fixed epochs a larger
  dataset silently gets more gradient updates, which would conflate "more data" with "more
  training". This matters most for B (all synthetic) vs C (a selected subset).
- `augmentation: "none"` - horizontal flip is forbidden; it inverts chest-X-ray anatomy.
- `seeds: [42, 43, 44]` -> 3 conditions x 3 seeds = 9 runs.
- `condition_d` - **inactive but retained**. The matched-random control is the documented fix for
  the one structural gap in the comparison (see the Stage 4 script's KNOWN LIMITATION note).
  Re-enabling it needs supervisor sign-off and `n_draws x 3` extra runs.
"""

EXPLANATIONS["configs/smoke_e2e.yaml"] = """
**What it is:** a fixture-only overlay, activated *only* through the `THESIS_CONFIG_OVERLAY`
environment variable (see `scripts/utils/config._smoke_overlay`). It shrinks everything - 64px
images, LoRA rank 2, 2 training steps, 4 inference steps, 2 MC-Dropout passes, tiny subset design -
so the whole Stage 1->5 chain can run on CPU in minutes.

**It cannot be activated by accident**: nothing loads it unless the env var points at it, and
`smoke_only: true` plus the `dev-smoke-v1` namespace mark every artifact it produces as
non-scientific.

The comments in this file are unusually valuable: they record exactly which smoke values had to be
raised or lowered and why (e.g. why `base_quota` went from 1 to 24, why the verification budget
went from 0.5 h to 1.5 h).
"""

EXPLANATIONS["configs/accelerate_config.yaml"] = """
**What it is:** the HuggingFace Accelerate launch configuration for Stage 1 - single process,
single GPU, bf16 mixed precision. Used as
`accelerate launch --config_file configs/accelerate_config.yaml ...`.
"""

# ---------------------------------------------------------------- shared utilities
EXPLANATIONS["scripts/utils/config.py"] = """
**Role:** the single entry point for reading YAML. Nothing else in the repository parses YAML
directly, so `PROJECT_ROOT` resolution stays consistent everywhere.

- `load_named_config(filename, section)` - the general loader used by Stage 2-5.
- `load_stage1_config(overrides)` - Stage 1, and accepts OmegaConf dotlist overrides
  (`training.train_batch_size=4`) so CLI experiments never edit the checked-in YAML.
- `_smoke_overlay(section)` - if `THESIS_CONFIG_OVERLAY` is set, merge that section over the base
  config. This is the **only** way smoke values enter the system.
- `ensure_dirs(config)` - idempotently create the output directories a script writes into.
"""

EXPLANATIONS["scripts/utils/manifest.py"] = """
**Role:** provenance. Every artifact in the project records enough to trace it back to the exact
code + config + data that produced it.

- `code_identity()` - hashes the *contents* of `scripts/`, `configs/`, `environment/`, including
  uncommitted and untracked files. A git SHA alone is not enough in a dirty research worktree; this
  is what lets a downstream stage refuse artifacts produced by a different source tree.
- `hash_dict` / `sha256_file` / `sha256_directory` - stable hashes for configs, files, directories.
- `make_run_id(config)` - `<UTC timestamp>_<git sha>_<config hash>`.
- `write_json` - **atomic** (temp file + `fsync` + `os.replace`), so an interrupted pod can never
  leave a half-written manifest.
- `write_frozen_json` - writes once and then refuses any overwrite, identical or not. This is the
  mechanism behind every "frozen" artifact in the project.
- `build_checkpoint_metadata` - the metadata block stored next to every Stage 1 checkpoint.
"""

EXPLANATIONS["scripts/utils/identifiers.py"] = """
**Role:** one tiny function, `sanitize_image_id`, that turns a CheXpert path into a flat id
(`patient00001__study1__view1_frontal`).

**Why it exists as its own module:** preprocessing (03) and caption generation (04) must produce
*identical* ids or the two outputs cannot be joined. Sharing the function makes drift impossible.
"""

EXPLANATIONS["scripts/utils/seed.py"] = """
**Role:** `set_seed(seed)` - seeds `random`, `PYTHONHASHSEED`, NumPy and torch (including CUDA) in
one call, tolerating missing optional dependencies. Called by every stage that has any randomness.
"""

EXPLANATIONS["scripts/utils/caption_builder.py"] = """
**Role:** the single source of truth for turning a CheXpert label row into a text prompt. Imported
**verbatim** by Stage 1 training (`04_generate_captions.py`), Stage 2 generation, and the Stage 1
probe sampler - any phrasing drift between train time and generation time directly degrades
conditioning fidelity.

Encoded policy decisions:

- **Uncertain (-1) labels are omitted** from the findings clause - never asserted present or absent.
- **`No Finding == 1` is authoritative** over any spuriously co-positive pathology column (a known
  artifact of CheXpert's NLP label extraction).
- **Support Devices is a device, not a pathology**, and gets its own separate clause.
- **Age is decade-bucketed** to reduce caption-vocabulary sparsity.
- **Paraphrase variant selection is deterministic** - `sha256(image_id + epoch) % n`, not an
  unseeded RNG, so captions are reproducible.
- `serialize_natural_list` produces "X, Y, and Z" grammar rather than a comma token-dump, which
  gives more stable CLIP embeddings.
"""

EXPLANATIONS["scripts/utils/labels.py"] = """
**Role:** the frozen label policy. Three label sets that are **deliberately not interchangeable**:

| Set | Size | Meaning |
|---|---:|---|
| `CLASSIFIER_TARGET_LABELS` | 14 | the classifier's output space; everything is predicted and reported |
| `PRIMARY_ENDPOINT_LABELS` | 11 | what macro-AUROC (the primary endpoint) is computed over |
| `GENERATION_TARGET_LABELS` | 12 | what Stage 2 is allowed to intentionally synthesize |

The 11 excludes `No Finding` (an absence-of-disease meta-label), `Support Devices` (a device label,
and the easiest/highest-prevalence column - including it would flatter the macro-average), and
`Pleural Other` (cannot meet the 50-negative-patient support rule in any split). The 12 adds
`Pleural Other` back, because support scarcity is a property of what can be reliably *evaluated*,
not a reason to stop *generating* examples of a condition.

**The uncertainty policy lives here** and is the most important thing in the file:
`build_label_arrays` returns three aligned arrays - `raw_label` (with `-2` for blank so the array
stays integer), `training_target`, and `mask`. Positions where the mask is False are meaningless
placeholders, **not** negatives. `-1` and blank contribute to neither loss nor metrics and are
never silently mapped to 0 or 1.

`patient_level_support` + `check_support_rule` implement the support rule at **patient** level -
image counts would let one patient with many studies manufacture apparent support.
"""

EXPLANATIONS["scripts/utils/chexpert_schema.py"] = """
**Role:** validate that a CheXpert CSV really is a CheXpert CSV before anything trusts it.

`validate_chexpert_frame` checks required columns exist, every label value is in {-1, 0, 1, blank},
no blank or duplicate image paths, every path yields a patient id, and (optionally) that every
referenced image file actually exists. Returns a `source_schema_hash` that downstream manifests
record. `resolve_chexpert_image` strips the `CheXpert-v1.0-small/` prefix variants and resolves a
CSV path value against the raw directory.
"""

EXPLANATIONS["scripts/utils/splits.py"] = """
**Role:** split loading, namespace resolution, and the **`final_eval_heldout` access guard** - the
single most important integrity mechanism in the project.

The honest version of the leakage rule is not "final_eval_heldout is never touched". It *is*
touched exactly once, during split construction, to assign patients, assert disjointness, run the
support check and hash the manifest. What must never happen is an **outcome-bearing read** before
Stage 5.

So access is mediated by an explicit **purpose token**:

- `NON_OUTCOME_PURPOSES` (split construction, support check, schema validation, existence check,
  hash verification) - always allowed.
- `OUTCOME_PURPOSES` (load images, load labels for evaluation, compute outcome statistics, model
  selection) - require a Stage 5 `--final-eval-run-id` **and** a registered, validated evaluation
  context. Any other caller fails loudly with `FinalEvalAccessViolation`.

Also here: `read_split_manifest` (version + namespace + self-hash integrity checks),
`split_provenance` (the small provenance block every downstream manifest embeds), and
`require_frozen_production_split_run` (the Stage 5 precondition: production class, frozen, support
check passed).
"""

EXPLANATIONS["scripts/utils/artifact_contracts.py"] = """
**Role:** the actual enforcement layer for "a stage must validate its upstream artifacts". Two
jobs:

1. **Namespace-safe paths.** `stage2_paths`, `stage3_paths`, `stage4_paths`,
   `auxiliary_checkpoint_path` - every artifact path in Stage 2-4 is derived here, always under a
   namespace directory. This is why a dev run and a production run cannot collide.
2. **Strict provenance contracts.** `require_manifest_fields` compares a manifest against expected
   values and lists every mismatch. `require_generation_complete` refuses to proceed unless the
   Stage 2 completion record matches the namespace, the split manifest hash, the generation
   manifest's own file hash, **and** the current `code_identity` hash. `require_score_artifact`
   does the same for each ASISM signal Parquet via its `.provenance.json` sidecar.
   `asism_score_provenance` composes the full expected chain: split -> generation -> LoRA checkpoint
   -> ASISM config -> code identity.

A config change alters the config hash, which forces a re-run from the affected stage forward
rather than silently mixing incompatible outputs.
"""

EXPLANATIONS["scripts/utils/experiment_registry.py"] = """
**Role:** one Parquet row per run, so every Stage 1-5 run is queryable in one place
(`outputs/experiments/experiment_registry.parquet`) instead of one manifest file at a time.

`parent_experiment_id` is the useful part: a Stage 4 run points at the Stage 3 selection run it
consumed, which points at Stage 2 generation, which points at the Stage 1 checkpoint - the whole
provenance chain as a single table.

`ExperimentRun` is a context manager: it appends a `running` row on entry and updates it to
`completed` or `failed` on exit (it never swallows the exception). Writes are
read -> modify -> atomic-replace; there is deliberately no locking layer, because the real execution
model is one researcher, one GPU, sequential runs.
"""

EXPLANATIONS["scripts/utils/metrics.py"] = """
**Role:** every Stage 5 metric and statistic. Pure NumPy - no torch, no GPU - which is what makes
the whole analysis layer developable and testable locally, off the pod.

- `auroc` - via the Mann-Whitney rank identity with correct mid-rank tie handling. Returns **NaN**
  when a class is absent, not a silent 0.5, so an undefined AUROC is excluded from a macro-average
  instead of dragging it toward chance.
- `average_precision`, `brier_score`, `expected_calibration_error`, `sensitivity_specificity_f1`.
- `macro_auroc_masked`, `full_metric_suite` - per-label metrics **with effective N attached**, so a
  metric over 40 usable patients is never silently compared to one over 400.
- `patient_level_bootstrap`, `paired_bootstrap_difference` - resampling is at **patient** level;
  bootstrapping images would let a patient with many studies dominate a confidence interval purely
  by study count.
- `holm_bonferroni` (confirmatory family) and `benjamini_hochberg` (exploratory) - the frozen
  multiplicity policy.

Everything is **masked**: uncertain and blank labels are excluded, never mapped to 0 or 1.
"""

EXPLANATIONS["scripts/utils/classifier.py"] = """
**Role:** one multi-label CXR classifier implementation with three consumers, so they cannot drift
apart: the auxiliary reference classifier (Stage 3 prerequisite), the ASISM proxy classifier
(used to measure subset utility), and the Stage 4 headline classifiers.

Two load-bearing design points:

- **MC Dropout needs dropout active at inference.** `model.eval()` disables it, so
  `enable_mc_dropout()` re-enables *only* the dropout modules and leaves BatchNorm in eval mode.
  Switching the whole model back to `train()` would make BatchNorm use batch statistics, corrupting
  the very predictions the uncertainty estimate is computed from.
- **Masked loss.** `masked_bce_loss` implements the frozen uncertainty policy: -1 and blank labels
  contribute to neither loss nor metric.

`TrainingBudget` expresses the budget in **optimizer steps, not epochs**, and `RunAccounting`
records what was actually spent - the fairness protocol that makes B vs C a data-composition
comparison rather than a training-length comparison.
"""

# ---------------------------------------------------------------- data preparation
EXPLANATIONS["scripts/data/00_download_dataset.py"] = """
**Pipeline step 1.** Downloads CheXpert-v1.0-small from Kaggle (`ashery/chexpert`) with
`kagglehub`, locates the real data root inside whatever nesting the archive used
(`find_data_root`), links or copies it into `data/chexpert/raw/` (`link_or_copy` - prefers a
symlink so the dataset is not duplicated on the persistent volume, falls back to copying on
Windows without symlink rights), then runs `01_verify_download.py` automatically.

**Idempotent**: never overwrites an existing entry. **Needs** Kaggle credentials configured for
`kagglehub` (`~/.kaggle/kaggle.json`, or `KAGGLE_USERNAME`/`KAGGLE_KEY`).
"""

EXPLANATIONS["scripts/data/01_verify_download.py"] = """
**Pipeline step 2 (integrity gate).** Refuses to let a broken or wrong-version download through.

Checks: both CSVs exist; `train.csv` row count within 1% of `expected_train_rows` and `valid.csv`
exactly `expected_valid_rows`; all 14 pathology columns present with values only in
{-1, 0, 1, NaN}; a random sample of images actually opens with plausible dimensions; and per-label
positive rates are printed for eyeball comparison against published CheXpert statistics.
"""

EXPLANATIONS["scripts/data/01b_build_dev_subset.py"] = """
**Optional step.** Builds a small, prevalence-faithful development subset so the pipeline can be
shaken down end to end without the full 224k-image cohort.

`select_dev_subset_patients` is **rarity-first greedy quota sampling**: sort pathology columns by
ascending prevalence (rarest first) and, for each, greedily draw *whole patients* carrying that
finding until its proportional quota is met. Because CheXpert pathologies co-occur, later quotas
are often partly satisfied for free; any shortfall is filled with uniform random patient draws.
All randomness comes from one seeded stream, and patients are never split.

`build_prevalence_report` then checks every column's prevalence against the full cohort within
`prevalence_tolerance_pct` and records the result (non-fatal, but written to a manifest).

Controlled by `stage1.dev_subset` in `configs/stage1_lora_sdxl.yaml`; a no-op when disabled.
"""

EXPLANATIONS["scripts/data/02_build_patient_splits.py"] = """
**SUPERSEDED - not a pipeline step.** This is the original schema-v1 **three-way** split builder
(`gen_train` / `gen_val` / `classifier_heldout`, with `asism_tuning_heldout` and
`final_eval_heldout` derived as *children* of `classifier_heldout`).

That architecture left **no population for a classifier development split**, which is why it was
replaced by the direct six-way builder (`02b_build_sixway_splits.py`). Its outputs are never read
by Stage 2-5 code.

Today it is kept alive by exactly one thing: `01b_build_dev_subset.py` imports its
`extract_patient_id` helper - a 4-line regex function that `02b` also defines identically. See the
audit log at the end of this notebook.
"""

EXPLANATIONS["scripts/data/02b_build_sixway_splits.py"] = """
**Pipeline step 3 - the real split builder, and the file everything downstream is hashed against.**

What it does, in order:

1. Validates the source CSV against the CheXpert schema contract (and, for the production
   namespace, against the full-cohort row-count contract - `verify_full_cohort`).
2. Extracts patient ids and builds per-patient positive **and negative** label vectors
   (`patient_label_vectors`, `patient_label_negative_vectors`).
3. `multilabel_partition_patients` - stratified assignment of *patients* into the six splits in one
   pass from one seed. It balances negative support as well as positive support, which is what the
   50-negative-patient rule actually needs.
4. `assert_disjoint_and_exact` - the partition is exactly disjoint and its union is every patient.
5. `build_support_report` - per-label positive/negative **patient** counts in every
   decision-bearing split, checked against the frozen support rule. `propose_fraction_adjustment`
   suggests a fix when it fails, instead of just erroring.
6. Writes `split_manifest_v2.json` with a self-hash, library versions, git commit, per-file
   SHA-256s, and `namespace_class`; `--freeze` makes it immutable.
7. Leaves a `SUPERSEDED_v1.md` note next to the old v1 artifacts rather than deleting them.

It touches `final_eval_heldout` here - legitimately - via
`assert_final_eval_access_allowed(purpose="split_construction")`.
"""

EXPLANATIONS["scripts/data/03_preprocess_images.py"] = """
**Pipeline step 4.** Turns raw CheXpert JPEGs into the fixed-size images Stage 1 trains on.

- **Frontal-only filter** (lateral views are dropped).
- `letterbox_resize` - aspect-preserving resize + pad. Chest anatomy must not be stretched.
- Quality gates: minimum source resolution, blank-image detection (`BLANK_STD_THRESHOLD`), decode
  failures - each rejection is counted and logged per split.
- `atomic_save_jpeg` - temp file + `os.replace`, so an interrupted run leaves no half-written JPEG.
- **Resumable and idempotent on real terms**: an existing output is skipped only after being fully
  decoded and validated (`validate_processed_jpeg` checks format, mode *and* dimensions), never on
  mere existence.
- `expected_manifest` + `check_manifest` - the staleness gate. If the config or the split hash
  changed, it refuses to mix new outputs into an old directory. It also refuses to reuse
  "unpinned" outputs that exist without a manifest.
"""

EXPLANATIONS["scripts/data/04_generate_captions.py"] = """
**Pipeline step 5.** Writes one caption record per preprocessed image, using
`scripts/utils/caption_builder.py` - the same module Stage 2 later generates with.

For each row it emits all configured paraphrase **variants** (`build_caption_variants`), so Stage 1
can pre-embed every caption an image might use and pick one deterministically per epoch. It also
stores `raw_label_vector` (the untouched ternary labels) next to the caption.

It imports `expected_manifest` and `validate_processed_jpeg` from `03_preprocess_images.py` rather
than re-implementing them, and refuses to caption images that preprocessing did not successfully
produce (`load_preprocessing_status`). `previous_caption_ids` makes re-runs incremental.
"""

# ---------------------------------------------------------------- stage 1
EXPLANATIONS["scripts/train/train_lora_sdxl.py"] = """
**Stage 1 - the generator.** SDXL fine-tuned with LoRA on CheXpert-derived captions. Adapted from
the diffusers `text_to_image_lora_sdxl` reference pattern, with the production concerns added:

- **UNet-only LoRA** on self- and cross-attention projections; both text encoders stay frozen.
- **bf16** mixed precision, PyTorch SDPA attention, gradient checkpointing.
- **One-time VAE latent + text-embedding caching** (`build_or_load_cache`). Captions are
  deterministic from labels, so nothing needs re-encoding every epoch - this is the single biggest
  speed win in the stage. `CachedSDXLDataset` serves from that cache and `set_epoch` selects the
  caption variant for the epoch.
- **`LoraEMA`** - exponential moving average of just the LoRA weights.
- **Resumable checkpointing** through Accelerate `save_state`/`load_state`, all paths under
  `PROJECT_ROOT`; `save_checkpoint` also writes `latest.json` and full metadata (config hash, split
  manifest hash, seed, library versions) and prunes to `keep_last_n`.
- **`validate_training_inputs`** - the upstream gate: split manifest, caption files and
  preprocessing manifests must all match what this run expects before a single step is taken.
- TensorBoard (and optional W&B) logging via `tracker_hparams`.
"""

EXPLANATIONS["scripts/train/launch_resumable.sh"] = """
**Stage 1 launcher, built for RunPod reality.** Reads
`checkpoints/stage1_lora_sdxl/latest_run.json` for an existing `run_id` and that run's
`latest.json` for the newest full checkpoint; resumes from it if present, otherwise starts fresh.
Runs under `nohup` so a dropped SSH session does not kill training, appending to a log on the
persistent volume.

**Note:** it *backgrounds* training and returns immediately - which is why `run_dev_subset.sh`
deliberately does **not** use it, and calls `accelerate launch` in the foreground instead so the
pipeline waits for Stage 1 to finish before Stage 2 starts.
"""

# ---------------------------------------------------------------- stage 2
EXPLANATIONS["scripts/generate/01_sample_label_recipes.py"] = """
**Stage 2a - decide *what* to generate, before generating anything.** The output is an auditable
recipe table, not images.

- `mine_cooccurrence` - counts how many distinct real **patients** exhibit each positive-label
  combination. Patient-level, so one patient with many studies cannot manufacture apparent support
  for a combination.
- `compute_label_quotas` - **rarity-aware** per-class quotas from real patient-level support, so
  rare findings are not drowned out by common ones.
- `blocked_by_medical_rule` - combinations forbidden by the configured medical block rules are
  never emitted.
- `build_recipes` - samples single- and multi-label compositions under the quotas, the
  co-occurrence support constraint and a cap on positives per recipe; attaches demographics
  (age bucket, sex, view) to each recipe.

Recipe eligibility keys off `GENERATION_TARGET_LABELS` (12 labels) - **not** the 11 primary
endpoint labels. Writes `label_recipes.csv`, `recipe_decisions.jsonl` (why each candidate was
accepted or rejected) and `recipes_manifest.json`, all atomically and all hashed.
"""

EXPLANATIONS["scripts/generate/02_generate_synthetic_images.py"] = """
**Stage 2b - generate the images.** Loads base SDXL plus the **explicitly pinned** Stage 1 LoRA
weights and produces one image per recipe.

- Captions come from `scripts/utils/caption_builder.py` **verbatim** - the same module Stage 1
  trained with. `recipe_to_caption_row` adapts a recipe into the row shape that module expects.
- `derive_seed(base_seed, recipe_id)` - a per-image deterministic seed, so any single image can be
  regenerated exactly.
- **The pilot gate.** `--mode pilot` generates a handful of images; `run_pilot_checks` computes
  automated sanity statistics; a human then runs `--approve-pilot --reviewer ... --notes ...`, which
  writes a signed approval manifest. `enforce_pilot_gate` makes `--mode full` refuse to run without
  it. Generating 5,000 images from a broken adapter is exactly the failure this prevents.
- **Resumable and idempotent** on the same terms as preprocessing: an existing image is skipped
  only after being decoded and validated (`validate_generated_image`), and every write is atomic.
- `validate_generation_inputs` / `check_upstream_ready` gate on the recipe manifest, split
  provenance and LoRA checkpoint hash. On success it writes `generation_complete.json` - the
  completion record that every Stage 3 script checks.

Each manifest row carries the image's `intended_label_vector`, which is the ground truth ASISM's
agreement signal scores against.
"""

# ---------------------------------------------------------------- stage 3: prerequisite + signals
EXPLANATIONS["scripts/classify/00_train_auxiliary_classifier.py"] = """
**Stage 3 prerequisite - and a file it is easy to misread.** This is **not** condition A.

It exists so the uncertainty, explainability and agreement signals have a real-data classifier to
query. It is trained on `gen_train` and validated on `gen_val` - **never** on `classifier_train` /
`classifier_val` (reserved unspent for the headline conditions) and never on a heldout split.
Using the generator's own splits for a scorer of the generator's output is the point: it leaves the
entire classifier development pool untouched.

Initialization is leakage-safe by construction: an ImageNet or random init only - a
CheXpert-pretrained backbone is rejected in code, because it would have seen the evaluation
patients.

Condition A is a *separate* experiment, trained later, from scratch, under the frozen Stage 4
protocol.
"""

EXPLANATIONS["scripts/asism/signals.py"] = """
**The five ASISM signals, as pure functions.** This is the scientific core of Stage 3 and the file
to read first when trying to understand the thesis.

| Signal | Function | What it measures |
|---|---|---|
| Similarity | `compute_similarity_scores` | k-NN similarity (k = 15) of a DINOv2 embedding against a **per-label real reference pool**; `hierarchical_reference_indices` falls back to looser label matches when exact-match references are scarce. Emits `similarity_knn_mean`, `similarity_top1`, `similarity_topk_spread`. |
| Image quality | `compute_iqa_scores` | blank / clipping / low-contrast / uniform-border / blur (`_laplacian_variance`) checks -> `iqa_composite`, `iqa_sharpness`, `iqa_contrast_std`. |
| Uncertainty | `compute_uncertainty_scores` | predictive std over MC-Dropout passes. Note the sign convention: **moderate uncertainty is not penalised** - a perfectly confident synthetic image is not automatically the most useful one. |
| Explainability | `region_overlap_score`, `expected_region_for`, `aggregate_pathology_overlaps` | Grad-CAM mass vs. the *empirically derived* expected pathology region (see `00c`), not a hand-drawn box. |
| Agreement | `compute_agreement_scores` | mean P(intended positives) minus penalty x mean P(confident **unintended** positives). |

Plus **distinctiveness** (`compute_distinctiveness_scores`): within-class k-NN redundancy,
`1 - mean top-k similarity`, and `duplicate_clusters` for grouping near-identical images.

The **near-duplicate / memorisation flag** fires at `similarity_top1 >= 0.95` (grounded in Dar et
al., *Nature Biomedical Engineering* 2025). Such images get no novelty credit and are rejected.

`write_score_artifact` writes each signal's Parquet plus a `.provenance.json` sidecar - the file
`artifact_contracts.require_score_artifact` later verifies.

**Design note:** IQA is deliberately pure NumPy/PIL. It is the one signal that needs no model and
no GPU, which is what keeps the whole selection layer developable off the pod.
"""

EXPLANATIONS["scripts/asism/00c_derive_expected_regions.py"] = """
**Stage 3 preparation - derive where each pathology actually appears, empirically.**

The explainability signal compares a synthetic image's Grad-CAM against an "expected region" for
its intended pathology. Hand-drawing those boxes from a textbook would be a researcher-injected
prior. Instead, this script averages Grad-CAM maps over **real positives** for each label and takes
the box enclosing `--mass-fraction` (default 0.80) of the averaged mass (`region_from_cam_mass`).

Two deliberate choices:

- `DERIVATION_SPLIT = "gen_train"` is **not configurable**. Deriving the regions from any
  decision-bearing split would leak.
- `_signals_module()` loads `01_compute_signals.py` by file path (its leading digit makes it
  unimportable by name) so the Grad-CAM used here is *literally the same code* the signal uses,
  never a re-implementation that could drift.

Output is written with `write_frozen_json` - derived once, then immutable.
"""

EXPLANATIONS["scripts/asism/01_compute_signals.py"] = """
**Stage 3a - compute the signals for every synthetic image.** Writes **five independent versioned
Parquet artifacts**, one per signal, rather than one shared table, so a signal can be recomputed,
audited or *excluded* by the Go/No-Go gate without disturbing the others.

- `--signal iqa` needs no GPU and no model. The other four query the frozen auxiliary classifier or
  the DINOv2 encoder and fail cleanly at that upstream gate when it is absent.
- `run_similarity` also produces **distinctiveness** internally, because it reuses the same
  synthetic embeddings - so distinctiveness costs no extra encoder passes and is not separately
  runnable.
- `require_generation_complete` + `asism_score_provenance` run first: the Stage 2 completion record,
  generation manifest hash, LoRA hash, split hash and code identity must all match.
- `validate_generation_manifest_rows` checks every manifest row before it is scored.

**Audit note:** this file's 19-line module docstring had been deleted by commit `0190fa6`, whose
message is only about adding research papers. Since the CLI builds its parser with
`description=__doc__`, `--help` printed nothing. Restored from `8b69933` -- see the audit log at the
end of this notebook.
"""

EXPLANATIONS["scripts/asism/02_gonogo.py"] = """
**Stage 3b - the Go/No-Go gate. The rule this script exists to enforce: a signal that fails its
gate is not forced into the final selector merely because it was implemented.**

Runs **before** any weights or thresholds are tuned. Each signal faces seven checks:

| Check | Question |
|---|---|
| `check_technical_validity` | is the artifact well-formed, with the expected columns? |
| `check_missing_rate` | how many images failed to score? |
| `check_numerical_stability` | enough distinct values; no degenerate constants or NaN storms |
| `check_directionality` | does the score move the way the design says it should? |
| `check_reproducibility` | does recomputing it give the same answer? |
| `check_redundancy` | is it just a copy of another signal (correlation)? |
| `check_usefulness` | does it separate anything downstream? |

`decide(checks)` then returns one of three verdicts:

- **`include`** - passed everything; eligible for the frozen selector.
- **`ablation_only`** - technically valid but weak or near-duplicate; retained for the
  leave-one-signal-out analysis, excluded from the selector.
- **`exclude`** - dropped.

Frozen into `gonogo_report.json` / `asism_frozen_manifest.json`. Everything downstream reads
`surviving_signals` from that report - which is how a rejected signal's feature columns disappear
from the learned ranker automatically.
"""

EXPLANATIONS["scripts/asism/03_tune_freeze_select.py"] = """
**NOT A PIPELINE STEP.** This is the *pre-learned* ASISM: a weighted-score selector that
normalises each signal, combines them with a weight vector (`composite_score`), searches the weight
grid in two bounded stages (coarse screening -> shortlist validation) with a proxy classifier
(`run_proxy_trial`), then freezes the winning weights and selects images.

It is historically important - it is the baseline the learned approach replaced - and its
engineering is worth reading: a compute-budget gate that must pass *before* any results are seen
(`enforce_compute_budget`), a resumable append-only search log (`append_search_log`,
`completed_trials`), and evaluation restricted to `asism_tuning_heldout` only.

But **nothing in the Stage 4/5 pipeline consumes its output**, and `run_dev_subset.sh` does not run
it. The `README` used to claim it was "retained for its shared signal-merge helpers only"; that was
stale -- the shared merge lives in `scripts/asism/candidate_pool.py`, and nothing imports
`load_merged_scores` except this file's own tests. See the audit log at the end.
"""

# ---------------------------------------------------------------- stage 3: learned ASISM
EXPLANATIONS["scripts/asism/candidate_pool.py"] = """
**The shared, provenance-checked loader for every learned-ASISM stage (04-09).** Small but
load-bearing.

It reads the Go/No-Go report, takes `surviving_signals`, verifies **each** signal Parquet against
its expected provenance (`require_score_artifact`), inner-joins them on `image_id`, loads every
image's `intended_label_vector` from the Stage 2 manifest, and attaches a `__stratum` key (the
sorted intended-positive label combination, or `__no_finding__`) used for stratified sampling.

The comment about hashing a **pristine** Stage 3 config is a real bug fix worth noting: callers
rewrite `cfg.paths` to namespaced locations, and hashing the *rewritten* config made every learned
stage reject valid score artifacts.
"""

EXPLANATIONS["scripts/asism/learned.py"] = """
**The pure-logic library behind learned ASISM - no torch, no I/O side effects, fully unit-testable.
The single densest file in the project.** Grouped by what it is for:

**Feature plumbing.** `FEATURE_COLUMNS_BY_SIGNAL`, `active_feature_columns` (intersect configured
columns with surviving signals - this is how Go/No-Go exclusion propagates), `contributing_signals`,
`safe_feature_frame`, `apply_feature_frame`.

**Subset design (04).** `split_image_pool` splits the candidate pool into **train-role** and
**val-role** image pools that are disjoint *by construction*. `build_controlled_subsets` /
`build_role_conditioned_subsets` build random / single-signal / mixed subsets across quantile
bands. `pool_feasibility_report` + `evaluate_subset_design_feasibility` are the fail-closed gate:
if the pool cannot support the design (too few candidates per class or per quantile band, too few
val subsets), it refuses rather than producing a degenerate design. `verify_built_subsets` checks
what was actually built.

**Ranking targets (05).** `normalize_ranking_targets` - the **C2** change: `standardize` (default),
`rank`, or `none`, plus optional winsorizing, all order-preserving, so one noisy proxy measurement
cannot dominate training on ~96 points. `banzhaf_msr_targets` - a Data-Banzhaf value estimator
computed from the *same* measured subsets at no extra GPU cost; `loo_vs_banzhaf_diagnostic` records
their Spearman agreement, and Banzhaf stays opt-in until that diagnostic is reviewed.
`validate_utility_results`, `spearman_with_reason`, `read_jsonl`.

**Threshold contexts (07).** `bootstrap_class_contexts` - per-class bootstrap resamples of one image
pool, each summarised as a 10-dim vector (score-distribution stats + real patient prevalence +
budget), honestly tagged `independent_clinical_sample: False`. `class_aware_context_vector`,
`real_class_support_context`, `hard_threshold_grid_search` (critic-guided candidate proposal),
`diversify_verification_candidates` (verify a *spread* of the grid, not just the critic's top-k).

**Governance (08).** `compute_verified_context_counts`, `eligible_classes_for_official_training`,
`determine_per_class_official_method` - the **three-tier per-class rule**: learned network only with
at least `min_verified_contexts_per_class` on *both* train and image-disjoint held-out sides *and*
passing frozen acceptance criteria; else the median best proxy-verified threshold
(`aggregate_hard_proxy_best_threshold`); else the fixed baseline. `freeze_acceptance_criteria` /
`enforce_acceptance_criteria` make those criteria pre-registered rather than chosen afterwards.

**Policies (08b/09).** `freematch_style_percentile_per_class` - the **C4** change: leniency scaled by
*range-normalised* real prevalence, so the rarest present class gets the most lenient threshold.
`enforce_per_class_selection_floor` - lowers a class's threshold if it would otherwise admit fewer
than `min_selected_per_label` (the one-time-curation analogue of a class-fairness term).
`build_policy_selected_manifest` combines per-class thresholds into the final multi-label selection.
`choose_full_policy` applies the pre-registered tie rule (`tie_noise_band` + `simplicity_order`).

`scientific_status_for_namespace` is the honesty tag: anything built in a dev/smoke namespace is
permanently marked non-scientific.
"""

EXPLANATIONS["scripts/asism/models.py"] = """
**The three trainable networks.** Deliberately small - the training signal is ~96 measured subsets,
not a large dataset.

- **`SetUtilityNetwork`** - a permutation-invariant Deep Sets model over an image subset. Default
  pooling is masked mean. `set_superset_summary` + `encode_set` implement the **C1** ablation
  (masked mean *and* std pooling plus a fixed full-pool summary concatenated to the utility head,
  after Xie et al., ICLR 2024) - flag `superset_conditioning`, **default off**.
  It learns *only* from measured subset-level downstream utility.
- **`MultiSignalUtilityRankingNetwork`** - an MLP `9 -> 128 -> 64 -> 32 -> 1`. The 9 inputs are the
  admitted feature columns (similarity x3, IQA x3, uncertainty, explainability, agreement;
  distinctiveness's column is computed and gated but not yet in the configured feature set). **One**
  scalar utility target, optimised by Smooth-L1 + 0.5 x `pairwise_ranking_loss`. It is *not* a
  multi-objective / Pareto model.
- **`AdaptiveThresholdNetwork`** - a 16-dim class embedding concatenated with the 10-dim context
  -> 64 -> 32 -> 1 -> sigmoid. Per-class, context-dependent admission thresholds.

The key anti-pseudo-replication point: image scores are **distilled from marginal set utility**. No
subset AUROC is ever copied onto each of its member images.

`soft_selection_gate` is a differentiable relaxation of thresholding - present and tested, but not
used by the pipeline, which thresholds hard.
"""

EXPLANATIONS["scripts/asism/04_build_utility_subsets.py"] = """
**Learned ASISM step 1 - design the experiment that measures utility.** Two phases, and the phase
split is the point.

`--phase feasibility` runs `preflight_check_candidate_pool_inputs` (are the signal artifacts and
the Go/No-Go report there?) and then `pool_feasibility_report` - can this candidate pool actually
support the configured subset design? It **fails closed** if not, before any GPU time is spent.

`--phase build` splits the pool into disjoint train-role and val-role image pools
(`split_image_pool`) and builds ~120 controlled subsets (`build_role_conditioned_subsets`): random,
single-signal and mixed compositions, sizes 100-250, drawn across quantile bands. Each subset
records its `role`, which is what later guarantees zero image overlap between what the Set-Utility
Network trains on and what it is early-stopped on.

`subset_design_config_hash` pins the design so a changed design forces a re-measure.
"""

EXPLANATIONS["scripts/asism/04b_evaluate_utility_subsets.py"] = """
**Learned ASISM step 2 - the expensive one. This is where "utility" stops being a proxy signal and
becomes a measurement.**

For every frozen subset recipe it trains a real (small) proxy classifier on
`classifier_train` + the subset, trains a real-only baseline with the **same** budget, and records
`augmented_macro_AUROC - real_only_macro_AUROC`. Evaluation is restricted to
`asism_tuning_heldout` - never `final_eval_heldout`.

Engineering that matters: `--phase estimate` computes the run count and GPU-hour estimate and
applies the budget gate *before* any training; `--phase run` is **resumable at
(subset, fold, seed) granularity**, so a dropped pod costs one cell, not the whole matrix.

After an exposure filter (each image must appear in at least `minimum_image_exposures` subsets),
roughly 96 usable `(subset, utility)` pairs remain. Those 96 numbers are the entire supervision
budget for everything that follows.
"""

EXPLANATIONS["scripts/asism/05_train_learned_asism.py"] = """
**Learned ASISM step 3 - turn 96 measured subset utilities into a per-image ranker.**

1. **Train the Set-Utility Network** on the measured subsets. The train/val split is **not** a
   random split of subsets - subsets share images, so that would leak. It splits by the `role` each
   subset carries from `04 --phase build` (`split_subsets_by_role`), and
   `image_overlap_fraction` is asserted to confirm zero overlap.
2. **Generate per-image targets** (`marginal_targets`): the size-normalised leave-one-out marginal
   utility `(U(S) - U(S\\i)) x (n-1)`, averaged over the subsets each image appeared in. The Banzhaf
   MSR alternative is computed from the same data and its Spearman agreement recorded in the
   manifest.
3. **Normalise the targets** (C2, default `standardize`) so one noisy measurement cannot dominate.
4. **Train the ranking network** (`train_ranker`) with Smooth-L1 + pairwise ranking loss, reporting
   `pairwise_ranking_accuracy`.

`padded_batch` handles variable-size subsets with a mask. **The script intentionally stops when
proxy results are missing** - that refusal is what prevents fabricated image targets.
"""

EXPLANATIONS["scripts/asism/06_learn_thresholds_select.py"] = """
**Learned ASISM step 4 - the fixed-ratio learned threshold. An intermediate/ablation artifact, not
a Stage 4 condition.**

`optimal_threshold` picks, per class, the threshold that best serves a frozen utility/budget
objective at a *fixed target selection ratio*; `context_vector` builds the class context it is
conditioned on. A hard technical safety gate always runs first, and targets are chosen on ASISM
tuning data only.

Its output (`learned/selected_manifest.jsonl`) is kept for the ablation table and is deliberately
left untouched by step 09. The **adaptive** path that actually feeds condition C continues at
`07_build_threshold_contexts.py`.
"""

EXPLANATIONS["scripts/asism/07_build_threshold_contexts.py"] = """
**Learned ASISM step 5 - build the contexts the threshold network will be conditioned on. Trains
nothing.**

It loads the **frozen** ranker and critic from step 05 as fixed, read-only models
(`load_frozen_models`), then for each of the 11 primary labels:

- builds bootstrap resample contexts from the train-role and val-role image pools **separately**
  (the same disjoint split as before), each a 10-dim vector of score-distribution statistics + real
  patient prevalence + budget, honestly tagged `independent_clinical_sample: False`;
- runs a critic-guided `hard_threshold_grid_search` over candidate thresholds;
- writes `threshold_contexts.jsonl` (one row per context),
  `threshold_candidate_evaluations.jsonl` (one row per threshold tried), and
  `proxy_verification_plan.json` - which candidates *would* be proxy-verified, plus the GPU estimate
  (`verification_compute_estimate`).

The plan diversifies candidates across the grid rather than taking only the critic's top-k, so
verification cannot be rigged by a critic that is systematically wrong in one direction.
"""

EXPLANATIONS["scripts/asism/07b_verify_thresholds_proxy.py"] = """
**Learned ASISM step 6 - the only script in the threshold pipeline that trains real proxy
classifiers.**

`--phase run` **refuses to start** - before touching any config, data or model - unless called with
`--i-understand-this-trains-real-models`. That flag exists specifically so that no test, CI job or
accidental invocation can trigger real GPU training. `--phase estimate` never requires it, because
it never trains.

For each planned `(context_id, candidate_threshold)`: `rebuild_hard_subset` reconstructs the
*identical* subset step 07's grid search scored (same rule, no drift), then `train_and_score` trains
real + subset against a real-only baseline and appends the measured utility. Resumable; evaluated on
`asism_tuning_heldout` only; gated by its own independent compute budget.
"""

EXPLANATIONS["scripts/asism/08_train_threshold_network.py"] = """
**Learned ASISM step 7 - train the Adaptive Threshold Network, with the project's strictest
governance. Two separate models that are never merged:**

- **`verified_only_official`** - trained only if at least one class is *eligible*: it independently
  clears `min_verified_contexts_per_class` on **both** verified train contexts and verified,
  **image-disjoint** held-out contexts. A class with zero verified train contexts never receives an
  official threshold from this path. Held-out measurements are used **only to evaluate** a threshold
  already built from train data - never to choose or construct one.
- **`critic_assisted_exploratory`** - a separate model on critic-only targets, always trained, always
  labelled exploratory.

Then the **three-tier per-class decision** (`determine_per_class_official_method`): sufficient
verified evidence *and* passing acceptance criteria -> the learned network; 1-2 verified contexts ->
the median best proxy-verified threshold; 0 -> the fixed baseline.

`evaluate_held_out`, `critic_predicted_utility_regret`, `threshold_stability` and
`compute_official_acceptance_evidence` produce the evidence; `enforce_acceptance_criteria` checks it
against criteria frozen *beforehand*. Finally `plan_full_policy_verification` writes the plan for
step 08b.
"""

EXPLANATIONS["scripts/asism/08b_verify_full_policy_proxy.py"] = """
**Learned ASISM step 8 - measure whole *policies*, not individual thresholds.** Same fail-closed
discipline as 07b (`--i-understand-this-trains-real-models`), with a **separate, independently
approved** compute budget that is never summed with 07b's.

For each candidate policy it builds the **final combined multi-label `selected_manifest`** (per-class
thresholds combined via `min(applicable thresholds)`), trains real + selection, and measures utility
on `asism_tuning_heldout` across several seeds. The five policies:

1. fixed baseline threshold
2. hard-proxy-best among verified
3. the adaptive network (`predict_network_thresholds_per_class`)
4. `literal_top_50_percent`
5. `freematch_style_adaptive_percentile` - the **C4** policy:
   `percentile_threshold_per_class` scaled by range-normalised real prevalence
   (`real_prevalence_contexts_from_threshold_contexts`), plus the per-class selection floor.

`validate_policy_selection` checks each policy's output is well-formed before it is trusted.
"""

EXPLANATIONS["scripts/asism/09_finalize_learned_selection.py"] = """
**Learned ASISM step 9 - commit to exactly one policy.** Trains nothing and never opens
`final_eval_heldout`.

It consumes the complete 08b proxy measurements, applies the **pre-registered tie rule**
(`tie_noise_band` + `simplicity_order`: within the noise band, prefer the simpler policy), commits to
one winner, and writes the frozen `adaptive_selected_manifest.jsonl` - the exact image list that
becomes Stage 4's condition C. The fixed-ratio output from step 06 is left untouched.

The tie rule being pre-registered is what stops "pick whichever policy won" from becoming
"pick whichever policy won on this particular noisy measurement".
"""

# ---------------------------------------------------------------- stage 4
EXPLANATIONS["scripts/classify/01_train_conditions.py"] = """
**Stage 4 - train the thesis conditions.**

| Condition | Training data |
|---|---|
| **A** | real only (`classifier_train`) |
| **B** | real + **all** Stage 2 synthetic images |
| **C** | real + the finalized ASISM adaptive selection |

**Primary comparison: C vs B** - does *selecting* synthetic images with ASISM beat using all of them
unselected?

**The KNOWN LIMITATION is stated in the file itself, and you should read it before reading the
results.** C is a strict subset of B, so a C-over-B gain has two competing explanations this design
cannot separate: (a) ASISM chose *good* images, or (b) using *fewer* synthetic images helps
regardless of which ones, because unselected synthetic data is noisy. Distinguishing them needs a
condition that draws |C| images at random with C's label profile - optional condition **D**. The
matched-draw machinery (`build_matched_random_draw`, `profile_of`, `condition_d.*`) is retained and
working for exactly that purpose, but D is not enabled by default.

Fairness machinery: `training_plan` computes the run matrix; every enabled condition and seed gets
the **same optimizer-step budget**; model selection is on `classifier_val` only. Four manifests are
frozen before training starts (protocol, label policy, threshold policy, seed plan), so the protocol
cannot be adjusted after seeing results. Runs are resumable per `(condition, seed, draw)` tag.
"""

# ---------------------------------------------------------------- stage 5
EXPLANATIONS["scripts/eval/stage5_evaluate.py"] = """
**Stage 5 - the protected final evaluation. The only place `final_eval_heldout` is read for
outcomes, and it is wrapped in more gates than anything else in the project.**

`enforce_preconditions` refuses to run unless *all* of these hold: the production split manifest
exists **and** is frozen **and** passed its support check; the recorded split hash matches the
current manifest; the Stage 4 frozen protocol manifest exists; the Stage 3 frozen selection manifest
exists; the required A/B/C checkpoints exist with their hashes recorded; the classification-threshold
policy is frozen; and an explicit `--final-eval-run-id` was supplied.

That run id is what unlocks outcome-bearing access through
`splits.assert_final_eval_access_allowed`. `register_final_evaluation` writes the evaluation context
first and `mark_final_outcome_access` records the moment the data was actually opened - so the single
permitted read is itself auditable.

Then: `predict_condition` runs inference per condition/seed, `select_thresholds` applies the frozen
threshold policy (`max_youden_j`, chosen on `classifier_val` **only**),
`validate_prediction_frame` checks the exact expected image and patient ids are present, and
`publish_predictions_atomic` + `validate_completed_predictions` write and re-verify the prediction
Parquets. It computes no comparisons itself - that is the next file's job.
"""

EXPLANATIONS["scripts/eval/compare_conditions.py"] = """
**Stage 5 analysis - the statistics and result tables.** Reads *only* the prediction Parquet files
written by `stage5_evaluate.py`. No torch, no GPU, so figures and interpretation can be iterated
locally, off the pod.

Implements the frozen statistical policy:

- patient-level **paired bootstrap** for effect sizes and 95% CIs;
- **Holm-Bonferroni** for the CONFIRMATORY family (primary endpoint, primary comparison);
- **Benjamini-Hochberg** FDR for EXPLORATORY analyses, labelled as such everywhere;
- an effect size reported alongside every p-value.

`parse_tag` decodes run tags (`C_seed43`, `D_draw2_seed44`), and condition D - if it was ever run -
is summarised **across** its draws (mean and spread), never as a single cherry-picked draw.
"""

EXPLANATIONS["scripts/eval/generate_probe_samples.py"] = """
**Stage 1 monitoring.** Generates a fixed-seed probe grid from a LoRA checkpoint: one prompt per
pathology positive, one no-finding prompt, and a few multi-label combinations
(`MULTI_LABEL_COMBOS`), regenerated with the **same seeds at every checkpoint** so the grids are
directly comparable checkpoint-to-checkpoint.

Prompts are built with `scripts/utils/caption_builder.py` - the same module used at training time -
so probe conditioning matches train-time phrasing exactly. This is a qualitative plausibility and
informal directional-conditioning check, not a metric.
"""

EXPLANATIONS["scripts/eval/compute_fid_clipscore.py"] = """
**Stage 1 monitoring - quantitative trend.** Computes **two** FID variants deliberately:

- **Inception FID** (`compute_inception_fid`) - field-standard and comparable to other work, but a
  poor fit for grayscale medical images, since Inception features are ImageNet-domain.
- **Domain FID** (`compute_domain_fid`) - `torchxrayvision` DenseNet121 features. The more
  literature-appropriate signal for this project, because standard FID does not reflect diagnostic
  content.

Plus CLIP score for caption-image agreement. **Both FIDs are tracked only as relative trends across
checkpoints, never as absolute quality bars.** The real reference set is `gen_val` - held out from
training but already touched by Stage 1; `classifier_heldout` must never be read here.
"""

# ---------------------------------------------------------------- smoke
EXPLANATIONS["scripts/smoke/00_build_fixture.py"] = """
**Creates the deterministic, synthetic, CheXpert-shaped input used only by the `dev-smoke-v1`
namespace** - a handful of generated grayscale images plus a matching `train.csv`/`valid.csv`.

No real patient data is involved, and everything it produces is permanently marked non-scientific.
This is what lets the whole Stage 1->5 chain be exercised on a laptop.
"""

EXPLANATIONS["scripts/smoke/run_smoke_pipeline.py"] = """
**The resumable Stage 1-5 fixture pipeline.** `--phase local` runs everything that needs no GPU;
the GPU phase adds real (tiny) SDXL training and generation.

`run_if_missing` makes each step skip when its output already exists, so the smoke run is resumable.
`require` / `require_successful_preprocessing` assert each step's outputs before the next step
starts, so a failure is reported at the step that caused it rather than three steps later.

This validates **engineering connectivity only** - that the artifacts, gates, hashes and manifests
all line up. It says nothing about scientific results, and its outputs are marked accordingly.
"""

EXPLANATIONS["scripts/smoke/01b_learned_asism_cpu_smoke.py"] = """
**A CPU-only integration smoke for the learned-ASISM pipeline: 04 -> 04b(fabricated) -> 05 -> 06.**

It exists because of a real coverage gap found in the supervisor review of 2026-08-21: the
end-to-end smoke never exercised `scripts/asism/04`-`09` - the thesis's novel contribution - because
doing so through the real pipeline requires a real (if tiny) SDXL run.

This script closes that gap **without touching SDXL**, by fabricating a self-consistent synthetic
candidate pool and score tables (`build_fixture`, `fabricate_utility_results`) and then running the
**real** 04, 05 and 06 scripts as subprocesses against them. It genuinely exercises the production
code path for: Go/No-Go feature removal (one signal is deliberately marked excluded), the subset
feasibility gate, role-disjoint pool splitting, set-utility training, LOO target generation, and
threshold selection.

What it does **not** validate: any scientific claim. The utilities are fabricated.
"""

# ---------------------------------------------------------------- tests
EXPLANATIONS["tests/run_all.py"] = """
**The production STOP gate.** `docs/runpod_stage2_to_5_commands.md` says "STOP unless every test
passes", and this is the command that decides.

Two hard-won details are worth reading:

- **Suites are discovered, never hand-listed.** A hardcoded list once omitted
  `test_learned_asism.py` - 85 tests covering the thesis's novel contribution - and this gate
  happily reported a green "78 passed" while running none of them.
- **Suites that need pytest go through pytest.** A suite using `monkeypatch`, `tmp_path` or
  `pytest.raises` cannot self-run; executing it as a bare script runs *zero* tests and returns 0.
  Detection is by the presence of a `__main__` self-runner.

A suite that produces no parseable summary is treated as **failed**, never as a silent pass.
"""

EXPLANATIONS["tests/fixture_workspace.py"] = """
**Per-test writable workspaces with unconditional cleanup**, under `tests/_runtime/`. A context
manager that creates a uniquely numbered directory, yields it, and removes it in a `finally` - so a
failing test cannot leave state behind that makes the *next* test pass or fail for the wrong reason.
"""

EXPLANATIONS["tests/test_metrics.py"] = """
**Tests for `scripts/utils/metrics.py`** - the statistics layer, tested against values computed by
hand rather than against another library's output.

Covers AUROC (including tie handling and the NaN-when-a-class-is-absent contract), average
precision at known values, calibration error, the masking behaviour, patient-level bootstrap, and
the Holm-Bonferroni / Benjamini-Hochberg procedures.
"""

EXPLANATIONS["tests/test_asism_signals.py"] = """
**Tests for `scripts/asism/signals.py`** - every signal's behaviour on constructed inputs where the
right answer is known: IQA on blank/clipped/blurred images, the hierarchical reference fallback,
agreement's unintended-positive penalty, the uncertainty sign convention, Grad-CAM region overlap,
distinctiveness and duplicate clustering.

It also **pins the `near_duplicate_similarity = 0.95` cut-off** (change C3) so the
literature-grounded threshold cannot drift silently.
"""

EXPLANATIONS["tests/test_learned_asism.py"] = """
**The largest suite in the project (~1,500 lines), and the one that matters most** - it covers the
thesis's actual contribution, `scripts/asism/learned.py` and `models.py`.

Among the invariants it enforces: train-role and val-role subsets are image-disjoint; the subset
feasibility gate fails closed; target normalization is order-preserving; LOO and Banzhaf targets are
computed from the same measured subsets; the three-tier per-class governance rule selects the right
threshold source for each evidence level; acceptance criteria are enforced against frozen values;
the FreeMatch policy's leniency ordering follows prevalence; the per-class selection floor actually
raises admission; and the tie rule in `choose_full_policy` prefers the simpler policy within the
noise band.

This is the suite that was silently not running under the old hand-listed `run_all.py`.
"""

EXPLANATIONS["tests/test_pipeline_contracts.py"] = """
**Tests the structural rules the pipeline depends on**, not individual functions: the label-set
definitions and their sizes, the patient-level support rule, split partition disjointness and
exactness, the `final_eval_heldout` purpose-token guard (including that outcome purposes are refused
without a run id), and the namespace validation rules.
"""

EXPLANATIONS["tests/test_provenance_contracts.py"] = """
**Tests the provenance chain** - that a stage really does refuse artifacts produced with different
settings. Covers manifest field mismatches, score-artifact hash mismatches, code-identity
mismatches, split-manifest integrity self-hashing, the `write_frozen_json` write-once rule, and the
final-evaluation context validation.
"""

EXPLANATIONS["tests/test_roundtrip_contracts.py"] = """
**End-to-end round-trip tests on real fixture workspaces**: write an artifact with the production
code, read it back with the production reader, and assert the contract holds - including that a
tampered or stale artifact is rejected rather than silently accepted.
"""

# ---------------------------------------------------------------- cloud
EXPLANATIONS["cloud/modal_stage1.py"] = """
**Stage 1 on Modal, running the repository's own scripts unmodified.**

The design note is the interesting part: rather than porting Stage 1 into a notebook, it runs
`02b_build_sixway_splits.py` and `train_lora_sdxl.py` *inside* the container, so the provenance chain
(frozen split manifest -> LoRA metadata -> generation manifest -> ...) is produced **natively** and
no Stage 1 -> Stage 2 adapter is needed.

Layout: `/root/repo` holds `scripts/`, `configs/`, `environment/` copied from the checkout (the code
identity); `/vol/project` is `PROJECT_ROOT` on a persistent Modal Volume (data, caches, checkpoints,
logs). Nothing under `scripts/` or `configs/` is edited - overlays are written instead
(`_write_overlay`). Entry points: `prepare_data`, `smoke`, `train` (with `_resume_args` /
`_latest_step` for resumption across container restarts), `status`.
"""

EXPLANATIONS["cloud/modal_bench.py"] = """
**Before/after benchmarks on Modal, with synthetic data.** Every benchmark runs the **original** and
the **optimized** behaviour through the same repo code, in the same container, on the same synthetic
inputs (`_synthetic_cxr_jpegs`, `_set_mode`), so the comparison isolates the change. No CheXpert data
and no Kaggle secret needed.

Reports per mode: wall time, throughput, GPU utilization sampled from `nvidia-smi` (`GpuSampler`),
peak VRAM, peak host RSS, estimated cost (`_cost`) - **plus an equivalence check of the outputs**,
which is what stops a "speed-up" that quietly changed the numbers from looking like a win.
"""

EXPLANATIONS["cloud/modal_gpu_probe.py"] = """
**A 20-line utility** that answers "which GPU types will this Modal workspace actually let me
launch?" by running `nvidia-smi` for a few seconds on the requested type. Written because a Starter
plan without a payment method refuses some GPU types at launch time.
"""

# ---------------------------------------------------------------- environment / drivers
EXPLANATIONS["environment/requirements.txt"] = """
**The dependency set.** After a verified working setup, freeze exact versions with
`pip freeze > environment/requirements-lock.txt` - the repository intentionally keeps the loose
requirements and the exact lock as two separate files.
"""

EXPLANATIONS["run_dev_subset.sh"] = """
**The canonical end-to-end pipeline, and the most useful single file for understanding execution
order.** A one-shot dev-subset shakedown of Stages 1-5 for namespace `dev-10k-v1`.

**Re-runnable by design:** every completed step drops a marker in `.pipeline_state/<namespace>/` and
is skipped next time, so if it dies at step N you just run it again and it resumes there. Delete a
marker to force one step to re-run. An `ERR` trap names the step that failed.

The real order it runs:

```
00_download -> 01b_dev_subset -> 02b_splits -> 03_preprocess -> 04_captions
  -> 10_stage1_lora (accelerate, FOREGROUND) -> 11_set_lora_ckpt
  -> 20_recipes -> 21_pilot -> 22_approve_pilot -> 23_generate_full
  -> 30_aux_classifier -> 31_signals -> 32_gonogo
  -> 33_feasibility -> 34_build_subsets -> 35_eval_subsets -> 36_train_learned
  -> 37_fixed_thresh -> 38_contexts -> 39_verify_thresh -> 40_thresh_net
  -> 41_full_policy -> 42_finalize
  -> 50_conditions -> 51_stage5_eval -> 52_compare
```

Note what is **absent**: `scripts/asism/03_tune_freeze_select.py`. It is not a pipeline step.

Two workarounds are documented inline and worth knowing: Stage 1 runs in the **foreground** (not via
`launch_resumable.sh`, which returns immediately and would let Stage 2 start before a checkpoint
exists), and `optimizer.name=adamw` is forced because the RunPod pytorch-2.8/cu128 image ships a
`bitsandbytes` without a CUDA binary.
"""

# --------------------------------------------------------------------------------------------
# Notebook layout: parts, intros, and the files in each.
# --------------------------------------------------------------------------------------------

PARTS: list[tuple[str, str, list[str]]] = [
    (
        "Part 1 - The configuration layer",
        "Every tunable in this project lives in YAML, not in code. Read these first: once you know "
        "what is configurable, the scripts read as plumbing around these numbers. Nothing here is "
        "loaded by hand - `scripts/utils/config.py` is the only reader.",
        [
            "configs/dataset_config.yaml",
            "configs/splits.yaml",
            "configs/stage1_lora_sdxl.yaml",
            "configs/stage2_generation.yaml",
            "configs/stage3_asism.yaml",
            "configs/stage4_classifier.yaml",
            "configs/smoke_e2e.yaml",
            "configs/accelerate_config.yaml",
        ],
    ),
    (
        "Part 2 - Shared utilities (`scripts/utils/`)",
        "The layer every stage stands on: config loading, provenance hashing, the label policy, the "
        "split access guard, artifact contracts, metrics, and the one shared classifier. If two "
        "stages must agree about something, that something is defined here exactly once.",
        [
            "scripts/utils/config.py",
            "scripts/utils/manifest.py",
            "scripts/utils/identifiers.py",
            "scripts/utils/seed.py",
            "scripts/utils/caption_builder.py",
            "scripts/utils/labels.py",
            "scripts/utils/chexpert_schema.py",
            "scripts/utils/splits.py",
            "scripts/utils/artifact_contracts.py",
            "scripts/utils/experiment_registry.py",
            "scripts/utils/metrics.py",
            "scripts/utils/classifier.py",
        ],
    ),
    (
        "Part 3 - Data preparation (`scripts/data/`)",
        "Acquire CheXpert, prove it is intact, cut the six patient-disjoint splits, preprocess the "
        "images, and write the captions. Everything downstream is hashed against the split manifest "
        "produced here.",
        [
            "scripts/data/00_download_dataset.py",
            "scripts/data/01_verify_download.py",
            "scripts/data/01b_build_dev_subset.py",
            "scripts/data/02_build_patient_splits.py",
            "scripts/data/02b_build_sixway_splits.py",
            "scripts/data/03_preprocess_images.py",
            "scripts/data/04_generate_captions.py",
        ],
    ),
    (
        "Part 4 - Stage 1: SDXL + LoRA fine-tuning",
        "Fine-tune the generator. One long GPU job, built to survive pod restarts and SSH drops, "
        "plus the two monitoring scripts used to watch it.",
        [
            "scripts/train/train_lora_sdxl.py",
            "scripts/train/launch_resumable.sh",
            "scripts/eval/generate_probe_samples.py",
            "scripts/eval/compute_fid_clipscore.py",
        ],
    ),
    (
        "Part 5 - Stage 2: Synthetic generation",
        "Decide what to generate (auditable label recipes), then generate it - behind a human pilot "
        "approval gate, with every image's intended label vector recorded.",
        [
            "scripts/generate/01_sample_label_recipes.py",
            "scripts/generate/02_generate_synthetic_images.py",
        ],
    ),
    (
        "Part 6 - Stage 3, first half: the auxiliary classifier, the signals, and the Go/No-Go gate",
        "This is where the thesis starts. Six per-image signals are computed, then each must earn "
        "its place through a gate that runs **before** any tuning.",
        [
            "scripts/classify/00_train_auxiliary_classifier.py",
            "scripts/asism/signals.py",
            "scripts/asism/00c_derive_expected_regions.py",
            "scripts/asism/01_compute_signals.py",
            "scripts/asism/02_gonogo.py",
        ],
    ),
    (
        "Part 7 - Stage 3, the superseded weighted-score selector",
        "Kept for the historical record and for its engineering patterns. **Not a pipeline step** - "
        "nothing downstream consumes its output.",
        ["scripts/asism/03_tune_freeze_select.py"],
    ),
    (
        "Part 8 - Stage 3, second half: Learned ASISM (04 -> 09)",
        "The novel contribution. Measure the real downstream utility of ~120 controlled subsets, "
        "learn a set-utility model from those measurements, distil it into a per-image ranker, then "
        "learn per-class admission thresholds under conservative, pre-registered governance - all on "
        "`asism_tuning_heldout`, never on `final_eval_heldout`.",
        [
            "scripts/asism/candidate_pool.py",
            "scripts/asism/learned.py",
            "scripts/asism/models.py",
            "scripts/asism/04_build_utility_subsets.py",
            "scripts/asism/04b_evaluate_utility_subsets.py",
            "scripts/asism/05_train_learned_asism.py",
            "scripts/asism/06_learn_thresholds_select.py",
            "scripts/asism/07_build_threshold_contexts.py",
            "scripts/asism/07b_verify_thresholds_proxy.py",
            "scripts/asism/08_train_threshold_network.py",
            "scripts/asism/08b_verify_full_policy_proxy.py",
            "scripts/asism/09_finalize_learned_selection.py",
        ],
    ),
    (
        "Part 9 - Stage 4: the thesis conditions",
        "Three classifiers under one frozen protocol. Only the training data differs.",
        ["scripts/classify/01_train_conditions.py"],
    ),
    (
        "Part 10 - Stage 5: protected final evaluation and analysis",
        "`final_eval_heldout` is opened exactly once, under an auditable run id - then the statistics "
        "run locally off the pod.",
        [
            "scripts/eval/stage5_evaluate.py",
            "scripts/eval/compare_conditions.py",
        ],
    ),
    (
        "Part 11 - Smoke pipelines",
        "How the whole chain gets exercised without CheXpert and without a GPU. These validate "
        "**engineering connectivity only**; their outputs are permanently marked non-scientific.",
        [
            "scripts/smoke/00_build_fixture.py",
            "scripts/smoke/run_smoke_pipeline.py",
            "scripts/smoke/01b_learned_asism_cpu_smoke.py",
        ],
    ),
    (
        "Part 12 - Tests",
        "The tests are where the project's invariants are written down executably. `tests/run_all.py` "
        "is the documented production STOP gate.",
        [
            "tests/run_all.py",
            "tests/fixture_workspace.py",
            "tests/test_metrics.py",
            "tests/test_asism_signals.py",
            "tests/test_learned_asism.py",
            "tests/test_pipeline_contracts.py",
            "tests/test_provenance_contracts.py",
            "tests/test_roundtrip_contracts.py",
        ],
    ),
    (
        "Part 13 - Cloud execution (Modal)",
        "Running the repository's own scripts on rented GPUs, without forking the code.",
        [
            "cloud/modal_stage1.py",
            "cloud/modal_bench.py",
            "cloud/modal_gpu_probe.py",
        ],
    ),
    (
        "Part 14 - Environment and drivers",
        "Dependencies, and the one script that runs the whole pipeline end to end.",
        [
            "environment/requirements.txt",
            "run_dev_subset.sh",
        ],
    ),
]

DOC_APPENDIX = [
    "README.md",
    "docs/project_overview.md",
    "docs/proposal.md",
    "docs/novelty_target_decision.md",
    "docs/stage1_plan.md",
    "docs/stages2_to_5_plan.md",
    "docs/introduction.md",
    "docs/literature_review.md",
    "docs/smoke_e2e.md",
    "docs/runpod_stage2_to_5_commands.md",
    "environment/RUNPOD_MATRIX.md",
    "cloud/BENCHMARKS.md",
]

AUDIT = """
---

# Part 16 - Audit log: contradictions found, and what was done

This notebook was built alongside a consistency audit of the whole repository. Recorded here so the
reasoning is not lost in the git log.

## Fixed: the F -> C rename was incomplete

Commit `a8691ce` renamed the ASISM-selected arm from **F** to **C** in the smoke config and smoke
scripts, but not in the documentation or the config comments. Left behind:

| Where | Said | Now says |
|---|---|---|
| `README.md` | "Primary comparison: **F vs. B**" | C vs. B |
| `docs/project_overview.md` | an **A/B/F/D** table, with D as a live arm, in a paragraph that then claimed "primary comparison C vs B" | A/B/C, with D explicitly marked INACTIVE |
| `docs/smoke_e2e.md` | "condition F", "`conditions: [A, B, F]` in the GPU overlay" | condition C; the overlay already said `[A, B, C]` |
| `docs/stages2_to_5_plan.md` | "Supporting comparisons: A vs. B and A vs. **F**" | A vs. C |
| `configs/stage4_classifier.yaml` | "B vs **F** (selected subset)"; the matched-random control "was matched to the removed condition **C**" and should be "re-pointed at condition **F**" | both inverted by the rename and corrected: C exists, F does not |
| 5 code docstrings, `configs/splits.yaml`, the plan's split tables | stale **A-E** / **A-G** condition ranges | A/B/C |

The code itself was already right: `01_train_conditions.py` and `compare_conditions.py` both record
`"C vs B"`. Only the prose had drifted. The `stages2_to_5_plan.md` note that records the *historical*
"F vs C" conflict is deliberately left as the record it is.

## Fixed: an accidentally deleted docstring

`scripts/asism/01_compute_signals.py` lost its 19-line module docstring in commit `0190fa6`, whose
message is *"Add new research papers for NeurIPS 2024 and 2025"* - the docstring deletion was the
commit's only change and is not mentioned. Because the CLI does
`argparse.ArgumentParser(description=__doc__)`, `--help` had been printing no description ever since.
Restored verbatim from `8b69933`.

Two related cases were repairs of *deliberate* removals that left dangling references:
`00c_derive_expected_regions.py` had a comment pointing at "SPLIT DISCIPLINE in the module docstring"
that no longer existed (the rule is now stated where it applies), and
`04_build_utility_subsets.py`'s `preflight_check_candidate_pool_inputs` had a whitespace-only line
where its docstring had been.

## Fixed: stale cross-references

- `03_preprocess_images.py` told the user to run `02_build_patient_splits.py` when a split file was
  missing - but it reads v2 namespaced splits, so the right command is
  `02b_build_sixway_splits.py --namespace <ns>`.
- `README.md`'s Stage 1 pipeline listed `02_build_patient_splits.py` with no namespace flags. The
  real sequence is `02b`, and `run_dev_subset.sh` is its executable form.
- `README.md`'s layout block listed only `scripts/{data,train,eval,utils}` - omitting `asism/`,
  `generate/`, `classify/` and `smoke/`, i.e. most of the thesis.
- `labels.py`'s `intended_vector_to_labels` docstring said agreement scores "all 12 predicted
  probabilities". Agreement is scored over the **11** `PRIMARY_ENDPOINT_LABELS` the function
  iterates; 12 is `GENERATION_TARGET_LABELS`, a different set.

## Removed: code with zero references

Each of these had exactly one occurrence in the repository - its own definition - and no test:

- **`scripts/utils/provenance_gate.py`, the whole module.** Its own docstring said "new Stage 2-5
  scripts should call `validate_upstream_artifact()`" - but Stage 2-5 was built on
  `artifact_contracts.require_manifest_fields` / `require_score_artifact` /
  `require_generation_complete`, which do the same job with per-field mismatch reporting.
  `docs/stages2_to_5_plan.md` 1.5 cited the unused helper as *the* enforcement mechanism and now
  names the ones actually in force.
- `splits.require_frozen_production_splits` - superseded by `require_frozen_production_split_run`.
- `experiment_registry.read_registry`
- `labels.SECONDARY_LABELS`
- `01_compute_signals.PRODUCED_SIGNALS`
- `01_sample_label_recipes.DEVICE_LABEL` - duplicated `caption_builder.DEVICE_COLUMN`
- `stage5_evaluate.file_hash` - the script uses `manifest.sha256_file` throughout

## Kept, but now labelled

Four helpers are unit-tested design alternatives that nothing on the current path calls. They were
kept - the reasoning is worth preserving - but each now says so in its own docstring instead of only
in the git history: `models.soft_selection_gate` (the pipeline thresholds hard),
`signals.duplicate_clusters` (near-duplicates are rejected individually),
`02b.partition_patients` (superseded by `multilabel_partition_patients`), and
`splits.freeze_patient_folds` / `build_patient_folds` (only `03_tune_freeze_select.py` calls them).

## Kept as-is: dormant by design, not dead

These look unused but are deliberate, and are documented as such in the code:

- **Condition D** and its whole matched-draw machinery (`build_matched_random_draw`, `profile_of`,
  the `condition_d` config block, `compare_conditions`' D handling). It is the documented fix for the
  thesis's one structural gap: C is a strict subset of B, so a C-over-B gain cannot be attributed to
  ASISM's *ranking* rather than to simply using *fewer* synthetic images. Re-enabling needs
  supervisor sign-off and `n_draws x 3` extra runs.
- **`06_learn_thresholds_select.py`** - the fixed-ratio threshold, an ablation reference, explicitly
  not a Stage 4 condition.
- **`03_tune_freeze_select.py`** - the historical weighted-score baseline. Not a pipeline step and
  not imported by anything; retained with that stated plainly.
- **`02_build_patient_splits.py`** - the superseded v1 three-way split builder, still imported by
  `01b_build_dev_subset.py` for its `extract_patient_id` helper (which `02b` duplicates identically).
- **The distinctiveness signal** - computed and Go/No-Go-gated, but its feature column is not yet in
  `configs/stage3_asism.yaml` -> `learned_asism.feature_columns`, so the ranker consumes 9 columns,
  not 10. A documented gap, consistent across `models.py`, the config and `project_overview.md`.
- **The superseded v1 split artifacts** and their `SUPERSEDED_v1.md` note - a deliberate audit trail.

## Still open, for a supervisor decision, not a code change

- **No matched-random control.** `docs/stages2_to_5_plan.md` 7.1, and stated in
  `01_train_conditions.py` itself. This is the one substantive limitation of the primary comparison.
- **Condition C has never been run end to end**, not even in smoke form, because the GPU smoke tier
  was built but never executed - no GPU was available. See `docs/smoke_e2e.md`.
"""

LANGUAGE_BY_SUFFIX = {".py": "python", ".yaml": "yaml", ".yml": "yaml", ".sh": "bash",
                      ".txt": "text", ".md": "markdown"}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, *, tags: list[str] | None = None) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": tags} if tags else {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def source_cells(relative: str) -> list[dict]:
    """One explanation cell, then one cell holding the whole file."""
    path = REPO / relative
    if not path.is_file():
        return [markdown(f"## `{relative}`\n\n> **Missing from this checkout.**")]
    body = path.read_text(encoding="utf-8")
    explanation = EXPLANATIONS.get(relative, "").strip("\n")
    header = f"## `{relative}`\n\n*{len(body.splitlines())} lines*\n"
    cells = [markdown(f"{header}\n{explanation}" if explanation else header)]
    suffix = path.suffix.lower()
    if suffix == ".py":
        cells.append(code(f"# ===== FILE: {relative} =====\n{body}", tags=["project-source"]))
    elif suffix == ".md":
        # Markdown files contain their own fenced code blocks, so wrapping them in another fence
        # would terminate it early. Emit them raw instead and let them render.
        cells.append(markdown(body))
    else:
        language = LANGUAGE_BY_SUFFIX.get(suffix, "text")
        cells.append(markdown(f"```{language}\n{body}\n```"))
    return cells


INTRO = """
# The Whole Project, One Notebook

**Adaptive Quality-Aware Synthetic Data Selection (ASISM)** - master's thesis, AI, Zewail City
University. SDXL fine-tuned with LoRA on CheXpert chest X-rays, used to generate synthetic images,
which are then *adaptively selected* before classifier training and final evaluation.

## How to read this

**One cell = one file.** Every source file in the repository appears exactly once, in full, in a
cell of its own. Each file's cell is preceded by a short explanation of what it does, where it sits
in the pipeline, what it reads and writes, and anything about it that is easy to misread.

**The order is execution order, not alphabetical order.** Reading top to bottom walks the pipeline:
config -> shared utilities -> data -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> smoke
-> tests -> cloud. The design documents are appended at the end, unmodified.

**The repository is the source of truth.** This notebook is generated from it by
`notebooks/build_full_project_notebook.py`. Re-run that script after changing any file - do not
edit source inside this notebook.

> **Do not execute a source cell in isolation.** Almost every script takes command-line arguments
> and enforces upstream artifact gates. Use the launcher cell below to run them properly, from the
> repository root. Source cells carry the tag `project-source`.
"""

PIPELINE = """
## The pipeline in one picture

```
                        configs/*.yaml  ------------ the only place tunables live
                              |
  scripts/data/00,01          v
  download + verify CheXpert ---> 02b: SIX patient-disjoint splits --> split_manifest_v2.json
                                        |                                   (everything
        +-------------------------------+                                 downstream is
        |                               |                                hashed against
        v                               v                                   this file)
  gen_train / gen_val          classifier_train / classifier_val
        |                               |        asism_tuning_heldout     final_eval_heldout
        |                               |                 |                       |
        v                               |                 |                       |
  03 preprocess --> 04 captions         |                 |                  (SEALED until
        |                               |                 |                    Stage 5)
        v                               |                 |                       |
  STAGE 1  train_lora_sdxl.py           |                 |                       |
        |  +--> frozen LoRA adapter     |                 |                       |
        v                               |                 |                       |
  STAGE 2  01 label recipes --> 02 generate (pilot gate)  |                       |
        |  +--> ~5k synthetic images + intended_label_vector                      |
        v                               |                 |                       |
  STAGE 3  classify/00 auxiliary classifier (gen_train / gen_val only)            |
        |  asism/00c empirically derived expected regions |                       |
        |  asism/01 six signals --> asism/02 GO/NO-GO gate|                       |
        |         |                                       |                       |
        |         v                                       |                       |
        |  LEARNED ASISM                                  |                       |
        |  04  design ~120 controlled subsets             |                       |
        |  04b MEASURE each subset's real utility --------+ (proxy training,      |
        |  05  set-utility net --> per-image ranker       |  evaluated here ONLY) |
        |  06  fixed-ratio threshold (ablation only)      |                       |
        |  07  bootstrap class contexts + grid search ----+                       |
        |  07b proxy-verify thresholds                    |                       |
        |  08  Adaptive Threshold Network + governance    |                       |
        |  08b proxy-verify whole POLICIES ---------------+                       |
        |  09  pre-registered tie rule --> ONE policy     |                       |
        |      +--> adaptive_selected_manifest.jsonl      |                       |
        v                                                                         |
  STAGE 4  classify/01    A: real only                                            |
                          B: real + ALL synthetic     identical protocol,         |
                          C: real + ASISM-selected    equal OPTIMIZER STEPS       |
        |                                                                         |
        v                                                                         v
  STAGE 5  eval/stage5_evaluate.py --- opens final_eval_heldout ONCE <------------ +
        +--> eval/compare_conditions.py --- Holm-Bonferroni / BH, patient bootstrap

  PRIMARY COMPARISON: C vs B - does SELECTING synthetic images beat using all of them?
```

### Four invariants to hold in your head while reading

1. **Image-disjoint everywhere.** Train-role and val-role subsets never share an image; threshold
   contexts come from one pool only.
2. **Proxy-only tuning.** Every Stage 3 decision is made on `asism_tuning_heldout`.
   `final_eval_heldout` is opened once, in Stage 5, behind a run-id guard.
3. **No pseudo-replication.** The ranker is never trained on a subset's AUROC copied onto its member
   images; it is distilled from measured *set-level* utility.
4. **Frozen manifests + hashes at every step.** A config change alters the config hash and forces a
   re-run from the affected stage forward.
"""

LAUNCHER_INTRO = """
## Launcher

The cell below finds the repository root and gives you `run_script()`, which runs the original
project scripts from the repository root with their normal command-line behaviour. Use it instead of
executing source cells.
"""

LAUNCHER_CODE = (
    "from pathlib import Path\n"
    "import subprocess\n"
    "import sys\n"
    "\n"
    "def find_repo_root(start=Path.cwd()):\n"
    "    for candidate in (start.resolve(), *start.resolve().parents):\n"
    "        if (candidate / 'scripts').is_dir() and (candidate / 'configs').is_dir():\n"
    "            return candidate\n"
    "    raise FileNotFoundError('Open this notebook from the project checkout.')\n"
    "\n"
    "REPO = find_repo_root()\n"
    "\n"
    "def run_script(relative_path, *args, env=None):\n"
    "    command = [sys.executable, str(REPO / relative_path), *map(str, args)]\n"
    "    print('Running:', subprocess.list2cmdline(command))\n"
    "    return subprocess.run(command, cwd=REPO, env=env, check=True)\n"
    "\n"
    "print('Repository:', REPO)"
)

SWITCH_INTRO = """
## Safe local checks

Both switches default to `False` so nothing long-running starts by accident. The smoke fixture is
explicitly non-scientific and does not represent thesis results.
"""

SWITCH_CODE = (
    "RUN_TESTS = False\n"
    "RUN_LOCAL_SMOKE = False\n"
    "\n"
    "if RUN_TESTS:\n"
    "    run_script('tests/run_all.py')\n"
    "if RUN_LOCAL_SMOKE:\n"
    "    run_script('scripts/smoke/run_smoke_pipeline.py', '--phase', 'local')\n"
    "if not (RUN_TESTS or RUN_LOCAL_SMOKE):\n"
    "    print('Nothing executed. Change a switch to True when ready.')"
)


def build() -> dict:
    cells: list[dict] = [
        markdown(INTRO),
        markdown(PIPELINE),
        markdown(LAUNCHER_INTRO),
        code(LAUNCHER_CODE),
        markdown(SWITCH_INTRO),
        code(SWITCH_CODE),
    ]

    for title, intro, files in PARTS:
        cells.append(markdown(f"---\n\n# {title}\n\n{intro}"))
        for relative in files:
            cells.extend(source_cells(relative))

    cells.append(
        markdown(
            "---\n\n# Part 15 - Design documents (verbatim)\n\n"
            "The rationale behind every decision above. Reproduced unmodified; the `.md` files in "
            "`docs/` remain the source of truth."
        )
    )
    for relative in DOC_APPENDIX:
        cells.extend(source_cells(relative))

    cells.append(markdown(AUDIT))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    notebook = build()
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    covered = sum(len(part[2]) for part in PARTS)
    described = sum(1 for part in PARTS for name in part[2] if EXPLANATIONS.get(name))
    print(f"Wrote {OUTPUT}")
    print(f"  {len(notebook['cells'])} cells")
    print(f"  {covered} source files ({described} with written explanations)"
          f" + {len(DOC_APPENDIX)} documents")

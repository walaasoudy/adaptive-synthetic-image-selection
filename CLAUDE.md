# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Master's thesis code: fine-tune SDXL with LoRA on medical images, generate synthetic images, select
them with ASISM (Adaptive Synthetic Image Selection Module), train a classifier, and compare
conditions. Two dataset tracks share one repository:

- **CheXpert** (the original track, described in `README.md`): scripts with numeric prefixes and no
  dataset prefix (`scripts/asism/04_…`, `scripts/classify/01_train_conditions.py`), configs
  `stage1_lora_sdxl.yaml` … `stage4_classifier.yaml`, `splits.yaml`.
- **HAM10000** (the active track): every file is prefixed `ham10000_` (or lives in
  `scripts/data/ham10000/`), configs `ham10000_*.yaml`, `splits_ham10000.yaml`. Split namespace
  `ham-stratified-v1`, LoRA run `ham-lora-v1`.

Do not mix the two tracks: a HAM10000 change should not touch the unprefixed CheXpert scripts.

## Commands

There is no build or lint step. Run everything from the repository root.

Local work is CPU only (no GPU). The default `python` has no dependencies; use the conda env
(`asism` and `thesis` are equivalent: Python 3.11, torch 2.8.0+cpu, pytest):

```bash
conda activate asism
python tests/run_all.py                                   # full gate: discovers every tests/test_*.py
python -m pytest tests/test_ham10000_stage4.py -q         # one suite
python -m pytest tests/test_ham10000_stage4.py -q -k name # one test
python -m pytest -q -p no:cacheprovider tests/test_asism_v2_*.py   # the ASISM v2 suites
```

- `tests/run_all.py` is the documented STOP gate before any pod run. It runs suites that have a
  `__main__` self-runner as scripts and the rest through pytest, so a new suite needs one or the
  other to be counted.
- Tests that use `tmp_path` fail at setup inside a sandbox that cannot list the Windows temp
  directory. That is environmental; re-run outside the sandbox before treating it as a failure.
- `python scripts/smoke/run_smoke_pipeline.py --phase local` runs the CPU fixture pipeline
  (`docs/smoke_e2e.md`). Smoke artifacts are stamped `scientific_evidence: false`.

GPU steps run on RunPod, not locally. The exact ordered commands are in
`docs/ham10000_runpod_commands.md` (HAM10000) and `docs/runpod_stage2_to_5_commands.md` /
`run_dev_subset.sh` (CheXpert). Scripts that train real models take an explicit phase or an
`--i-understand-this-trains-real-models` flag; `estimate` / `feasibility` / `plan` phases are CPU.

## Architecture

**Configuration.** `configs/*.yaml` is the single source of truth for tunables. Scripts load it
through `scripts/utils/config.py` (OmegaConf), never by parsing YAML directly. Paths resolve against
the `PROJECT_ROOT` environment variable, so one checkout works on a pod (`/workspace/...`) and
locally. `THESIS_CONFIG_OVERLAY` merges an overlay file (used by the smoke lane). CLI overrides are
OmegaConf dotlist entries (`training.max_train_steps=2000`).

**Splits and the protected test set.** Splits are six-way and frozen under a namespace:
`gen_train`, `gen_val`, `classifier_train`, `classifier_val`, `asism_tuning_heldout`,
`final_eval_heldout`. Each has a fixed permitted use (table in
`docs/ham10000_v2_experimental_contract.md` §2). `final_eval_heldout` is reachable only through
`assert_final_eval_access_allowed` in `scripts/utils/splits.py`; no new code path may read it, and
`classifier_val` is never used to select or rank synthetic images.

**Stage flow (HAM10000).** The ordered stage commands are in `docs/ham10000_runpod_commands.md`.
`scripts/asism/candidate_pool.py` is the shared signal merge that later stages read. Stage 4
conditions are A real, B real + all, C real + selected, D real + matched random. Stages communicate
through files with provenance: each consumer checks the producer's config, split and checkpoint
hashes and refuses stale inputs.

**Three generations of the selection module.** They coexist; do not "clean up" the older ones.

- *v1* (`scripts/asism/ham10000_03…05`, `ham10000_ranking.py`, `ham10000_thresholds.py`): ran end to
  end; its results are recorded and its artifacts are never rewritten.
- *Follow-up diagnostics* (`scripts/followup/`): noise floor, instrument check, noise decomposition,
  headroom, the E4 quantity curve and its consequences, and the v2 select / Stage 4 grid / compare
  scripts.
- *ASISM v2* (`scripts/asism_v2/`): `features.py` (four signals, train-only normalisation),
  `models.py` / `pipeline.py` (size-aware set utility; the per-image score is a one-hidden-layer network, the linear score its reported baseline), `stopping.py` (per-class stopping on the
  lower 95% bound of marginal utility), `contracts.py` (exclusive-create outputs, split-role and
  grid validation), `instrument.py`, `preserve.py`, `prereg.py` (the approved configuration, frozen in code),
  `supervision.py`, `gates.py`, `persist.py`, `selection_files.py`. The runner is
  `scripts/followup/ham10000_asism_v2_utility.py` (plan, measure, G1) followed by
  `ham10000_asism_v2_learned_select.py` (accept, fit, select, stability). Its GPU measure phase
  is refused in code until approved; `scripts/smoke/asism_v2_ranker_dry_run.py` runs the whole
  path on CPU with a planted formula in place of the classifier, and
  `scripts/smoke/asism_v2_learned_e2e_smoke.py` carries it on through Stage 4, the aggregator and
  the comparison on fixture images. Pod order: `docs/ham10000_asism_v2_learned_pod_commands.md`.

## Experimental rules that constrain code changes

These come from `docs/ham10000_v2_experimental_contract.md` and apply to any change that can affect
which images are selected, how many, or how a result is judged:

- Every such parameter must be traceable to the contract. Items marked PENDING block whatever
  depends on them; `NOT APPROVED` sections (e.g. `docs/ham10000_asism_v2_final_protocol.md` §2) are
  enforced in code and must not be bypassed.
- A threshold is written before the run it judges and is never relaxed afterwards. A change is a new
  dated amendment written before the next run.
- Historical artifacts and reports are never overwritten; new outputs go to new directories. For
  example, `ham10000_02_gonogo.py` requires `--out` when given a non-canonical `--scores-dir`.
- Each change gets its own branch, with code and tests in separate commits.
- Stage 4 training is not bit-deterministic (`cudnn.deterministic` is not set); results reproduce as
  distributions over seeds, and tests should not assume otherwise.

## Repository conventions

- A `prepare-commit-msg` hook appends `[AI n%]` to the subject line from `.git/AI_PERCENT`. Set that
  file before committing and restore its previous value afterwards.
- Datasets, checkpoints, logs and outputs are gitignored. Run artifacts and evidence copies live
  outside the repo (`C:\Users\walaa\ham10000_work`, `C:\Users\walaa\ham10000_preservation`).
- Current status and open decisions: `docs/ham10000_status_2026-10-02.md`,
  `reports/Thesis audit and completion plan.md`, `docs/ham10000_results_and_limitations.md`.

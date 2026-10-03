# CheXpert track (the original pipeline)

This is the README as it stood while the project targeted CheXpert chest X-rays. The scripts it
names (numeric prefix, no `ham10000_` prefix) are still in the repository and still tested. The
active experiments are on HAM10000; see the top-level `README.md`.

---


Master's thesis project (AI, Zewail City University): fine-tuning SDXL with LoRA on CheXpert
chest X-rays, generating synthetic data, and adaptively selecting/weighting it (ASISM) before
classifier training and evaluation. See `docs/proposal.md` for the full 5-stage framework and
`docs/literature_review.md` for the grounding literature.

The repository contains executable code for Stages 1--5. Production thesis results still require
the real CheXpert data, trained checkpoints, generated images, and GPU execution; committed smoke
artifacts validate engineering connectivity only.

Stage 5 compares three conditions on `final_eval_heldout` (`configs/stage4_classifier.yaml` →
`conditions: [A, B, C]`): **A** real only, **B** real + all synthetic, **C** real + ASISM-
selected synthetic. There is no separate weighted-baseline selector or condition — ASISM is the full
module (`docs/stages2_to_5_plan.md` §4.9), including the Multi-Signal Utility Ranking Network and
Adaptive Threshold Learning below. Primary comparison: **C vs. B**. See
`docs/stages2_to_5_plan.md` §7.1 for a known limitation of that comparison (no matched-random
control) still pending a supervisor decision.

Learned-ASISM order (after Stage 2, auxiliary classifier, signal computation, and Go/No-Go):

```bash
python scripts/asism/04_build_utility_subsets.py --phase feasibility
python scripts/asism/04_build_utility_subsets.py --phase build
python scripts/asism/04b_evaluate_utility_subsets.py --phase estimate
python scripts/asism/04b_evaluate_utility_subsets.py --phase run
python scripts/asism/05_train_learned_asism.py
python scripts/asism/06_learn_thresholds_select.py
# 06 is the fixed-ratio learned threshold — an intermediate/ablation artifact, not a Stage 4
# condition. The adaptive path that feeds condition C continues with:
python scripts/asism/07_build_threshold_contexts.py
python scripts/asism/07b_verify_thresholds_proxy.py --phase estimate
python scripts/asism/07b_verify_thresholds_proxy.py --phase run --i-understand-this-trains-real-models
python scripts/asism/08_train_threshold_network.py
python scripts/asism/08b_verify_full_policy_proxy.py --phase estimate
python scripts/asism/08b_verify_full_policy_proxy.py --phase run --i-understand-this-trains-real-models
python scripts/asism/09_finalize_learned_selection.py
```

`scripts/asism/03_tune_freeze_select.py` (the pre-learned weighted-score selector) is retained as
the historical baseline the learned module replaced, and is exercised only by its own tests. Nothing
imports it and nothing in the Stage 4/5 pipeline consumes its output — do not run it as a pipeline
step. The shared signal-merge used by the learned stages (04–09) is
`scripts/asism/candidate_pool.py`, not this script.

## Environment

Target hardware: RunPod, 1x NVIDIA A40 (48GB), PyTorch 2.8.0 / CUDA 12.8 template. All paths that
must survive a pod restart (data, checkpoints, logs, HF cache) are expected to live on the
persistent volume — set `PROJECT_ROOT` (see `configs/stage1_lora_sdxl.yaml`) to wherever this repo
is cloned (e.g. `/workspace/chest-synth-thesis` on RunPod, or the local checkout when developing).

```bash
pip install -r environment/requirements.txt
```

After a verified working setup, freeze exact versions:

```bash
pip freeze > environment/requirements-lock.txt
```

## Pipeline (Stage 1)

Run from the repo root, in order. `NS` is the split namespace every later stage is pinned to
(`dev-<run>` or `production-<run>`):

```bash
NS=production-thesis-v1
python scripts/data/00_download_dataset.py                  # kagglehub download + link into data/chexpert/raw/ (idempotent)
python scripts/data/01_verify_download.py                   # sanity-check the extracted CheXpert-v1.0-small archive
python scripts/data/02b_build_sixway_splits.py --namespace production --run-id "$NS" --freeze
python scripts/data/03_preprocess_images.py --namespace "$NS" --splits gen_train gen_val classifier_train classifier_val asism_tuning_heldout final_eval_heldout
python scripts/data/04_generate_captions.py --namespace "$NS" --splits gen_train gen_val
bash scripts/train/launch_resumable.sh                      # resumable SDXL LoRA training (checks checkpoints/.../latest.json)
```

`scripts/data/02_build_patient_splits.py` is the superseded schema-v1 three-way builder and is not a
pipeline step; `configs/splits.yaml` replaced it. For the full Stage 1–5 sequence, including the dev
namespace, see `run_dev_subset.sh` — it is the executable version of this list.

Evaluation/monitoring:

```bash
python scripts/eval/generate_probe_samples.py --checkpoint <path>
python scripts/eval/compute_fid_clipscore.py --checkpoint <path>
```

## Dataset

CheXpert-v1.0-small, obtained from Kaggle (`ashery/chexpert`) via `scripts/data/00_download_dataset.py`,
which downloads it with `kagglehub`, symlinks (or copies) it into `data/chexpert/raw/`, and runs
`01_verify_download.py` automatically. Requires Kaggle API credentials configured for `kagglehub`
(e.g. `~/.kaggle/kaggle.json` or the `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars). Not included in this repo.
See `docs/stage1_plan.md` §6 for the exact expected structure.

## Layout

```
data/chexpert/{raw,processed}/   # dataset (gitignored except splits/manifests)
configs/                         # YAML configs — single source of truth for tunables
scripts/data/                    # download, verify, six-way splits, preprocess, captions
scripts/train/                   # Stage 1 SDXL + LoRA
scripts/generate/                # Stage 2 label recipes + synthetic generation
scripts/asism/                   # Stage 3 signals, Go/No-Go, learned ASISM (04–09)
scripts/classify/                # auxiliary classifier + Stage 4 conditions
scripts/eval/                    # Stage 1 monitoring + Stage 5 evaluation and analysis
scripts/smoke/                   # CPU/GPU fixture pipelines (non-scientific by construction)
scripts/utils/                   # shared config, provenance, labels, splits, metrics, classifier
tests/                           # contract + unit suites; tests/run_all.py is the STOP gate
cloud/                           # Modal entry points (Stage 1, benchmarks, GPU probe)
notebooks/                       # thesis_full_project_code.ipynb — the whole project, one notebook
checkpoints/stage1_lora_sdxl/    # resumable + inference-ready LoRA checkpoints (gitignored, per-run metadata tracked)
logs/, outputs/                  # TensorBoard logs, probe sample grids (gitignored)
docs/                            # proposal, literature review, this stage's design plan
```

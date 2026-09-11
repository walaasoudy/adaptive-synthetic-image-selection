# Adaptive Quality-Aware Synthetic Data Selection Framework

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
Adaptive Threshold Learning below. Primary comparison: **F vs. B**. See
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

`scripts/asism/03_tune_freeze_select.py` (the pre-learned weighted-score selector) is retained for
its shared signal-merge helpers only. Nothing in the Stage 4/5 pipeline consumes its output — do not
run it as a pipeline step.

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

Run from the repo root, in order:

```bash
python scripts/data/00_download_dataset.py      # kagglehub download + link into data/chexpert/raw/ (idempotent)
python scripts/data/01_verify_download.py      # sanity-check the extracted CheXpert-v1.0-small archive
python scripts/data/02_build_patient_splits.py  # patient-level gen_train / gen_val / classifier_heldout split
python scripts/data/03_preprocess_images.py     # frontal-only filter, aspect-preserving resize+pad, quality checks
python scripts/data/04_generate_captions.py     # structured label-to-text captions (scripts/utils/caption_builder.py)
bash scripts/train/launch_resumable.sh          # resumable SDXL LoRA training (checks checkpoints/.../latest.json)
```

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
scripts/{data,train,eval,utils}/ # pipeline code
checkpoints/stage1_lora_sdxl/    # resumable + inference-ready LoRA checkpoints (gitignored, per-run metadata tracked)
logs/, outputs/                  # TensorBoard logs, probe sample grids (gitignored)
docs/                            # proposal, literature review, this stage's design plan
```

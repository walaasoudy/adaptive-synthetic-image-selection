# Adaptive Quality-Aware Synthetic Data Selection Framework

Master's thesis project (AI, Zewail City University): fine-tuning SDXL with LoRA on CheXpert
chest X-rays, generating synthetic data, and adaptively selecting/weighting it (ASISM) before
classifier training and evaluation. See `docs/proposal.md` for the full 5-stage framework and
`docs/literature_review.md` for the grounding literature.

The repository contains executable code for Stages 1--5. Production thesis results still require
the real CheXpert data, trained checkpoints, generated images, and GPU execution; committed smoke
artifacts validate engineering connectivity only.

Stage 3 preserves two separate selectors: the original proxy-tuned weighted baseline and the
learned ASISM proposal (set-utility model, marginal-utility image ranker, and class-aware adaptive
threshold network). They intentionally write separate manifests so Stage 4 can compare them.

Learned-ASISM order (after Stage 2 and signal computation):

```bash
python scripts/asism/04_build_utility_subsets.py
python scripts/asism/04b_evaluate_utility_subsets.py
python scripts/asism/05_train_learned_asism.py
python scripts/asism/06_learn_thresholds_select.py
# 06 is the frozen fixed-ratio learned baseline. The adaptive path continues with:
python scripts/asism/07_build_threshold_contexts.py
python scripts/asism/07b_verify_thresholds_proxy.py --phase estimate
python scripts/asism/08_train_threshold_network.py
python scripts/asism/08b_verify_full_policy_proxy.py --phase estimate
python scripts/asism/09_finalize_learned_selection.py
```

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

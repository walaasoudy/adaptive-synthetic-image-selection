# Adaptive Synthetic Image Selection (ASISM)

Master's thesis project (AI, Zewail City University). A diffusion model generates synthetic medical
images, and the Adaptive Synthetic Image Selection Module (ASISM) decides which of them, and how
many, are added to the real training set of a classifier.

The experiments are on **HAM10000** (dermatoscopy, 10,015 images, 7 classes). The pipeline was first
built for CheXpert chest X-rays; that track is still in the repository and is described in
[docs/chexpert_pipeline.md](docs/chexpert_pipeline.md).

## Framework

| Stage | What it does |
|---|---|
| 1 | Fine-tune Stable Diffusion XL with LoRA on the real images |
| 2 | Generate synthetic images |
| 3 | **ASISM**: score every synthetic image, rank it, and select |
| 4 | Train a DenseNet-121 classifier on real + selected synthetic images |
| 5 | Evaluate against real only and real + all synthetic |

ASISM uses four signals per image:

- similarity to real images of the same class (DINOv2, k-NN);
- image quality assessment (blur, contrast, border artifacts);
- uncertainty (Monte Carlo dropout, 20 passes);
- explainability (Grad-CAM typicality against real images of the same class).

A ranking network learns a weight for each image from these signals, and a stopping rule decides how
many images of each class are kept. No count or ratio is given to it.

Stage 5 compares four conditions:

| Condition | Training data |
|---|---|
| A | real images only |
| B | real + all synthetic images |
| C | real + images selected by ASISM |
| D | real + a random draw with the same number of images per class as C |

C against D isolates the effect of selection.

## Status

- **Stages 1 and 2:** done. One rank-32 LoRA (8,000 steps); 3,168 candidate images.
- **Signals:** computed for all candidates; the four signals passed the Go/No-Go gate.
- **Version 1** ran end to end. Balanced accuracy on the held-out test set (3 seeds):
  A 0.566, B 0.645, C 0.612. C used 616 images and did not beat B. The results and their
  limitations are in [docs/ham10000_results_and_limitations.md](docs/ham10000_results_and_limitations.md).
- **Follow-up experiments** (noise floor, instrument check, selection headroom, quantity curve) are
  in `scripts/followup/`.
- **Version 2** is developed on the branch
  [`docs/readme-ham10000-asism`](https://github.com/walaasoudy/adaptive-synthetic-image-selection/tree/docs/readme-ham10000-asism), not on `main`. Its ranking and stopping code
  (`scripts/asism_v2/`) is written and tested on toy data. It has not been trained on real
  measurements yet. Design:
  [experimental contract](https://github.com/walaasoudy/adaptive-synthetic-image-selection/blob/docs/readme-ham10000-asism/docs/ham10000_v2_experimental_contract.md),
  [quantity design check](https://github.com/walaasoudy/adaptive-synthetic-image-selection/blob/docs/readme-ham10000-asism/docs/asism_v2_quantity_design_check_2026-10-03.md).

A short summary of the whole project is in
[Thesis_Summary_ASISM.docx](https://github.com/walaasoudy/adaptive-synthetic-image-selection/blob/docs/readme-ham10000-asism/docs/Thesis_Summary_ASISM.docx) on that branch.

## Setup

```bash
pip install -r environment/requirements.txt
```

GPU stages run on RunPod. Set `PROJECT_ROOT` to the checkout; every config path resolves against it.
CPU steps and the tests run locally in a Python 3.11 environment with the same requirements.

## Tests

```bash
python tests/run_all.py                                # every suite
python -m pytest tests/test_ham10000_stage4.py -q      # one suite
```

## Running the HAM10000 pipeline

The full ordered command list, with what to check after each step, is in
[docs/ham10000_runpod_commands.md](docs/ham10000_runpod_commands.md). In outline:

```bash
NS=ham-stratified-v1
python scripts/data/ham10000/00_download_dataset.py
python scripts/data/ham10000/01_build_splits.py --namespace production --run-id $NS --freeze
python scripts/data/ham10000/02_preprocess_images.py --namespace $NS
python scripts/data/ham10000/04_prepare_lora_inputs.py --namespace $NS
python scripts/train/ham10000_train_lora_sdxl.py --run-id ham-lora-v1                 # Stage 1
python scripts/generate/ham10000_generate_synthetic_images.py --build-recipes         # Stage 2
python scripts/classify/ham10000_train_auxiliary_classifier.py --namespace $NS
python scripts/asism/ham10000_01_compute_signals.py --namespace $NS                   # Stage 3 signals
python scripts/asism/ham10000_02_gonogo.py --namespace $NS
```

Stage 2 generates a small pilot first and needs a recorded approval before the full run. Stage 4 is
`scripts/classify/ham10000_train_conditions.py`; Stage 5 is `scripts/eval/ham10000_stage5_evaluate.py`
followed by `scripts/eval/ham10000_compare_conditions.py`.

## Data splits

HAM10000 is split at lesion level into six parts with fixed uses:

| Split | Images | Used for |
|---|---|---|
| `gen_train` | 3,586 | LoRA training; reference for similarity and Grad-CAM |
| `gen_val` | 388 | monitoring Stage 1 |
| `classifier_train` | 1,641 | the real part of every classifier |
| `classifier_val` | 1,401 | monitoring Stage 4 |
| `asism_tuning_heldout` | 1,377 | utility measurements inside ASISM |
| `final_eval_heldout` | 1,622 | the final evaluation only |

## Layout

```
configs/            YAML configs (ham10000_*.yaml for HAM10000)
scripts/data/       download, splits, preprocessing
scripts/train/      Stage 1
scripts/generate/   Stage 2
scripts/asism/      Stage 3: signals, gate, version 1 ranking and thresholds
scripts/followup/   experiments run after version 1
scripts/classify/   auxiliary classifier and Stage 4
scripts/eval/       Stage 5
scripts/utils/      shared code
tests/              test suites
docs/               design documents, results, thesis text
```

The dataset, checkpoints, generated images and run outputs are not in the repository.

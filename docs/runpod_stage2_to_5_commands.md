# RunPod production execution handoff

Use one immutable split run ID everywhere. The examples use `production-thesis-v1`; changing any
split assignment requires a new ID. Dev commands use `dev-smoke-v1`. Outputs from the two IDs are
physically separate and every consumer verifies the split, configuration, checkpoint, source-tree,
and upstream artifact hashes.

## Local, no GPU

```powershell
python -m compileall -q scripts tests
python tests/run_all.py
python scripts/data/02b_build_sixway_splits.py --help
python scripts/asism/03_tune_freeze_select.py --help
python scripts/classify/01_train_conditions.py --plan-only --namespace dev-smoke-v1
```

The plan-only command still requires a valid `dev-smoke-v1` split manifest. A fixture/dev build is:

```powershell
python scripts/data/02b_build_sixway_splits.py --namespace dev --run-id dev-smoke-v1 --input-csv data/chexpert/raw/train.csv
python scripts/data/03_preprocess_images.py --namespace dev-smoke-v1 --splits gen_train gen_val classifier_train classifier_val asism_tuning_heldout final_eval_heldout
python scripts/data/04_generate_captions.py --namespace dev-smoke-v1 --splits gen_train gen_val
```

Preprocessing resumes independently per split and validates its manifest before skipping an image.
Captions are intentionally generated only for the generator splits. These commands need CPU and
roughly the size of the selected source images plus processed JPEGs; they do not establish scientific
validity.

## RunPod setup and immutable inputs

Provision a persistent volume of at least 200 GB. A40 48 GB is the reviewed target. Application
dependencies and immutable model revisions are in `environment/requirements.txt` and
`environment/RUNPOD_MATRIX.md`.

```bash
cd /workspace/chest-synth-thesis
export PROJECT_ROOT=/workspace/chest-synth-thesis
pip install -r environment/requirements.txt
python -c "import torch,torchvision,diffusers,transformers,accelerate,peft,timm,pyarrow; print(torch.__version__,torch.version.cuda,torchvision.__version__,torch.cuda.is_available())"
python -m compileall -q scripts tests
python tests/run_all.py
nvidia-smi
```

STOP unless the matrix is Python 3.11, torch 2.8.0, torchvision 0.23.0, CUDA 12.8, CUDA available,
and every test passes.

## Exact production order

1. Verify the complete CheXpert source and image tree (CPU/I/O):

   ```bash
   python scripts/data/01_verify_download.py
   ```

2. Build and atomically freeze a new patient split (CPU/RAM, normally minutes). This refuses an
   existing run directory and refuses production freeze if the support gate fails:

   ```bash
   python scripts/data/02b_build_sixway_splits.py --namespace production --run-id production-thesis-v1 --freeze
   ```

   STOP and manually review `data/chexpert/processed/splits/production-thesis-v1/split_manifest_v2.json`,
   prevalence deviations, rare-label support, hashes, and patient counts.

3. Process all real splits (CPU/I/O; expect several hours and tens of GB) and build Stage 1 captions:

   ```bash
   python scripts/data/03_preprocess_images.py --namespace production-thesis-v1 --splits gen_train gen_val classifier_train classifier_val asism_tuning_heldout final_eval_heldout
   python scripts/data/04_generate_captions.py --namespace production-thesis-v1 --splits gen_train gen_val
   ```

4. Stage 1 smoke and explicit resume test (GPU). Both use the same immutable namespace:

   ```bash
   bash scripts/train/launch_resumable.sh split.namespace=production-thesis-v1 training.max_train_steps=20 training.checkpointing_steps=10
   # wait for step 10+, stop the worker once, then repeat the identical command and verify continuity
   bash scripts/train/launch_resumable.sh split.namespace=production-thesis-v1 training.max_train_steps=20 training.checkpointing_steps=10
   ```

   STOP and inspect loss, samples, checkpoint metadata, and resume continuity. Then start the frozen
   full Stage 1 configuration with `split.namespace=production-thesis-v1`. Select one explicit LoRA
   checkpoint only after probe review; record its path in `configs/stage2_generation.yaml`.

5. Stage 2 recipes, pilot, manual gate, and full generation (GPU; potentially tens of GPU-hours and
   tens of GB depending on recipe count):

   ```bash
   python scripts/generate/01_sample_label_recipes.py --namespace production-thesis-v1
   python scripts/generate/02_generate_synthetic_images.py --mode pilot --namespace production-thesis-v1
   ```

   STOP for visual/manual pilot review. Approval is irreversible provenance, not an automatic check:

   ```bash
   python scripts/generate/02_generate_synthetic_images.py --approve-pilot --reviewer "<name>" --namespace production-thesis-v1
   python -u scripts/generate/02_generate_synthetic_images.py --mode full --namespace production-thesis-v1
   ```

6. Auxiliary classifier and Stage 3 signals (GPU except generic IQA):

   ```bash
   python scripts/classify/00_train_auxiliary_classifier.py --namespace production-thesis-v1 --max-steps 8000
   python scripts/asism/01_compute_signals.py --signal all --namespace production-thesis-v1
   python scripts/asism/02_gonogo.py --namespace production-thesis-v1
   ```

   STOP after Go/No-Go. Review which signals were admitted before continuing: the learned-ASISM
   feature set is derived from exactly those signals, and every stage below is frozen against them.

   `03_tune_freeze_select.py` (the weighted-score selector) is **not part of the thesis pipeline** —
   the thesis's ASISM is the full module including the two learned components, so there is no
   weighted-baseline condition to produce. The script is retained for reference only; do not run it.

6b. Learned ASISM — set-utility model, ranking network, adaptive threshold network
   (`docs/stages2_to_5_plan.md` §4.9). Each `--phase estimate` prints a GPU-hour estimate against
   `configs/stage3_asism.yaml`'s compute-budget gates; `--phase run` is refused without the explicit
   confirmation flag, so nothing here can accidentally launch paid GPU time:

   ```bash
   python scripts/asism/04_build_utility_subsets.py --phase feasibility --namespace production-thesis-v1
   python scripts/asism/04_build_utility_subsets.py --phase build --namespace production-thesis-v1
   python scripts/asism/04b_evaluate_utility_subsets.py --phase estimate --namespace production-thesis-v1
   python scripts/asism/04b_evaluate_utility_subsets.py --phase run --namespace production-thesis-v1
   python scripts/asism/05_train_learned_asism.py --namespace production-thesis-v1
   python scripts/asism/06_learn_thresholds_select.py --namespace production-thesis-v1
   ```

   STOP at `--phase feasibility`: it fails closed if the candidate pool is too small for the
   configured `subset_design.subset_sizes`. Roughly **3,750+ usable synthetic candidates** are needed
   under the current sizes `[100, 200, 250]`. Fix by revising `subset_sizes` (or generating more in
   Stage 2) and re-running feasibility — never by editing the thresholds to make a failing report
   pass.

   `06` produces the fixed-ratio learned threshold — an intermediate artifact and ablation
   reference, not a Stage 4 condition. The adaptive path that feeds condition F continues:

   ```bash
   python scripts/asism/07_build_threshold_contexts.py --namespace production-thesis-v1
   python scripts/asism/07b_verify_thresholds_proxy.py --phase estimate --namespace production-thesis-v1
   python scripts/asism/07b_verify_thresholds_proxy.py --phase run --i-understand-this-trains-real-models --namespace production-thesis-v1
   python scripts/asism/08_train_threshold_network.py --namespace production-thesis-v1
   python scripts/asism/08b_verify_full_policy_proxy.py --phase estimate --namespace production-thesis-v1
   python scripts/asism/08b_verify_full_policy_proxy.py --phase run --i-understand-this-trains-real-models --namespace production-thesis-v1
   python scripts/asism/09_finalize_learned_selection.py --namespace production-thesis-v1
   ```

7. Stage 4 plan and 9 classifier runs — the thesis's three arms A/B/F × 3 seeds
   (`docs/stages2_to_5_plan.md` §7 v3 revision note; GPU; budget before launch, commonly many
   GPU-hours):

   ```bash
   python scripts/classify/01_train_conditions.py --plan-only --namespace production-thesis-v1
   python scripts/classify/01_train_conditions.py --condition all --namespace production-thesis-v1
   ```

   `--condition all` trains every condition in `configs/stage4_classifier.yaml` → `conditions`
   (currently `[A, B, F]`). F will hit an `UPSTREAM GATE` error if step 6b was skipped — that is the
   intended fail-closed behavior, not a bug.

   **Before launching, confirm the §7.1 decision:** F is a strict subset of B, so with only A/B/F
   there is no matched-random control and an F-over-B gain cannot be attributed to ASISM's ranking
   rather than to using fewer synthetic images. Adding that control costs `n_draws × 3` more runs.
   Settle it now — it cannot be added after results are seen.

   Each run resumes from its optimizer checkpoint; validated completed runs are skipped.

8. Register and run the one protected final evaluation (GPU inference), then analyze locally (CPU):

   ```bash
   python scripts/eval/stage5_evaluate.py --final-eval-run-id final-thesis-v1 --namespace production-thesis-v1
   python scripts/eval/compare_conditions.py --run-dir outputs/stage5/production-thesis-v1/final-thesis-v1
   ```

   STOP before registration for final protocol/checkpoint/hash review. The first outcome-bearing read
   changes registration state. Technical resume must reuse the identical run ID and hashes; a second
   methodological evaluation is refused.

## Resume units

| Producer | Resume unit | Completion contract |
|---|---|---|
| preprocessing | image within each split | decoded image plus matching per-split manifest |
| Stage 1 | optimizer step | model/optimizer/RNG plus exact split and config identity |
| Stage 2 | image | valid JPEG plus unique provenance-bearing manifest row |
| auxiliary/proxy classifier | optimizer step | model/optimizer/scheduler/RNG/sampler and best model |
| ASISM signals | signal | Parquet hash plus provenance sidecar |
| ASISM search | trial/fold | compatible trial log and checkpoint provenance |
| Stage 4 | optimizer step and completed run | best checkpoint and atomic completion record |
| Stage 5 | condition | validated Parquet plus completion sidecar |

Current preserved root-level Stage 2 recipes and manifests are legacy artifacts. Current code
explicitly refuses them; regenerate them under the selected versioned namespace.

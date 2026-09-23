# HAM10000 on RunPod — the exact command order

The CheXpert runbook (`docs/runpod_stage2_to_5_commands.md`) does not apply here: its 200 GB volume
and its stage order are sized for a dataset forty times larger. This is the HAM10000 one.

**Namespace:** `ham-stratified-v1` — frozen. Every command below names it explicitly.

**Four pod sessions, not one.** The steps between them are CPU work and human review; an idle pod
costs the same as a working one. Terminate at the end of each session — the Network Volume is what
makes that safe.

| Session | Steps | GPU hours | Gate before the next one |
|---|---|---|---|
| 1 | setup, data, Stage 1 LoRA, Stage 2 pilot | 6 – 11 | **you review 35 generated images** |
| 2 | Stage 2 full, aux classifier, CAM reference, signals, Go/No-Go | 2 – 4 | **you read the Go/No-Go report** |
| 3 | utility measurement, ranking network, thresholds | 2 – 3 | **you read the selection manifest** |
| 4 | Stage 4 A/B/C, Stage 5 | 5 – 8 | — |

Total ≈ 13 – 25 GPU hours ≈ $7 – 14 at ~$0.40/hr, plus ~$3.50/month for a 50 GB Network Volume.

---

## Storage, once

Create a **Network Volume** (not a pod volume — that is deleted with the pod). **50 GB** is enough:
raw 3 GB, preprocessed 1 GB, SDXL cache 13 GB, candidates 2 GB, checkpoints and signals 2 GB.

A Network Volume is locked to one datacenter, so check GPU availability there *before* creating it.
Storage bills whether or not a pod is running.

## Pod session 1 — setup

The volume mounts at `/workspace`.

```bash
cd /workspace && git clone <your-repo-url> master
```

```bash
echo 'export PROJECT_ROOT=/workspace/master' >> ~/.bashrc
echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

`PROJECT_ROOT` is read by `scripts/utils/config.py`; without it outputs land somewhere ephemeral.
`HF_HOME` on the volume stops SDXL (~13 GB) from re-downloading on every pod.

The environment goes on the volume too, or it is reinstalled every session:

```bash
python -m venv /workspace/venv && echo 'source /workspace/venv/bin/activate' >> ~/.bashrc
```

```bash
source /workspace/venv/bin/activate && pip install -r environment/requirements.txt
```

### The gate — do not continue if this fails

```bash
python -c "import torch,torchvision,diffusers,transformers,accelerate,peft,timm,pyarrow; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

```bash
python -m compileall -q scripts tests && python -m pytest tests/ -q
```

Expect Python 3.11, torch 2.8.0, CUDA available, and 555 passing tests.

---

# Session 1 — data and the generator

## 1.1 Data (CPU on the pod; the download is faster here than at home)

Needs a Kaggle token at `~/.kaggle/kaggle.json`.

```bash
python scripts/data/ham10000/00_download_dataset.py
```

```bash
python scripts/data/ham10000/01_build_splits.py --namespace production --run-id ham-stratified-v1 --freeze
```

`--namespace` accepts only `production` or `dev`; the split's own name is the `--run-id`, and that is
what every later `--namespace ham-stratified-v1` refers to. For this thesis the splits were **not**
rebuilt on the pod: the local `ham-stratified-v1` was uploaded unchanged (md5 identical). The command
is for reproducing the project from scratch.

> **Run this once, ever.** Every artifact downstream is keyed to these splits. Rebuilding them later
> invalidates everything already computed. Back the manifest up immediately:
>
> ```bash
> cp -r /workspace/master/data/ham10000/processed/splits /workspace/BACKUP_splits
> ```

```bash
python scripts/data/ham10000/02_preprocess_images.py --namespace ham-stratified-v1
```

Resumable per image, so an interrupted run can simply be repeated.

```bash
python scripts/data/ham10000/03_smoke_subset.py --namespace ham-stratified-v1 --per-class-lesions 0
```

`--per-class-lesions 0` runs the check on every `gen_train` lesion. With the default sample of 40
lesions per class, one of the 27 checks failed on the pod; on the whole of `gen_train` all 27 passed.

This reads real pixels from `gen_train` only and prints numbers, not just pass/fail. It is the last
cheap chance to catch a greyscale conversion, a content box that never reaches the measurement, or
calibration touching evaluation data. **Read its output before continuing.**

```bash
python scripts/data/ham10000/04_prepare_lora_inputs.py --namespace ham-stratified-v1
```

Unpadded 4:3 images — not the letterboxed canvases, or the LoRA would learn to draw grey bars.

## 1.2 Stage 1 — SDXL + LoRA  (4 – 8 h)

```bash
python scripts/train/ham10000_train_lora_sdxl.py --run-id ham-lora-v1
```

> **Never use `--auto-resume`.** It ignores `--run-id` and continues whichever run
> `latest_run.json` names — after a memory probe or a test run, that is the probe, and the real
> training would be written into the probe's directory. Resume explicitly instead:
>
> ```bash
> python scripts/train/ham10000_train_lora_sdxl.py --run-id ham-lora-v1 --resume-from checkpoints/ham10000/stage1_lora_sdxl/ham-lora-v1/checkpoint-<N>
> ```

8000 optimizer steps, effective batch 16, 768×576, LoRA rank 32.

**Calibrate the estimate early:** watch the `tqdm` rate after ~100 steps and extrapolate. If it is far
below ~1 step/s, the bottleneck is volume I/O — see *If training is slow* below. Resuming with
`--resume-from` means a terminated pod loses only the steps since the last checkpoint.

## 1.3 Stage 2 — recipes and pilot  (~10 min)

```bash
python scripts/generate/ham10000_generate_synthetic_images.py --build-recipes
```

Per-class quotas from `gen_train` only, inverted by rarity: roughly nv 100, bkl 268, mel 292,
bcc 400, akiec 479, vasc 731, df 847 — about 3,120 candidates in total.

```bash
python scripts/generate/ham10000_generate_synthetic_images.py --mode pilot checkpoint.lora_weights_dir=checkpoints/ham10000/stage1_lora_sdxl/ham-lora-v1/final
```

The LoRA must be named explicitly: `checkpoint.lora_weights_dir` is `null` in
`configs/ham10000_stage2.yaml`, so pilot, approval and full generation refuse to start without it.
`final` is the snapshot chosen for this thesis; use the same directory in all three commands.

### ⛔ Stop here. Terminate the pod.

35 images, five per class, in the pilot output directory. Look at them yourself.

Full generation is ~3 GPU hours, and every stage after it is built on these images. If the pilot
looks wrong, the cheap fix is to retrain Stage 1 with different settings — not to generate 3,120
bad images first. `--mode full` refuses to run without an approval for these same inputs, by design.

---

# Session 2 — candidates and signals

```bash
cd /workspace/master && source /workspace/venv/bin/activate && git pull && nvidia-smi
```

## 2.1 Approve the pilot, then generate  (1.5 – 3 h)

Only after you have actually looked:

```bash
python scripts/generate/ham10000_generate_synthetic_images.py --approve-pilot --reviewer "Walaa" --notes "<what you checked>" checkpoint.lora_weights_dir=checkpoints/ham10000/stage1_lora_sdxl/ham-lora-v1/final
```

```bash
python scripts/generate/ham10000_generate_synthetic_images.py --mode full checkpoint.lora_weights_dir=checkpoints/ham10000/stage1_lora_sdxl/ham-lora-v1/final
```

```bash
python scripts/generate/ham10000_standardize_generated.py --input-dir <raw_generated> --output-dir <candidates>
```

Images that break the geometry contract go to `rejected_geometry.csv` and never enter the pool with
a guessed content box. Check that file is small.

## 2.2 The auxiliary classifier  (0.5 – 1 h)

```bash
python scripts/classify/ham10000_train_auxiliary_classifier.py --namespace ham-stratified-v1
```

The real-only model Stage 3 scores *with*. It trains on `classifier_train`, never on `gen_train`,
because `gen_train` is the explainability reference split — a model trained on it would have
memorised the images whose attention it defines. Its `cam_model_id` is pinned by every calibrated
row downstream.

## 2.3 IQA calibration  (CPU, minutes)

```bash
python scripts/asism/ham10000_calibrate_iqa_blur.py --namespace ham-stratified-v1
```

```bash
python scripts/asism/ham10000_calibrate_iqa_border.py --namespace ham-stratified-v1
```

The blur threshold is the 2nd percentile of `gen_train` Laplacian variance; there is no constant
fallback, so this must exist before any IQA run. The border script changes no config — it writes
evidence and contact sheets.

## 2.4 Grad-CAM reference  (0.25 – 0.5 h)

```bash
python scripts/asism/ham10000_00_build_cam_reference.py --namespace ham-stratified-v1
```

Without it every `explainability_calibrated_typicality` is NaN and the feature is refused at
selection.

## 2.5 The five signals  (0.5 – 1.5 h)

```bash
python scripts/asism/ham10000_01_compute_signals.py --namespace ham-stratified-v1
```

Five separate parquet files, each with a provenance sidecar. The expensive part is MC Dropout:
20 passes over ~3,120 candidates.

## 2.6 Go/No-Go  (CPU, seconds)

```bash
python scripts/asism/ham10000_02_gonogo.py --namespace ham-stratified-v1
```

### ⛔ Stop. Terminate the pod, and read the report.

Each signal comes back `include`, `ablation_only`, or `exclude`, with a reason. This is a result in
its own right and belongs in the thesis — including the exclusions. Do not skim it.

---

# Session 3 — ranking and selection

```bash
cd /workspace/master && source /workspace/venv/bin/activate && git pull
```

## 3.1 Subsets  (CPU, seconds)

```bash
python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase feasibility
```

```bash
python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase build
```

`build` refuses unless the newest feasibility report matches this config with zero failures — so a
broken design cannot reach the GPU phase.

## 3.2 Measure  (1.5 – 3 h)

```bash
python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase measure
```

80 proxy trainings at 224 px, 300 steps each — about a minute apiece. This is the only honest source
of a target for the ranking network: nothing here estimates utility from the signals themselves,
which would be circular.

## 3.3 Ranking network  (~5 min)

```bash
python scripts/asism/ham10000_04_train_ranking_network.py --namespace ham-stratified-v1
```

Deep Sets utility model, then per-image targets distilled as size-normalised leave-one-out marginals,
then the ranking network fitted to those from signal features alone. A Banzhaf MSR cross-check runs
alongside the distillation.

## 3.4 Adaptive thresholds  (CPU, seconds)

```bash
python scripts/asism/ham10000_05_adaptive_thresholds.py --namespace ham-stratified-v1 --policy network
```

Writes `asism_selected.csv` — the only file Stage 4 condition C reads. Order is fixed: safety,
quality floor, class-aware threshold, class floor and cap.

### ⛔ Stop. Read the manifest before spending session 4.

Three things to check:

* **`residual_summary`** — `in_sample_mean` vs `leave_one_class_out_mean`. A large gap means the
  network reproduced the frozen search rather than learning a transferable rule from class context.
  That is a reportable limitation, already written up in `docs/ham10000_asism_methodology.md` §3.5;
  it does not invalidate the thresholds actually applied.
* **`quality_floor_attrition_per_class`** — any class losing ≥50% at the floor.
* **The per-class selected counts** — if the rare classes came out near their floor, condition C is
  close to condition A and the comparison will be weak.

---

# Session 4 — classifier and evaluation

```bash
cd /workspace/master && source /workspace/venv/bin/activate && git pull
```

## 4.1 Stage 4 — nine runs  (4 – 7 h)

```bash
for C in A B C; do for S in 42 43 44; do python scripts/classify/ham10000_train_conditions.py --condition $C --seed $S; done; done
```

A = real only · B = real + all synthetic · C = real + selected. Equal **optimizer steps** (3000), not
equal epochs: at fixed epochs the larger B set would silently get more gradient updates, conflating
"more data" with "more training" in exactly the comparison the thesis makes.

```bash
python scripts/classify/ham10000_aggregate_conditions.py --namespace ham-stratified-v1
```

Verifies that only the data differed, over `FAIRNESS_KEYS`. **If this fails, stop** — do not evaluate
an unfair comparison on the protected split. There is only one chance at it.

## 4.2 Stage 5 — the protected evaluation  (~15 min)

```bash
python scripts/eval/ham10000_stage5_evaluate.py --namespace ham-stratified-v1 --final-eval-run-id <choose-one-id>
```

The only read of `final_eval_heldout` for an outcome in the whole pipeline. It writes prediction
tables and computes no comparison and no p-value — that separation is what makes "we looked once" a
checkable statement rather than an intention.

```bash
cp -r <results_dir>/ham-stratified-v1 /workspace/BACKUP_stage5
```

### Terminate the pod. Nothing after this needs a GPU.

## 4.3 The analysis  (on your laptop)

```bash
conda run -n asism python scripts/eval/ham10000_compare_conditions.py --run-dir <run_dir>
```

Reads the prediction tables only — no torch, and the protected split is never reopened, so figures
and intervals can be iterated freely.

* **C vs B** is the confirmatory comparison: same generator, same candidates, same recipe — only the
  *selection* differs. Family-wise error controlled with Holm–Bonferroni.
* Everything else is exploratory, corrected with BH-FDR, and labelled as such wherever it appears.

---

# Operational notes

## If training is slow

Stage 4 reads ~96,000 images per run — 864,000 across nine runs. Over a network volume that I/O can
dominate and multiply wall-clock several times over.

The container disk is local NVMe and much faster. Preprocessed images are read-only inputs (~1 GB),
so losing them with the pod costs nothing:

```bash
mkdir -p /root/fast && cp -r /workspace/master/data/ham10000/processed /root/fast/
```

Point the run at `/root/fast/processed`. **Outputs stay on `/workspace`** so they survive termination.

## What to download to your own machine

The volume is one copy at one company — it is storage, not a backup. These are small and expensive
to recreate:

```
splits manifests · *.parquet signals · gonogo report · ranking model
asism_selected.csv · stage 5 prediction tables · every *.json manifest
```

Synthetic images are large and can be regenerated. `asism_selected.csv` and the reports cannot.

## Judging results before Stage 5

If the numbers look wrong and you want to change the training recipe, judge on
`asism_tuning_heldout` and re-run. Reading Stage 5 results and *then* adjusting is the one thing the
whole protocol exists to prevent — the four leakage defences would no longer mean anything.

## Resume points

| Step | Resume unit | Cost of interruption |
|---|---|---|
| preprocessing | per image | negligible — rerun |
| Stage 1 LoRA | optimizer step (`--resume-from`, never `--auto-resume`) | since last checkpoint |
| Stage 2 generation | per image | per-image, manifest-tracked |
| signals | per signal | recompute one signal |
| `--phase measure` | per subset | one subset (~1 min) |
| Stage 4 | per (condition, seed) run | one run (~30 min) |
| Stage 5 | per condition | one condition |

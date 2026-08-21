# Tiny Stage 1-5 end-to-end smoke run

This lane is engineering validation only. It creates 60 deterministic, patient-disjoint synthetic
CheXpert-shaped records under `outputs/smoke/dev-smoke-v1`; it never reads `data/chexpert`, never
uses a production namespace, and stamps the workspace `scientific_evidence: false`.

## One orchestrated command

From the repository root:

```bash
python scripts/smoke/run_smoke_pipeline.py --phase all
```

The command prints every subprocess, stops at the first failure, validates each producer artifact,
and resumes completed work. Its first invocation deliberately stops after generating the four pilot
images. Inspect:

```text
outputs/smoke/dev-smoke-v1/data/chexpert/synthetic/dev-smoke-v1/pilot/images/
```

Then resume the same pipeline with explicit fixture-pilot approval:

```bash
python scripts/smoke/run_smoke_pipeline.py --phase all --approve-smoke-pilot
```

The approval is valid only inside the marked smoke workspace and cannot unlock production.

## Local-only preparation

```bash
python tests/run_all.py
python scripts/smoke/run_smoke_pipeline.py --phase local
```

Expected CPU outputs:

| Stage | Input | Validated output |
|---|---|---|
| fixture | deterministic generator seed/rules | `outputs/smoke/dev-smoke-v1/SMOKE_ONLY.json`, 60 source JPEGs and CSV rows |
| split | fixture `train.csv` | six CSVs plus `split_manifest_v2.json` under the smoke workspace |
| preprocessing | six fixture split CSVs | six image directories, manifests, and failure-free logs |
| captions | processed `gen_train`/`gen_val` | two nonempty JSONL caption files and provenance manifests |

No GPU is required. On the tested local machine this phase took under one minute after code import.

## RunPod GPU phase

Use the pinned environment described in `environment/RUNPOD_MATRIX.md`, then either run the entire
command on the pod or copy the smoke workspace from local storage and run only the GPU phase:

```bash
cd /workspace/chest-synth-thesis
pip install -r environment/requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
python tests/run_all.py
python scripts/smoke/run_smoke_pipeline.py --phase all
# inspect the four pilot images; then:
python scripts/smoke/run_smoke_pipeline.py --phase all --approve-smoke-pilot
```

If the CPU workspace was copied to the identical repository-relative location:

```bash
python scripts/smoke/run_smoke_pipeline.py --phase runpod
# inspect pilot, then resume:
python scripts/smoke/run_smoke_pipeline.py --phase runpod --approve-smoke-pilot
```

Expected GPU-path outputs:

| Stage | Input | Validated output |
|---|---|---|
| Stage 1 | smoke images/captions, pinned SDXL | two-step LoRA final checkpoint plus resumable optimizer checkpoints |
| Stage 2 | explicit LoRA and four recipes | pilot approval record, four full images, generation completion manifest |
| auxiliary | smoke `gen_train`/`gen_val` | two-step resume checkpoint and separate best-validation checkpoint |
| Stage 3 signals | four synthetic images | five Parquets and five hash/provenance sidecars |
| Go/No-Go | score artifacts | `gonogo_report.json` with admitted/excluded signals |
| Stage 4 | conditions **A/B only**, one seed | two two-step best checkpoints and atomic completion records |
| Stage 5 | separate three-patient fixture final split | registered protected context, predictions with sidecars, comparison report/tables |

Cold-cache estimate on one A40 48 GB: approximately 30–60 minutes, dominated by model downloads,
SDXL initialization, and the tiny ASISM proxy matrix. Warm-cache resume should be materially faster.
Allow approximately 20–35 GB for pinned model caches, checkpoints, environment packages, and smoke
outputs. These are operational estimates, not measured production costs.

STOP if any artifact gate fails, if a path contains `production`, if the workspace lacks
`SMOKE_ONLY.json`, if pilot review fails, or if the orchestrator attempts to open real thesis data.

## Learned ASISM CPU integration smoke (04 → 05 → 06)

```bash
python scripts/smoke/run_smoke_pipeline.py --phase local   # builds the split/caption fixture this needs
python scripts/smoke/01b_learned_asism_cpu_smoke.py
```

CPU-only, no SDXL, under a minute. Fabricates 900 synthetic candidate images (label vectors only —
no JPEGs) and four of the five score signals (`explainability` is deliberately omitted), then runs
the **real** `04_build_utility_subsets.py`, `05_train_learned_asism.py`, and
`06_learn_thresholds_select.py` as subprocesses against that fixture, with `utility_results.jsonl`
fabricated (never a real proxy-classifier measurement) so `05` has something to train against
without a GPU. Explicitly verifies:

- Go/No-Go feature removal: `active_feature_columns()` narrows the ranking network's inputs to the
  4 surviving signals rather than crashing (the bug fixed 2026-08-21) or silently keeping the 5th.
- `contributing_signals` and `learned_variant_status` in the training manifest correctly report
  4 signals / `"primary"` (the primary-vs-`reduced_variant` boundary itself is unit-tested in
  `tests/test_learned_asism.py`, not here).
- Train/val image pools are disjoint (`image_overlap_fraction == 0.0`).
- A ranking-model checkpoint and a non-empty selected-images manifest are produced.

## Scientific limitations

This run demonstrates software connectivity only. It does not validate medical quality,
statistical power, rare-label support, or thesis outcomes. Nothing under this heading — including
the learned-ASISM CPU smoke above — is scientific evidence about ASISM's selection quality.

**REMAINING COVERAGE GAP.** The GPU-path table above stops after Go/No-Go and trains only
`conditions: [A, B]` (`configs/smoke_e2e.yaml`); it does not run `07`–`09` or train **condition F**.
Combined with the CPU tier above, current coverage is:

| Tier | Covers | Status |
|---|---|---|
| Unit tests | Every learned-ASISM function in isolation | ✅ `tests/test_learned_asism.py`, 88 tests |
| CPU integration smoke | `04` → `05` → `06` connect; Go/No-Go feature removal | ✅ `01b_learned_asism_cpu_smoke.py` |
| GPU smoke | `07`–`09`, real (tiny) proxy verification, condition **F** trained | ❌ not built yet |

Closing the last row needs a GPU-smoke extension of the existing RunPod phase: real (tiny) proxy
classifiers for `04b`/`07b`/`08b` instead of the CPU tier's fabricated numbers, plus
`conditions: [A, B, F]` in the GPU overlay. Until it exists, condition F and the thesis's primary
comparison (F vs. B) have never been run end to end anywhere, including in smoke form.

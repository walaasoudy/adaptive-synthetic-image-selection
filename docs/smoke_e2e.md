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
| Go/No-Go and tuning | score artifacts and smoke tuning folds | report, bounded baseline search log, frozen baseline selector and selected IDs |
| Stage 4 | conditions A/B/C, one seed | three two-step best checkpoints and atomic completion records |
| Stage 5 | separate three-patient fixture final split | registered protected context, predictions with sidecars, comparison report/tables |

Cold-cache estimate on one A40 48 GB: approximately 30–60 minutes, dominated by model downloads,
SDXL initialization, and the tiny ASISM proxy matrix. Warm-cache resume should be materially faster.
Allow approximately 20–35 GB for pinned model caches, checkpoints, environment packages, and smoke
outputs. These are operational estimates, not measured production costs.

STOP if any artifact gate fails, if a path contains `production`, if the workspace lacks
`SMOKE_ONLY.json`, if pilot review fails, or if the orchestrator attempts to open real thesis data.

## Scientific limitations

This run demonstrates software connectivity only. It does not validate medical quality,
statistical power, rare-label support, or thesis outcomes. A trainable Multi-Objective Ranking
Network, learned Adaptive Threshold model, and Pareto ranking are not implemented. The current
Stage 3 combination remains explicitly a weighted-score/proxy-tuned baseline pending the supervisor
target decision in `docs/novelty_target_decision.md`.

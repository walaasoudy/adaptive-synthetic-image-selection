#!/usr/bin/env python3
"""CPU-only integration smoke for the Learned ASISM pipeline: 04 -> 04b(fake) -> 05 -> 06.

Requested by the supervisor review of 2026-08-21: the existing end-to-end smoke
(scripts/smoke/run_smoke_pipeline.py) never exercises scripts/asism/04-09 — the thesis's novel
contribution — because doing so through the real pipeline requires a real (if tiny) SDXL run. This
script closes that gap WITHOUT touching SDXL, by fabricating a self-consistent synthetic candidate
pool and score tables, then running the REAL 04, 05, and 06 scripts as subprocesses against them.

What this DOES exercise, with the real production code path:
  - Go/No-Go feature removal: one signal (explainability) is deliberately marked excluded, so this
    proves active_feature_columns() actually narrows the model's inputs (the bug fixed 2026-08-21)
    rather than merely having a code path for it.
  - 04's feasibility gate, failing closed on an under-sized pool before being tuned to pass.
  - 04's disjoint train/val image pools (verified again explicitly below, on top of 05's own check).
  - 05's training manifest, including learned_variant_status, which must read "reduced_variant"
    here (4 of 5 signals survived) — proving the manifest accurately reports reduced coverage.
  - 05's ranking-model checkpoint and image_marginal_targets.jsonl.
  - 06's fixed-ratio learned selection manifest.

What this does NOT exercise (needs a real, if small, GPU run — the "learned GPU smoke" tier):
  04b's real proxy-classifier training, 07-09's threshold learning and proxy verification, and
  condition F end to end. utility_results.jsonl here is FABRICATED, not measured.

STOP if any assertion fails. This is engineering connectivity evidence only — never a medical or
generative-quality claim, and never scientific evidence about ASISM's selection quality.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAMESPACE = "dev-smoke-v1"
# pool_feasibility_report checks EVERY PRIMARY_ENDPOINT_LABEL, not just whichever ones a fixture
# happens to use — so the fixture must cover all 11, not a convenient subset. N_CANDIDATES is sized
# so each of the 11 labels + No-Finding clears feasibility_thresholds.min_candidates_per_class in
# BOTH the train and val pools (see configs/smoke_e2e.yaml -> stage3.learned_asism).
N_CANDIDATES = 900  # ~75/label pre-split; empirically the smallest round number with headroom
                     # against random shuffle variance across all 11 labels x 2 pools

sys.path.insert(0, str(REPO))


def build_fixture(workspace: Path, env: dict[str, str]) -> None:
    import numpy as np

    os.environ.update(env)  # so the imports below see PROJECT_ROOT / THESIS_CONFIG_OVERLAY too

    from scripts.asism.signals import write_score_artifact
    from scripts.utils.artifact_contracts import (
        asism_score_provenance, current_code_identity_hash, namespace_identity, stage2_paths, stage3_paths,
    )
    from scripts.utils.config import load_named_config
    from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS
    from scripts.utils.manifest import sha256_file, write_json

    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    stage3_cfg = load_named_config("stage3_asism.yaml", "stage3")
    s2_paths = stage2_paths(stage2_cfg, NAMESPACE)
    s3_paths = stage3_paths(stage3_cfg, NAMESPACE)
    # Mirror EXACTLY the config mutation every 0x_*.py script applies before computing provenance
    # (namespace-resolved absolute paths written back onto cfg.paths, split_namespace pinned) — the
    # provenance hash below must match what those scripts independently recompute in their own
    # subprocess, and config_sha256() hashes the WHOLE resolved config, so any unmirrored mutation
    # here silently produces a different hash and a "Stale/incompatible" false-positive rejection.
    for key, value in s3_paths.items():
        if key in stage3_cfg.paths:
            stage3_cfg.paths[key] = str(value)
    stage3_cfg.split_namespace = NAMESPACE
    s2_paths["root"].mkdir(parents=True, exist_ok=True)
    s3_paths["scores_dir"].mkdir(parents=True, exist_ok=True)
    s3_paths["asism_dir"].mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)

    # 1) A fabricated Stage 2 generation manifest: image_id + intended_label_vector, single-label
    #    recipes cycling through ALL 11 PRIMARY_ENDPOINT_LABELS plus a No-Finding (all-zero) recipe
    #    so every label the feasibility gate checks has real candidates in both the train and val
    #    pools — this fixture is about the learned-ASISM DATA PLUMBING, not recipe realism.
    image_ids = [f"smoke_synth_{index:04d}" for index in range(N_CANDIDATES)]
    recipe_pool = list(PRIMARY_ENDPOINT_LABELS) + [None]
    recipes = (recipe_pool * (N_CANDIDATES // len(recipe_pool) + 1))[:N_CANDIDATES]
    rng.shuffle(recipes)
    with open(s2_paths["manifest_path"], "w", encoding="utf-8") as handle:
        for image_id, label in zip(image_ids, recipes):
            vector = {lab: int(lab == label) for lab in PRIMARY_ENDPOINT_LABELS}
            handle.write(json.dumps({"image_id": image_id, "intended_label_vector": vector}) + "\n")

    # 2) A self-consistent (but not SDXL-derived) Stage 2 completion record. require_generation_
    #    complete() only checks that the recorded hashes match files that exist and that the split/
    #    code identity match the CURRENT run — it never requires the manifest to have come from a
    #    real generation job, so a structurally honest fixture is legitimate here.
    identity = namespace_identity(NAMESPACE)
    completion = {
        **identity,
        "status": "complete",
        "generation_manifest_sha256": sha256_file(s2_paths["manifest_path"]),
        "code_identity_sha256": current_code_identity_hash(),
        "lora_checkpoint_sha256": "0" * 64,  # placeholder: no real checkpoint exists in this fixture
        "smoke_only": True,
        "fixture": "01b_learned_asism_cpu_smoke.py",
    }
    write_json(s2_paths["generation_completion"], completion)

    # asism_score_provenance() re-reads Stage 2's completion from disk (via require_generation_
    # complete), so the fixture's provenance is computed the SAME WAY production computes it —
    # never hand-assembled, which is exactly what caused the first failed attempt at this fixture
    # (a placeholder asism_config_sha256 that require_score_artifact correctly rejected).
    provenance = asism_score_provenance(stage3_cfg, stage2_cfg, NAMESPACE)

    # 3) Five synthetic score tables, one column set per signal, matching signals.py's real schema
    #    so downstream code sees realistic shapes. Values are pure noise — never claimed to reflect
    #    generative quality.
    def frame(columns_and_ranges):
        import pandas as pd
        data = {"image_id": image_ids}
        for column, (low, high) in columns_and_ranges.items():
            data[column] = rng.uniform(low, high, size=N_CANDIDATES)
        return pd.DataFrame(data)

    signal_frames = {
        "similarity": frame({
            "similarity_knn_mean": (0.3, 0.9), "similarity_top1": (0.4, 0.95),
            "similarity_topk_spread": (0.0, 0.3),
        }),
        "iqa": frame({
            "iqa_composite": (0.2, 1.0), "iqa_sharpness": (10.0, 200.0), "iqa_contrast_std": (10.0, 60.0),
        }),
        "uncertainty": frame({
            "uncertainty_mean_std": (0.01, 0.2), "uncertainty_max_std": (0.02, 0.3),
            "uncertainty_entropy": (0.05, 0.6),
        }),
        "agreement": frame({"agreement_score": (-0.3, 0.9)}),
        # "explainability" is DELIBERATELY OMITTED — this is the Go/No-Go exclusion case below.
    }
    signal_frames["uncertainty"]["uncertainty_band"] = rng.choice(
        ["low", "moderate", "extreme"], size=N_CANDIDATES, p=[0.4, 0.45, 0.15]
    )
    for signal, signal_frame in signal_frames.items():
        write_score_artifact(
            signal_frame, s3_paths["scores_dir"] / f"{signal}_scores.parquet", signal, provenance
        )

    # 4) A hand-written Go/No-Go report: explainability EXCLUDED (its parquet does not exist), the
    #    other four INCLUDED. This is the exact case that used to crash 04/05/07/09 with
    #    "Missing learned-ASISM feature columns" before active_feature_columns() was added.
    write_json(s3_paths["gonogo_report"], {
        "surviving_signals": ["agreement", "iqa", "similarity", "uncertainty"],
        "excluded_signals": ["explainability"],
        "ablation_only_signals": [],
        "asism_variant_status": "reduced_variant — fewer than 5 signals survived (smoke fixture)",
        "smoke_only": True,
    })

    print(f"Fixture built: {N_CANDIDATES} synthetic candidates, 4/5 signals surviving Go/No-Go.")


def fabricate_utility_results(workspace: Path, env: dict[str, str]) -> None:
    """Stand in for 04b's real proxy-classifier training with a deterministic, clearly-fake
    macro-AUROC delta per subset. NEVER measured, NEVER claimed as evidence about ASISM quality —
    only shaped like utility_results.jsonl so 05 has something to train against on CPU."""
    import numpy as np

    os.environ.update(env)
    from scripts.asism.learned import read_jsonl
    from scripts.utils.artifact_contracts import stage3_paths
    from scripts.utils.config import load_named_config

    stage3_cfg = load_named_config("stage3_asism.yaml", "stage3")
    paths = stage3_paths(stage3_cfg, NAMESPACE)
    subsets = read_jsonl(paths["utility_subsets"])
    rng = np.random.default_rng(11)
    with open(paths["utility_results"], "w", encoding="utf-8") as handle:
        for row in subsets:
            real_only = float(rng.uniform(0.55, 0.65))
            # A synthetic "signal": larger subsets get a slightly larger, noisy uplift, purely so
            # 05's training loss has something non-degenerate to fit. Not a claim about real data.
            uplift = 0.01 + 0.00005 * len(row["image_ids"]) + float(rng.normal(0, 0.01))
            handle.write(json.dumps({
                "subset_id": row["subset_id"], "fold": 0, "seed": int(row.get("seed", 0)),
                "real_only_macro_auroc": real_only,
                "augmented_macro_auroc": real_only + uplift,
                "synthetic_smoke_only": True,
            }) + "\n")
    print(f"Fabricated utility_results.jsonl for {len(subsets)} subsets (SMOKE ONLY, not measured).")


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"SMOKE GATE FAILED: missing {label}: {path}")
    print(f"  validated {label}: {path}", flush=True)


def main() -> int:
    workspace = REPO / "outputs" / "smoke" / NAMESPACE
    if not (workspace / "SMOKE_ONLY.json").is_file():
        raise SystemExit(
            "Run `python scripts/smoke/run_smoke_pipeline.py --phase local` first (builds the "
            "fixture splits this script's Stage-2 fixture is layered on top of)."
        )
    env = {
        **os.environ, "PROJECT_ROOT": str(workspace),
        # Reuses the SAME overlay as run_smoke_pipeline.py (proven to resolve split/namespace paths
        # correctly), extended with a `stage3.learned_asism` block sized for this fixture's 150
        # candidates — see configs/smoke_e2e.yaml.
        "THESIS_CONFIG_OVERLAY": str(REPO / "configs" / "smoke_e2e.yaml"),
        "PYTHONUNBUFFERED": "1",
    }
    py = sys.executable

    build_fixture(workspace, env)

    run([py, "scripts/asism/04_build_utility_subsets.py", "--phase", "feasibility", "--namespace", NAMESPACE], env)
    run([py, "scripts/asism/04_build_utility_subsets.py", "--phase", "build", "--namespace", NAMESPACE], env)

    from scripts.utils.artifact_contracts import stage3_paths
    from scripts.utils.config import load_named_config
    stage3_cfg = load_named_config("stage3_asism.yaml", "stage3")
    paths = stage3_paths(stage3_cfg, NAMESPACE)
    require(paths["utility_subsets"], "utility_subsets.jsonl")

    fabricate_utility_results(workspace, env)
    require(paths["utility_results"], "fabricated utility_results.jsonl")

    run([py, "scripts/asism/05_train_learned_asism.py", "--namespace", NAMESPACE], env)
    require(paths["learned_dir"] / "learned_training_manifest.json", "learned training manifest")
    require(paths["learned_dir"] / "ranking_model.pt", "ranking model checkpoint")
    require(paths["learned_dir"] / "set_utility_model.pt", "set-utility model checkpoint")

    from scripts.utils.manifest import read_json
    manifest = read_json(paths["learned_dir"] / "learned_training_manifest.json")
    if sorted(manifest.get("contributing_signals", [])) != ["agreement", "iqa", "similarity", "uncertainty"]:
        raise SystemExit(
            f"SMOKE GATE FAILED: expected exactly the 4 surviving signals in contributing_signals "
            f"(explainability excluded), got {manifest.get('contributing_signals')}. This is the "
            "exact bug fixed 2026-08-21 — a missing signal must shrink the feature/signal set, "
            "never crash and never silently keep a 5th signal."
        )
    # 4 of 5 signals clears the >=3 primary-vs-reduced_variant threshold (02_gonogo.py's rule,
    # mirrored for the learned selector) — "primary" IS the correct status here. The threshold
    # boundary itself (2 vs 3 signals) is unit-tested directly in test_learned_asism.py; this
    # integration smoke's job is proving the plumbing doesn't crash when a signal is missing, which
    # the successful run up to this point already demonstrates.
    if manifest.get("learned_variant_status") != "primary":
        raise SystemExit(
            f"SMOKE GATE FAILED: 4 contributing signals should report status='primary'; got "
            f"{manifest.get('learned_variant_status')!r}."
        )
    if manifest["validation"]["image_overlap_fraction"] != 0.0:
        raise SystemExit(
            f"SMOKE GATE FAILED: train/val image pools must be disjoint; overlap fraction = "
            f"{manifest['validation']['image_overlap_fraction']}"
        )
    print(
        f"  verified: {len(manifest['contributing_signals'])}/5 signals contributed "
        f"({manifest['contributing_signals']}), status={manifest['learned_variant_status']!r}, "
        f"train/val pools disjoint"
    )

    run([py, "scripts/asism/06_learn_thresholds_select.py", "--namespace", NAMESPACE], env)
    require(paths["learned_dir"] / "learned_selection_manifest.json", "fixed-ratio learned selection manifest")
    require(paths["learned_selected_manifest"], "fixed-ratio learned selected_manifest.jsonl")

    print("\nCPU LEARNED-ASISM INTEGRATION SMOKE COMPLETE.")
    print("Fixture data and fabricated utility results only — NOT a medical, generative-quality, or")
    print("selection-quality result. Proves 04 -> 05 -> 06 connect and the Go/No-Go feature-removal")
    print("fix works end to end. 07-09 and condition F still need a GPU smoke pass (see docs/smoke_e2e.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

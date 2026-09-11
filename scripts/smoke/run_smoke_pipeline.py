#!/usr/bin/env python3
"""Resumable Stage 1-5 fixture pipeline; outputs are permanently marked non-scientific."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAMESPACE = "dev-smoke-v1"


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"SMOKE GATE FAILED: missing {label}: {path}")
    print(f"  validated {label}: {path}", flush=True)


def require_successful_preprocessing(root: Path, split: str) -> None:
    log = root / f"{split}_preprocessing_log.jsonl"
    require(log, f"{split} preprocessing log")
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    failed = [row for row in rows if row.get("action") == "failed"]
    kept = [row for row in rows if row.get("kept")]
    if failed or not kept:
        raise SystemExit(f"SMOKE GATE FAILED: {split} has failed={len(failed)}, kept={len(kept)}")


def run(command: list[str], env: dict[str, str]) -> None:
    print("\n+ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def run_if_missing(command: list[str], output_path: Path, env: dict[str, str]) -> None:
    """Skip a step whose output already exists, rather than reruning it into a FileExistsError.

    Several learned-ASISM scripts (04 build, 06, 09) write their single output file with mode "x"
    (exclusive create) — a deliberate guard against silently overwriting a real result. That is
    correct for a hand-invoked production run, but an ORCHESTRATOR that may be resumed after a
    partial failure must check first, exactly like this file already does for Stage 1's LoRA
    checkpoint above. Without this, resuming this pipeline after any later step fails throws away
    all of this step's (possibly expensive) work for no reason.
    """
    if output_path.exists():
        print(f"  skip (already exists): {subprocess.list2cmdline(command)}", flush=True)
        return
    run(command, env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["local", "runpod", "all"], default="all")
    parser.add_argument("--workspace", default=str(REPO / "outputs" / "smoke" / NAMESPACE))
    parser.add_argument("--approve-smoke-pilot", action="store_true",
                        help="Explicitly approve fixture pilot after visual review; never approves production")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    if workspace.name != NAMESPACE or "production" in str(workspace).lower():
        raise SystemExit("Smoke workspace must end in dev-smoke-v1 and must not contain 'production'")
    workspace.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PROJECT_ROOT": str(workspace), "THESIS_CONFIG_OVERLAY": str(REPO / "configs" / "smoke_e2e.yaml"),
           "THESIS_SMOKE_MODE": "1", "PYTHONUNBUFFERED": "1"}
    py = sys.executable

    if args.phase in {"local", "all"}:
        run([py, "scripts/smoke/00_build_fixture.py"], env)
        split_manifest = workspace / "data/chexpert/processed/splits" / NAMESPACE / "split_manifest_v2.json"
        if not split_manifest.is_file():
            run([py, "scripts/data/02b_build_sixway_splits.py", "--namespace", "dev", "--run-id", NAMESPACE], env)
        require(split_manifest, "six-way split manifest")
        run([py, "scripts/data/03_preprocess_images.py", "--namespace", NAMESPACE], env)
        for split in ("gen_train", "gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"):
            require(workspace / "data/chexpert/processed/images_768" / NAMESPACE / f"{split}_preprocessing_manifest.json", f"{split} preprocessing manifest")
            require_successful_preprocessing(workspace / "data/chexpert/processed/images_768" / NAMESPACE, split)
        run([py, "scripts/data/04_generate_captions.py", "--namespace", NAMESPACE, "--splits", "gen_train", "gen_val"], env)
        for split in ("gen_train", "gen_val"):
            caption_path = workspace / "data/chexpert/processed/captions" / NAMESPACE / f"{split}_captions.jsonl"
            require(caption_path, f"{split} captions")
            if not caption_path.read_text(encoding="utf-8").strip():
                raise SystemExit(f"SMOKE GATE FAILED: empty captions at {caption_path}")
        print("\nLOCAL SMOKE PHASE COMPLETE. No thesis data was accessed.", flush=True)
        if args.phase == "local":
            return 0

    require(workspace / "SMOKE_ONLY.json", "fixture-only marker")
    split_manifest = workspace / "data/chexpert/processed/splits" / NAMESPACE / "split_manifest_v2.json"
    require(split_manifest, "split manifest")

    stage1_run = workspace / "checkpoints/stage1_lora_sdxl/smoke-stage1-v1"
    final_lora = stage1_run / "final"
    if not (final_lora / "metadata.json").is_file():
        command = ["accelerate", "launch", "--config_file", "configs/accelerate_config.yaml",
                   "scripts/train/train_lora_sdxl.py", "--run-id", "smoke-stage1-v1"]
        latest = stage1_run / "latest.json"
        if latest.is_file():
            checkpoint_dir = json.loads(latest.read_text(encoding="utf-8"))["checkpoint_dir"]
            command += ["--resume-from", checkpoint_dir]
        run(command, env)
    require(final_lora / "metadata.json", "Stage 1 final LoRA metadata")
    env["SMOKE_LORA_DIR"] = str(final_lora)

    # Up to 400 recipes (the smoke quota yields ~270): the learned-ASISM feasibility gate below needs
    # several candidates per label in BOTH the train and the 20% val image pools; 40 left most labels
    # with 0-1 val candidates (measured on the A10 GPU smoke). Pilot review itself still only
    # samples stage2.pilot.num_images (4) regardless of this limit.
    run([py, "scripts/generate/01_sample_label_recipes.py", "--namespace", NAMESPACE, "--limit", "400"], env)
    synthetic = workspace / "data/chexpert/synthetic" / NAMESPACE
    require(synthetic / "recipes_manifest.json", "Stage 2 recipe manifest")
    run([py, "scripts/generate/02_generate_synthetic_images.py", "--mode", "pilot", "--namespace", NAMESPACE], env)
    require(synthetic / "pilot/pilot_approval_manifest.json", "pilot checks")
    if not args.approve_smoke_pilot:
        print("\nSTOP: inspect the four fixture pilot images, then rerun this same command with --approve-smoke-pilot.", flush=True)
        return 3
    run([py, "scripts/generate/02_generate_synthetic_images.py", "--approve-pilot", "--reviewer", "smoke-fixture-review",
         "--notes", "Fixture-only pipeline validation", "--namespace", NAMESPACE], env)
    run([py, "scripts/generate/02_generate_synthetic_images.py", "--mode", "full", "--namespace", NAMESPACE], env)
    require(synthetic / "generation_complete.json", "Stage 2 completion")

    run([py, "scripts/classify/00_train_auxiliary_classifier.py", "--namespace", NAMESPACE, "--max-steps", "2", "--batch-size", "2"], env)
    require(workspace / "checkpoints/auxiliary_classifier" / NAMESPACE / "auxiliary_classifier.best.pt", "auxiliary best checkpoint")
    run([py, "scripts/asism/01_compute_signals.py", "--signal", "all", "--namespace", NAMESPACE], env)
    for signal in ("similarity", "iqa", "uncertainty", "explainability", "agreement"):
        require(synthetic / "scores" / f"{signal}_scores.provenance.json", f"{signal} provenance")
    run([py, "scripts/asism/02_gonogo.py", "--namespace", NAMESPACE], env)
    require(synthetic / "asism/gonogo_report.json", "Go/No-Go report")

    # Learned ASISM (docs/stages2_to_5_plan.md §4.9) on the REAL signals just computed above — the
    # thesis's novel contribution, and the ONE part of this file never previously exercised
    # end-to-end (the CPU-only tier in 01b_learned_asism_cpu_smoke.py covers 04->05->06 on a
    # FABRICATED pool; this covers the full 04-09 chain, including real (tiny) proxy verification,
    # on REAL generated images). UNVERIFIED BY EXECUTION as of 2026-08-21 — no GPU was available to
    # run this locally. If `--phase feasibility` below fails, that is expected on a first attempt:
    # fix subset_design/feasibility_thresholds in configs/smoke_e2e.yaml (stage3.learned_asism) and
    # rerun feasibility ONLY — it needs no GPU and does not repeat Stage 1/2/aux/signals.
    learned = synthetic / "asism/learned"
    run([py, "scripts/asism/04_build_utility_subsets.py", "--phase", "feasibility", "--namespace", NAMESPACE], env)
    require(learned / "subset_design_report.json", "subset design feasibility report")
    run_if_missing(
        [py, "scripts/asism/04_build_utility_subsets.py", "--phase", "build", "--namespace", NAMESPACE],
        learned / "utility_subsets.jsonl", env,
    )
    require(learned / "utility_subsets.jsonl", "utility subsets")
    run([py, "scripts/asism/04b_evaluate_utility_subsets.py", "--phase", "estimate", "--namespace", NAMESPACE], env)
    run_if_missing(
        [py, "scripts/asism/04b_evaluate_utility_subsets.py", "--phase", "run", "--namespace", NAMESPACE],
        learned / "utility_results.jsonl", env,
    )
    require(learned / "utility_results.jsonl", "measured utility results")
    run([py, "scripts/asism/05_train_learned_asism.py", "--namespace", NAMESPACE], env)
    require(learned / "learned_training_manifest.json", "learned training manifest")
    require(learned / "ranking_model.pt", "ranking model checkpoint")
    run_if_missing(
        [py, "scripts/asism/06_learn_thresholds_select.py", "--namespace", NAMESPACE],
        learned / "selected_manifest.jsonl", env,
    )
    require(learned / "selected_manifest.jsonl", "fixed-ratio learned selection (condition-G ablation reference)")

    run([py, "scripts/asism/07_build_threshold_contexts.py", "--namespace", NAMESPACE], env)
    require(learned / "threshold_contexts.jsonl", "threshold contexts")
    run([py, "scripts/asism/07b_verify_thresholds_proxy.py", "--phase", "estimate", "--namespace", NAMESPACE], env)
    run_if_missing(
        [py, "scripts/asism/07b_verify_thresholds_proxy.py", "--phase", "run",
         "--i-understand-this-trains-real-models", "--namespace", NAMESPACE],
        learned / "threshold_proxy_measurements.jsonl", env,
    )
    require(learned / "threshold_proxy_measurements.jsonl", "threshold proxy measurements")
    run([py, "scripts/asism/08_train_threshold_network.py", "--namespace", NAMESPACE], env)
    require(learned / "adaptive_threshold_manifest.json", "adaptive threshold manifest")
    run([py, "scripts/asism/08b_verify_full_policy_proxy.py", "--phase", "estimate", "--namespace", NAMESPACE], env)
    run_if_missing(
        [py, "scripts/asism/08b_verify_full_policy_proxy.py", "--phase", "run",
         "--i-understand-this-trains-real-models", "--namespace", NAMESPACE],
        learned / "full_policy_verification_results.jsonl", env,
    )
    require(learned / "full_policy_verification_results.jsonl", "full-policy verification results")
    run_if_missing(
        [py, "scripts/asism/09_finalize_learned_selection.py", "--namespace", NAMESPACE],
        learned / "adaptive_selected_manifest.jsonl", env,
    )
    require(learned / "adaptive_selected_manifest.jsonl", "condition C: finalized Learned ASISM selection")

    run([py, "scripts/classify/01_train_conditions.py", "--condition", "all", "--namespace", NAMESPACE], env)
    require(workspace / "outputs/stage4" / NAMESPACE / "stage4_training_results.json", "Stage 4 A/B/C results")
    run([py, "scripts/eval/stage5_evaluate.py", "--final-eval-run-id", "smoke-final-v1", "--namespace", NAMESPACE,
         "--bootstrap-resamples", "20"], env)
    stage5 = workspace / "outputs/stage5" / NAMESPACE / "smoke-final-v1"
    require(stage5 / "final_eval_run_manifest.json", "Stage 5 smoke manifest")
    run([py, "scripts/eval/compare_conditions.py", "--run-dir", str(stage5), "--bootstrap-resamples", "20"], env)
    require(stage5 / "stage5_comparison_report.json", "Stage 5 comparison report")
    print("\nEND-TO-END SMOKE COMPLETE — FIXTURE ENGINEERING EVIDENCE ONLY, NEVER THESIS RESULTS.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

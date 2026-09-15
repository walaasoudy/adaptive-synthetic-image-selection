#!/usr/bin/env python3
"""Execute the proxy_verification_plan.json produced by 07_build_threshold_contexts.py.

This is the ONLY script in the threshold-learning pipeline that trains real proxy classifiers.
--phase run REFUSES to start (before touching any config, data, or model) unless called with
--i-understand-this-trains-real-models. That flag exists specifically so nothing — a test, a CI
job, an accidental invocation — can trigger real GPU training by mistake. --phase estimate never
requires it, because it never trains anything.

For each planned (context_id, candidate_threshold): rebuilds the identical hard subset (same rule
as 07's grid search), trains real + hard-subset vs a real-only baseline, and appends the MEASURED
utility to threshold_proxy_measurements.jsonl — a SEPARATE file from
threshold_candidate_evaluations.jsonl. This script never opens threshold_candidate_evaluations.jsonl
for writing: that file is 07's frozen, critic-only artifact, and mixing measured results into it is
exactly the confusion this split exists to prevent.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import read_jsonl  # noqa: E402
from scripts.utils.artifact_contracts import stage2_paths, stage3_paths  # noqa: E402
from scripts.utils.classifier import (  # noqa: E402
    TrainingBudget, evaluate_classifier, records_from_split, records_from_synthetic_manifest, train_classifier,
)
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.manifest import hash_dict, read_json  # noqa: E402
from scripts.utils.splits import load_split  # noqa: E402


def proxy_run_count(n_planned: int) -> dict:
    """How many proxy trainings --phase run performs. Informational only: there is no GPU-hour cap."""
    return {
        "n_planned_evaluations": n_planned,
        "total_proxy_runs": 1 + n_planned,  # 1 shared real-only baseline + 1 run per planned evaluation
    }


def rebuild_hard_subset(context: dict, threshold: float, ranking_scores: dict[str, float], selected_count: int) -> list[str]:
    """Deterministically reproduces the exact hard subset 07's grid search selected for this
    (context, threshold) pair — same rule: score >= threshold, truncated to the top `selected_count`
    by ranking_score."""
    candidates = [image_id for image_id in context["image_ids"] if ranking_scores.get(image_id, 0.0) >= threshold]
    return sorted(candidates, key=lambda i: -ranking_scores.get(i, 0.0))[:selected_count]


def train_and_score(train_records, eval_records, cfg, seed, checkpoint, tag):
    budget = TrainingBudget(max_steps=int(cfg.tuning.coarse.proxy_max_steps),
                            batch_size=int(cfg.tuning.proxy.batch_size),
                            learning_rate=float(cfg.tuning.proxy.learning_rate), seed=seed,
                            eval_every_n_steps=0)
    model, _, _ = train_classifier(
        train_records=train_records, val_records=[], budget=budget,
        dropout_p=float(cfg.tuning.proxy.dropout_p), pretrained_source=str(cfg.tuning.proxy.pretrained_source),
        resolution=int(cfg.tuning.proxy.resolution), progress_desc=tag,
        checkpoint_path=checkpoint, resume_from=checkpoint if checkpoint.is_file() else None,
        checkpoint_metadata={"dataset_id": tag, "condition_or_draw_id": tag,
                             "config_hash": hash_dict({"tag": tag, "seed": seed}, length=64),
                             "label_policy": "chexpert_mask_uncertain_v1"})
    return float(evaluate_classifier(model, eval_records, int(cfg.tuning.proxy.resolution))["macro_auroc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--phase", choices=["estimate", "run"], default="estimate")
    parser.add_argument(
        "--i-understand-this-trains-real-models", action="store_true",
        help="Required for --phase run. Confirms real GPU proxy-classifier training is about to start.",
    )
    args = parser.parse_args()

    if args.phase == "run" and not args.i_understand_this_trains_real_models:
        raise SystemExit(
            "REFUSED: --phase run launches real proxy-classifier training (GPU/RunPod cost).\n"
            "Pass --i-understand-this-trains-real-models to confirm. This guard runs before any "
            "config, data, or model is touched — nothing about this command can accidentally train."
        )

    cfg = load_named_config("stage3_asism.yaml", "stage3")
    stage2 = load_named_config("stage2_generation.yaml", "stage2")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths: cfg.paths[key] = str(value)
    cfg.split_namespace = namespace
    for key, value in stage2_paths(stage2, namespace).items():
        if key in stage2.paths: stage2.paths[key] = str(value)

    plan_path = Path(cfg.paths.proxy_verification_plan)
    if not plan_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: no proxy verification plan at {plan_path}.\n"
            "Run: python scripts/asism/07_build_threshold_contexts.py"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    n_planned = len(plan["planned_evaluations"])

    if args.phase == "estimate":
        print(json.dumps(proxy_run_count(n_planned), indent=2))
        return 0

    print(f"{proxy_run_count(n_planned)['total_proxy_runs']} proxy runs planned", flush=True)

    contexts = {row["context_id"]: row for row in read_jsonl(Path(cfg.paths.threshold_contexts))}
    evaluations = read_jsonl(Path(cfg.paths.threshold_candidate_evaluations))  # read-only reference
    evaluations_by_key = {(row["context_id"], row["candidate_threshold"]): row for row in evaluations}
    measurements_path = Path(cfg.paths.threshold_proxy_measurements)
    already_measured = {
        (row["context_id"], row["candidate_threshold"])
        for row in (read_jsonl(measurements_path) if measurements_path.is_file() else [])
    }

    baseline_module_path = Path(__file__).with_name("07_build_threshold_contexts.py")
    spec = importlib.util.spec_from_file_location("build_threshold_contexts_07", baseline_module_path)
    context_builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_builder)
    context_builder.preflight_check_phase1_outputs(cfg)
    merged, _, _ = context_builder.load_candidate_pool(cfg)
    columns = list(read_json(Path(cfg.paths.learned_dir) / "learned_training_manifest.json")["feature_columns"])
    _, ranker, training_manifest = context_builder.load_frozen_models(cfg, columns)
    normalized = context_builder.apply_feature_frame(merged, columns, training_manifest["normalization"])
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32)))
    ranking_scores: dict[str, float] = dict(
        zip(merged.image_id.astype(str), (1.0 / (1.0 + np.exp(-raw.numpy())))))

    stage1 = load_stage1_config()
    image_root = Path(stage1.paths.images_dir) / namespace
    real = records_from_split(
        load_split("classifier_train", namespace, purpose="schema_validation", caller="threshold_proxy_verification"),
        image_root / "classifier_train",
    )
    evaluation_records = records_from_split(
        load_split("asism_tuning_heldout", namespace, purpose="schema_validation", caller="threshold_proxy_verification"),
        image_root / "asism_tuning_heldout",
    )
    checkpoint_dir = Path(cfg.paths.learned_dir) / "threshold_proxy_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.learned_asism.subset_design.seed)
    baseline_checkpoint = checkpoint_dir / f"real_only_seed{seed}.pt"
    baseline_score = train_and_score(real, evaluation_records, cfg, seed, baseline_checkpoint, f"threshold-real-only-s{seed}")

    stage2_manifest = Path(stage2.paths.manifest_path)
    stage2_images = Path(stage2.paths.images_dir)

    with open(measurements_path, "a", encoding="utf-8") as handle:
        for index, planned in enumerate(plan["planned_evaluations"]):
            key = (planned["context_id"], planned["candidate_threshold"])
            if key in already_measured:
                continue
            context = contexts[planned["context_id"]]
            evaluation = evaluations_by_key.get(key)
            if evaluation is None:
                continue
            subset_ids = rebuild_hard_subset(context, planned["candidate_threshold"], ranking_scores,
                                             evaluation["selected_count"])
            synthetic = records_from_synthetic_manifest(stage2_manifest, stage2_images, subset_ids)
            tag = f"threshold-{planned['context_id']}-t{planned['candidate_threshold']:.2f}-s{seed}"
            checkpoint = checkpoint_dir / f"{hash_dict({'tag': tag}, length=32)}.pt"
            measured = train_and_score(real + synthetic, evaluation_records, cfg, seed, checkpoint, tag)
            row = {
                "context_id": planned["context_id"],
                "candidate_threshold": planned["candidate_threshold"],
                "selection_reasons": planned.get("selection_reasons", []),
                "selected_count": evaluation["selected_count"],
                "real_only_macro_auroc": baseline_score,
                "augmented_macro_auroc": measured,
                "measured_utility": measured - baseline_score,
                "seed": seed,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[{index + 1}/{n_planned}] {tag}: measured_utility={row['measured_utility']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

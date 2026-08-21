#!/usr/bin/env python3
"""Execute full_policy_verification_plan.json produced by 08_train_threshold_network.py.

Same fail-closed discipline as 07b_verify_thresholds_proxy.py: --phase run REFUSES to start (before
touching any config, data, or model) unless called with --i-understand-this-trains-real-models.
--phase estimate never requires it.

INDEPENDENT budget from 07b: learned_asism.threshold_network.full_policy_verification.compute_budget.
Never summed with 07b's per-context threshold-verification budget — they are separately approved,
separately executed production steps.

For each planned policy variant, builds the FINAL combined multi-label selected_manifest (per-class
thresholds combined via min(applicable_thresholds) — see build_policy_selected_manifest), then trains
real + that manifest's synthetic images vs a shared real-only baseline, for the configured number of
seeds, and appends results to full_policy_verification_results.jsonl.
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
from scripts.asism.learned import build_policy_selected_manifest, class_aware_context_vector, read_jsonl  # noqa: E402
from scripts.asism.models import AdaptiveThresholdNetwork  # noqa: E402
from scripts.utils.artifact_contracts import stage2_paths, stage3_paths  # noqa: E402
from scripts.utils.classifier import (  # noqa: E402
    TrainingBudget, evaluate_classifier, records_from_split, records_from_synthetic_manifest, train_classifier,
)
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import hash_dict, read_json  # noqa: E402
from scripts.utils.splits import load_split  # noqa: E402


def compute_budget_estimate(cfg, n_variants: int, n_seeds: int) -> dict:
    budget = cfg.learned_asism.threshold_network.full_policy_verification.compute_budget
    hours_each = float(budget.hours_per_proxy_run_estimate)
    total_runs = 1 + n_variants * n_seeds  # 1 shared real-only baseline + variant*seed runs
    estimated_hours = total_runs * hours_each
    max_hours = float(budget.max_gpu_hours)
    return {
        "n_variants": n_variants,
        "n_seeds": n_seeds,
        "total_runs": total_runs,
        "hours_per_proxy_run_estimate": hours_each,
        "estimated_gpu_hours": round(estimated_hours, 3),
        "max_gpu_hours": max_hours,
        "within_budget": bool(estimated_hours <= max_hours),
    }


def enforce_compute_budget(cfg, n_variants: int, n_seeds: int) -> dict:
    estimate = compute_budget_estimate(cfg, n_variants, n_seeds)
    if not estimate["within_budget"]:
        raise SystemExit(
            "COMPUTE BUDGET GATE: full-policy verification exceeds its declared, INDEPENDENT budget "
            "(learned_asism.threshold_network.full_policy_verification.compute_budget).\n"
            f"  estimated: {estimate['estimated_gpu_hours']}h ({estimate['total_runs']} runs), "
            f"budget: {estimate['max_gpu_hours']}h\n"
            "Reduce full_policy_verification.n_variants/n_seeds and re-run "
            "08_train_threshold_network.py to regenerate the plan."
        )
    return estimate


def percentile_threshold_per_class(percentile: float, ranking_scores: dict, intended_by_id: dict,
                                   image_ids: list, primary_labels: list) -> dict:
    """literal_top_50_percent (percentile=50.0): the per-class score value at the given percentile
    among that class's own candidates — a real, computed threshold, not a fixed constant."""
    result = {}
    for label in primary_labels:
        class_scores = [ranking_scores[image_id] for image_id in image_ids
                        if int(intended_by_id.get(image_id, {}).get(label, 0)) == 1]
        result[label] = float(np.percentile(class_scores, percentile)) if class_scores else None
    return result


def real_prevalence_contexts_from_threshold_contexts(contexts: list[dict], primary_labels: list[str]) -> dict:
    """Recover and validate the exact frozen prevalence inputs embedded by 07."""
    result = {}
    for label in primary_labels:
        values = [row.get("real_prevalence_context") for row in contexts if row.get("label") == label]
        if not values or any(value is None for value in values):
            raise ValueError(f"missing frozen real_prevalence_context for {label}; rebuild contexts with 07")
        canonical = json.dumps(values[0], sort_keys=True)
        if any(json.dumps(value, sort_keys=True) != canonical for value in values[1:]):
            raise ValueError(f"inconsistent frozen real_prevalence_context for {label}")
        result[label] = values[0]
    return result


def validate_policy_selection(policy_name: str, thresholds: dict, selected_ids: list[str], primary_labels: list[str]) -> None:
    """Reject unresolved or empty synthetic policies before any proxy training starts."""
    missing = [label for label in primary_labels if thresholds.get(label) is None]
    if missing:
        raise ValueError(f"policy {policy_name} has unresolved thresholds for: {missing}")
    if not selected_ids:
        raise ValueError(f"policy {policy_name} selected zero synthetic images")


def predict_network_thresholds_per_class(manifest: dict, checkpoint_path: Path, ranking_scores: dict,
                                          intended_by_id: dict, image_ids: list, primary_labels: list,
                                          real_prevalence_by_label: dict) -> dict:
    """adaptive_threshold_network: inference from the OFFICIAL checkpoint on a deployment-time
    context vector (built from the CURRENT full candidate pool for that class, not a training-time
    bootstrap resample). Only classes governance actually assigned this method get a prediction;
    every other class is None here regardless of what the network might output for it — this
    function must never be used to smuggle a network opinion into a class it wasn't approved for."""
    result = {label: None for label in primary_labels}
    if not checkpoint_path.is_file():
        return result
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = AdaptiveThresholdNetwork(
        checkpoint["number_of_classes"], checkpoint["context_dim"], checkpoint["embedding_dim"],
        tuple(checkpoint["hidden_dims"]), checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    per_class_official_method = manifest["per_class_official_method"]
    for class_id, label in enumerate(primary_labels):
        if per_class_official_method.get(label) != "adaptive_threshold_network":
            continue
        class_scores = [ranking_scores[image_id] for image_id in image_ids
                        if int(intended_by_id.get(image_id, {}).get(label, 0)) == 1]
        if not class_scores:
            continue
        context = class_aware_context_vector(
            class_scores, real_prevalence_by_label[label],
            {"target_synthetic_to_real_ratio": 1.0},
        )
        with torch.no_grad():
            prediction = model(torch.tensor([class_id]), torch.tensor([context], dtype=torch.float32))
        result[label] = float(prediction.item())
    return result


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
    for key, value in stage2_paths(stage2, namespace).items():
        if key in stage2.paths: stage2.paths[key] = str(value)

    plan_path = Path(cfg.paths.full_policy_verification_plan)
    if not plan_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: no full-policy verification plan at {plan_path}.\n"
            "Run: python scripts/asism/08_train_threshold_network.py"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    n_variants, n_seeds = len(plan["policy_variants"]), int(plan["n_seeds"])

    if args.phase == "estimate":
        estimate = compute_budget_estimate(cfg, n_variants, n_seeds)
        print(json.dumps(estimate, indent=2))
        return 0 if estimate["within_budget"] else 2

    enforce_compute_budget(cfg, n_variants, n_seeds)

    manifest = read_json(Path(cfg.paths.adaptive_threshold_manifest))
    context_builder_path = Path(__file__).with_name("07_build_threshold_contexts.py")
    spec = importlib.util.spec_from_file_location("build_threshold_contexts_07", context_builder_path)
    context_builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_builder)
    context_builder.preflight_check_phase1_outputs(cfg)
    merged, intended, _ = context_builder.load_candidate_pool(cfg)
    columns = list(read_json(Path(cfg.paths.learned_dir) / "learned_training_manifest.json")["feature_columns"])
    _, ranker, training_manifest = context_builder.load_frozen_models(cfg, columns)
    normalized = context_builder.apply_feature_frame(merged, columns, training_manifest["normalization"])
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32)))
    ranking_scores: dict[str, float] = dict(
        zip(merged.image_id.astype(str), (1.0 / (1.0 + np.exp(-raw.numpy())))))
    image_ids = list(ranking_scores)

    stage1 = load_stage1_config()
    image_root = Path(stage1.paths.images_dir) / namespace
    real = records_from_split(
        load_split("classifier_train", namespace, purpose="schema_validation", caller="full_policy_verification"),
        image_root / "classifier_train",
    )
    evaluation_records = records_from_split(
        load_split("asism_tuning_heldout", namespace, purpose="schema_validation", caller="full_policy_verification"),
        image_root / "asism_tuning_heldout",
    )
    checkpoint_dir = Path(cfg.paths.learned_dir) / "full_policy_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_and_score(train_records, checkpoint, tag, seed):
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
        return float(evaluate_classifier(model, evaluation_records, int(cfg.tuning.proxy.resolution))["macro_auroc"])

    results_path = Path(cfg.paths.full_policy_verification_results)
    completed = {(r["policy"], r["seed"]) for r in read_jsonl(results_path)} if results_path.is_file() else set()
    stage2_manifest, stage2_images = Path(stage2.paths.manifest_path), Path(stage2.paths.images_dir)
    primary_labels = list(PRIMARY_ENDPOINT_LABELS)
    threshold_context_rows = read_jsonl(Path(cfg.paths.threshold_contexts))
    real_prevalence_by_label = real_prevalence_contexts_from_threshold_contexts(
        threshold_context_rows, primary_labels,
    )

    # Resolved ONCE — thresholds don't vary by seed. The OLD baseline uses its own frozen selection
    # artifact directly (06's learned_selected_manifest.jsonl), not a per-class threshold combined
    # here, because its selection logic is not the min(applicable_thresholds) rule this script uses
    # for the other variants.
    selected_ids_by_policy: dict[str, list[str]] = {}
    baseline_selection_manifest_path = Path(cfg.paths.learned_dir) / "learned_selection_manifest.json"
    if not baseline_selection_manifest_path.is_file():
        raise SystemExit("FULL-POLICY GATE: fixed-ratio selection manifest is missing; run 06 first")
    baseline_thresholds = dict(read_json(baseline_selection_manifest_path)["thresholds"])
    for policy_name in plan["policy_variants"]:
        if policy_name == "hard_proxy_best_among_verified":
            thresholds = dict(baseline_thresholds)
            thresholds.update({k: v for k, v in manifest["hard_proxy_best_thresholds"].items() if v is not None})
        elif policy_name == "literal_top_50_percent":
            thresholds = percentile_threshold_per_class(50.0, ranking_scores, intended, image_ids, primary_labels)
        elif policy_name == "adaptive_threshold_network":
            network_thresholds = predict_network_thresholds_per_class(
                manifest, Path(cfg.paths.official_threshold_checkpoint), ranking_scores, intended,
                image_ids, primary_labels, real_prevalence_by_label,
            )
            thresholds = dict(baseline_thresholds)
            thresholds.update({k: v for k, v in manifest["hard_proxy_best_thresholds"].items() if v is not None})
            thresholds.update({k: v for k, v in network_thresholds.items() if v is not None})
        elif policy_name == "fixed_target_ratio_threshold_distillation_baseline_v1":
            baseline_manifest_path = Path(cfg.paths.learned_selected_manifest)
            selected_ids_by_policy[policy_name] = (
                sorted(row["image_id"] for row in read_jsonl(baseline_manifest_path))
                if baseline_manifest_path.is_file() else []
            )
            continue
        else:
            thresholds = {label: None for label in primary_labels}
        selected_ids_by_policy[policy_name] = build_policy_selected_manifest(
            thresholds, ranking_scores, intended, image_ids, primary_labels,
        )
        validate_policy_selection(policy_name, thresholds, selected_ids_by_policy[policy_name], primary_labels)

    if not selected_ids_by_policy.get("fixed_target_ratio_threshold_distillation_baseline_v1", []):
        raise SystemExit("FULL-POLICY GATE: fixed-ratio baseline selection is missing or empty; run 06 first")

    with open(results_path, "a", encoding="utf-8") as handle:
        for seed_offset in range(n_seeds):
            seed = int(cfg.learned_asism.subset_design.seed) + seed_offset
            baseline_checkpoint = checkpoint_dir / f"real_only_seed{seed}.pt"
            baseline_score = train_and_score(real, baseline_checkpoint, f"full-policy-real-only-s{seed}", seed)

            for policy_name in plan["policy_variants"]:
                if (policy_name, seed) in completed:
                    continue
                selected_ids = selected_ids_by_policy[policy_name]
                synthetic = records_from_synthetic_manifest(stage2_manifest, stage2_images, selected_ids)
                tag = f"full-policy-{policy_name}-s{seed}"
                checkpoint = checkpoint_dir / f"{hash_dict({'tag': tag}, length=32)}.pt"
                measured = train_and_score(real + synthetic, checkpoint, tag, seed)
                row = {
                    "policy": policy_name, "seed": seed, "n_selected": len(selected_ids),
                    "real_only_macro_auroc": baseline_score, "augmented_macro_auroc": measured,
                    "measured_utility": measured - baseline_score,
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(f"[{policy_name} seed={seed}] measured_utility={row['measured_utility']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

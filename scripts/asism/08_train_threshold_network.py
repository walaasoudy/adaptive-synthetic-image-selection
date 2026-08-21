#!/usr/bin/env python3
"""Train AdaptiveThresholdNetwork(s) for learned ASISM (Phase 2) — two SEPARATE, never-merged paths.

verified_only_official: trained ONLY if at least one class is ELIGIBLE — meaning it independently
    clears min_verified_contexts_per_class on BOTH verified TRAIN contexts and verified,
    image-disjoint HELD-OUT contexts (eligible_classes_for_official_training). A class with zero
    verified TRAIN contexts never gets an official threshold from this path or from
    hard_proxy_best_among_verified — held-out measurements are NEVER used to decide or construct a
    threshold, only to evaluate one already built from train data.
critic_assisted_exploratory: a SEPARATE model trained on critic-only targets, always run (contexts
    always exist), written to its own checkpoint and its own target file. Never merged into, or
    substituted for, the official path's output.

Given this pass has 07b never executed (no GPU training), the expected, CORRECT outcome is that
every class falls back to fixed_target_ratio_threshold_distillation_baseline_v1: zero proxy-verified
targets exist anywhere, so official_trained is False and eligible_classes is empty. That is not a
bug in this script — it is what "no production evidence yet" is supposed to look like.

Full-policy verification (after per-class thresholds are combined across all 11 diseases) is
PLANNED here (plan_full_policy_verification) but never executed — see
08b_verify_full_policy_proxy.py, which has its own independent compute budget and the same
fail-closed --i-understand-this-trains-real-models guard as 07b.
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
from scripts.asism.learned import (  # noqa: E402
    aggregate_hard_proxy_best_threshold, compute_verified_context_counts, critic_proxy_correlation_per_class,
    determine_per_class_official_method, eligible_classes_for_official_training, enforce_acceptance_criteria,
    filter_targets_to_eligible_classes,
    freeze_acceptance_criteria, held_out_generalization_metrics, read_jsonl,
    resolve_critic_assisted_exploratory_targets, resolve_verified_only_targets,
)
from scripts.asism.models import AdaptiveThresholdNetwork  # noqa: E402
from scripts.utils.artifact_contracts import stage3_paths  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import read_json, sha256_file, write_frozen_json  # noqa: E402


def train_threshold_network(model, targets: list[dict], device, epochs: int, learning_rate: float):
    """Full-batch weighted-MSE distillation. weight_decay=0.0 explicitly: AdamW's default (0.01)
    would move parameters every step even with a zero-weight/zero-gradient loss."""
    if not targets:
        raise ValueError("train_threshold_network: no training targets")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    class_ids = torch.tensor([t["class_id"] for t in targets], dtype=torch.long, device=device)
    context = torch.tensor([t["context_features"] for t in targets], dtype=torch.float32, device=device)
    target_values = torch.tensor([t["target_threshold"] for t in targets], dtype=torch.float32, device=device)
    weights = torch.tensor([t["supervision_weight"] for t in targets], dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        prediction = model(class_ids, context)
        loss = (weights * (prediction - target_values) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


@torch.no_grad()
def predict_thresholds(model, targets: list[dict], device) -> list[float]:
    if not targets:
        return []
    model.eval()
    class_ids = torch.tensor([t["class_id"] for t in targets], dtype=torch.long, device=device)
    context = torch.tensor([t["context_features"] for t in targets], dtype=torch.float32, device=device)
    return model(class_ids, context).cpu().numpy().tolist()


def critic_predicted_utility_regret(critic_utility_fn, ranking_scores: dict, context: dict,
                                    grid_best_utility: float, predicted_threshold: float,
                                    max_selected: int) -> float:
    candidates = [i for i in context["image_ids"] if ranking_scores.get(i, 0.0) >= predicted_threshold]
    if not candidates:
        return float("inf")
    if len(candidates) > max_selected:
        candidates = sorted(candidates, key=lambda i: -ranking_scores.get(i, 0.0))[:max_selected]
    return grid_best_utility - float(critic_utility_fn(candidates))


def threshold_stability(targets: list[dict], predictions: list[float]) -> dict:
    by_class: dict[int, list[float]] = {}
    for target, prediction in zip(targets, predictions):
        by_class.setdefault(target["class_id"], []).append(prediction)
    stds = {class_id: float(np.std(values)) for class_id, values in by_class.items() if len(values) >= 2}
    return {"per_class_std": stds, "median_std": float(np.median(list(stds.values()))) if stds else None}


def evaluate_held_out(model, held_out_targets, contexts_by_id, ranking_scores, evaluations_by_context,
                      critic_utility_fn, max_selected, device):
    predictions = predict_thresholds(model, held_out_targets, device)
    results = []
    for target, predicted in zip(held_out_targets, predictions):
        context = contexts_by_id[target["context_id"]]
        scores = {image_id: ranking_scores.get(image_id, 0.0) for image_id in context["image_ids"]}
        metrics = held_out_generalization_metrics(float(predicted), target["target_threshold"], scores)
        grid_best = min(evaluations_by_context.get(target["context_id"], [{"critic_predicted_utility": 0.0}]),
                        key=lambda row: row.get("rank_within_context", 0))
        metrics["critic_predicted_utility_regret"] = critic_predicted_utility_regret(
            critic_utility_fn, ranking_scores, context, grid_best["critic_predicted_utility"],
            float(predicted), max_selected,
        )
        results.append({**metrics, "context_id": target["context_id"], "label": target["label"]})
    return results, predictions


def compute_official_acceptance_evidence(held_out_generalization: list[dict], correlation: dict, stability: dict) -> dict:
    """Acceptance evidence using explicitly critic-predicted (not proxy-measured) utility regret
    by the network's own threshold vs the grid optimum), NEVER from absolute_threshold_error (a
    probability-space distance between two thresholds, not a utility measure) — conflating the two
    was a real bug in an earlier version of this script."""
    finite_regrets = [abs(entry["critic_predicted_utility_regret"]) for entry in held_out_generalization
                      if np.isfinite(entry["critic_predicted_utility_regret"])]
    return {
        "median_spearman": correlation["median_spearman"],
        "median_critic_predicted_utility_regret": float(np.median(finite_regrets)) if finite_regrets else None,
        "threshold_stability_std": stability["median_std"],
    }


def save_checkpoint(model, path: Path, path_label: str, tn_cfg, context_dim: int) -> str:
    torch.save({
        "state_dict": model.state_dict(),
        "schema_version": 1,
        "path": path_label,
        "number_of_classes": len(PRIMARY_ENDPOINT_LABELS),
        "context_dim": context_dim,
        "embedding_dim": int(tn_cfg.class_embedding_dim),
        "hidden_dims": list(tn_cfg.hidden_dims),
        "dropout": float(tn_cfg.dropout),
    }, path)
    return sha256_file(path)


def plan_full_policy_verification(per_class_official_method: dict[str, str], full_policy_cfg) -> dict:
    """PLAN ONLY — no execution. Which full-policy variants would need proxy verification once
    thresholds are combined across all 11 diseases, and the independent GPU-hour cost of doing so.
    Actual execution is a separate, fail-closed step: 08b_verify_full_policy_proxy.py --phase run."""
    variants = sorted({"fixed_target_ratio_threshold_distillation_baseline_v1", "literal_top_50_percent"}
                      | set(per_class_official_method.values()))
    n_seeds = int(full_policy_cfg.n_seeds)
    n_runs = len(variants) * n_seeds
    budget = full_policy_cfg.compute_budget
    hours_each = float(budget.hours_per_proxy_run_estimate)
    estimated_hours = n_runs * hours_each
    max_hours = float(budget.max_gpu_hours)
    return {
        "status": "plan_only_not_executed",
        "policy_variants": variants,
        "n_seeds": n_seeds,
        "estimated_runs": n_runs,
        "hours_per_proxy_run_estimate": hours_each,
        "estimated_gpu_hours": round(estimated_hours, 3),
        "max_gpu_hours": max_hours,
        "within_budget": bool(estimated_hours <= max_hours),
        "next_step": (
            "python scripts/asism/08b_verify_full_policy_proxy.py --phase run "
            "--i-understand-this-trains-real-models (only after independent budget sign-off)"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths: cfg.paths[key] = str(value)

    contexts_path = Path(cfg.paths.threshold_contexts)
    evaluations_path = Path(cfg.paths.threshold_candidate_evaluations)
    if not contexts_path.is_file() or not evaluations_path.is_file():
        raise SystemExit(
            "UPSTREAM GATE: threshold contexts/evaluations missing.\n"
            "Run: python scripts/asism/07_build_threshold_contexts.py"
        )
    contexts = read_jsonl(contexts_path)
    evaluations = read_jsonl(evaluations_path)
    measurements_path = Path(cfg.paths.threshold_proxy_measurements)
    measurements = read_jsonl(measurements_path) if measurements_path.is_file() else []
    tn_cfg = cfg.learned_asism.threshold_network
    labels = list(PRIMARY_ENDPOINT_LABELS)

    context_builder_path = Path(__file__).with_name("07_build_threshold_contexts.py")
    spec = importlib.util.spec_from_file_location("build_threshold_contexts_07", context_builder_path)
    context_builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(context_builder)
    context_builder.preflight_check_phase1_outputs(cfg)
    merged, _, _ = context_builder.load_candidate_pool(cfg)
    columns = list(read_json(Path(cfg.paths.learned_dir) / "learned_training_manifest.json")["feature_columns"])
    critic, ranker, training_manifest = context_builder.load_frozen_models(cfg, columns)
    normalized = context_builder.apply_feature_frame(merged, columns, training_manifest["normalization"])
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32)))
    ranking_scores: dict[str, float] = dict(
        zip(merged.image_id.astype(str), (1.0 / (1.0 + np.exp(-raw.numpy())))))
    feature_lookup = {
        str(image_id): row.to_numpy(np.float32)
        for image_id, (_, row) in zip(merged.image_id, normalized.iterrows())
    }
    critic_utility_fn = context_builder.make_critic_utility_fn(critic, feature_lookup, torch.device("cpu"))

    features_by_context = {c["context_id"]: c["context_features"] for c in contexts}
    contexts_by_id = {c["context_id"]: c for c in contexts}
    evaluations_by_context: dict[str, list[dict]] = {}
    for row in evaluations:
        evaluations_by_context.setdefault(row["context_id"], []).append(row)
    device = torch.device("cpu")
    min_verified = int(tn_cfg.min_verified_contexts_per_class)

    # ---- Path A: verified_only_official ------------------------------------------------------
    verified_targets = resolve_verified_only_targets(contexts, measurements)
    for target in verified_targets:
        target["context_features"] = features_by_context[target["context_id"]]
    verified_train = [t for t in verified_targets if t["context_source"] == "bootstrap_train_pool"]
    verified_held_out = [t for t in verified_targets if t["context_source"] == "bootstrap_val_pool"]

    train_counts, held_out_counts = compute_verified_context_counts(verified_train, verified_held_out, labels)
    eligible_classes = eligible_classes_for_official_training(train_counts, held_out_counts, min_verified)
    official_train = filter_targets_to_eligible_classes(verified_train, eligible_classes)
    official_held_out = filter_targets_to_eligible_classes(verified_held_out, eligible_classes)
    # Training requires (a) at least one class independently clearing the minimum on BOTH train and
    # held-out sides, and (b) actual verified train data to fit on. Neither alone is enough.
    official_trained = bool(eligible_classes) and bool(official_train)

    official_held_out_generalization: list[dict] = []
    official_stability: dict = {"per_class_std": {}, "median_std": None}
    official_correlation = critic_proxy_correlation_per_class(evaluations, measurements, contexts)
    acceptance_gate: dict = {"passed": False, "criteria_status": "not_evaluated_no_eligible_class"}
    official_checkpoint_hash = None

    if official_trained:
        official_model = AdaptiveThresholdNetwork(
            len(labels), len(official_train[0]["context_features"]),
            int(tn_cfg.class_embedding_dim), tuple(tn_cfg.hidden_dims), float(tn_cfg.dropout),
        ).to(device)
        train_threshold_network(official_model, official_train, device, int(tn_cfg.epochs), float(tn_cfg.learning_rate))
        official_held_out_generalization, official_predictions = evaluate_held_out(
            official_model, official_held_out, contexts_by_id, ranking_scores, evaluations_by_context,
            critic_utility_fn, int(tn_cfg.max_selected_per_label), device,
        )
        official_stability = threshold_stability(official_held_out, official_predictions) if official_held_out else official_stability
        evidence = compute_official_acceptance_evidence(official_held_out_generalization, official_correlation, official_stability)
        frozen_criteria = freeze_acceptance_criteria(dict(tn_cfg.acceptance_criteria))
        acceptance_gate = enforce_acceptance_criteria(evidence, frozen_criteria)

        official_checkpoint_hash = save_checkpoint(
            official_model, Path(cfg.paths.official_threshold_checkpoint), "verified_only_official",
            tn_cfg, len(official_train[0]["context_features"]),
        )

    # ---- Path B: critic_assisted_exploratory (always runs, never official) ------------------
    exploratory_targets = resolve_critic_assisted_exploratory_targets(contexts, evaluations)
    for target in exploratory_targets:
        target["context_features"] = features_by_context[target["context_id"]]
    exploratory_train = [t for t in exploratory_targets if t["context_source"] == "bootstrap_train_pool"]
    exploratory_held_out = [t for t in exploratory_targets if t["context_source"] == "bootstrap_val_pool"]
    exploratory_held_out_generalization: list[dict] = []
    exploratory_checkpoint_hash = None
    if exploratory_train:
        exploratory_model = AdaptiveThresholdNetwork(
            len(labels), len(exploratory_train[0]["context_features"]),
            int(tn_cfg.class_embedding_dim), tuple(tn_cfg.hidden_dims), float(tn_cfg.dropout),
        ).to(device)
        train_threshold_network(exploratory_model, exploratory_train, device, int(tn_cfg.epochs), float(tn_cfg.learning_rate))
        exploratory_held_out_generalization, _ = evaluate_held_out(
            exploratory_model, exploratory_held_out, contexts_by_id, ranking_scores, evaluations_by_context,
            critic_utility_fn, int(tn_cfg.max_selected_per_label), device,
        )
        exploratory_checkpoint_hash = save_checkpoint(
            exploratory_model, Path(cfg.paths.exploratory_threshold_checkpoint), "critic_assisted_exploratory",
            tn_cfg, len(exploratory_train[0]["context_features"]),
        )

    # ---- Governance + hard-proxy-best aggregation (TRAIN contexts only, never held-out) -------
    per_class_official_method = determine_per_class_official_method(
        verified_train, verified_held_out, labels, min_verified, bool(acceptance_gate["passed"]),
    )
    hard_proxy_best_thresholds = {
        label: aggregate_hard_proxy_best_threshold(verified_train, label)
        for label, method in per_class_official_method.items()
        if method == "hard_proxy_best_among_verified"
    }
    full_policy_plan = plan_full_policy_verification(per_class_official_method, tn_cfg.full_policy_verification)

    # ---- Write SEPARATE verified/exploratory target files (never one shared file) -------------
    output_dir = Path(cfg.paths.learned_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.paths.threshold_training_targets_verified), "x", encoding="utf-8") as handle:
        for target in verified_targets:
            handle.write(json.dumps(target, sort_keys=True) + "\n")
    with open(Path(cfg.paths.threshold_training_targets_exploratory), "x", encoding="utf-8") as handle:
        for target in exploratory_targets:
            handle.write(json.dumps(target, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 3,
        "method": "verified_hard_class_aware_threshold_optimization_with_supervised_distillation",
        "frozen": True,
        "official_path": {
            "trained": official_trained,
            "eligible_classes": sorted(eligible_classes),
            "train_counts": train_counts,
            "held_out_counts": held_out_counts,
            "n_train_targets": len(verified_train),
            "n_held_out_targets": len(verified_held_out),
            "n_eligible_train_targets": len(official_train),
            "n_eligible_held_out_targets": len(official_held_out),
            "checkpoint_path": str(cfg.paths.official_threshold_checkpoint) if official_trained else None,
            "checkpoint_sha256": official_checkpoint_hash,
            "held_out_generalization": official_held_out_generalization,
            "threshold_stability": official_stability,
            "critic_proxy_correlation": official_correlation,
            "acceptance_gate": acceptance_gate,
        },
        "exploratory_path": {
            "trained": bool(exploratory_train),
            "note": "critic_assisted_exploratory — NEVER used as the official selector, ablation only",
            "n_train_targets": len(exploratory_train),
            "n_held_out_targets": len(exploratory_held_out),
            "checkpoint_path": str(cfg.paths.exploratory_threshold_checkpoint) if exploratory_train else None,
            "checkpoint_sha256": exploratory_checkpoint_hash,
            "held_out_generalization": exploratory_held_out_generalization,
        },
        "per_class_official_method": per_class_official_method,
        "hard_proxy_best_thresholds": hard_proxy_best_thresholds,
        "any_class_uses_network_officially": any(
            v == "adaptive_threshold_network" for v in per_class_official_method.values()
        ),
        "full_policy_verification_plan": full_policy_plan,
        "targets_verified_sha256": sha256_file(Path(cfg.paths.threshold_training_targets_verified)),
        "targets_exploratory_sha256": sha256_file(Path(cfg.paths.threshold_training_targets_exploratory)),
    }
    write_frozen_json(Path(cfg.paths.adaptive_threshold_manifest), manifest)
    with open(Path(cfg.paths.full_policy_verification_plan), "x", encoding="utf-8") as handle:
        json.dump(full_policy_plan, handle, indent=2, sort_keys=True)

    print(f"Official path trained: {official_trained} (eligible classes: {sorted(eligible_classes)})")
    print(f"Per-class official method: {per_class_official_method}")
    print(f"hard_proxy_best_thresholds: {hard_proxy_best_thresholds}")
    print(f"Full-policy verification plan (NOT executed): {full_policy_plan['estimated_runs']} runs, "
          f"{full_policy_plan['estimated_gpu_hours']}h, within_budget={full_policy_plan['within_budget']}")
    print(f"-> {cfg.paths.adaptive_threshold_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build threshold contexts + hard-grid-search candidate thresholds for learned ASISM (Phase 2).

Uses the FROZEN ranker + critic from 05_train_learned_asism.py as fixed, read-only models: this
script never trains anything. For each of the 11 primary labels, builds bootstrap resample contexts
from the train-role and val-role image pools SEPARATELY (same disjoint split as Phase 1b), runs a
hard threshold grid search scored by the critic, and writes:

  threshold_contexts.jsonl              — one row per context (see bootstrap_class_contexts)
  threshold_candidate_evaluations.jsonl — one row per threshold tried within a context
  proxy_verification_plan.json          — which top-K candidates WOULD be proxy-verified, and the
                                          GPU-hour cost of doing so. Nothing here is executed;
                                          07b_verify_thresholds_proxy.py --phase run does that,
                                          separately, and refuses to run without an explicit flag.

critic_predicted_utility is exactly what it says: the frozen critic's prediction, not a measured
value. Nothing here is presented as ground truth.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import (  # noqa: E402
    apply_feature_frame, bootstrap_class_contexts, diversify_verification_candidates,
    hard_threshold_grid_search, real_class_support_context, split_image_pool,
)
from scripts.asism.models import MultiSignalUtilityRankingNetwork, SetUtilityNetwork  # noqa: E402
from scripts.utils.artifact_contracts import stage3_paths  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, sha256_file  # noqa: E402


def preflight_check_phase1_outputs(cfg) -> None:
    """Phase 2 depends on Phase 1's frozen ranker/critic. Fail with a clear message, not a raw
    FileNotFoundError, if they don't exist yet."""
    learned_dir = Path(cfg.paths.learned_dir)
    required = ["set_utility_model.pt", "ranking_model.pt", "learned_training_manifest.json"]
    missing = [name for name in required if not (learned_dir / name).is_file()]
    if missing:
        raise SystemExit(
            "UPSTREAM GATE: threshold-context building needs Phase 1's trained models, missing:\n"
            + "\n".join(f"  - {learned_dir / name}" for name in missing)
            + "\n\n-> python scripts/asism/05_train_learned_asism.py"
        )


def load_frozen_models(cfg, columns: list[str]):
    learned_dir = Path(cfg.paths.learned_dir)
    training_manifest = read_json(learned_dir / "learned_training_manifest.json")

    set_cfg = cfg.learned_asism.set_utility_network
    critic_checkpoint = torch.load(learned_dir / "set_utility_model.pt", map_location="cpu", weights_only=True)
    critic = SetUtilityNetwork(
        len(columns), tuple(set_cfg.image_hidden_dims), tuple(set_cfg.utility_hidden_dims),
        superset_conditioning=bool(critic_checkpoint.get("superset_conditioning", False)),
    )
    critic.load_state_dict(critic_checkpoint["state_dict"])
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    rank_cfg = cfg.learned_asism.ranking_network
    ranker = MultiSignalUtilityRankingNetwork(len(columns), tuple(rank_cfg.hidden_dims), float(rank_cfg.dropout))
    ranker_checkpoint = torch.load(learned_dir / "ranking_model.pt", map_location="cpu", weights_only=True)
    ranker.load_state_dict(ranker_checkpoint["state_dict"])
    ranker.eval()

    return critic, ranker, training_manifest


def make_critic_utility_fn(critic, feature_lookup: dict, device):
    @torch.no_grad()
    def critic_utility_fn(image_ids: list[str]) -> float:
        vectors = [feature_lookup[image_id] for image_id in image_ids if image_id in feature_lookup]
        if not vectors:
            return float("-inf")
        features = torch.tensor(np.asarray(vectors), dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.ones(1, len(vectors), dtype=torch.bool, device=device)
        return float(critic(features, mask).item())
    return critic_utility_fn


def verification_compute_estimate(cfg, n_planned: int) -> dict:
    """Per-context threshold verification ONLY. Full-policy verification (after per-class thresholds
    are combined across all 11 diseases) has its OWN independent budget and its own fail-closed
    execution gate — see 08_train_threshold_network.py's plan_full_policy_verification and
    08b_verify_full_policy_proxy.py. The two are never summed into one number: they are separately
    approved, separately budgeted production steps."""
    budget = cfg.learned_asism.threshold_network.verification_compute_budget
    hours_each = float(budget.hours_per_proxy_run_estimate)
    estimated_hours = n_planned * hours_each
    max_hours = float(budget.max_gpu_hours)
    return {
        "n_planned_threshold_evaluations": n_planned,
        "hours_per_proxy_run_estimate": hours_each,
        "estimated_gpu_hours": round(estimated_hours, 3),
        "max_gpu_hours": max_hours,
        "within_budget": bool(estimated_hours <= max_hours),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths:
            cfg.paths[key] = str(value)
    cfg.split_namespace = namespace

    preflight_check_phase1_outputs(cfg)

    baseline_path = Path(__file__).with_name("04_build_utility_subsets.py")
    spec = importlib.util.spec_from_file_location("build_utility_subsets_04", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    baseline.preflight_check_candidate_pool_inputs(cfg)
    merged, intended, _ = baseline.load_candidate_pool(cfg)

    tn_cfg = cfg.learned_asism.threshold_network
    columns = list(read_json(Path(cfg.paths.learned_dir) / "learned_training_manifest.json")["feature_columns"])
    critic, ranker, training_manifest = load_frozen_models(cfg, columns)
    ranking_checkpoint_hash = sha256_file(Path(cfg.paths.learned_dir) / "ranking_model.pt")
    critic_checkpoint_hash = sha256_file(Path(cfg.paths.learned_dir) / "set_utility_model.pt")

    normalized = apply_feature_frame(merged, columns, training_manifest["normalization"])
    device = torch.device("cpu")
    feature_lookup = {
        str(image_id): row.to_numpy(np.float32)
        for image_id, (_, row) in zip(merged.image_id, normalized.iterrows())
    }
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32)))
    merged = merged.copy()
    merged["learned_ranking_score"] = (1.0 / (1.0 + np.exp(-raw.numpy())))
    ranking_scores = dict(zip(merged.image_id.astype(str), merged["learned_ranking_score"]))
    critic_utility_fn = make_critic_utility_fn(critic, feature_lookup, device)

    val_fraction = float(cfg.learned_asism.subset_design.val_pool_fraction)
    seed = int(cfg.learned_asism.subset_design.seed)
    train_pool, val_pool = split_image_pool(merged, val_fraction, seed)

    budget_context = {
        "target_synthetic_to_real_ratio": float(tn_cfg.target_synthetic_to_real_ratio),
        "min_selected_per_label": int(tn_cfg.min_selected_per_label),
        "max_selected_per_label": int(tn_cfg.max_selected_per_label),
    }
    t_grid = np.arange(0.0, 1.0 + 1e-9, float(tn_cfg.t_grid_step))
    top_k = int(tn_cfg.top_k_per_context)
    quantile_fractions = [float(fraction) for fraction in tn_cfg.verification_quantile_fractions]

    all_contexts: list[dict] = []
    all_evaluations: list[dict] = []
    planned_evaluations: list[dict] = []

    for class_id, label in enumerate(PRIMARY_ENDPOINT_LABELS):
        real_prevalence = real_class_support_context(namespace, [label])["labels"][label]
        for pool, source in ((train_pool, "bootstrap_train_pool"), (val_pool, "bootstrap_val_pool")):
            contexts = bootstrap_class_contexts(
                pool, class_id, label, intended, "learned_ranking_score", source,
                int(tn_cfg.n_bootstrap_contexts_per_class), float(tn_cfg.bootstrap_min_fraction),
                float(tn_cfg.bootstrap_max_fraction), ranking_checkpoint_hash, critic_checkpoint_hash,
                real_prevalence, budget_context, seed + class_id,
            )
            all_contexts.extend(contexts)
            for context in contexts:
                try:
                    evaluations = hard_threshold_grid_search(
                        critic_utility_fn, ranking_scores, context["image_ids"], t_grid,
                        float(tn_cfg.budget_weight), int(tn_cfg.max_selected_per_label),
                        min_selected=int(tn_cfg.min_selected_per_label),
                    )
                except ValueError:
                    continue  # every threshold produced an empty subset for this context; skip it
                for evaluation in evaluations:
                    evaluation["context_id"] = context["context_id"]

                diversified = diversify_verification_candidates(
                    evaluations, top_k, quantile_fractions, len(context["image_ids"]),
                )
                planned_by_threshold = {entry["candidate_threshold"]: entry for entry in diversified}

                for evaluation in evaluations:
                    planned_entry = planned_by_threshold.get(evaluation["candidate_threshold"])
                    evaluation["proxy_verification_status"] = "planned" if planned_entry else "not_planned"
                    evaluation["selection_reasons"] = planned_entry["selection_reasons"] if planned_entry else []
                    all_evaluations.append(evaluation)
                    if planned_entry:
                        # Same selection rule as hard_threshold_grid_search's own truncation, so
                        # 07b can deterministically re-derive the identical hard subset later.
                        candidates = [image_id for image_id in context["image_ids"]
                                     if ranking_scores.get(image_id, 0.0) >= evaluation["candidate_threshold"]]
                        subset_image_ids = sorted(candidates, key=lambda i: -ranking_scores.get(i, 0.0))[
                            :evaluation["selected_count"]]
                        planned_evaluations.append({
                            "context_id": context["context_id"],
                            "candidate_threshold": evaluation["candidate_threshold"],
                            "selection_reasons": evaluation["selection_reasons"],
                            "hard_subset_image_ids_hash": hash_dict({"image_ids": sorted(subset_image_ids)}, length=32),
                        })

    output_dir = Path(cfg.paths.learned_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive-create ("x"): these are FROZEN, critic-only artifacts. 07b measures proxy results
    # into a completely separate file (threshold_proxy_measurements.jsonl) and must never be able to
    # append to or rewrite this one.
    with open(Path(cfg.paths.threshold_contexts), "x", encoding="utf-8") as handle:
        for context in all_contexts:
            handle.write(json.dumps(context, sort_keys=True) + "\n")
    with open(Path(cfg.paths.threshold_candidate_evaluations), "x", encoding="utf-8") as handle:
        for evaluation in all_evaluations:
            handle.write(json.dumps(evaluation, sort_keys=True) + "\n")

    estimate = verification_compute_estimate(cfg, len(planned_evaluations))
    reason_counts: dict[str, int] = {}
    for entry in planned_evaluations:
        for reason in entry["selection_reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    plan = {
        "schema_version": 1,
        "frozen": True,
        "top_k_per_context": top_k,
        "verification_quantile_fractions": quantile_fractions,
        "n_contexts": len(all_contexts),
        "n_planned_by_selection_reason": reason_counts,
        "planned_evaluations": [
            {"context_id": e["context_id"], "candidate_threshold": e["candidate_threshold"],
             "selection_reasons": e["selection_reasons"]}
            for e in planned_evaluations
        ],
        **estimate,
        "git_commit_hash": get_git_commit_hash(),
    }
    with open(Path(cfg.paths.proxy_verification_plan), "x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)

    print(f"{len(all_contexts)} contexts, {len(all_evaluations)} threshold evaluations, "
          f"{len(planned_evaluations)} planned for proxy verification (by reason: {reason_counts})")
    print(f"Threshold-verification budget: {estimate['n_planned_threshold_evaluations']} runs "
          f"({estimate['estimated_gpu_hours']}h, within_budget={estimate['within_budget']}). "
          "Full-policy verification is a separate, independently-budgeted step (see 08's plan).")
    print(f"-> {cfg.paths.threshold_contexts}\n-> {cfg.paths.threshold_candidate_evaluations}\n"
          f"-> {cfg.paths.proxy_verification_plan}")
    print("Nothing was executed. Next (on RunPod, after budget sign-off): "
          "07b_verify_thresholds_proxy.py --phase run --i-understand-this-trains-real-models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

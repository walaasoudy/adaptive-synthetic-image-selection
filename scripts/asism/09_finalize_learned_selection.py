#!/usr/bin/env python3
"""Freeze the winning learned-ASISM policy and publish its Stage-4 selection.

This step never trains a model and never opens final_eval_heldout. It consumes the complete
full-policy proxy measurements made on asism_tuning_heldout, applies the pre-registered tie rule,
and writes a new adaptive manifest. The fixed-ratio output from 06 remains untouched.
"""
from __future__ import annotations

import importlib.util
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import (  # noqa: E402
    build_policy_selected_manifest, choose_full_policy, enforce_per_class_selection_floor,
    freematch_style_percentile_per_class, read_jsonl,
)
from scripts.utils.artifact_contracts import (  # noqa: E402
    current_code_identity_hash, namespace_identity, require_generation_complete, stage3_paths,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import read_json, sha256_file, write_frozen_json  # noqa: E402


def _load_neighbor(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

    required = [
        Path(cfg.paths.full_policy_verification_plan),
        Path(cfg.paths.full_policy_verification_results),
        Path(cfg.paths.adaptive_threshold_manifest),
        Path(cfg.paths.threshold_contexts),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"FINALIZATION GATE: missing upstream artifacts: {missing}")

    plan = read_json(Path(cfg.paths.full_policy_verification_plan))
    results = read_jsonl(Path(cfg.paths.full_policy_verification_results))
    fp_cfg = cfg.learned_asism.threshold_network.full_policy_verification
    seeds = [int(cfg.learned_asism.subset_design.seed) + index for index in range(int(plan["n_seeds"]))]
    decision = choose_full_policy(
        results, list(plan["policy_variants"]), seeds,
        float(fp_cfg.finalization.tie_noise_band), list(fp_cfg.finalization.simplicity_order),
    )

    context_builder = _load_neighbor("07_build_threshold_contexts.py", "threshold_context_builder_07")
    verifier = _load_neighbor("08b_verify_full_policy_proxy.py", "full_policy_verifier_08b")
    context_builder.preflight_check_phase1_outputs(cfg)
    merged, intended, _ = context_builder.load_candidate_pool(cfg)
    # Provenance recorded in the frozen selection manifest below; Stage 4 checks these before it
    # trains on the selection. Resolved before anything is written, so a mismatch leaves no output.
    identity = namespace_identity(namespace)
    _, generation = require_generation_complete(load_named_config("stage2_generation.yaml", "stage2"), namespace)
    columns = list(read_json(Path(cfg.paths.learned_dir) / "learned_training_manifest.json")["feature_columns"])
    _, ranker, training_manifest = context_builder.load_frozen_models(cfg, columns)
    normalized = context_builder.apply_feature_frame(merged, columns, training_manifest["normalization"])
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32)))
    ranking_scores = dict(zip(merged.image_id.astype(str), 1.0 / (1.0 + np.exp(-raw.numpy()))))
    image_ids = list(ranking_scores)
    labels = list(PRIMARY_ENDPOINT_LABELS)
    threshold_manifest = read_json(Path(cfg.paths.adaptive_threshold_manifest))

    baseline_manifest_path = Path(cfg.paths.learned_dir) / "learned_selection_manifest.json"
    baseline_ids_path = Path(cfg.paths.learned_selected_manifest)
    if not baseline_manifest_path.is_file() or not baseline_ids_path.is_file():
        raise SystemExit("FINALIZATION GATE: fixed-ratio baseline artifacts are missing; run 06 first")
    baseline_thresholds = dict(read_json(baseline_manifest_path)["thresholds"])
    winner = decision["winning_policy"]
    thresholds = None
    if winner == "fixed_target_ratio_threshold_distillation_baseline_v1":
        selected_ids = sorted(row["image_id"] for row in read_jsonl(baseline_ids_path))
    else:
        if winner == "literal_top_50_percent":
            thresholds = verifier.percentile_threshold_per_class(50.0, ranking_scores, intended, image_ids, labels)
        elif winner == "freematch_style_adaptive_percentile":
            contexts = read_jsonl(Path(cfg.paths.threshold_contexts))
            prevalence = verifier.real_prevalence_contexts_from_threshold_contexts(contexts, labels)
            freematch_cfg = cfg.learned_asism.threshold_network.freematch_style
            thresholds = freematch_style_percentile_per_class(
                ranking_scores, intended, image_ids, labels, prevalence,
                float(freematch_cfg.base_percentile), float(freematch_cfg.min_percentile),
            )
            thresholds = enforce_per_class_selection_floor(
                thresholds, ranking_scores, intended, image_ids, labels,
                int(cfg.learned_asism.threshold_network.min_selected_per_label),
            )
        elif winner == "hard_proxy_best_among_verified":
            thresholds = dict(baseline_thresholds)
            thresholds.update({k: v for k, v in threshold_manifest["hard_proxy_best_thresholds"].items() if v is not None})
        elif winner == "adaptive_threshold_network":
            contexts = read_jsonl(Path(cfg.paths.threshold_contexts))
            prevalence = verifier.real_prevalence_contexts_from_threshold_contexts(contexts, labels)
            network = verifier.predict_network_thresholds_per_class(
                threshold_manifest, Path(cfg.paths.official_threshold_checkpoint), ranking_scores,
                intended, image_ids, labels, prevalence,
            )
            thresholds = dict(baseline_thresholds)
            thresholds.update({k: v for k, v in threshold_manifest["hard_proxy_best_thresholds"].items() if v is not None})
            thresholds.update({k: v for k, v in network.items() if v is not None})
        else:
            raise SystemExit(f"FINALIZATION GATE: unsupported winning policy {winner}")
        selected_ids = build_policy_selected_manifest(thresholds, ranking_scores, intended, image_ids, labels)
        verifier.validate_policy_selection(winner, thresholds, selected_ids, labels)

    if not selected_ids:
        raise SystemExit(f"FINALIZATION GATE: winning policy {winner} selected zero images")
    with open(Path(cfg.paths.adaptive_selected_manifest), "x", encoding="utf-8") as handle:
        for image_id in selected_ids:
            handle.write(json.dumps({
                "image_id": image_id,
                "ranking_score": float(ranking_scores[image_id]),
                "selector": "adaptive_learned_asism_v2",
                "winning_policy": winner,
            }, sort_keys=True) + "\n")

    write_frozen_json(Path(cfg.paths.adaptive_selection_manifest), {
        "schema_version": 1,
        "method": "class_aware_adaptive_threshold_v2",
        "frozen": True,
        "split_namespace": namespace,
        "namespace_class": identity["namespace_class"],
        "split_manifest_hash": identity["split_manifest_hash"],
        "generation_manifest_sha256": generation["generation_manifest_sha256"],
        "code_identity_sha256": current_code_identity_hash(),
        "winning_policy": winner,
        "policy_decision": decision,
        "thresholds": thresholds,
        "selected_count": len(selected_ids),
        "full_policy_results_sha256": sha256_file(Path(cfg.paths.full_policy_verification_results)),
        "adaptive_threshold_manifest_sha256": sha256_file(Path(cfg.paths.adaptive_threshold_manifest)),
        "ranking_checkpoint_sha256": sha256_file(Path(cfg.paths.learned_dir) / "ranking_model.pt"),
        "selected_manifest_sha256": sha256_file(Path(cfg.paths.adaptive_selected_manifest)),
    })
    print(f"Final adaptive policy: {winner}; selected {len(selected_ids)} images")
    print(f"-> {cfg.paths.adaptive_selected_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Learn class-aware thresholds from utility-derived ranking scores and select images.

Threshold targets are chosen on ASISM tuning data only by a frozen utility/budget objective, then
distilled into the requested adaptive network. A hard technical safety gate always runs first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import apply_feature_frame  # noqa: E402
from scripts.asism.candidate_pool import load_candidate_pool  # noqa: E402
from scripts.asism.models import AdaptiveThresholdNetwork, MultiSignalUtilityRankingNetwork  # noqa: E402
from scripts.utils.artifact_contracts import stage3_paths  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import read_json, sha256_file, write_frozen_json  # noqa: E402
from scripts.utils.seed import set_seed  # noqa: E402


def context_vector(scores: np.ndarray, prevalence: float, candidates: int, target_count: int) -> np.ndarray:
    q25, q50, q75 = np.quantile(scores, [0.25, 0.5, 0.75]) if len(scores) else (0, 0, 0)
    return np.asarray([prevalence, np.log1p(candidates), scores.mean() if len(scores) else 0,
                       scores.std() if len(scores) else 0, q25, q50, q75,
                       target_count / max(candidates, 1)], dtype=np.float32)


def optimal_threshold(scores: np.ndarray, target_count: int, budget_weight: float) -> float:
    """Frozen one-dimensional search: retain utility while respecting the target count."""
    if not len(scores): return 1.0
    best = None
    for threshold in np.linspace(0.0, 1.0, 101):
        selected = scores[scores >= threshold]
        utility = float(selected.mean()) if len(selected) else -1.0
        budget_error = abs(len(selected) - target_count) / max(target_count, 1)
        objective = utility - budget_weight * budget_error
        candidate = (objective, -threshold, threshold)
        if best is None or candidate > best: best = candidate
    return float(best[2])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths: cfg.paths[key] = str(value)
    cfg.split_namespace = namespace
    # BUG FIXED 2026-08-21: this script never seeded torch's global RNG — AdaptiveThresholdNetwork's
    # init was non-deterministic across runs. Confirmed to occasionally produce thresholds so high
    # that ZERO images clear them (see the guard added below). Same fix as 05/08_train_*.py.
    set_seed(int(cfg.learned_asism.subset_design.seed))
    learned_dir = Path(cfg.paths.learned_dir)
    training_manifest = read_json(learned_dir / "learned_training_manifest.json")

    merged, intended, _ = load_candidate_pool(cfg)
    columns = list(training_manifest["feature_columns"])
    normalized = apply_feature_frame(merged, columns, training_manifest["normalization"])
    checkpoint = torch.load(learned_dir / "ranking_model.pt", map_location="cpu", weights_only=True)
    rank_cfg = cfg.learned_asism.ranking_network
    ranker = MultiSignalUtilityRankingNetwork(len(columns), tuple(rank_cfg.hidden_dims), float(rank_cfg.dropout))
    ranker.load_state_dict(checkpoint["state_dict"]); ranker.eval()
    with torch.no_grad():
        raw = ranker(torch.tensor(normalized.to_numpy(np.float32))).numpy()
    merged = merged.copy(); merged["learned_ranking_score"] = 1.0 / (1.0 + np.exp(-raw))

    threshold_cfg = cfg.learned_asism.threshold_network
    contexts, targets, class_ids = [], [], []
    per_class_candidates = {}
    for class_id, label in enumerate(PRIMARY_ENDPOINT_LABELS):
        ids = [image_id for image_id in merged.image_id if int(intended.get(str(image_id), {}).get(label, 0)) == 1]
        candidates = merged.loc[merged.image_id.isin(ids), "learned_ranking_score"].to_numpy()
        per_class_candidates[label] = set(map(str, ids))
        target_count = min(int(threshold_cfg.max_selected_per_label), max(int(threshold_cfg.min_selected_per_label), len(candidates) // 2))
        contexts.append(context_vector(candidates, len(candidates) / max(len(merged), 1), len(candidates), target_count))
        targets.append(optimal_threshold(candidates, target_count, float(threshold_cfg.budget_weight)))
        class_ids.append(class_id)
    x = torch.tensor(np.asarray(contexts)); y = torch.tensor(targets, dtype=torch.float32); ids_tensor = torch.tensor(class_ids)
    model = AdaptiveThresholdNetwork(len(PRIMARY_ENDPOINT_LABELS), x.shape[1], int(threshold_cfg.class_embedding_dim),
                                     tuple(threshold_cfg.hidden_dims), float(threshold_cfg.dropout))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(threshold_cfg.learning_rate))
    model.train()
    for _ in range(int(threshold_cfg.epochs)):
        prediction = model(ids_tensor, x)
        loss = torch.nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad(): learned_thresholds = model(ids_tensor, x).numpy()

    selected, rejected = set(), []
    safety = cfg.learned_asism.safety
    for _, row in merged.iterrows():
        image_id = str(row.image_id)
        if bool(safety.reject_invalid_iqa) and "iqa_valid" in row and not bool(row.iqa_valid):
            rejected.append({"image_id": image_id, "reason": "invalid_iqa_safety_gate"}); continue
        if bool(safety.reject_near_duplicates) and bool(row.get("novelty_is_near_duplicate", False)):
            rejected.append({"image_id": image_id, "reason": "near_duplicate_safety_gate"}); continue
        applicable = [learned_thresholds[i] for i, label in enumerate(PRIMARY_ENDPOINT_LABELS)
                      if image_id in per_class_candidates[label]]
        threshold = min(applicable) if applicable else float(np.median(learned_thresholds))
        if float(row.learned_ranking_score) >= threshold:
            selected.add(image_id)
        else:
            rejected.append({"image_id": image_id, "reason": "below_learned_class_threshold",
                             "ranking_score": float(row.learned_ranking_score), "threshold": float(threshold)})
    if not selected:
        # Matches 09_finalize_learned_selection.py's identical guard for the adaptive policy. A
        # silently-empty selected_manifest.jsonl is a valid-looking file that downstream code
        # (Stage 4 condition builders) would only catch later via an unrelated "no synthetic images
        # found" error, far from this actual cause — fail here, at the source, instead.
        raise SystemExit(
            "FINALIZATION GATE: fixed-ratio learned threshold selected ZERO images. This is a real "
            "failure, not empty-by-design: check learned_thresholds against learned_ranking_score's "
            "actual distribution (a degenerate ranker or an unreasonably strict "
            "threshold_network.budget_weight/min_selected_per_label are the usual causes)."
        )
    with open(Path(cfg.paths.learned_selected_manifest), "x", encoding="utf-8") as handle:
        for image_id in sorted(selected):
            score = float(merged.loc[merged.image_id.astype(str) == image_id, "learned_ranking_score"].iloc[0])
            handle.write(json.dumps({"image_id": image_id, "ranking_score": score, "selector": "learned_asism_v1"}) + "\n")
    with open(Path(cfg.paths.learned_rejected_log), "x", encoding="utf-8") as handle:
        for row in rejected: handle.write(json.dumps(row) + "\n")
    torch.save({"state_dict": model.state_dict(), "context_dim": x.shape[1]}, learned_dir / "threshold_model.pt")
    write_frozen_json(learned_dir / "learned_selection_manifest.json", {
        "schema_version": 1, "method": "fixed_target_ratio_threshold_distillation_baseline_v1", "frozen": True,
        "thresholds": dict(zip(PRIMARY_ENDPOINT_LABELS, map(float, learned_thresholds))),
        "threshold_search_targets": dict(zip(PRIMARY_ENDPOINT_LABELS, map(float, targets))),
        "selected_count": len(selected), "ranking_checkpoint_sha256": sha256_file(learned_dir / "ranking_model.pt")})
    print(f"Learned ASISM selected {len(selected)}/{len(merged)} images -> {cfg.paths.learned_selected_manifest}")
    return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 3e — learn the class-aware thresholds and publish the selection Stage 4 condition C uses.

This is the last step of ASISM and the only one that writes `asism_selected.csv`. Everything before
it scores and ranks; this decides how high is high enough, per class, and which candidates are
therefore kept.

THE ORDER IS THE METHOD, and it is fixed here:

    1. safety      Invalid images and memorised near-copies of real patients are gone before
                   anything is ranked — re-checked here against the CURRENT artifacts, not trusted
                   from the ranking run, because an image that became unsafe must not stay selected.
    2. quality     An absolute floor on the pooled ranking score. Below it a candidate is rejected
                   whatever its class. Rarity never buys admission for a bad image.
    3. threshold   The class-aware cut, from the configured policy.
    4. floor       A rare class's cut is lowered just enough to reach its minimum — over the
                   candidates that already cleared step 2, so this can only admit a class's best
                   remaining images, never its worst.

TWO POLICIES, AND THE CHOICE IS PRE-REGISTERED. `selection.threshold_policy` picks which one writes
the manifest; the other is computed anyway and its thresholds and counts are recorded side by side,
so the comparison exists in the artifact rather than being assembled afterwards from whichever run
was kept. Which policy actually produces better classifiers is a Stage 4 question, and it is
answered by running condition C under each — not by anything this script can see.

final_eval_heldout is never read. The real class counts come from the classifier training split.

Usage:
    python scripts/asism/ham10000_05_adaptive_thresholds.py --namespace ham-stratified-v1
    python scripts/asism/ham10000_05_adaptive_thresholds.py --namespace ham-stratified-v1 --policy percentile
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_ranking import PoolGate, load_candidate_pool  # noqa: E402
from scripts.asism.ham10000_thresholds import (  # noqa: E402
    CONTEXT_FEATURES,
    class_context,
    enforce_class_floor,
    normalise_scores,
    quality_floor,
    rarity_scaled_percentiles,
    search_threshold,
    target_counts,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402

POLICIES = ("network", "percentile")
FORBIDDEN_SPLIT = "final_eval_heldout"


def real_class_counts(splits_dir: Path, split: str) -> dict[str, int]:
    if split == FORBIDDEN_SPLIT:
        raise PoolGate("the protected split cannot be used to set thresholds")
    path = Path(splits_dir) / f"{split}.csv"
    if not path.is_file():
        raise PoolGate(f"UPSTREAM GATE: missing split {path}; build and freeze the splits first.")
    frame = pd.read_csv(path)
    counts = pd.Series([normalize_diagnosis(value) for value in frame["dx"]]).value_counts()
    return {label: int(counts.get(label, 0)) for label in CLASSIFIER_TARGET_LABELS}


def percentile_thresholds(scores_by_class, real_counts, labels, selection_cfg) -> tuple[dict, dict]:
    percentiles = rarity_scaled_percentiles(
        real_counts,
        labels,
        float(selection_cfg.percentile_policy.base_percentile),
        float(selection_cfg.percentile_policy.min_percentile),
    )
    thresholds = {}
    for label in labels:
        values = np.asarray(scores_by_class.get(label, []), dtype=np.float64)
        thresholds[label] = float(np.percentile(values, percentiles[label])) if values.size else 1.0
    return thresholds, percentiles


def network_thresholds(scores_by_class, real_counts, targets, labels, selection_cfg, seed: int) -> tuple[dict, dict]:
    """Distil `AdaptiveThresholdNetwork` from the frozen search's per-class answers.

    The search alone would already give seven numbers. The network is what makes them a RULE: it is
    fitted to predict a class's threshold from that class's context — how rare it really is, the
    shape of its candidates' scores, how many it is budgeted — so the cut a class receives is
    consistent with how every other class was treated, rather than being seven independent fits.
    Where the network and the search disagree, the disagreement is recorded; it is the distillation
    residual, and a large one means the contexts do not explain the searched thresholds.
    """
    import torch

    from scripts.asism.models import AdaptiveThresholdNetwork
    from scripts.utils.seed import set_seed

    set_seed(int(seed))
    budget_weight = float(selection_cfg.network_policy.budget_weight)

    present = [label for label in labels if len(scores_by_class.get(label, []))]
    total_real = sum(real_counts.values()) or 1
    searched = {
        label: search_threshold(scores_by_class[label], targets[label], budget_weight)
        for label in present
    }
    contexts = {
        label: class_context(scores_by_class[label], real_counts.get(label, 0) / total_real, targets[label])
        for label in present
    }

    network = AdaptiveThresholdNetwork(len(CLASSIFIER_TARGET_LABELS), len(CONTEXT_FEATURES))
    optimizer = torch.optim.AdamW(network.parameters(), lr=float(selection_cfg.network_policy.learning_rate))
    class_ids = torch.tensor([CLASSIFIER_TARGET_LABELS.index(label) for label in present], dtype=torch.long)
    context = torch.tensor(np.stack([contexts[label] for label in present]), dtype=torch.float32)
    target = torch.tensor([searched[label] for label in present], dtype=torch.float32)

    network.train()
    for _ in range(int(selection_cfg.network_policy.epochs)):
        prediction = network(class_ids, context)
        loss = torch.nn.functional.mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    network.eval()
    with torch.no_grad():
        predicted = network(class_ids, context).cpu().numpy()

    thresholds = {label: 1.0 for label in labels}
    thresholds.update({label: float(value) for label, value in zip(present, predicted)})
    diagnostics = {
        "searched_thresholds": {label: float(value) for label, value in searched.items()},
        "distillation_residual": {
            label: float(abs(predicted[index] - searched[label])) for index, label in enumerate(present)
        },
        "budget_weight": budget_weight,
        "context_features": list(CONTEXT_FEATURES),
        "final_loss": float(loss.item()),
    }
    return thresholds, diagnostics


def apply_thresholds(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    cuts = frame["dx"].map(lambda label: thresholds.get(label, 1.0)).astype(float)
    return frame["selection_score"] >= cuts


def run(namespace: str, policy: str | None = None) -> dict:
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    selection_cfg = stage3.selection
    policy = str(policy or selection_cfg.threshold_policy)
    if policy not in POLICIES:
        raise ValueError(f"unknown threshold policy {policy!r}; expected one of {list(POLICIES)}")

    namespace_dir = Path(stage3.paths.outputs_dir) / namespace
    learned_dir = namespace_dir / "learned"
    ranking_path = learned_dir / "ranking_scores.parquet"
    if not ranking_path.is_file():
        raise PoolGate(
            f"UPSTREAM GATE: no ranking scores at {ranking_path}.\n"
            "Run: python scripts/asism/ham10000_04_train_ranking_network.py --namespace " + namespace
        )
    ranking = pd.read_parquet(ranking_path)

    # Re-derive the safe pool from the CURRENT artifacts rather than trusting the ranking run's.
    # A candidate that has since been flagged invalid or as a near-duplicate must not stay selected
    # because it was safe when it was scored.
    pool, surviving, pool_report = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root))
    safe_ids = set(pool["image_id"].astype(str))
    dropped = sorted(set(ranking["image_id"].astype(str)) - safe_ids)
    frame = ranking[ranking["image_id"].astype(str).isin(safe_ids)].reset_index(drop=True)
    if frame.empty:
        raise PoolGate("no ranked candidate survives the current safety gate")

    scores, score_transform = normalise_scores(frame["ranking_score"].to_numpy())
    frame["selection_score"] = scores
    if score_transform["degenerate"]:
        raise PoolGate(
            "every candidate received the same ranking score, so there is no ordering to threshold. "
            "Investigate the ranking network before selecting."
        )

    labels = list(CLASSIFIER_TARGET_LABELS)
    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    real_counts = real_class_counts(splits_dir, str(stage3.learned_asism.real_train_split))
    targets = target_counts(
        real_counts,
        float(selection_cfg.target_synthetic_to_real_ratio),
        int(selection_cfg.min_accepted_per_class),
        int(selection_cfg.max_accepted_per_class),
        labels,
    )

    # 2. Quality floor — absolute, pooled, and applied BEFORE any class is considered.
    floor = quality_floor(frame["selection_score"].to_numpy(), float(selection_cfg.quality_floor_percentile))
    above_floor = frame[frame["selection_score"] >= floor].reset_index(drop=True)
    scores_by_class = {
        label: above_floor.loc[above_floor["dx"] == label, "selection_score"].to_numpy()
        for label in labels
    }

    # 3. Both policies are computed; the configured one decides.
    percentile_cuts, percentiles = percentile_thresholds(scores_by_class, real_counts, labels, selection_cfg)
    network_cuts, network_diagnostics = network_thresholds(
        scores_by_class, real_counts, targets, labels, selection_cfg, int(stage3.learned_asism.seed)
    )
    chosen = dict(network_cuts if policy == "network" else percentile_cuts)

    # 4. Per-class floor, over the candidates that already cleared the quality floor.
    final_cuts, lowered = enforce_class_floor(chosen, scores_by_class, int(selection_cfg.min_accepted_per_class))
    selected = above_floor[apply_thresholds(above_floor, final_cuts)].reset_index(drop=True)

    # Cap: a class whose threshold admits more than its budget keeps its highest-scoring candidates.
    maximum = int(selection_cfg.max_accepted_per_class)
    selected = (
        selected.sort_values("selection_score", ascending=False)
        .groupby("dx", group_keys=False)
        .head(maximum)
        .sort_values(["dx", "selection_score"], ascending=[True, False])
        .reset_index(drop=True)
    )

    manifest_frame = pd.read_csv(Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv")
    paths = dict(zip(manifest_frame["image_id"].astype(str), manifest_frame["image_path"].astype(str)))
    output = pd.DataFrame(
        {
            "image_id": selected["image_id"].astype(str),
            "image_path": [paths[image_id] for image_id in selected["image_id"].astype(str)],
            "dx": selected["dx"].astype(str),
            "selection_score": selected["selection_score"].astype(float),
            "ranking_score": selected["ranking_score"].astype(float),
        }
    )

    selected_path = namespace_dir / "asism_selected.csv"
    output.to_csv(selected_path, index=False)
    # Stage 4 condition C reads a fixed, un-namespaced path. The copy carries a provenance sidecar
    # naming the namespace it came from, so a file left over from a different namespace is visible
    # rather than silently training condition C on another experiment's selection.
    stage4_path = Path(stage3.paths.outputs_dir) / "asism_selected.csv"
    stage4_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_path, stage4_path)

    per_class_selected = output["dx"].value_counts().to_dict()
    counts_under = {
        "network": {
            label: int((scores_by_class[label] >= network_cuts[label]).sum()) for label in labels
        },
        "percentile": {
            label: int((scores_by_class[label] >= percentile_cuts[label]).sum()) for label in labels
        },
    }
    # A class with NOTHING above the pooled quality floor selects nothing, and that is the floor
    # working as intended — it is absolute, and rarity does not buy admission for a bad image. But it
    # is a different finding from "short of its budget" and gets its own line: it says the generator
    # produced nothing usable for that class, which Stage 4 has to be read knowing.
    eliminated = {
        label: int((frame["dx"] == label).sum())
        for label in labels
        if int((frame["dx"] == label).sum()) > 0 and not len(scores_by_class[label])
    }
    short_of_target = {
        label: {"selected": int(per_class_selected.get(label, 0)), "target": targets[label]}
        for label in labels
        if int(per_class_selected.get(label, 0)) < targets[label]
    }

    manifest = {
        "stage": "ham10000_adaptive_thresholds",
        "namespace": namespace,
        "threshold_policy": policy,
        "policy_choice": "pre-registered in configs/ham10000_stage3.yaml selection.threshold_policy",
        "order": ["safety", "quality_floor", "class_threshold", "per_class_floor", "budget_cap"],
        "surviving_signals": surviving,
        "pool": pool_report,
        "candidates_ranked": int(len(ranking)),
        "candidates_dropped_by_current_safety_gate": len(dropped),
        "score_transform": score_transform,
        "quality_floor_percentile": float(selection_cfg.quality_floor_percentile),
        "quality_floor_value": float(floor),
        "candidates_above_quality_floor": int(len(above_floor)),
        "real_class_counts": real_counts,
        "real_train_split": str(stage3.learned_asism.real_train_split),
        "target_counts": targets,
        "thresholds_applied": final_cuts,
        "thresholds_network": network_cuts,
        "thresholds_percentile": percentile_cuts,
        "percentile_per_class": percentiles,
        "network_diagnostics": network_diagnostics,
        "selected_under_each_policy_before_floor": counts_under,
        "classes_lowered_to_meet_the_floor": lowered,
        "selected_per_class": {label: int(per_class_selected.get(label, 0)) for label in labels},
        # Reported, never silently fixed: a class that cannot reach its target is a finding about
        # the generator or the gate, and Stage 4 has to be read knowing it.
        "classes_short_of_target": short_of_target,
        "classes_eliminated_by_the_quality_floor": eliminated,
        "n_selected": int(len(output)),
        "selected_path": str(selected_path),
        "stage4_path": str(stage4_path),
        "evidence": "ranking scores and the real training split; final_eval_heldout never read",
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(namespace_dir / "asism_selection_manifest.json", manifest)
    write_json(
        stage4_path.with_suffix(".provenance.json"),
        {
            "split_namespace": namespace,
            "threshold_policy": policy,
            "n_selected": int(len(output)),
            "source": str(selected_path),
            "git_commit_hash": manifest["git_commit_hash"],
        },
    )

    print(f"Adaptive thresholds ({policy} policy)", flush=True)
    print("=" * 72, flush=True)
    for label in labels:
        print(
            f"  {label:<6} cut={final_cuts[label]:.4f}  selected={per_class_selected.get(label, 0):>5}"
            f"  target={targets[label]:>5}  real={real_counts[label]:>5}",
            flush=True,
        )
    print("=" * 72, flush=True)
    print(f"  quality floor  p{selection_cfg.quality_floor_percentile} = {floor:.4f} "
          f"({len(above_floor)}/{len(frame)} candidates above it)", flush=True)
    if eliminated:
        print(f"  ELIMINATED by the quality floor (no candidate above it): {eliminated}", flush=True)
    if short_of_target:
        print(f"  short of target: {short_of_target}", flush=True)
    print(f"  selected {len(output)} -> {selected_path}", flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--policy", default=None, choices=list(POLICIES))
    args = parser.parse_args()

    manifest = run(args.namespace, args.policy)
    print(json.dumps({key: manifest[key] for key in ("threshold_policy", "n_selected", "selected_per_class")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

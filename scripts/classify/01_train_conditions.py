#!/usr/bin/env python3
"""Stage 4 — train the thesis conditions A/B/C.

The three arms the thesis specifies for Stage 5:

    A  real only                     classifier_train
    B  real + ALL synthetic          classifier_train + every Stage 2 image
    C  real + ASISM-selected         classifier_train + the finalized adaptive selection

Primary comparison: C vs B — does selecting synthetic images with ASISM beat using all of them
unselected?

KNOWN LIMITATION — no matched-random control. C is a strict subset of B, so a C-over-B gain has two
competing explanations that this design cannot separate: (a) ASISM chose *good* images, or (b) using
*fewer* synthetic images helps regardless of which ones, because unselected synthetic data is noisy.
Distinguishing them needs a condition drawing |C| images at random with C's label profile
(optional condition D). The matched-draw machinery below (build_matched_random_draw, profile_of,
condition_d.*) is retained and working for exactly that purpose, but D is not enabled by default.
See docs/stages2_to_5_plan.md §7.

Fairness (§7.1): every run — all conditions, all seeds — uses the SAME optimizer-step budget, batch
size, augmentation, and checkpoint-selection rule. All model selection is on classifier_val;
final_eval_heldout is never touched here.

Usage:
    python scripts/classify/01_train_conditions.py --condition A
    python scripts/classify/01_train_conditions.py --condition all
    python scripts/classify/01_train_conditions.py --plan-only            # run/cost accounting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.experiment_registry import ExperimentRun  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, get_library_versions, hash_dict, read_json, sha256_file, write_frozen_json, write_json  # noqa: E402
from scripts.utils.splits import load_split, split_provenance  # noqa: E402
from scripts.utils.artifact_contracts import (  # noqa: E402
    current_code_identity_hash, namespace_identity, require_generation_complete,
    require_manifest_fields, stage2_paths, stage3_paths, stage4_paths,
)

CONDITIONS = ["A", "B", "C"]


def load_configs():
    stage4 = load_named_config("stage4_classifier.yaml", "stage4")
    stage3 = load_named_config("stage3_asism.yaml", "stage3")
    stage2 = load_named_config("stage2_generation.yaml", "stage2")
    return stage4, stage3, stage2


def load_all_synthetic_ids(stage2_cfg) -> list[str]:
    path = Path(stage2_cfg.paths.manifest_path)
    if not path.is_file():
        raise SystemExit(f"UPSTREAM GATE: missing generation manifest {path}")
    ids = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["image_id"])
    return ids


def load_selected_ids(stage3_cfg, selector: str = "weighted") -> list[str]:
    paths = {
        "weighted": stage3_cfg.paths.selected_manifest,
        "adaptive": stage3_cfg.paths.adaptive_selected_manifest,
        "fixed_ratio": stage3_cfg.paths.learned_selected_manifest,
    }
    if selector not in paths:
        raise ValueError(f"unknown selector {selector}")
    path = Path(paths[selector])
    if not path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: ASISM selection not found at {path}\n"
            "Run the weighted selector or the learned-ASISM selection pipeline, as applicable."
        )
    ids = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["image_id"])
    return ids


def intended_lookup(stage2_cfg) -> dict[str, dict]:
    lookup = {}
    with open(Path(stage2_cfg.paths.manifest_path), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                lookup[entry["image_id"]] = entry["intended_label_vector"]
    return lookup


def profile_of(image_ids: list[str], intended: dict[str, dict]) -> dict:
    """Class-distribution profile used to match a random-draw control against a reference condition."""
    per_label = {label: 0 for label in PRIMARY_ENDPOINT_LABELS}
    single, multi, none = 0, 0, 0
    for image_id in image_ids:
        vector = intended.get(image_id, {})
        positives = [l for l in PRIMARY_ENDPOINT_LABELS if int(vector.get(l, 0)) == 1]
        for label in positives:
            per_label[label] += 1
        if len(positives) == 0:
            none += 1
        elif len(positives) == 1:
            single += 1
        else:
            multi += 1
    total = max(len(image_ids), 1)
    return {
        "total": len(image_ids),
        "per_label_positive": per_label,
        "single_label": single,
        "multi_label": multi,
        "no_finding": none,
        "single_fraction": single / total,
        "multi_fraction": multi / total,
    }


def build_matched_random_draw(
    all_ids: list[str],
    target_profile: dict,
    intended: dict[str, dict],
    seed: int,
    config,
) -> tuple[list[str], dict]:
    """One matched-random draw for condition D.

    Greedy per-label matching against the target's marginal positive counts, then a top-up to the
    exact total. Joint label-vector matching is ATTEMPTED (by preferring candidates whose exact
    vector the target also contains) but never allowed to make matching infeasible — a strict joint
    match is unachievable for rare combinations and would silently shrink the draw.

    Residual imbalance is measured and returned, never hidden.
    """
    matching = config.condition_d.matching
    rng = np.random.default_rng(seed)
    available = [i for i in all_ids]
    rng.shuffle(available)

    target_counts = dict(target_profile["per_label_positive"])
    target_total = int(target_profile["total"])

    by_vector: dict[tuple, list[str]] = {}
    for image_id in available:
        vector = intended.get(image_id, {})
        key = tuple(sorted(l for l in PRIMARY_ENDPOINT_LABELS if int(vector.get(l, 0)) == 1))
        by_vector.setdefault(key, []).append(image_id)

    chosen: list[str] = []
    chosen_set: set[str] = set()
    counts = {label: 0 for label in PRIMARY_ENDPOINT_LABELS}

    if bool(matching.match_per_label_marginal):
        for label in sorted(PRIMARY_ENDPOINT_LABELS, key=lambda l: target_counts[l]):
            need = target_counts[label]
            if need <= 0:
                continue
            candidates = [
                image_id
                for image_id in available
                if image_id not in chosen_set
                and int(intended.get(image_id, {}).get(label, 0)) == 1
            ]
            for image_id in candidates:
                if counts[label] >= need or len(chosen) >= target_total:
                    break
                chosen.append(image_id)
                chosen_set.add(image_id)
                for other in PRIMARY_ENDPOINT_LABELS:
                    if int(intended.get(image_id, {}).get(other, 0)) == 1:
                        counts[other] += 1

    # Top up to the exact total, preferring No-Finding vectors if the target has them.
    if len(chosen) < target_total:
        remaining = [i for i in available if i not in chosen_set]
        need_no_finding = target_profile["no_finding"] - sum(
            1 for i in chosen if not any(
                int(intended.get(i, {}).get(l, 0)) == 1 for l in PRIMARY_ENDPOINT_LABELS
            )
        )
        no_finding_pool = [
            i for i in remaining
            if not any(int(intended.get(i, {}).get(l, 0)) == 1 for l in PRIMARY_ENDPOINT_LABELS)
        ]
        for image_id in no_finding_pool[: max(0, need_no_finding)]:
            if len(chosen) >= target_total:
                break
            chosen.append(image_id)
            chosen_set.add(image_id)
        for image_id in remaining:
            if len(chosen) >= target_total:
                break
            if image_id not in chosen_set:
                chosen.append(image_id)
                chosen_set.add(image_id)

    chosen = chosen[:target_total]

    # Deterministic repair: hill-climb swaps against the frozen marginal and single/multi target.
    # No reseeding/retry is used; ties follow the seeded `available` order above.
    def positive_set(image_id):
        return frozenset(label for label in PRIMARY_ENDPOINT_LABELS if int(intended.get(image_id, {}).get(label, 0)) == 1)
    chosen_vectors = [positive_set(image_id) for image_id in chosen]
    current_counts = {label: sum(label in vector for vector in chosen_vectors) for label in PRIMARY_ENDPOINT_LABELS}
    current_single = sum(len(vector) == 1 for vector in chosen_vectors)
    current_multi = sum(len(vector) > 1 for vector in chosen_vectors)
    def mismatch_counts(label_counts, single, multi):
        label_error = sum(abs(label_counts[label] - target_counts[label]) / max(target_counts[label], 1) for label in PRIMARY_ENDPOINT_LABELS)
        shape_error = abs(single / max(target_total, 1) - target_profile["single_fraction"]) + abs(multi / max(target_total, 1) - target_profile["multi_fraction"])
        return label_error + shape_error
    current_score = mismatch_counts(current_counts, current_single, current_multi)
    outside = [image_id for image_id in available if image_id not in chosen_set][:500]
    for _ in range(min(100, max(target_total, 1))):
        best = None
        for chosen_index, old_id in list(enumerate(chosen))[:500]:
            old_vector = chosen_vectors[chosen_index]
            for new_id in outside:
                new_vector = positive_set(new_id)
                trial_counts = {label: current_counts[label] - (label in old_vector) + (label in new_vector) for label in PRIMARY_ENDPOINT_LABELS}
                trial_single = current_single - (len(old_vector) == 1) + (len(new_vector) == 1)
                trial_multi = current_multi - (len(old_vector) > 1) + (len(new_vector) > 1)
                score = mismatch_counts(trial_counts, trial_single, trial_multi)
                if score + 1e-12 < current_score and (best is None or score < best[0]):
                    best = (score, chosen_index, old_id, new_id, new_vector, trial_counts, trial_single, trial_multi)
        if best is None:
            break
        current_score, chosen_index, old_id, new_id, new_vector, current_counts, current_single, current_multi = best
        chosen[chosen_index] = new_id
        chosen_vectors[chosen_index] = new_vector
        outside.remove(new_id)
        outside.append(old_id)
        chosen_set.remove(old_id)
        chosen_set.add(new_id)
    achieved = profile_of(chosen, intended)

    tolerance = float(matching.per_label_tolerance_fraction)
    residual = {}
    within_tolerance = True
    for label in PRIMARY_ENDPOINT_LABELS:
        target_count = target_counts[label]
        achieved_count = achieved["per_label_positive"][label]
        allowed = max(1.0, tolerance * max(target_count, 1))
        delta = achieved_count - target_count
        residual[label] = {
            "target": target_count,
            "achieved": achieved_count,
            "delta": delta,
            "within_tolerance": bool(abs(delta) <= allowed),
        }
        if abs(delta) > allowed:
            within_tolerance = False

    single_delta = abs(achieved["single_fraction"] - target_profile["single_fraction"])
    single_ok = single_delta <= float(matching.single_multi_tolerance_fraction)

    valid_pool = set(all_ids)
    constraints = {
        "exact_total": achieved["total"] == target_total,
        "per_label_marginals": within_tolerance,
        "single_multi_proportion": single_ok,
        "no_duplicate_ids": len(chosen) == len(set(chosen)),
        "all_ids_in_valid_pool": set(chosen).issubset(valid_pool),
    }
    unsatisfied = [name for name, passed in constraints.items() if not passed]
    return chosen, {
        "seed": seed,
        "target_total": target_total,
        "achieved_total": achieved["total"],
        "count_matched_exactly": bool(achieved["total"] == target_total),
        "per_label_residual": residual,
        "all_labels_within_tolerance": within_tolerance,
        "single_fraction_target": target_profile["single_fraction"],
        "single_fraction_achieved": achieved["single_fraction"],
        "single_fraction_delta": single_delta,
        "single_multi_within_tolerance": bool(single_ok),
        "joint_vector_matching_attempted": bool(matching.attempt_joint_vector_matching),
        "note": "Residual imbalance is recorded, not hidden; exact joint matching is not forced.",
        "constraints": constraints,
        "passed": not unsatisfied,
        "unsatisfied_constraints": unsatisfied,
        "draw_id_hash": hash_dict({"seed": seed, "ids": chosen}, length=64),
    }


def training_plan(stage4_cfg) -> dict:
    seeds = list(stage4_cfg.seeds)
    n_draws = int(stage4_cfg.condition_d.n_draws)
    all_runs = {
        "A": len(seeds), "B": len(seeds), "C": len(seeds),
        "D": n_draws * len(seeds),
    }
    enabled = list(stage4_cfg.get("conditions", CONDITIONS))
    runs = {condition: all_runs[condition] for condition in enabled}
    total = sum(runs.values())
    return {
        "runs_per_condition": runs,
        "total_runs": total,
        "seeds": seeds,
        "n_d_draws": n_draws,
        "optimizer_steps_per_run": int(stage4_cfg.training.max_steps),
        "total_optimizer_steps": total * int(stage4_cfg.training.max_steps),
        "formula": "sum of enabled per-condition runs; D contributes n_draws x |seeds|",
    }


def build_records(condition: str, draw_ids: list[str] | None, cfgs, namespace: str):
    from scripts.utils.classifier import records_from_split, records_from_synthetic_manifest

    stage4_cfg, stage3_cfg, stage2_cfg = cfgs
    stage1_cfg = load_stage1_config()
    images_dir = Path(stage1_cfg.paths.images_dir) / namespace

    frame = load_split(
        "classifier_train", namespace, purpose="schema_validation", caller="stage4"
    )
    real_records = records_from_split(frame, images_dir / "classifier_train")
    if not real_records:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed classifier_train images under "
            f"{images_dir / 'classifier_train'}"
        )

    synthetic_ids: list[str] | None = None
    if condition == "B":
        synthetic_ids = None  # all
    elif condition == "C":
        synthetic_ids = load_selected_ids(stage3_cfg, selector="adaptive")
    elif condition == "D":
        synthetic_ids = draw_ids

    synthetic_records = []
    if condition != "A":
        synthetic_records = records_from_synthetic_manifest(
            Path(stage2_cfg.paths.manifest_path),
            Path(stage2_cfg.paths.images_dir),
            synthetic_ids,
        )
        if not synthetic_records:
            raise SystemExit(
                f"UPSTREAM GATE: condition {condition} needs synthetic images but none were found."
            )

    val_frame = load_split("classifier_val", namespace, purpose="schema_validation", caller="stage4")
    val_records = records_from_split(val_frame, images_dir / "classifier_val")
    if not val_records:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed classifier_val images under "
            f"{images_dir / 'classifier_val'}"
        )

    return real_records + synthetic_records, val_records


def run_one(condition: str, seed: int, draw_index: int | None, draw_ids, cfgs, namespace: str) -> dict:
    from scripts.utils.classifier import TrainingBudget, train_classifier

    stage4_cfg, _, _ = cfgs
    train_records, val_records = build_records(condition, draw_ids, cfgs, namespace)

    tag = f"{condition}" + (f"_draw{draw_index}" if draw_index is not None else "") + f"_seed{seed}"
    checkpoint_path = Path(stage4_cfg.paths.checkpoints_dir) / f"condition_{tag}.pt"

    budget = TrainingBudget(
        max_steps=int(stage4_cfg.training.max_steps),
        batch_size=int(stage4_cfg.training.batch_size),
        learning_rate=float(stage4_cfg.training.learning_rate),
        weight_decay=float(stage4_cfg.training.weight_decay),
        eval_every_n_steps=int(stage4_cfg.training.eval_every_n_steps),
        seed=seed,
    )

    config_block = {
        "stage4": OmegaConf.to_container(stage4_cfg, resolve=True),
        "condition": condition,
        "draw_index": draw_index,
        "seed": seed,
    }

    with ExperimentRun(
        stage=f"stage4_condition_{condition}",
        config=config_block,
        dataset_version=split_provenance(namespace)["split_manifest_hash"],
    ) as run:
        model, accounting, history = train_classifier(
            train_records=train_records,
            val_records=val_records,
            budget=budget,
            dropout_p=float(stage4_cfg.model.dropout_p),
            pretrained_source=str(stage4_cfg.model.pretrained_source),
            resolution=int(stage4_cfg.model.resolution),
            progress_desc=f"cond-{tag}",
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path if checkpoint_path.is_file() else None,
            checkpoint_metadata={
                "split_manifest_hash": split_provenance(namespace)["split_manifest_hash"],
                "split_namespace": namespace, "dataset_id": tag,
                "condition_or_draw_id": f"{condition}:{draw_index}",
                "config_hash": hash_dict(config_block), "label_policy": "chexpert_mask_uncertain_v1",
                "code_version": get_git_commit_hash(), "code_identity_sha256": current_code_identity_hash(),
            },
        )
        accounting.dataset_id = tag
        accounting.config_hash = hash_dict(config_block)
        accounting.sampling_policy = str(stage4_cfg.training.sampling_policy)
        best = max(history, key=lambda entry: entry["macro_auroc"]) if history else {}
        run.set_checkpoint_path(checkpoint_path)
        run.set_metrics({"best_val_macro_auroc": best.get("macro_auroc"), **accounting.to_dict()})

    return {
        "tag": tag,
        "condition": condition,
        "draw_index": draw_index,
        "seed": seed,
        "checkpoint": accounting.extra.get("best_checkpoint_path", str(checkpoint_path)),
        "resume_checkpoint": str(checkpoint_path),
        "best_val_macro_auroc": best.get("macro_auroc"),
        "accounting": accounting.to_dict(),
        "eval_history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--plan-only", action="store_true", help="Print run/cost accounting and exit")
    args = parser.parse_args()

    cfgs = load_configs()
    stage4_cfg, stage3_cfg, stage2_cfg = cfgs
    namespace = args.namespace or str(stage4_cfg.split_namespace)
    stage4_cfg.split_namespace = namespace
    for key, value in stage4_paths(stage4_cfg, namespace).items():
        stage4_cfg.paths[key] = str(value)
    for key, value in stage3_paths(stage3_cfg, namespace).items():
        if key in stage3_cfg.paths:
            stage3_cfg.paths[key] = str(value)
    for key, value in stage2_paths(stage2_cfg, namespace).items():
        if key in stage2_cfg.paths:
            stage2_cfg.paths[key] = str(value)

    plan = training_plan(stage4_cfg)
    print("Stage 4 run plan (§7 frozen seed policy)", flush=True)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return 0

    identity = namespace_identity(namespace)
    _, generation_completion = require_generation_complete(stage2_cfg, namespace)
    # The frozen Stage 3 selection is 09's Learned-ASISM manifest (the sole selector; Stage 5 checks
    # the same file). The weighted selector's asism_frozen_manifest.json (03) is not produced by the
    # pipeline, so gating on it made Stage 4 unrunnable.
    require_manifest_fields(Path(stage3_cfg.paths.adaptive_selection_manifest), {
        "split_namespace": namespace, "namespace_class": identity["namespace_class"],
        "split_manifest_hash": identity["split_manifest_hash"],
        "generation_manifest_sha256": generation_completion["generation_manifest_sha256"],
        "code_identity_sha256": current_code_identity_hash(), "frozen": True,
    }, "frozen Learned ASISM selection")

    results_dir = Path(stage4_cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Freeze the experiment protocol before any training run starts (§9 step 11).
    protocol = {
        "stage": "stage4_frozen_protocol",
        "frozen": True,
        "split_provenance": split_provenance(namespace),
        "model": OmegaConf.to_container(stage4_cfg.model, resolve=True),
        "training": OmegaConf.to_container(stage4_cfg.training, resolve=True),
        "seeds": list(stage4_cfg.seeds),
        "condition_d": OmegaConf.to_container(stage4_cfg.condition_d, resolve=True),
        "threshold_policy": OmegaConf.to_container(stage4_cfg.threshold_policy, resolve=True),
        "label_policy": {
            "classifier_target_labels": 14,
            "primary_endpoint_labels": PRIMARY_ENDPOINT_LABELS,
            "no_finding_in_primary": False,
            "support_devices_in_primary": False,
        },
        "uncertainty_policy": "raw preserved; -1 and blank masked in loss and metrics",
        "fairness_protocol": "equal optimizer steps across every enabled condition and seed",
        "primary_endpoint": f"macro-AUROC over the {len(PRIMARY_ENDPOINT_LABELS)} primary labels on final_eval_heldout",
        "primary_comparison": "C vs B (ASISM-selected synthetic images vs all synthetic images)",
        "known_limitation": (
            "C is a strict subset of B, and no matched-random control condition is enabled. A "
            "C-over-B gain therefore cannot distinguish ASISM's selection quality from the effect "
            "of simply using fewer synthetic images."
        ),
        "multiplicity": {"confirmatory": "holm_bonferroni", "exploratory": "benjamini_hochberg"},
        "run_plan": plan,
        "git_commit_hash": get_git_commit_hash(),
        "code_identity_sha256": current_code_identity_hash(),
        "environment_versions": get_library_versions(),
        "serialized_config": OmegaConf.to_container(stage4_cfg, resolve=True),
    }
    protocol_path = Path(stage4_cfg.paths.protocol_manifest)
    if protocol_path.exists():
        existing = read_json(protocol_path)
        if existing != protocol:
            raise SystemExit("Frozen experiment protocol already exists with different content; use a new versioned protocol/run ID")
    else:
        write_frozen_json(protocol_path, protocol)
    freeze_components = {
        Path(stage4_cfg.paths.label_policy_manifest): protocol["label_policy"],
        Path(stage4_cfg.paths.threshold_policy_manifest): protocol["threshold_policy"],
        Path(stage4_cfg.paths.seed_plan_manifest): {"seeds": protocol["seeds"], "run_plan": plan},
        Path(stage4_cfg.paths.d_matching_policy_manifest): protocol["condition_d"],
    }
    for artifact_path, payload in freeze_components.items():
        frozen_payload = {"schema_version": 1, "protocol_manifest_hash": sha256_file(protocol_path),
                          "config": payload, "environment_versions": protocol.get("environment_versions", {})}
        if artifact_path.exists():
            existing = read_json(artifact_path)
            if existing != {**frozen_payload, "frozen": True}:
                raise SystemExit(f"Frozen methodology artifact differs and cannot be overwritten: {artifact_path}")
        else:
            write_frozen_json(artifact_path, frozen_payload)
    print(f"\nFrozen protocol -> {stage4_cfg.paths.protocol_manifest}", flush=True)

    enabled_conditions = list(stage4_cfg.get("conditions", CONDITIONS))
    if args.condition != "all" and args.condition not in enabled_conditions:
        raise SystemExit(f"Condition {args.condition} is disabled by this explicit configuration: {enabled_conditions}")
    conditions = enabled_conditions if args.condition == "all" else [args.condition]
    if "C" in conditions:
        require_manifest_fields(
            Path(stage3_cfg.paths.adaptive_selection_manifest),
            {"frozen": True, "method": "class_aware_adaptive_threshold_v2"},
            "adaptive learned ASISM selection",
        )
    seeds = list(stage4_cfg.seeds)
    all_results = []
    completed_dir = results_dir / "completed_runs"
    completed_dir.mkdir(exist_ok=True)

    def execute_or_resume(condition, seed, draw_index=None, ids=None):
        tag = f"{condition}" + (f"_draw{draw_index}" if draw_index is not None else "") + f"_seed{seed}"
        expected_run_identity = {
            "split_manifest_hash": split_provenance(namespace)["split_manifest_hash"],
            "split_namespace": namespace, "tag": tag,
            "code_identity_sha256": current_code_identity_hash(),
        }
        record_path = completed_dir / f"{tag}.json"
        if record_path.is_file():
            record = read_json(record_path)
            checkpoint = Path(record.get("checkpoint", ""))
            history_hash = hash_dict({"history": record.get("result", {}).get("eval_history", [])}, length=64)
            compatible = all(record.get(key) == value for key, value in expected_run_identity.items())
            if (record.get("status") == "complete" and compatible and checkpoint.is_file()
                    and record.get("checkpoint_hash") == sha256_file(checkpoint)
                    and record.get("metric_history_hash") == history_hash):
                return record["result"]
            raise SystemExit(f"Invalid/corrupt completed-run record for {tag}; refusing silent reuse: {record_path}")
        result = run_one(condition, seed, draw_index, ids, cfgs, namespace)
        checkpoint = Path(result["checkpoint"])
        history_hash = hash_dict({"history": result["eval_history"]}, length=64)
        completion = {"schema_version": 2, "status": "complete", **expected_run_identity, "checkpoint": str(checkpoint),
                      "checkpoint_hash": sha256_file(checkpoint), "metric_history_hash": history_hash,
                      "accounting": result["accounting"], "split_provenance": split_provenance(namespace), "result": result}
        write_json(record_path, completion)
        return result

    # Matched-random draws are built against the reference condition's actual selection.
    draw_specs: list[tuple[int, list[str], dict]] = []
    if "D" in conditions:
        intended = intended_lookup(stage2_cfg)
        selected_ids = load_selected_ids(stage3_cfg)
        target_profile = profile_of(selected_ids, intended)
        all_ids = load_all_synthetic_ids(stage2_cfg)
        pool = [i for i in all_ids if i not in set(selected_ids)] or all_ids

        for draw_index, draw_seed in enumerate(list(stage4_cfg.condition_d.draw_seeds)):
            ids, report = build_matched_random_draw(
                pool, target_profile, intended, draw_seed, stage4_cfg
            )
            draw_specs.append((draw_index, ids, report))
            if not report["passed"]:
                raise SystemExit(
                    f"Condition D matching infeasible for draw {draw_index}, seed {draw_seed}; "
                    f"refusing training. Unsatisfied constraints: {report['unsatisfied_constraints']}; "
                    f"details={json.dumps(report, sort_keys=True)}"
                )
            write_json(results_dir / f"condition_d_draw_{draw_index}.json", {"draw_index": draw_index, "image_ids": ids, **report})

        write_json(
            results_dir / "condition_d_matching_report.json",
            {
                "target_profile_from_condition_C": target_profile,
                "n_draws": len(draw_specs),
                "draws": [report for _, _, report in draw_specs],
            },
        )
        print(f"Condition D: {len(draw_specs)} matched draws built", flush=True)
        for _, _, report in draw_specs:
            print(
                f"  seed {report['seed']}: n={report['achieved_total']}/{report['target_total']} "
                f"labels_within_tolerance={report['all_labels_within_tolerance']}",
                flush=True,
            )

    for condition in conditions:
        if condition == "D":
            for draw_index, ids, _ in draw_specs:
                for seed in seeds:
                    all_results.append(execute_or_resume(condition, seed, draw_index, ids))
        else:
            for seed in seeds:
                all_results.append(execute_or_resume(condition, seed))

    write_json(
        results_dir / "stage4_training_results.json",
        {
            "results": all_results,
            "run_plan": plan,
            "protocol_manifest": str(stage4_cfg.paths.protocol_manifest),
            "git_commit_hash": get_git_commit_hash(),
        },
    )
    print(f"\n{len(all_results)} runs complete -> {results_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

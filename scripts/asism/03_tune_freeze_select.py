#!/usr/bin/env python3
"""Stage 3c — ASISM bounded tuning, freeze, and selection (docs/stages2_to_5_plan.md §4.7, §4.8).

Three phases, each independently invocable:

  --phase estimate   Compute the proxy-run count and GPU-hour estimate and apply the compute-budget
                     gate. Runs no training. This is the gate that must pass BEFORE any results are
                     seen — if the estimate is over budget, the search space is reduced here, never
                     after observing performance.

  --phase tune       Two-stage bounded search (coarse screening -> shortlist validation). Proxy
                     trains on classifier_train + the ACTUALLY-CONSTRUCTED candidate synthetic
                     subset and evaluates on asism_tuning_heldout ONLY. Fixed optimizer-step budget,
                     no candidate-specific early stopping. classifier_val is not used to rank
                     candidates. final_eval_heldout is never touched. Resumable at trial granularity.

  --phase select     Apply the frozen configuration: quality floor FIRST, then class quota. Writes
                     selected_manifest.jsonl and rejected_log.jsonl with a reason per rejected image.

NOT PART OF THE THESIS PIPELINE (docs/stages2_to_5_plan.md §4.9 / §7 v3 revision notes). This is the
weighted-score selector, which predates the learned ASISM components. The thesis defines ASISM as the
full module including the Multi-Objective Ranking Network and Adaptive Threshold Learning, so there
is no weighted-baseline Stage 4 condition and nothing consumes this script's selected_manifest.jsonl.
Retained for reference and for the Stage 3 signal-merge helpers other code still imports.

Usage:
    python scripts/asism/03_tune_freeze_select.py --phase estimate
    python scripts/asism/03_tune_freeze_select.py --phase tune
    python scripts/asism/03_tune_freeze_select.py --phase select
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.signals import SCHEMA_VERSION  # noqa: E402
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS, normalize_label  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, write_frozen_json, write_json  # noqa: E402
from scripts.utils.splits import load_split, split_provenance  # noqa: E402
from scripts.utils.artifact_contracts import (  # noqa: E402
    asism_score_provenance, require_score_artifact, require_generation_complete,
    stage2_paths, stage3_paths,
)

# Directional signals eligible for the weighted composite (uncertainty is banded, not directional).
DIRECTIONAL_SCORES = {
    "similarity": "similarity_knn_mean",
    "iqa": "iqa_composite",
    "explainability": "explainability_region_overlap",
    "agreement": "agreement_score",
}


def load_config():
    return load_named_config("stage3_asism.yaml", "stage3")


def compute_budget_estimate(config) -> dict:
    """Pre-execution accounting (§4.7). Must be computed and checked before any trial runs."""
    tuning = config.tuning
    coarse_runs = int(tuning.coarse.trials) * int(tuning.coarse.folds) * int(tuning.coarse.seeds)
    shortlist_runs = (
        int(tuning.shortlist.size) * int(tuning.shortlist.folds) * int(tuning.shortlist.seeds)
    )
    total_runs = coarse_runs + shortlist_runs
    hours_each = float(tuning.compute_budget.hours_per_proxy_run_estimate)
    estimated_hours = total_runs * hours_each
    max_hours = float(tuning.compute_budget.max_gpu_hours)

    return {
        "coarse_runs": coarse_runs,
        "shortlist_runs": shortlist_runs,
        "total_proxy_runs": total_runs,
        "hours_per_proxy_run_estimate": hours_each,
        "estimated_gpu_hours": round(estimated_hours, 3),
        "max_gpu_hours": max_hours,
        "within_budget": bool(estimated_hours <= max_hours),
        "formula": (
            "total = coarse_trials*coarse_folds*coarse_seeds "
            "+ shortlist_size*shortlist_folds*shortlist_seeds"
        ),
    }


def enforce_compute_budget(config) -> dict:
    estimate = compute_budget_estimate(config)
    if not estimate["within_budget"]:
        raise SystemExit(
            "COMPUTE BUDGET GATE: the ASISM search exceeds its declared budget.\n"
            f"  estimated: {estimate['estimated_gpu_hours']} GPU-hours "
            f"({estimate['total_proxy_runs']} proxy runs x "
            f"{estimate['hours_per_proxy_run_estimate']}h)\n"
            f"  budget:    {estimate['max_gpu_hours']} GPU-hours\n\n"
            "Reduce the search space in configs/stage3_asism.yaml (tuning.coarse.trials, "
            "shortlist.size, folds, or seeds) BEFORE running.\n"
            "Reducing it after seeing results would let the outcome influence the search design "
            "(docs/stages2_to_5_plan.md §4.7)."
        )
    return estimate


def load_merged_scores(config, surviving: list[str]) -> pd.DataFrame:
    """Merge the surviving signals' artifacts by image_id (§4.0)."""
    scores_dir = Path(config.paths.scores_dir)
    merged = None
    for signal in surviving:
        path = scores_dir / f"{signal}_scores.parquet"
        if not path.is_file():
            raise SystemExit(f"Surviving signal {signal!r} has no artifact at {path}")
        require_score_artifact(path, signal, config._expected_score_provenance)
        frame = pd.read_parquet(path)
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="inner")

    # Uncertainty is always merged when present: §4.8 uses its band even though it is not part of
    # the directional composite.
    uncertainty_path = scores_dir / "uncertainty_scores.parquet"
    if uncertainty_path.is_file() and "uncertainty_band" not in (merged.columns if merged is not None else []):
        require_score_artifact(uncertainty_path, "uncertainty", config._expected_score_provenance)
        uncertainty = pd.read_parquet(uncertainty_path)
        merged = uncertainty if merged is None else merged.merge(uncertainty, on="image_id", how="left")

    if merged is None or merged.empty:
        raise SystemExit("No ASISM scores available to merge.")
    return merged


def normalize_series(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    low, high = values.min(), values.max()
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - low) / (high - low)


def composite_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    score = pd.Series(np.zeros(len(frame)), index=frame.index)
    for signal, weight in weights.items():
        column = DIRECTIONAL_SCORES.get(signal)
        if weight <= 0 or column is None or column not in frame.columns:
            continue
        score = score + weight * normalize_series(frame[column])
    return score / total_weight


def candidate_weight_grid(surviving: list[str], n_trials: int, seed: int) -> list[dict[str, float]]:
    """Coarse grid over weights for the directional surviving signals (§4.7 Stage A).

    Deliberately coarse: fine-grained optimization against asism_tuning_heldout is exactly the
    overfitting risk the bounded-search design exists to limit.
    """
    directional = [s for s in surviving if s in DIRECTIONAL_SCORES]
    if not directional:
        return [{}]

    levels = [0.0, 0.5, 1.0]
    combinations = [
        dict(zip(directional, values))
        for values in product(levels, repeat=len(directional))
        if sum(values) > 0
    ]
    equal = {signal: 1.0 for signal in directional}
    required = [equal]
    required += [{name: (1.0 if name == signal else 0.0) for name in directional} for signal in directional]
    if "uncertainty" in surviving:
        # Uncertainty is intentionally banded rather than treated as monotonically good/bad. This
        # baseline therefore ranks only by the frozen preferred-band bonus inside class quotas.
        required.append({**{name: 0.0 for name in directional}, "__uncertainty_band_only__": 1.0})
    if len(directional) > 1:
        required += [{name: (0.0 if name == omitted else 1.0) for name in directional} for omitted in directional]
    # De-duplicate while retaining deterministic baseline order.
    required = list({tuple(sorted(item.items())): item for item in required}.values())
    if n_trials < len(required):
        raise ValueError(f"coarse.trials={n_trials} cannot cover {len(required)} required equal/single-signal/leave-one-out baselines")

    rng = np.random.default_rng(seed)
    if len(combinations) > n_trials:
        rest = [c for c in combinations if c not in required]
        chosen = rng.choice(len(rest), size=n_trials - len(required), replace=False)
        combinations = required + [rest[i] for i in chosen]
    else:
        combinations = required + [c for c in combinations if c not in required]
    return combinations[:n_trials]


def select_images(
    frame: pd.DataFrame,
    weights: dict[str, float],
    config,
    intended_by_id: dict[str, dict],
) -> tuple[list[str], list[dict]]:
    """§4.8 ordered policy: quality floor FIRST, then per-label quota.

    A poor image is never accepted merely because its pathology is rare — rarity changes how many
    images compete for a quota, never whether the floor applies.
    """
    selection_cfg = config.selection
    scored = frame.copy()
    scored["composite"] = composite_score(scored, weights)

    rejected: list[dict] = []

    # ---- Step 1: absolute quality floor, applied to everything.
    floor_value = np.percentile(scored["composite"], float(selection_cfg.quality_floor_percentile))
    below_floor = scored["composite"] < floor_value
    for image_id in scored.loc[below_floor, "image_id"]:
        rejected.append({"image_id": image_id, "reason": "below_quality_floor"})

    # Memorized near-duplicates are rejected outright: they add no new information and carry a
    # patient-privacy risk that no quota should be able to override.
    eligible = scored.loc[~below_floor].copy()
    if "novelty_is_near_duplicate" in eligible.columns:
        memorized = eligible["novelty_is_near_duplicate"].astype(bool)
        for image_id in eligible.loc[memorized, "image_id"]:
            rejected.append({"image_id": image_id, "reason": "near_duplicate_memorization"})
        eligible = eligible.loc[~memorized]

    # ---- Step 2: per-label quota among floor-passing images.
    preference = str(selection_cfg.uncertainty_band_preference)
    if preference != "any" and "uncertainty_band" in eligible.columns:
        # Preference re-ranks within the quota; it never overrides the floor.
        eligible["band_bonus"] = (eligible["uncertainty_band"] == preference).astype(float) * 0.05
        eligible["composite"] = eligible["composite"] + eligible["band_bonus"]

    min_per_label = int(selection_cfg.min_accepted_per_label)
    max_per_label = int(selection_cfg.max_accepted_per_label)

    accepted: set[str] = set()
    per_label_counts: dict[str, int] = {}

    for label in PRIMARY_ENDPOINT_LABELS:
        label_ids = [
            image_id
            for image_id in eligible["image_id"]
            if int(intended_by_id.get(image_id, {}).get(label, 0)) == 1
        ]
        if not label_ids:
            per_label_counts[label] = 0
            continue
        subset = eligible[eligible["image_id"].isin(label_ids)].nlargest(max_per_label, "composite")
        chosen = subset["image_id"].tolist()
        accepted.update(chosen)
        per_label_counts[label] = len(chosen)

    # No-Finding recipes compete as their own group.
    no_finding_ids = [
        image_id
        for image_id in eligible["image_id"]
        if sum(int(v) for v in intended_by_id.get(image_id, {}).values()) == 0
    ]
    if no_finding_ids:
        subset = eligible[eligible["image_id"].isin(no_finding_ids)].nlargest(max_per_label, "composite")
        accepted.update(subset["image_id"].tolist())
        per_label_counts["__no_finding__"] = len(subset)

    for image_id in eligible["image_id"]:
        if image_id not in accepted:
            rejected.append({"image_id": image_id, "reason": "did_not_win_class_quota"})

    starved = {
        label: count
        for label, count in per_label_counts.items()
        if 0 < count < min_per_label
    }
    if starved:
        print(
            f"  NOTE: {len(starved)} label(s) below min_accepted_per_label={min_per_label}: "
            f"{starved}\n  (not back-filled — the quality floor is absolute, §4.8)",
            flush=True,
        )

    return sorted(accepted), rejected


def build_intended_lookup(config) -> dict[str, dict]:
    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    manifest_path = stage2_paths(stage2_cfg, str(config.split_namespace))["manifest_path"]
    if not manifest_path.is_file():
        raise SystemExit(f"UPSTREAM GATE: missing generation manifest {manifest_path}")
    lookup = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                lookup[entry["image_id"]] = entry["intended_label_vector"]
    return lookup


def run_proxy_trial(
    weights: dict[str, float],
    merged: pd.DataFrame,
    intended_by_id: dict[str, dict],
    config,
    namespace: str,
    seed: int,
    fold: int,
    trial_key: str,
) -> float:
    """One proxy evaluation (§4.7): train on classifier_train + candidate subset, evaluate on
    asism_tuning_heldout. Returns macro-AUROC over the primary endpoint labels."""
    from scripts.utils.classifier import (
        TrainingBudget,
        evaluate_classifier,
        records_from_split,
        records_from_synthetic_manifest,
        train_classifier,
    )

    stage1_cfg = load_stage1_config()
    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    tuning = config.tuning

    selected_ids, _ = select_images(merged, weights, config, intended_by_id)

    # Real component is classifier_train — NOT gen_train (§4.7 frozen data flow).
    real_frame = load_split(
        "classifier_train", namespace, purpose="schema_validation", caller="asism_proxy"
    )
    real_records = records_from_split(
        real_frame, Path(stage1_cfg.paths.images_dir) / namespace / "classifier_train"
    )
    s2paths = stage2_paths(stage2_cfg, namespace)
    synthetic_records = records_from_synthetic_manifest(s2paths["manifest_path"], s2paths["images_dir"], selected_ids)

    eval_frame = load_split(
        "asism_tuning_heldout", namespace, purpose="schema_validation", caller="asism_proxy"
    )
    fold_manifest_hash = None
    if fold > 0:
        fold_manifest = read_json(Path(config.paths.fold_manifest))
        fold_key = str(fold - 1)
        if fold_key not in fold_manifest["folds"]:
            raise SystemExit(f"ASISM fold {fold} is absent from the frozen fold manifest")
        patient_ids = set(fold_manifest["folds"][fold_key]["patient_ids"])
        eval_frame = eval_frame[eval_frame["patient_id"].astype(str).isin(patient_ids)].copy()
        fold_manifest_hash = fold_manifest["fold_manifest_hash"]
    eval_records = records_from_split(
        eval_frame, Path(stage1_cfg.paths.images_dir) / namespace / "asism_tuning_heldout"
    )

    budget = TrainingBudget(
        max_steps=int(tuning.coarse.proxy_max_steps) if fold == 0 else int(tuning.shortlist.proxy_max_steps),
        batch_size=int(tuning.proxy.batch_size),
        learning_rate=float(tuning.proxy.learning_rate),
        seed=seed,
        eval_every_n_steps=0,  # no candidate-specific early stopping (§4.7)
    )
    checkpoint_path = Path(config.paths.asism_dir) / "proxy_checkpoints" / f"{hash_dict({'trial_key': trial_key}, length=32)}.pt"
    checkpoint_provenance = {
        "split_manifest_hash": split_provenance(namespace)["split_manifest_hash"],
        "split_namespace": namespace, "dataset_id": trial_key,
        "condition_or_draw_id": trial_key, "config_hash": hash_dict({"weights": weights, "budget": budget.__dict__}, length=64),
        "label_policy": "chexpert_mask_uncertain_v1", "fold_manifest_hash": fold_manifest_hash,
        "code_identity_sha256": config._expected_score_provenance["code_identity_sha256"],
    }
    model, _, _ = train_classifier(
        train_records=real_records + synthetic_records,
        val_records=[],
        budget=budget,
        dropout_p=float(tuning.proxy.dropout_p),
        pretrained_source=str(tuning.proxy.pretrained_source),
        resolution=int(tuning.proxy.resolution),
        progress_desc=f"proxy s{seed}f{fold}",
        checkpoint_path=checkpoint_path,
        resume_from=checkpoint_path if checkpoint_path.is_file() else None,
        checkpoint_metadata=checkpoint_provenance,
    )
    metrics = evaluate_classifier(model, eval_records, int(tuning.proxy.resolution))
    return float(metrics["macro_auroc"]), fold_manifest_hash


def append_search_log(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def completed_trials(path: Path, expected_provenance: dict) -> dict[str, float]:
    """Resume at trial granularity (§4.7): an interrupted search re-reads its own log."""
    if not path.is_file():
        return {}
    done = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                for key, value in expected_provenance.items():
                    if entry.get(key) != value:
                        raise SystemExit(f"Stale ASISM search log at {path}:{key}: expected {value!r}, got {entry.get(key)!r}")
                done[entry["trial_key"]] = entry["proxy_macro_auroc"]
    return done


def phase_tune(config, namespace: str) -> int:
    estimate = enforce_compute_budget(config)

    gonogo_path = Path(config.paths.gonogo_report)
    if not gonogo_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: Go/No-Go report not found at {gonogo_path}\n"
            "Signals must clear the gate before tuning (§4.6): python scripts/asism/02_gonogo.py"
        )
    gonogo = read_json(gonogo_path)
    for key, value in config._expected_score_provenance.items():
        if gonogo.get(key) != value:
            raise SystemExit(f"Go/No-Go provenance mismatch at {key}: expected {value!r}, got {gonogo.get(key)!r}")
    surviving = list(gonogo["surviving_signals"])
    if not surviving:
        raise SystemExit("No signals survived the Go/No-Go gate — nothing to tune.")

    merged = load_merged_scores(config, surviving)
    intended_by_id = build_intended_lookup(config)
    search_log = Path(config.paths.search_log)
    done = completed_trials(search_log, dict(config._expected_score_provenance))

    tuning = config.tuning
    from scripts.utils.splits import freeze_patient_folds
    fold_path = Path(config.paths.fold_manifest)
    if not fold_path.exists():
        fold_frame = load_split("asism_tuning_heldout", namespace, purpose="schema_validation", caller="asism_fold_freeze")
        freeze_patient_folds(fold_path, fold_frame["patient_id"].astype(str).tolist(), int(tuning.shortlist.folds), int(tuning.search_seed), split_provenance(namespace)["split_manifest_hash"])
    fold_manifest = read_json(fold_path)
    grid = candidate_weight_grid(surviving, int(tuning.coarse.trials), int(tuning.search_seed))

    print(f"Compute estimate: {estimate['total_proxy_runs']} runs, "
          f"{estimate['estimated_gpu_hours']}h (budget {estimate['max_gpu_hours']}h)", flush=True)
    print(f"Surviving signals: {surviving}", flush=True)
    print(f"\nStage A — coarse screening ({len(grid)} candidates)", flush=True)

    coarse_results = []
    for index, weights in enumerate(grid):
        key = f"coarse:{index}:{hash_dict(weights)}"
        if key in done:
            score = done[key]
            print(f"  [{index}] resumed  {weights} -> {score:.4f}", flush=True)
        else:
            score, _ = run_proxy_trial(
                weights, merged, intended_by_id, config, namespace,
                seed=int(tuning.search_seed), fold=0, trial_key=key,
            )
            append_search_log(search_log, {
                "trial_key": key, "phase": "coarse", "weights": weights,
                "proxy_macro_auroc": score, "seed": int(tuning.search_seed), "fold": 0,
                **dict(config._expected_score_provenance),
            })
            print(f"  [{index}] {weights} -> {score:.4f}", flush=True)
        coarse_results.append({"weights": weights, "score": score})

    coarse_results.sort(key=lambda item: item["score"], reverse=True)
    shortlist = coarse_results[: int(tuning.shortlist.size)]
    print(f"\nStage B — shortlist validation ({len(shortlist)} candidates)", flush=True)

    shortlist_results = []
    for index, candidate in enumerate(shortlist):
        scores = []
        for fold in range(1, int(tuning.shortlist.folds) + 1):
            for seed_offset in range(int(tuning.shortlist.seeds)):
                seed = int(tuning.search_seed) + seed_offset
                key = f"shortlist:{index}:{fold}:{seed}:{hash_dict(candidate['weights'])}"
                if key in done:
                    scores.append(done[key])
                    continue
                score, fold_hash = run_proxy_trial(
                    candidate["weights"], merged, intended_by_id, config, namespace,
                    seed=seed, fold=fold, trial_key=key,
                )
                append_search_log(search_log, {
                    "trial_key": key, "phase": "shortlist", "weights": candidate["weights"],
                    "proxy_macro_auroc": score, "seed": seed, "fold": fold,
                    "fold_manifest_hash": fold_hash,
                    **dict(config._expected_score_provenance),
                })
                scores.append(score)
        mean_score = float(np.mean(scores))
        shortlist_results.append({
            "weights": candidate["weights"], "mean_score": mean_score,
            "std_score": float(np.std(scores)), "n_runs": len(scores),
        })
        print(f"  [{index}] {candidate['weights']} -> {mean_score:.4f} "
              f"(+/-{np.std(scores):.4f}, n={len(scores)})", flush=True)

    # Frozen selection rule: best mean; ties within the noise band resolve to the SIMPLER
    # configuration (fewer active signals, then smaller total weight).
    band = float(tuning.tie_noise_band)
    finite_results = [item for item in shortlist_results if np.isfinite(item["mean_score"])]
    selection_fallback = None
    if finite_results:
        best_score = max(item["mean_score"] for item in finite_results)
        tied = [item for item in finite_results if best_score - item["mean_score"] <= band]
    elif os.environ.get("THESIS_SMOKE_MODE") == "1":
        # A tiny fixture fold can contain only one class for every label, making macro-AUROC
        # undefined. Keep the engineering smoke moving with the already-finite coarse result;
        # production remains fail-closed below.
        best_coarse = max(coarse_results, key=lambda item: item["score"])
        tied = [{**shortlist_results[0], "mean_score": float(best_coarse["score"])}]
        selection_fallback = "smoke_only_all_shortlist_aurocs_undefined_used_best_coarse_score"
    else:
        raise SystemExit(
            "ASISM tuning produced no finite shortlist macro-AUROC values. Increase held-out "
            "class support or revise the predeclared fold plan; no production winner was frozen."
        )
    winner = min(
        tied,
        key=lambda item: (
            sum(1 for w in item["weights"].values() if w > 0),
            sum(item["weights"].values()),
        ),
    )

    frozen = {
        "schema_version": SCHEMA_VERSION,
        "stage": "asism_frozen",
        "surviving_signals": surviving,
        "ablation_only_signals": gonogo.get("ablation_only_signals", []),
        "excluded_signals": gonogo.get("excluded_signals", []),
        "asism_variant_status": gonogo.get("asism_variant_status"),
        "frozen_weights": winner["weights"],
        "selection_policy": OmegaConf.to_container(config.selection, resolve=True),
        "tie_break_rule": "best mean proxy macro-AUROC; ties within noise band -> simpler config",
        "tie_noise_band": band,
        "n_tied_candidates": len(tied),
        "selection_fallback": selection_fallback,
        "winner_mean_proxy_macro_auroc": winner["mean_score"],
        "shortlist_results": shortlist_results,
        "coarse_results": coarse_results,
        "required_baselines": {
            "equal_weight": [item for item in coarse_results if len(set(item["weights"].values())) == 1 and any(item["weights"].values())],
            "individual_signal": [item for item in coarse_results if sum(value > 0 for value in item["weights"].values()) == 1],
            "leave_one_signal_out": [item for item in coarse_results if sum(value == 0 for value in item["weights"].values()) == 1],
            "uncertainty_band_only": [item for item in coarse_results if item["weights"].get("__uncertainty_band_only__", 0) > 0],
        },
        "compute_accounting": estimate,
        "proxy_data_flow": {
            "train_real": "classifier_train",
            "train_synthetic": "candidate subset (actually constructed)",
            "evaluate": "asism_tuning_heldout",
            "classifier_val_used_for_ranking": False,
            "final_eval_heldout_used": False,
        },
        "search_log": str(search_log),
        "fold_manifest_hash": fold_manifest["fold_manifest_hash"],
        "split_provenance": split_provenance(namespace),
        "git_commit_hash": get_git_commit_hash(),
        "frozen": True,
        **dict(config._expected_score_provenance),
    }
    write_frozen_json(Path(config.paths.frozen_manifest), frozen)

    print(f"\nFROZEN weights: {winner['weights']}", flush=True)
    print(f"  proxy macro-AUROC: {winner['mean_score']:.4f}", flush=True)
    print(f"  manifest -> {config.paths.frozen_manifest}", flush=True)
    return 0


def phase_select(config, namespace: str) -> int:
    frozen_path = Path(config.paths.frozen_manifest)
    if not frozen_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: ASISM is not frozen ({frozen_path} missing).\n"
            "Run --phase tune first."
        )
    frozen = read_json(frozen_path)
    for key, value in config._expected_score_provenance.items():
        if frozen.get(key) != value:
            raise SystemExit(f"Frozen ASISM provenance mismatch at {key}: expected {value!r}, got {frozen.get(key)!r}")
    if not frozen.get("frozen"):
        raise SystemExit("ASISM manifest exists but is not marked frozen.")

    merged = load_merged_scores(config, frozen["surviving_signals"])
    intended_by_id = build_intended_lookup(config)
    selected, rejected = select_images(merged, frozen["frozen_weights"], config, intended_by_id)

    selected_path = Path(config.paths.selected_manifest)
    rejected_path = Path(config.paths.rejected_log)
    selected_path.parent.mkdir(parents=True, exist_ok=True)

    def atomic_jsonl(path: Path, records: list[dict]) -> None:
        fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)

    atomic_jsonl(
        selected_path,
        [
            {
                "image_id": image_id,
                "intended_label_vector": intended_by_id.get(image_id, {}),
                "asism_frozen_weights": frozen["frozen_weights"],
                "asism_manifest_hash": hash_dict(frozen["frozen_weights"]),
            }
            for image_id in selected
        ],
    )
    atomic_jsonl(rejected_path, rejected)

    reasons = pd.Series([r["reason"] for r in rejected]).value_counts().to_dict() if rejected else {}
    write_json(
        selected_path.with_name("selection_summary.json"),
        {
            "n_candidates": len(merged),
            "n_selected": len(selected),
            "n_rejected": len(rejected),
            "rejection_reasons": reasons,
            "frozen_weights": frozen["frozen_weights"],
            "selection_policy": frozen["selection_policy"],
            "split_provenance": split_provenance(namespace),
            "git_commit_hash": get_git_commit_hash(),
        },
    )

    print(f"Candidates: {len(merged)}", flush=True)
    print(f"Selected:   {len(selected)}", flush=True)
    print(f"Rejected:   {len(rejected)}  {reasons}", flush=True)
    print(f"-> {selected_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["estimate", "tune", "select"], required=True)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    config = load_config()
    namespace = args.namespace or str(config.split_namespace)
    stage2_config = load_named_config("stage2_generation.yaml", "stage2")
    expected_score_provenance = asism_score_provenance(config, stage2_config, namespace)
    config.split_namespace = namespace
    paths = stage3_paths(config, namespace)
    for key, value in paths.items():
        if key in config.paths:
            config.paths[key] = str(value)
    config._expected_score_provenance = expected_score_provenance

    if args.phase == "estimate":
        estimate = compute_budget_estimate(config)
        print(json.dumps(estimate, indent=2), flush=True)
        if not estimate["within_budget"]:
            print(
                "\nOVER BUDGET — reduce the search space in configs/stage3_asism.yaml before "
                "running --phase tune.",
                flush=True,
            )
            return 2
        print("\nWithin budget. Proceed: --phase tune", flush=True)
        return 0

    if args.phase == "tune":
        return phase_tune(config, namespace)
    return phase_select(config, namespace)


if __name__ == "__main__":
    raise SystemExit(main())

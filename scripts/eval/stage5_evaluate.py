#!/usr/bin/env python3
"""Stage 5 — protected final evaluation on final_eval_heldout (docs/stages2_to_5_plan.md §8).

Refuses to run unless every precondition in §8 holds:
  - the production split manifest exists AND is frozen AND passed its support check;
  - the recorded split hash matches the current manifest;
  - the frozen experiment-protocol manifest exists (Stage 4);
  - the frozen Learned-ASISM selection manifest exists (Stage 3);
  - the required A/B/F checkpoints exist, with their hashes recorded;
  - the classification-threshold policy is frozen;
  - an explicit --final-eval-run-id is supplied.

The run id is what unlocks outcome-bearing access to final_eval_heldout through
scripts/utils/splits.assert_final_eval_access_allowed(). No other code path can obtain it.

TECHNICAL RESUME vs. METHODOLOGICAL RE-EVALUATION (§8.4): predictions are written incrementally,
and an interrupted run may resume from the same frozen artifacts and complete the same partial
prediction files without altering the protocol. That is an engineering property. What is NOT
permitted is inspecting results and then re-running under a changed method — a genuine
methodological error requires returning to the development stage, re-freezing, and declaring a NEW
run id, which produces a separate result rather than overwriting this one.

Usage:
    python scripts/eval/stage5_evaluate.py --final-eval-run-id stage5-2026-08-07-a --namespace production
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.labels import CLASSIFIER_TARGET_LABELS, build_label_arrays  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import (  # noqa: E402
    load_split,
    read_split_manifest,
    require_frozen_production_split_run,
    split_provenance,
)
from scripts.utils.artifact_contracts import current_code_identity_hash, stage3_paths, stage4_paths  # noqa: E402


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def require_smoke_final_eval_authorization(namespace: str, manifest: dict, project_root: Path) -> dict:
    """Authorize fixture or real-data engineering smoke without weakening production guards.

    The historical fixture smoke keeps its existing marker.  A real-data smoke requires a
    namespace- and split-hash-bound marker, so a marker created for one development split cannot
    unlock another split (and a production-class namespace is always refused).
    """
    if manifest.get("namespace_class") != "dev":
        raise SystemExit("Smoke Stage 5 requires a dev-class split")

    if namespace == "dev-smoke-v1" and (project_root / "SMOKE_ONLY.json").is_file():
        return {"kind": "fixture", "upstream_code_identity_sha256": current_code_identity_hash()}

    marker_path = project_root / "REAL_DATA_SMOKE_ONLY.json"
    if not marker_path.is_file():
        raise SystemExit(
            "Real-data smoke Stage 5 requires REAL_DATA_SMOKE_ONLY.json in PROJECT_ROOT"
        )
    marker = read_json(marker_path)
    expected = {
        "schema_version": 1,
        "purpose": "real-data-engineering-smoke-only",
        "namespace": namespace,
        "split_manifest_hash": manifest.get("manifest_hash"),
        "not_scientific_evidence": True,
    }
    mismatches = {
        key: {"expected": value, "actual": marker.get(key)}
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"Real-data smoke authorization marker mismatch: {mismatches}")
    upstream_identity = marker.get("upstream_code_identity_sha256")
    if not isinstance(upstream_identity, str) or len(upstream_identity) != 64:
        raise SystemExit(
            "Real-data smoke authorization marker requires a 64-character "
            "upstream_code_identity_sha256"
        )
    return {"kind": "real_data", "upstream_code_identity_sha256": upstream_identity}


def enforce_preconditions(namespace: str, run_id: str) -> dict:
    """All of §8's execution guards. Any failure aborts before final_eval_heldout is opened."""
    failures: list[str] = []
    evidence: dict = {"final_eval_run_id": run_id, "namespace": namespace}

    if not run_id:
        failures.append("no --final-eval-run-id supplied")

    # 1-2. Production splits frozen, support check passed, hash recorded.
    try:
        smoke_mode = os.environ.get("THESIS_SMOKE_MODE") == "1"
        if smoke_mode:
            manifest = read_split_manifest(namespace)
            smoke_authorization = require_smoke_final_eval_authorization(
                namespace, manifest, Path(os.environ.get("PROJECT_ROOT", "."))
            )
            evidence["smoke_only_not_scientific_evidence"] = True
            evidence["smoke_kind"] = smoke_authorization["kind"]
            evidence["upstream_code_identity_sha256"] = smoke_authorization[
                "upstream_code_identity_sha256"
            ]
        else:
            manifest = require_frozen_production_split_run(namespace)
        evidence["split_manifest_hash"] = manifest.get("manifest_hash")
        evidence["split_support_check_passed"] = manifest.get("support_check", {}).get("passed")
        evidence["split_frozen"] = manifest.get("frozen")
    except SystemExit as exc:
        failures.append(f"split manifest: {exc}")

    expected_code_identity = evidence.get(
        "upstream_code_identity_sha256", current_code_identity_hash()
    )

    stage4_cfg = load_named_config("stage4_classifier.yaml", "stage4")
    stage3_cfg = load_named_config("stage3_asism.yaml", "stage3")
    for key, value in stage4_paths(stage4_cfg, namespace).items():
        stage4_cfg.paths[key] = str(value)
    for key, value in stage3_paths(stage3_cfg, namespace).items():
        if key in stage3_cfg.paths:
            stage3_cfg.paths[key] = str(value)

    # 3. Frozen experiment protocol.
    protocol_path = Path(stage4_cfg.paths.protocol_manifest)
    if not protocol_path.is_file():
        failures.append(f"frozen experiment-protocol manifest missing: {protocol_path}")
    else:
        protocol = read_json(protocol_path)
        if not protocol.get("frozen"):
            failures.append("experiment-protocol manifest exists but is not marked frozen")
        evidence["protocol_manifest_hash"] = sha256_file(protocol_path)
        if protocol.get("split_provenance", {}).get("split_manifest_hash") != evidence.get("split_manifest_hash"):
            failures.append("experiment protocol split hash differs from the requested frozen split")
        if protocol.get("code_identity_sha256") != expected_code_identity:
            failures.append("experiment protocol code identity differs from the current source tree")
        # 6. Threshold policy frozen.
        if not protocol.get("threshold_policy"):
            failures.append("classification-threshold policy is not recorded in the frozen protocol")
        else:
            evidence["threshold_policy"] = protocol["threshold_policy"]
        frozen_component_hashes = {}
        for key in ("label_policy_manifest", "threshold_policy_manifest", "seed_plan_manifest", "d_matching_policy_manifest"):
            component_path = Path(stage4_cfg.paths[key])
            if not component_path.is_file():
                failures.append(f"frozen methodology component missing: {component_path}")
                continue
            component = read_json(component_path)
            if not component.get("frozen") or component.get("protocol_manifest_hash") != evidence["protocol_manifest_hash"]:
                failures.append(f"frozen methodology component incompatible: {component_path}")
            frozen_component_hashes[key] = sha256_file(component_path)
        evidence["frozen_component_hashes"] = frozen_component_hashes

    # 4. Learned-ASISM selection is the sole Stage-3 selector in this thesis protocol.
    asism_path = Path(stage3_cfg.paths.adaptive_selection_manifest)
    if not asism_path.is_file():
        failures.append(f"ASISM freeze manifest missing: {asism_path}")
    else:
        asism = read_json(asism_path)
        if not asism.get("frozen") or asism.get("method") != "class_aware_adaptive_threshold_v2":
            failures.append("Learned ASISM selection manifest is not a compatible frozen artifact")
        evidence["learned_asism_manifest_hash"] = sha256_file(asism_path)

    # F is the finalized Learned-ASISM selector.
    if "F" in list(stage4_cfg.get("conditions", [])):
        learned_path = Path(stage3_cfg.paths.adaptive_selection_manifest)
        if not learned_path.is_file():
            failures.append(f"adaptive learned ASISM selection manifest missing: {learned_path}")
        else:
            learned = read_json(learned_path)
            if not learned.get("frozen") or learned.get("method") != "class_aware_adaptive_threshold_v2":
                failures.append("adaptive learned ASISM selection manifest is not a compatible frozen artifact")

    # 5. Required checkpoints.
    results_path = Path(stage4_cfg.paths.results_dir) / "stage4_training_results.json"
    if not results_path.is_file():
        failures.append(f"Stage 4 training results missing: {results_path}")
    else:
        results = read_json(results_path)["results"]
        import torch
        checkpoint_hashes = {}
        missing = []
        for entry in results:
            path = Path(entry["checkpoint"])
            if path.is_file():
                checkpoint_hashes[entry["tag"]] = sha256_file(path)
                payload = torch.load(path, map_location="cpu", weights_only=False)
                provenance = payload.get("provenance", {})
                if payload.get("checkpoint_role") != "best_validation_model":
                    failures.append(f"checkpoint is not a persisted best-validation model: {entry['tag']}")
                if provenance.get("split_manifest_hash") != evidence.get("split_manifest_hash"):
                    failures.append(f"checkpoint split provenance mismatch: {entry['tag']}")
                if provenance.get("code_identity_sha256") != expected_code_identity:
                    failures.append(f"checkpoint code identity mismatch: {entry['tag']}")
            else:
                missing.append(entry["tag"])
        if missing:
            failures.append(f"{len(missing)} required checkpoint(s) missing: {missing[:5]}")
        expected = read_json(results_path).get("run_plan", {}).get("total_runs")
        if expected and len(results) != expected:
            failures.append(f"expected {expected} runs, found {len(results)}")
        evidence["checkpoint_hashes"] = checkpoint_hashes
        evidence["n_checkpoints"] = len(checkpoint_hashes)

    if failures:
        raise SystemExit(
            "STAGE 5 REFUSED — preconditions not met (docs/stages2_to_5_plan.md §8):\n"
            + "\n".join(f"  - {failure}" for failure in failures)
            + "\n\nfinal_eval_heldout was NOT opened."
        )
    return evidence


def predict_condition(entry: dict, records: list[dict], stage4_cfg) -> np.ndarray:
    from scripts.utils.classifier import build_model, predict_probabilities, require_torch

    torch = require_torch()
    payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    model = build_model(
        len(CLASSIFIER_TARGET_LABELS),
        float(stage4_cfg.model.dropout_p),
        str(stage4_cfg.model.pretrained_source),
    )
    model.load_state_dict(payload["model_state"])
    probabilities, _ = predict_probabilities(
        model, records, int(stage4_cfg.model.resolution)
    )
    return probabilities


def select_thresholds(entry: dict, namespace: str, stage4_cfg) -> dict[str, float]:
    """Thresholds chosen on classifier_val ONLY (frozen policy), never on final_eval_heldout."""
    from scripts.utils.classifier import records_from_split
    from scripts.utils.metrics import sensitivity_specificity_f1

    stage1_cfg = load_stage1_config()
    val_frame = load_split("classifier_val", namespace, purpose="schema_validation", caller="stage5")
    val_records = records_from_split(
        val_frame, Path(stage1_cfg.paths.images_dir) / namespace / "classifier_val"
    )
    probabilities = predict_condition(entry, val_records, stage4_cfg)

    frame = pd.DataFrame([record["labels"] for record in val_records])
    for label in CLASSIFIER_TARGET_LABELS:
        if label not in frame.columns:
            frame[label] = np.nan
    _, targets, masks = build_label_arrays(frame[CLASSIFIER_TARGET_LABELS], CLASSIFIER_TARGET_LABELS)

    thresholds = {}
    for index, label in enumerate(CLASSIFIER_TARGET_LABELS):
        mask = masks[:, index].astype(bool)
        if mask.sum() == 0:
            thresholds[label] = 0.5
            continue
        y_true = targets[mask, index].astype(int)
        y_score = probabilities[mask, index]
        best_threshold, best_j = 0.5, -np.inf
        for candidate in np.linspace(0.05, 0.95, 19):
            sensitivity, specificity, _ = sensitivity_specificity_f1(y_score, y_true, candidate)
            if np.isnan(sensitivity) or np.isnan(specificity):
                continue
            youden_j = sensitivity + specificity - 1
            if youden_j > best_j:
                best_j, best_threshold = youden_j, float(candidate)
        thresholds[label] = best_threshold
    return thresholds


def validate_prediction_frame(frame: pd.DataFrame, expected_image_ids: list[str], expected_patient_ids: list[str]) -> None:
    required = ["image_id", "patient_id", *CLASSIFIER_TARGET_LABELS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction schema missing columns: {missing}")
    if len(frame) != len(expected_image_ids):
        raise ValueError(f"wrong prediction row count: {len(frame)} != {len(expected_image_ids)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("duplicate prediction image IDs")
    if set(frame["image_id"].astype(str)) != set(map(str, expected_image_ids)):
        raise ValueError("prediction image-ID set mismatch")
    if set(frame["patient_id"].astype(str)) != set(map(str, expected_patient_ids)):
        raise ValueError("prediction patient-ID set mismatch")


def publish_predictions_atomic(path: Path, frame: pd.DataFrame, provenance: dict, expected_image_ids: list[str], expected_patient_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.parquet", dir=path.parent)
    os.close(fd)
    temporary = Path(temp_name)
    try:
        frame.to_parquet(temporary, index=False)
        validate_prediction_frame(pd.read_parquet(temporary), expected_image_ids, expected_patient_ids)
        os.replace(temporary, path)
        sidecar = {"schema_version": 1, "status": "complete", "prediction_sha256": sha256_file(path),
                   "row_count": len(frame), "image_id_set_hash": hash_dict({"ids": sorted(map(str, expected_image_ids))}, length=64),
                   "patient_id_set_hash": hash_dict({"ids": sorted(set(map(str, expected_patient_ids)))}, length=64), **provenance}
        write_json(path.with_suffix(".complete.json"), sidecar)
    finally:
        temporary.unlink(missing_ok=True)


def validate_completed_predictions(path: Path, expected_provenance: dict, expected_image_ids: list[str], expected_patient_ids: list[str]) -> None:
    sidecar_path = path.with_suffix(".complete.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise ValueError("prediction file and completion sidecar are both required")
    sidecar = read_json(sidecar_path)
    if sidecar.get("prediction_sha256") != sha256_file(path):
        raise ValueError("prediction file hash mismatch/corruption")
    mismatch = {k: (sidecar.get(k), v) for k, v in expected_provenance.items() if sidecar.get(k) != v}
    if mismatch:
        raise ValueError(f"prediction provenance mismatch: {mismatch}")
    validate_prediction_frame(pd.read_parquet(path), expected_image_ids, expected_patient_ids)


def register_final_evaluation(context_path: Path, registration: dict) -> dict:
    """Register a new run or validate an identical technical resume."""
    identity_keys = (
        "final_eval_run_id", "namespace", "split_manifest_hash", "asism_manifest_hash",
        "protocol_manifest_hash", "checkpoint_hashes", "threshold_policy_hash",
        "code_git_version", "code_identity_sha256",
    )
    if context_path.exists():
        existing = read_json(context_path)
        for key in identity_keys:
            if existing.get(key) != registration.get(key):
                raise SystemExit(f"Registered final-evaluation context is incompatible at {key}; technical resume refused")
        return existing
    for prior_path in context_path.parent.parent.glob("*/final_eval_registration.json"):
        prior = read_json(prior_path)
        if prior.get("split_manifest_hash") == registration["split_manifest_hash"] and prior.get("status") in {"outcome_accessed", "complete"}:
            raise SystemExit("Final outcomes were already accessed under another run ID; a corrected run requires a new linked protocol freeze")
    write_json(context_path, registration)
    return registration


def mark_final_outcome_access(context_path: Path) -> dict:
    registration = read_json(context_path)
    registration["status"] = "outcome_accessed"
    registration["first_outcome_access_at_utc"] = registration.get("first_outcome_access_at_utc") or datetime.now(timezone.utc).isoformat()
    write_json(context_path, registration)
    return registration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-eval-run-id", required=True)
    parser.add_argument("--namespace", default="production")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()

    run_id = args.final_eval_run_id
    namespace = args.namespace

    print("Stage 5 precondition checks...", flush=True)
    evidence = enforce_preconditions(namespace, run_id)
    print("  all preconditions satisfied", flush=True)

    stage4_cfg = load_named_config("stage4_classifier.yaml", "stage4")
    for key, value in stage4_paths(stage4_cfg, namespace).items():
        stage4_cfg.paths[key] = str(value)
    stage1_cfg = load_stage1_config()

    output_dir = Path(stage4_cfg.paths.project_root) / "outputs" / "stage5" / namespace / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    results = read_json(Path(stage4_cfg.paths.results_dir) / "stage4_training_results.json")["results"]
    # Freeze concrete classifier_val-selected values before any outcome-bearing final-data access.
    thresholds_path = output_dir / "frozen_threshold_values.json"
    if not thresholds_path.exists():
        threshold_values = {entry["tag"]: select_thresholds(entry, namespace, stage4_cfg) for entry in results}
        write_json(thresholds_path, {"frozen": True, "selected_on": "classifier_val", "values": threshold_values,
                                    "policy": dict(stage4_cfg.threshold_policy)})
    threshold_hash = sha256_file(thresholds_path)

    context_path = output_dir / "final_eval_registration.json"
    registration = {
        "final_eval_run_id": run_id, "namespace": namespace, "split_manifest_hash": evidence["split_manifest_hash"],
        "asism_manifest_hash": evidence["learned_asism_manifest_hash"], "protocol_manifest_hash": evidence["protocol_manifest_hash"],
        "checkpoint_hashes": evidence["checkpoint_hashes"], "threshold_policy_hash": threshold_hash,
        "code_git_version": get_git_commit_hash(), "code_identity_sha256": current_code_identity_hash(),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(), "status": "registered",
    }
    registration = register_final_evaluation(context_path, registration)

    # This is the ONLY outcome-bearing read of final_eval_heldout in the codebase, and it is
    # unlocked solely by the explicit run id (§1.6).
    from scripts.utils.classifier import records_from_split

    eval_frame = load_split(
        "final_eval_heldout",
        namespace,
        purpose="load_labels_for_evaluation",
        final_eval_run_id=run_id,
        final_eval_context_path=context_path,
        caller="stage5_evaluate",
    )
    registration = mark_final_outcome_access(context_path)
    eval_records = records_from_split(
        eval_frame, Path(stage1_cfg.paths.images_dir) / namespace / "final_eval_heldout"
    )
    if not eval_records:
        raise SystemExit(
            "UPSTREAM GATE: no preprocessed final_eval_heldout images found. "
            "Stage 1 preprocessing must have run for this split."
        )
    print(f"  final_eval_heldout: {len(eval_records)} images, "
          f"{eval_frame['patient_id'].nunique()} patients", flush=True)

    # Technical resume: per-condition prediction files are written incrementally and reused.
    for entry in results:
        prediction_path = predictions_dir / f"{entry['tag']}.parquet"
        prediction_provenance = {
            "checkpoint_hash": evidence["checkpoint_hashes"][entry["tag"]],
            "split_hash": evidence["split_manifest_hash"], "protocol_hash": evidence["protocol_manifest_hash"],
            "threshold_hash": threshold_hash, "code_version": get_git_commit_hash(),
        }
        if prediction_path.is_file():
            try:
                validate_completed_predictions(prediction_path, prediction_provenance,
                                               [r["image_id"] for r in eval_records], [r["patient_id"] for r in eval_records])
            except Exception as exc:
                raise SystemExit(f"Existing predictions for {entry['tag']} are partial/corrupt/incompatible; refusing reuse: {exc}") from exc
            print(f"  [{entry['tag']}] safely resumed from validated predictions", flush=True)
            continue
        print(f"  [{entry['tag']}] predicting...", flush=True)
        probabilities = predict_condition(entry, eval_records, stage4_cfg)
        frame = pd.DataFrame(probabilities, columns=CLASSIFIER_TARGET_LABELS)
        frame.insert(0, "image_id", [record["image_id"] for record in eval_records])
        frame.insert(1, "patient_id", [record["patient_id"] for record in eval_records])
        publish_predictions_atomic(prediction_path, frame, prediction_provenance,
                                   [r["image_id"] for r in eval_records], [r["patient_id"] for r in eval_records])

    write_json(
        output_dir / "final_eval_run_manifest.json",
        {
            "final_eval_run_id": run_id,
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "namespace": namespace,
            "preconditions": evidence,
            "n_eval_images": len(eval_records),
            "n_eval_patients": int(eval_frame["patient_id"].nunique()),
            "predictions_dir": str(predictions_dir),
            "split_provenance": split_provenance(namespace),
            "git_commit_hash": get_git_commit_hash(),
            "protocol_note": (
                "One frozen final-evaluation protocol. Technical resume is permitted; "
                "methodological re-evaluation is not. A genuine methodological error requires "
                "re-freezing and a NEW run id, reported alongside this one."
            ),
        },
    )
    registration["status"] = "complete"
    registration["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(context_path, registration)

    print(f"\nPredictions written -> {predictions_dir}", flush=True)
    print("Next (no GPU required):", flush=True)
    print(f"  python scripts/eval/compare_conditions.py --run-dir {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

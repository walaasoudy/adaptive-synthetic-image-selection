"""Namespace-safe Stage 2/3 artifact paths and strict provenance contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.utils.manifest import code_identity, hash_dict, read_json, sha256_file
from scripts.utils.splits import read_split_manifest, resolve_splits_dir


class ArtifactContractError(SystemExit):
    pass


def validate_namespace_run_id(namespace: str, namespace_class: str) -> None:
    prefix = f"{namespace_class}-"
    if namespace_class not in {"dev", "production"} or not namespace.startswith(prefix) or len(namespace) <= len(prefix):
        raise ArtifactContractError(
            f"Invalid immutable split run ID {namespace!r}: a {namespace_class!r} run must be versioned as {prefix}<run-or-version>"
        )


def namespace_identity(namespace: str) -> dict[str, Any]:
    manifest = read_split_manifest(namespace)
    namespace_class = manifest.get("namespace_class")
    if namespace_class not in {"dev", "production"}:
        raise ArtifactContractError(
            f"Split run {namespace!r} has no valid namespace_class; rebuild it with the current six-way builder"
        )
    validate_namespace_run_id(namespace, namespace_class)
    return {
        "split_namespace": namespace,
        "namespace_class": namespace_class,
        "split_manifest_version": manifest["manifest_version"],
        "split_manifest_hash": manifest["manifest_hash"],
        "split_manifest_file_sha256": sha256_file(resolve_splits_dir(namespace) / "split_manifest_v2.json"),
        "split_frozen": bool(manifest.get("frozen", False)),
    }


def stage2_paths(config, namespace: str) -> dict[str, Path]:
    root = Path(config.paths.synthetic_root) / namespace
    pilot = root / "pilot"
    return {
        "root": root,
        "images_dir": root / "images",
        "recipes_path": root / "label_recipes.csv",
        "recipe_decisions_path": root / "recipe_decisions.jsonl",
        "recipes_manifest": root / "recipes_manifest.json",
        "manifest_path": root / "generation_manifest.jsonl",
        "generation_provenance": root / "generation_provenance.json",
        "generation_completion": root / "generation_complete.json",
        "pilot_dir": pilot,
        "pilot_images": pilot / "images",
        "pilot_manifest": pilot / "pilot_generation_manifest.jsonl",
        "pilot_approval_manifest": pilot / "pilot_approval_manifest.json",
    }


def stage3_paths(config, namespace: str) -> dict[str, Path]:
    root = Path(config.paths.synthetic_root) / namespace
    asism = root / "asism"
    return {
        "root": root,
        "scores_dir": root / "scores",
        "asism_dir": asism,
        "gonogo_report": asism / "gonogo_report.json",
        "search_log": asism / "proxy_search_log.jsonl",
        "frozen_manifest": asism / "asism_frozen_manifest.json",
        "fold_manifest": asism / "asism_patient_folds.json",
        "selected_manifest": asism / "selected_manifest.jsonl",
        "rejected_log": asism / "rejected_log.jsonl",
    }


def auxiliary_checkpoint_path(config, namespace: str) -> Path:
    configured = Path(config.auxiliary_classifier.checkpoint_path)
    return configured.parent / namespace / configured.name


def stage4_paths(config, namespace: str) -> dict[str, Path]:
    checkpoints = Path(config.paths.checkpoints_dir) / namespace
    results = Path(config.paths.results_dir) / namespace
    return {
        "checkpoints_dir": checkpoints, "results_dir": results,
        "protocol_manifest": results / "frozen_experiment_protocol.json",
        "label_policy_manifest": results / "frozen_label_policy.json",
        "threshold_policy_manifest": results / "frozen_threshold_policy.json",
        "seed_plan_manifest": results / "frozen_seed_run_plan.json",
        "d_matching_policy_manifest": results / "frozen_d_matching_policy.json",
    }


def config_sha256(config_section) -> str:
    return hash_dict(OmegaConf.to_container(config_section, resolve=True), length=64)


def require_manifest_fields(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArtifactContractError(f"Missing {label} manifest: {path}")
    actual = read_json(path)
    mismatches = {key: {"expected": value, "actual": actual.get(key)} for key, value in expected.items() if actual.get(key) != value}
    if mismatches:
        raise ArtifactContractError(f"Stale/incompatible {label} artifact at {path}: {mismatches}")
    return actual


def current_code_identity_hash() -> str:
    return code_identity()["source_tree_sha256"]


def require_generation_complete(config, namespace: str) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = stage2_paths(config, namespace)
    identity = namespace_identity(namespace)
    expected = {"split_namespace": namespace, "namespace_class": identity["namespace_class"],
                "split_manifest_hash": identity["split_manifest_hash"], "status": "complete"}
    completion = require_manifest_fields(paths["generation_completion"], expected, "Stage 2 generation completion")
    if not paths["manifest_path"].is_file() or completion.get("generation_manifest_sha256") != sha256_file(paths["manifest_path"]):
        raise ArtifactContractError("Generation manifest is missing or differs from its completion record")
    if completion.get("code_identity_sha256") != current_code_identity_hash():
        raise ArtifactContractError("Generation artifacts were produced by a different source-tree identity")
    return paths, completion


def require_score_artifact(path: Path, signal: str, expected_provenance: dict[str, Any]) -> dict[str, Any]:
    sidecar = path.with_suffix(".provenance.json")
    expected = {"schema_version": 2, "signal": signal, **expected_provenance}
    manifest = require_manifest_fields(sidecar, expected, f"ASISM {signal} score")
    if not path.is_file() or manifest.get("parquet_sha256") != sha256_file(path):
        raise ArtifactContractError(f"ASISM {signal} Parquet hash mismatch: {path}")
    return manifest


def asism_score_provenance(stage3_config, stage2_config, namespace: str) -> dict[str, Any]:
    stage2, generation = require_generation_complete(stage2_config, namespace)
    return {
        **namespace_identity(namespace), "asism_config_sha256": config_sha256(stage3_config),
        "generation_manifest_sha256": sha256_file(stage2["manifest_path"]),
        "generation_completion_sha256": sha256_file(stage2["generation_completion"]),
        "lora_checkpoint_sha256": generation["lora_checkpoint_sha256"],
        "code_identity_sha256": current_code_identity_hash(),
    }

"""Split loading, namespace resolution, and the programmatic `final_eval_heldout` access guard
(docs/stages2_to_5_plan.md §1.4 and §1.6).

The guard exists because the honest version of the leakage rule is not "final_eval_heldout is never
touched" — it IS touched, exactly once, during split construction, to assign patients, assert
disjointness, run the support check, and hash the manifest. What must never happen is an
outcome-bearing read before Stage 5: loading its images, loading its labels to compute outcome
statistics, or letting any model/method decision depend on it.

So access is mediated by an explicit purpose token. Non-outcome-bearing purposes are always
allowed; outcome-bearing purposes require a Stage 5 final-evaluation run id, which only the Stage 5
entry point supplies. Any other caller asking for an outcome-bearing read fails loudly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from scripts.utils.config import load_named_config
from scripts.utils.manifest import hash_dict, read_json, sha256_file, write_frozen_json
import random

SPLIT_NAMES = [
    "gen_train",
    "gen_val",
    "classifier_train",
    "classifier_val",
    "asism_tuning_heldout",
    "final_eval_heldout",
]

FINAL_EVAL_SPLIT = "final_eval_heldout"

SPLIT_MANIFEST_FILENAME = "split_manifest_v2.json"

# Purposes that do NOT read outcomes: allowed at any time, including before Stage 5.
NON_OUTCOME_PURPOSES = {
    "split_construction",    # deterministic assignment, disjointness/partition assertions
    "support_check",         # positive/negative patient counts (a property of the split, not a result)
    "schema_validation",     # column/dtype checks
    "existence_check",       # does the file exist
    "hash_verification",     # manifest hashing / hash comparison
}

# Purposes that DO read outcomes: require an explicit Stage 5 final-evaluation run id.
OUTCOME_PURPOSES = {
    "load_images",
    "load_labels_for_evaluation",
    "compute_outcome_statistics",
    "model_selection",
}


class FinalEvalAccessViolation(SystemExit):
    """Raised (as SystemExit, so it surfaces as a clean CLI failure) when code attempts an
    outcome-bearing read of final_eval_heldout outside a declared Stage 5 evaluation run."""


def assert_final_eval_access_allowed(
    purpose: str,
    final_eval_run_id: str | None = None,
    caller: str = "unknown",
) -> None:
    """Gate every access to final_eval_heldout behind an explicit declared purpose.

    Non-outcome purposes (§1.6's enumerated split-construction activities) pass unconditionally.
    Outcome purposes require `final_eval_run_id` — supplied only by the Stage 5 entry point.
    """
    if purpose in NON_OUTCOME_PURPOSES:
        return

    if purpose not in OUTCOME_PURPOSES:
        raise FinalEvalAccessViolation(
            f"[{caller}] Unrecognized final_eval_heldout access purpose {purpose!r}. "
            f"Declare one of: non-outcome {sorted(NON_OUTCOME_PURPOSES)} "
            f"or outcome-bearing {sorted(OUTCOME_PURPOSES)}."
        )

    if not final_eval_run_id:
        raise FinalEvalAccessViolation(
            f"[{caller}] Refusing outcome-bearing access to {FINAL_EVAL_SPLIT} "
            f"(purpose={purpose!r}) without an explicit final-evaluation run id.\n"
            "final_eval_heldout may only be read for outcomes inside a declared Stage 5 run "
            "(docs/stages2_to_5_plan.md §1.6). If this is Stage 5, pass --final-eval-run-id."
        )


def load_splits_config() -> DictConfig:
    return load_named_config("splits.yaml", "splits")


def resolve_splits_dir(namespace: str, config: DictConfig | None = None) -> Path:
    """Namespaced split directory: <splits_root>/<namespace>/ (plan §1.4).

    Keeping `dev` and `production` in separate directories is what makes it structurally impossible
    for a dev-subset split to overwrite a production split.
    """
    config = config if config is not None else load_splits_config()
    if not namespace or namespace in {".", ".."} or any(ch in namespace for ch in "/\\"):
        raise ValueError(f"Invalid explicit split namespace/run id {namespace!r}")
    return Path(config.paths.splits_root) / namespace


def split_manifest_path(namespace: str, config: DictConfig | None = None) -> Path:
    return resolve_splits_dir(namespace, config) / SPLIT_MANIFEST_FILENAME


def read_split_manifest(namespace: str, config: DictConfig | None = None) -> dict[str, Any]:
    path = split_manifest_path(namespace, config)
    if not path.is_file():
        raise SystemExit(
            f"Missing split manifest: {path}\n"
            f"Build it first: python scripts/data/02b_build_sixway_splits.py --namespace {namespace}"
        )
    manifest = read_json(path)
    expected = int((config or load_splits_config()).manifest_version)
    if manifest.get("manifest_version") != expected:
        raise SystemExit(f"Split manifest version mismatch at {path}: expected {expected}, got {manifest.get('manifest_version')!r}")
    if manifest.get("split_namespace") != namespace:
        raise SystemExit(f"Split namespace mismatch at {path}: expected {namespace!r}, got {manifest.get('split_namespace')!r}")
    if manifest.get("manifest_hash_scope") == "full_manifest_without_manifest_hash_v2":
        declared = manifest.get("manifest_hash")
        unhashed = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        actual = hash_dict(unhashed, length=64)
        if declared != actual:
            raise SystemExit(f"Split manifest integrity mismatch at {path}: declared {declared}, computed {actual}")
    return manifest


def split_metadata(split_name: str, namespace: str, config: DictConfig | None = None) -> dict[str, Any]:
    """Return non-outcome metadata without parsing a label-bearing CSV."""
    manifest = read_split_manifest(namespace, config)
    path = resolve_splits_dir(namespace, config) / f"{split_name}.csv"
    if not path.is_file():
        raise SystemExit(f"Missing split file: {path}")
    outputs = manifest.get("output_files", {})
    declared = outputs.get(split_name, {})
    actual_hash = sha256_file(path)
    expected_hash = declared.get("sha256")
    if expected_hash and actual_hash != expected_hash:
        raise SystemExit(f"Split hash mismatch for {split_name}: expected {expected_hash}, got {actual_hash}")
    return {
        "exists": True, "path": str(path), "sha256": actual_hash,
        "row_count": declared.get("row_count", manifest.get("images_per_split", {}).get(split_name)),
        "schema_hash": manifest.get("source_schema_hash"),
        "manifest_hash": manifest.get("manifest_hash"), "namespace": namespace,
    }


def validate_final_eval_context(context_path: str | Path, run_id: str, namespace: str) -> dict[str, Any]:
    path = Path(context_path)
    if not path.is_file():
        raise FinalEvalAccessViolation(f"Final-evaluation context is not registered: {path}")
    record = read_json(path)
    if record.get("final_eval_run_id") != run_id or record.get("namespace") != namespace:
        raise FinalEvalAccessViolation("Final-evaluation context run ID/namespace mismatch")
    if record.get("status") not in {"registered", "outcome_accessed", "in_progress", "complete"}:
        raise FinalEvalAccessViolation(f"Final-evaluation context has invalid status {record.get('status')!r}")
    required = ("split_manifest_hash", "asism_manifest_hash", "protocol_manifest_hash", "threshold_policy_hash", "checkpoint_hashes", "code_git_version", "code_identity_sha256")
    missing = [key for key in required if not record.get(key)]
    if missing:
        raise FinalEvalAccessViolation(f"Final-evaluation context is incomplete: {missing}")
    return record


def load_split(
    split_name: str,
    namespace: str,
    config: DictConfig | None = None,
    purpose: str = "load_labels_for_evaluation",
    final_eval_run_id: str | None = None,
    final_eval_context_path: str | Path | None = None,
    caller: str = "unknown",
) -> pd.DataFrame:
    """Load one split CSV, enforcing the final_eval_heldout access rule.

    `purpose` is only consulted for final_eval_heldout; the other five splits are ordinary
    development data with no such restriction.
    """
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {split_name!r}; expected one of {SPLIT_NAMES}")

    if split_name == FINAL_EVAL_SPLIT:
        assert_final_eval_access_allowed(
            purpose=purpose, final_eval_run_id=final_eval_run_id, caller=caller
        )
        if purpose in NON_OUTCOME_PURPOSES:
            raise FinalEvalAccessViolation(
                "Metadata-only final-evaluation checks must use split_metadata(); load_split() "
                "never returns final_eval_heldout rows before a validated evaluation context"
            )
        if purpose in OUTCOME_PURPOSES:
            if not final_eval_context_path or not final_eval_run_id:
                raise FinalEvalAccessViolation("Outcome-bearing final data access requires a validated registered context")
            validate_final_eval_context(final_eval_context_path, final_eval_run_id, namespace)

    path = resolve_splits_dir(namespace, config) / f"{split_name}.csv"
    if not path.is_file():
        raise SystemExit(
            f"Missing split file: {path}\n"
            f"Build it first: python scripts/data/02b_build_sixway_splits.py --namespace {namespace}"
        )
    return pd.read_csv(path)


def split_provenance(namespace: str, config: DictConfig | None = None) -> dict[str, Any]:
    """The provenance block every downstream stage records and validates against, so a stage can
    never silently consume outputs built from a different split partition.

    Deliberately small: the manifest hash pins the entire partition (seed, fractions, patient
    assignment) without every downstream manifest having to embed all of it.
    """
    manifest = read_split_manifest(namespace, config)
    return {
        "split_namespace": namespace,
        "split_manifest_version": manifest.get("manifest_version"),
        "split_manifest_hash": manifest.get("manifest_hash"),
        "split_seed": manifest.get("split_seed"),
        "frozen": manifest.get("frozen", False),
    }


def build_patient_folds(patient_ids: list[str], fold_count: int, seed: int) -> dict[int, set[str]]:
    """Deterministic patient-level validation folds; disjoint with an exact union."""
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    ordered = sorted(set(map(str, patient_ids)))
    random.Random(seed).shuffle(ordered)
    folds = {index: set() for index in range(fold_count)}
    for index, patient_id in enumerate(ordered):
        folds[index % fold_count].add(patient_id)
    union = set().union(*folds.values())
    if union != set(ordered) or sum(len(v) for v in folds.values()) != len(union):
        raise AssertionError("fold construction is not an exact disjoint patient partition")
    return folds


def freeze_patient_folds(path: str | Path, patient_ids: list[str], fold_count: int, seed: int, split_manifest_hash: str) -> dict[str, Any]:
    folds = build_patient_folds(patient_ids, fold_count, seed)
    core = {
        "schema_version": 1, "assignment_unit": "patient_id", "fold_count": fold_count,
        "assignment_seed": seed, "split_manifest_hash": split_manifest_hash,
        "pool_patient_set_hash": hash_dict({"patient_ids": sorted(set(map(str, patient_ids)))}, length=64),
        "folds": {str(i): {"patient_ids": sorted(ids), "patient_set_hash": hash_dict({"patient_ids": sorted(ids)}, length=64)} for i, ids in folds.items()},
    }
    core["fold_manifest_hash"] = hash_dict(core, length=64)
    write_frozen_json(path, core)
    return {**core, "frozen": True}


def require_frozen_production_splits(config: DictConfig | None = None) -> dict[str, Any]:
    """Stage 5 precondition (plan §8): the production split manifest must exist AND be frozen."""
    manifest = read_split_manifest("production", config)
    if not manifest.get("frozen", False):
        raise SystemExit(
            "Production split manifest exists but is not frozen. "
            "Freeze it before any final evaluation: "
            "python scripts/data/02b_build_sixway_splits.py --namespace production --freeze"
        )
    if not manifest.get("support_check", {}).get("passed", False):
        raise SystemExit(
            "Production split manifest is frozen but its support check did not pass — "
            "final evaluation on splits that fail the frozen support rule is not permitted "
            "(docs/stages2_to_5_plan.md §1.3)."
        )
    return manifest


def require_frozen_production_split_run(namespace: str, config: DictConfig | None = None) -> dict[str, Any]:
    """Require production class, freeze state, and support for any versioned production run ID."""
    manifest = read_split_manifest(namespace, config)
    if manifest.get("namespace_class") != "production":
        raise SystemExit(f"Split run {namespace!r} is not a production namespace (namespace_class={manifest.get('namespace_class')!r})")
    if not manifest.get("frozen", False):
        raise SystemExit(f"Production split run {namespace!r} is not frozen")
    if not manifest.get("support_check", {}).get("passed", False):
        raise SystemExit(f"Production split run {namespace!r} failed the frozen patient-support rule")
    return manifest

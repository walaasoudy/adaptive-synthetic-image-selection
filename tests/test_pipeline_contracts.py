"""Contract and guard tests for the Stage 2-5 pipeline (docs/stages2_to_5_plan.md).

Two things are verified here:

1. The LEAKAGE GUARDS actually refuse. A guard that is only documented is not a guard, so each one
   is exercised and asserted to fail.
2. The Stage 5 ANALYSIS LAYER works end-to-end on explicitly-labelled FIXTURES. Fixtures are not
   scientific validation and are never presented as results — they prove the statistical code paths
   execute and produce the expected structure, nothing more.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
sys.dont_write_bytecode = True
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.utils.labels import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    GENERATION_TARGET_LABELS,
    INSUFFICIENT_SUPPORT_LABELS,
    PRIMARY_ENDPOINT_LABELS,
    build_label_arrays,
    check_support_rule,
    normalize_label,
    patient_level_support,
)
from scripts.utils.metrics import full_metric_suite  # noqa: E402
from scripts.utils.splits import (  # noqa: E402
    FinalEvalAccessViolation,
    assert_final_eval_access_allowed,
)
from fixture_workspace import fixture_workspace


# ------------------------------------------------------------------ label policy (§5, §6)

def test_primary_endpoint_excludes_no_finding_and_support_devices():
    assert len(CLASSIFIER_TARGET_LABELS) == 14
    assert len(PRIMARY_ENDPOINT_LABELS) == 11
    assert "No Finding" not in PRIMARY_ENDPOINT_LABELS
    assert "Support Devices" not in PRIMARY_ENDPOINT_LABELS
    assert "Pleural Other" not in PRIMARY_ENDPOINT_LABELS
    assert "No Finding" in CLASSIFIER_TARGET_LABELS
    assert "Support Devices" in CLASSIFIER_TARGET_LABELS
    assert "Pleural Other" in CLASSIFIER_TARGET_LABELS


def test_generation_target_labels_restores_insufficient_support_labels():
    assert len(GENERATION_TARGET_LABELS) == 12
    assert "Pleural Other" in GENERATION_TARGET_LABELS
    assert set(GENERATION_TARGET_LABELS) == set(PRIMARY_ENDPOINT_LABELS) | set(INSUFFICIENT_SUPPORT_LABELS)
    assert "No Finding" not in GENERATION_TARGET_LABELS
    assert "Support Devices" not in GENERATION_TARGET_LABELS


def test_uncertain_and_blank_are_masked_never_converted():
    frame = pd.DataFrame({label: [1, 0, -1, None] for label in CLASSIFIER_TARGET_LABELS})
    raw, target, mask = build_label_arrays(frame)
    # Row 0 = positive, row 1 = negative -> masked in; rows 2 (-1) and 3 (blank) -> masked out.
    assert mask[0, 0] and mask[1, 0]
    assert not mask[2, 0], "-1 must be masked, never mapped to 0 or 1"
    assert not mask[3, 0], "blank must be masked"
    assert raw[2, 0] == -1, "raw -1 must be preserved verbatim"
    assert raw[3, 0] == -2, "blank must stay distinguishable from an explicit -1"


def test_normalize_label_handles_all_forms():
    assert normalize_label(1) == 1
    assert normalize_label("0") == 0
    assert normalize_label(-1.0) == -1
    assert normalize_label(None) is None
    assert normalize_label("") is None
    assert normalize_label(float("nan")) is None


def test_patient_level_support_counts_patients_not_images():
    # One patient with 5 positive studies must count as ONE positive patient.
    frame = pd.DataFrame(
        {
            "patient_id": ["p1"] * 5 + ["p2", "p3"],
            "Cardiomegaly": [1, 1, 1, 1, 1, 0, 0],
        }
    )
    support = patient_level_support(frame, ["Cardiomegaly"])
    row = support.iloc[0]
    assert row["positive_patients"] == 1, "patient-level, not image-level"
    assert row["positive_images"] == 5
    assert row["negative_patients"] == 2


def test_patient_positive_wins_over_negative():
    # A patient with one positive and one negative study is a POSITIVE patient.
    frame = pd.DataFrame({"patient_id": ["p1", "p1"], "Edema": [1, 0]})
    support = patient_level_support(frame, ["Edema"])
    assert support.iloc[0]["positive_patients"] == 1
    assert support.iloc[0]["negative_patients"] == 0


def test_support_rule_reports_all_failures_not_just_first():
    support = pd.DataFrame(
        [
            {"label": "A", "positive_patients": 10, "negative_patients": 500},
            {"label": "B", "positive_patients": 500, "negative_patients": 5},
            {"label": "C", "positive_patients": 500, "negative_patients": 500},
        ]
    )
    passed, failures = check_support_rule(support, 50, 50)
    assert passed is False
    assert len(failures) == 2, "must report every failing label, not abort on the first"
    assert {f["label"] for f in failures} == {"A", "B"}


# ------------------------------------------------------------------ leakage guards (§1.6)

def test_final_eval_guard_allows_split_construction():
    assert_final_eval_access_allowed("split_construction", caller="test") is None
    assert_final_eval_access_allowed("support_check", caller="test") is None
    assert_final_eval_access_allowed("hash_verification", caller="test") is None


def test_final_eval_guard_refuses_outcome_access_without_run_id():
    for purpose in ("load_images", "load_labels_for_evaluation", "compute_outcome_statistics",
                    "model_selection"):
        try:
            assert_final_eval_access_allowed(purpose, caller="test")
            raise AssertionError(f"guard did not refuse outcome purpose {purpose!r}")
        except FinalEvalAccessViolation:
            pass


def test_final_eval_guard_allows_outcome_access_with_run_id():
    assert_final_eval_access_allowed(
        "load_labels_for_evaluation", final_eval_run_id="stage5-run-1", caller="test"
    )


def test_final_eval_guard_rejects_unknown_purpose():
    try:
        assert_final_eval_access_allowed("sneaky_peek", caller="test")
        raise AssertionError("guard accepted an undeclared purpose")
    except FinalEvalAccessViolation:
        pass


def test_stage5_refuses_without_frozen_preconditions():
    """Stage 5 must abort BEFORE opening final_eval_heldout when preconditions are unmet."""
    with fixture_workspace("stage5-refusal") as workspace:
        environment = {**os.environ, "PROJECT_ROOT": str(workspace)}
        result = subprocess.run(
            [sys.executable, "scripts/eval/stage5_evaluate.py",
             "--final-eval-run-id", "test-run", "--namespace", "production-isolated-v1"],
            cwd=REPO, capture_output=True, text=True, env=environment,
        )
    assert result.returncode != 0, "Stage 5 must refuse when preconditions are unmet"
    combined = result.stdout + result.stderr
    assert "STAGE 5 REFUSED" in combined or "Missing split manifest" in combined
    assert "final_eval_heldout was NOT opened" in combined or "split manifest" in combined


def test_chexpert_pretrained_initialization_is_rejected():
    """A CheXpert-pretrained checkpoint cannot be shown disjoint from our evaluation patients (§2)."""
    from scripts.utils.classifier import build_model

    try:
        build_model(14, 0.2, "chexpert_densenet121")
        raise AssertionError("CheXpert-pretrained initialization was not rejected")
    except SystemExit as exc:
        assert "Refusing CheXpert-pretrained" in str(exc)


# ------------------------------------------------------------------ split builder (§1)

def test_split_fractions_sum_to_one():
    from scripts.utils.splits import SPLIT_NAMES, load_splits_config

    config = load_splits_config()
    total = sum(float(config.fractions[name]) for name in SPLIT_NAMES)
    assert abs(total - 1.0) < 1e-9, f"frozen fractions must sum to 1.0, got {total}"


def test_fid_real_reference_reads_the_namespaced_gen_val_preprocessing_writes():
    import importlib.util
    from scripts.utils.config import load_stage1_config

    spec = importlib.util.spec_from_file_location("fid", REPO / "scripts" / "eval" / "compute_fid_clipscore.py")
    fid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fid)

    cfg = load_stage1_config()
    # 03_preprocess_images.py writes <images_dir>/<namespace>/<split>/; the old un-namespaced
    # <images_dir>/gen_val never exists, so FID found no real reference images.
    assert fid.real_reference_dir(cfg, "production-thesis-v1") == (
        Path(cfg.paths.images_dir) / "production-thesis-v1" / "gen_val"
    )


def test_six_way_partition_is_exact_and_disjoint():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "builder", REPO / "scripts" / "data" / "02b_build_sixway_splits.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    patients = [f"patient{i:05d}" for i in range(1000)]
    fractions = {
        "gen_train": 0.60, "gen_val": 0.10, "classifier_train": 0.15,
        "classifier_val": 0.05, "asism_tuning_heldout": 0.05, "final_eval_heldout": 0.05,
    }
    groups = builder.partition_patients(patients, fractions, seed=42)

    assert sum(len(ids) for ids in groups.values()) == 1000, "partition must be exact"
    builder.assert_disjoint_and_exact(groups, set(patients))  # raises on any leak

    # Deterministic: same seed -> same partition.
    again = builder.partition_patients(patients, fractions, seed=42)
    assert groups == again


def test_partition_detects_injected_leak():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "builder", REPO / "scripts" / "data" / "02b_build_sixway_splits.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)

    groups = {"a": {"p1", "p2"}, "b": {"p2", "p3"}}  # p2 leaks across both
    try:
        builder.assert_disjoint_and_exact(groups, {"p1", "p2", "p3"})
        raise AssertionError("disjointness assertion failed to detect an injected leak")
    except SystemExit as exc:
        assert "SPLIT LEAK" in str(exc)


# ------------------------------------------------------------------ Stage 5 analysis on FIXTURES

def test_stage5_analysis_layer_on_fixtures():
    """End-to-end statistical layer on FIXTURE data.

    This is explicitly NOT a scientific result: probabilities are synthesised so that one condition
    is genuinely better than another, and the test asserts the machinery detects that. It proves the
    code path works; it says nothing about ASISM.
    """
    rng = np.random.default_rng(0)
    n_patients, per_patient = 120, 2
    n_rows = n_patients * per_patient

    patient_ids = np.repeat([f"p{i:04d}" for i in range(n_patients)], per_patient)
    truth = pd.DataFrame(
        {label: rng.integers(0, 2, n_rows) for label in CLASSIFIER_TARGET_LABELS}
    )
    _, targets, masks = build_label_arrays(truth)

    def make_probabilities(signal_strength: float) -> np.ndarray:
        probabilities = np.zeros((n_rows, len(CLASSIFIER_TARGET_LABELS)))
        for index, label in enumerate(CLASSIFIER_TARGET_LABELS):
            y = truth[label].to_numpy()
            noise = rng.uniform(0, 1, n_rows)
            probabilities[:, index] = np.clip(
                signal_strength * y + (1 - signal_strength) * noise, 0.001, 0.999
            )
        return probabilities

    strong = make_probabilities(0.75)
    weak = make_probabilities(0.05)

    suite = full_metric_suite(strong, targets, masks)
    assert suite["macro_auroc_primary"] > 0.7
    assert len(suite["primary_labels"]) == 11
    assert set(suite["secondary_labels"]) == {"No Finding", "Support Devices", "Pleural Other"}
    for label, entry in suite["per_label"].items():
        assert "effective_n" in entry, f"{label} must report effective N"

    from scripts.utils.metrics import auroc, paired_bootstrap_difference

    def macro(probabilities, rows):
        values = []
        for label in PRIMARY_ENDPOINT_LABELS:
            index = CLASSIFIER_TARGET_LABELS.index(label)
            mask = masks[rows, index].astype(bool)
            y = targets[rows][mask, index].astype(int)
            if (y == 1).sum() and (y == 0).sum():
                values.append(auroc(probabilities[rows][mask, index], y))
        return float(np.mean(values)) if values else float("nan")

    comparison = paired_bootstrap_difference(
        patient_ids,
        lambda rows: macro(strong, rows),
        lambda rows: macro(weak, rows),
        n_resamples=200,
        seed=1,
    )
    assert comparison["effect_size"] > 0.1
    assert comparison["ci_lower"] > 0, "a real difference must produce a CI excluding zero"
    assert comparison["p_value"] < 0.05


def test_fixture_data_is_never_written_to_production_paths():
    """Guard against a test accidentally polluting real artifact directories."""
    from scripts.utils.splits import load_splits_config

    config = load_splits_config()
    production_dir = Path(config.paths.splits_root) / "production"
    # The test suite must not have created production splits as a side effect.
    if production_dir.exists():
        manifest = production_dir / "split_manifest_v2.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            assert data.get("split_namespace") == "production"
            assert data.get("input_csv_rows", 0) > 100000, (
                "a production split manifest exists but was built from a small CSV — "
                "it must only be built from the full cohort"
            )


if __name__ == "__main__":
    import traceback

    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

#!/usr/bin/env python3
"""The Stage 3a driver: five signals, five artifacts, and the gates in front of each.

Run against a laid-out PROJECT_ROOT on CPU. The similarity signal is exercised only through its
pure scoring function elsewhere (test_ham10000_selection_signals.py) — running it here would
download a pinned DINOv2 checkpoint, which does not belong in a test suite.

What is under test is everything a unit test of the scoring functions cannot see: that each signal
writes its OWN artifact with provenance, that a missing calibration or reference stops the run
instead of being defaulted, and that a candidate whose class has no reference still appears in the
pool carrying NaN rather than vanishing from it.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from test_ham10000_cam_reference_build import (  # noqa: E402
    CLASSES,
    NAMESPACE,
    RESOLUTION,
    _overlay,
    _workspace,
    _write_checkpoint,
    _write_images,
)

N_CANDIDATES_PER_CLASS = 4
CONTENT_BOX = [0.0, 0.125, 1.0, 0.875]


def _write_candidates(root: Path, diagnoses=CLASSES) -> Path:
    import pandas as pd

    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

    candidate_dir = root / "outputs/ham10000/stage2" / NAMESPACE / "full/candidates"
    rows, image_ids = [], []
    for diagnosis in diagnoses:
        for index in range(N_CANDIDATES_PER_CLASS):
            image_id = f"SYN_{diagnosis}_{index:03d}"
            image_ids.append(image_id)
            rows.append(
                {
                    "image_id": image_id,
                    "status": "accepted",
                    "dx": diagnosis,
                    "class_index": CLASSIFIER_TARGET_LABELS.index(diagnosis),
                    "image_path": str(candidate_dir / f"{image_id}.jpg"),
                    "content_box": json.dumps(CONTENT_BOX),
                    "seed": index,
                }
            )
    _write_images(candidate_dir, image_ids)
    path = root / "outputs/ham10000/stage2" / NAMESPACE / "all_candidates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_blur_calibration(root: Path, threshold: float = 1.0) -> Path:
    """The artifact ham10000_calibrate_iqa_blur.py produces. The threshold is deliberately low so
    the fixture's noise images are not all flagged — this test is about wiring, not about the
    calibration, which has its own suite."""
    path = root / "outputs/ham10000/stage3" / NAMESPACE / "iqa_blur_calibration.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_split": "gen_train",
                "rule": "lower_quantile",
                "quantile": 0.02,
                "laplacian_blur_threshold": threshold,
            }
        ),
        encoding="utf-8",
    )
    return path


def _build_reference(root: Path):
    from scripts.asism.ham10000_00_build_cam_reference import build

    return build(NAMESPACE, device="cpu")


def _run(signals: list[str], **kwargs):
    from scripts.asism.ham10000_01_compute_signals import run

    return run(NAMESPACE, signals, device="cpu", **kwargs)


def _artifact(root: Path, name: str):
    import pandas as pd

    path = root / "outputs/ham10000/stage3" / NAMESPACE / "signals" / f"{name}_scores.parquet"
    sidecar = path.with_suffix(".provenance.json")
    return pd.read_parquet(path), json.loads(sidecar.read_text(encoding="utf-8"))


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("compute-signals") as root:
        _workspace(root)
        _write_checkpoint(root)
        _write_candidates(root)
        _write_blur_calibration(root)
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        yield root


# ==============================================================================================
# IQA — the one signal that needs no model
# ==============================================================================================


def test_iqa_writes_its_own_artifact_naming_the_calibration_it_used(workspace):
    _run(["iqa"])
    frame, provenance = _artifact(workspace, "iqa")

    assert len(frame) == N_CANDIDATES_PER_CLASS * len(CLASSES)
    assert provenance["signal"] == "iqa"
    assert provenance["n_candidates_scored"] == len(frame)
    # A score whose threshold cannot be traced back to an artifact cannot be defended later.
    assert provenance["laplacian_blur_threshold"] == 1.0
    assert provenance["blur_calibration_sha256"]
    assert provenance["candidates_csv_sha256"]


def test_iqa_refuses_to_run_without_the_blur_calibration(workspace):
    """There is no constant fallback on purpose: the inherited CheXpert threshold flagged a third
    of real, clinically accepted dermoscopy as blurry."""
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "iqa_blur_calibration.json").unlink()
    with pytest.raises(SystemExit, match="calibrate_iqa_blur"):
        _run(["iqa"])


def test_a_candidate_without_a_content_box_stops_the_run(workspace):
    import pandas as pd

    path = workspace / "outputs/ham10000/stage2" / NAMESPACE / "all_candidates.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "content_box"] = None
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit, match="content_box"):
        _run(["iqa"])


# ==============================================================================================
# Uncertainty and agreement — one set of passes, two independent artifacts
# ==============================================================================================


def test_uncertainty_and_agreement_are_written_as_separate_gateable_artifacts(workspace):
    _run(["uncertainty", "agreement"])

    uncertainty, uncertainty_provenance = _artifact(workspace, "uncertainty")
    agreement, agreement_provenance = _artifact(workspace, "agreement")

    assert set(uncertainty["image_id"]) == set(agreement["image_id"])
    assert uncertainty_provenance["signal"] == "uncertainty"
    assert agreement_provenance["signal"] == "agreement"
    # Same model, same passes: the two artifacts cannot disagree about what the model predicted.
    assert uncertainty_provenance["cam_model_id"] == agreement_provenance["cam_model_id"]
    assert uncertainty_provenance["mc_dropout_passes"] == agreement_provenance["mc_dropout_passes"]
    assert agreement_provenance["probabilities"] == "mc_dropout_mean"

    assert uncertainty_provenance["band_quantity"] == "normalised_mutual_information"
    assert (uncertainty["uncertainty_mutual_information"] >= 0).all()
    assert set(uncertainty["uncertainty_band"]) <= {"low", "moderate", "extreme"}


def test_agreement_scores_each_candidate_against_the_class_its_recipe_asked_for(workspace):
    _run(["agreement"])
    frame, _ = _artifact(workspace, "agreement")

    intended = dict(zip(frame["image_id"], frame["agreement_intended_diagnosis"]))
    for image_id, diagnosis in intended.items():
        assert image_id.startswith(f"SYN_{diagnosis}_")
    assert frame["agreement_intended_prob"].between(0, 1).all()


def test_uncertainty_alone_does_not_write_an_agreement_artifact(workspace):
    _run(["uncertainty"])
    signals_dir = workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals"
    assert (signals_dir / "uncertainty_scores.parquet").is_file()
    assert not (signals_dir / "agreement_scores.parquet").exists()


# ==============================================================================================
# Explainability — calibrated against the real reference, or NaN and visible
# ==============================================================================================


def test_explainability_is_calibrated_against_the_reference_artifact(workspace):
    _build_reference(workspace)
    _run(["explainability"])
    frame, provenance = _artifact(workspace, "explainability")

    assert len(frame) == N_CANDIDATES_PER_CLASS * len(CLASSES)
    assert frame["explainability_calibrated"].astype(bool).all()
    assert provenance["classes_without_a_reference"] == {}
    assert provenance["reference_artifact_sha256"]
    # Exact boxes both sides: the calibrated column is only defined when the candidate was measured
    # the same way the reference was.
    assert frame["explainability_calibrated_typicality"].notna().all()
    assert frame["explainability_calibrated_typicality"].between(0, 1).all()


def test_explainability_refuses_to_run_before_the_reference_exists(workspace):
    with pytest.raises(SystemExit, match="build_cam_reference"):
        _run(["explainability"])


def test_a_reference_from_a_different_cam_model_is_refused(workspace):
    """Calibrating one model's attention against another's is not a comparison. The check is by
    model hash, so replacing the checkpoint after the reference was built is caught."""
    _build_reference(workspace)
    reference_path = workspace / "outputs/ham10000/stage3" / NAMESPACE / "explainability_reference.json"
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    payload["cam_model_id"] = "ham10000-classifier:" + "f" * 64
    reference_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="different CAM model"):
        _run(["explainability"])


def test_a_candidate_whose_class_has_no_reference_stays_in_the_pool_carrying_NaN(workspace):
    """Dropping it would silently shrink the candidate pool; giving it a number would calibrate it
    against another lesion's attention. It is kept, uncalibrated, and the count is recorded."""
    import pandas as pd

    # Build the reference from nv only, then score candidates of both classes.
    split_path = workspace / "data/ham10000/processed/splits" / NAMESPACE / "gen_train.csv"
    full = pd.read_csv(split_path)
    full[full["dx"] == "nv"].to_csv(split_path, index=False)
    _build_reference(workspace)
    full.to_csv(split_path, index=False)

    _run(["explainability"])
    frame, provenance = _artifact(workspace, "explainability")

    assert len(frame) == N_CANDIDATES_PER_CLASS * len(CLASSES)
    assert provenance["classes_without_a_reference"] == {"mel": N_CANDIDATES_PER_CLASS}

    mel_rows = frame[frame["image_id"].str.startswith("SYN_mel")]
    nv_rows = frame[frame["image_id"].str.startswith("SYN_nv")]
    assert mel_rows["explainability_calibrated_typicality"].isna().all()
    assert not mel_rows["explainability_calibrated"].astype(bool).any()
    assert nv_rows["explainability_calibrated_typicality"].notna().all()


# ==============================================================================================
# The candidate pool itself
# ==============================================================================================


def test_a_candidate_image_missing_from_disk_stops_the_run(workspace):
    """Scoring whatever happens to be present would silently change which images selection could
    choose from, and nothing downstream compares the pool against the manifest."""
    candidate = workspace / "outputs/ham10000/stage2" / NAMESPACE / "full/candidates" / "SYN_nv_000.jpg"
    candidate.unlink()
    with pytest.raises(SystemExit, match="absent on disk"):
        _run(["iqa"])


def test_a_missing_candidate_manifest_names_the_command_that_produces_it(workspace):
    (workspace / "outputs/ham10000/stage2" / NAMESPACE / "all_candidates.csv").unlink()
    with pytest.raises(SystemExit, match="ham10000_generate_synthetic_images"):
        _run(["iqa"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

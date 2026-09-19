#!/usr/bin/env python3
"""End-to-end build of the Grad-CAM reference on a tiny CPU workspace.

Separate from test_ham10000_cam_reference.py, which tests the artifact contract in isolation. This
one runs the actual entry point against a laid-out PROJECT_ROOT — splits, preprocessed images,
content boxes and a checkpoint — because the failures it is looking for are wiring failures that
unit tests of the pieces cannot see: a reference built from the model's PREDICTED class instead of
the true one, a missing content box silently replaced by the generic band, or an image from
final_eval_heldout reaching the reference through the split loader.

The model here is randomly initialised, so the reference VALUES are meaningless. The wiring,
provenance and refusals are what is under test.
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

NAMESPACE = "ham-fixture-v1"
RESOLUTION = 64
PER_CLASS = 8
CLASSES = ("mel", "nv")


def _write_images(directory: Path, image_ids: list[str]) -> None:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for image_id in image_ids:
        array = rng.integers(0, 255, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        Image.fromarray(array).save(directory / f"{image_id}.jpg")


def _workspace(root: Path) -> dict:
    """A PROJECT_ROOT laid out the way the real pipeline leaves it after preprocessing."""
    import pandas as pd

    from scripts.utils.ham10000_geometry import write_content_boxes

    splits_dir = root / "data/ham10000/processed/splits" / NAMESPACE
    images_dir = root / "data/ham10000/processed/images" / NAMESPACE
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows, image_ids = [], []
    for diagnosis in CLASSES:
        for index in range(PER_CLASS):
            image_id = f"ISIC_{diagnosis}_{index:03d}"
            image_ids.append(image_id)
            rows.append({"image_id": image_id, "lesion_id": f"HAM_{diagnosis}_{index}", "dx": diagnosis})
    pd.DataFrame(rows).to_csv(splits_dir / "gen_train.csv", index=False)

    final_eval = [
        {"image_id": f"ISIC_final_{index:03d}", "lesion_id": f"HAM_final_{index}", "dx": "nv"}
        for index in range(4)
    ]
    pd.DataFrame(final_eval).to_csv(splits_dir / "final_eval_heldout.csv", index=False)

    _write_images(images_dir / "gen_train", image_ids)
    # The content box the real preprocessing writes: a square source letterboxed to the canvas
    # occupies the whole canvas, which is the simplest box that is still exact.
    write_content_boxes(
        images_dir / "gen_train_content_boxes.csv",
        [
            {
                "image_id": image_id,
                "source_width": RESOLUTION,
                "source_height": RESOLUTION,
                "x0": 0.0, "y0": 0.125, "x1": 1.0, "y1": 0.875,
            }
            for image_id in image_ids
        ],
    )
    return {"splits_dir": splits_dir, "images_dir": images_dir, "image_ids": image_ids}


def _write_checkpoint(root: Path, *, train_split: str = "classifier_train") -> Path:
    import torch

    from scripts.utils.classifier import build_model
    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

    out_dir = root / "checkpoints/ham10000/asism_auxiliary" / NAMESPACE
    out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(len(CLASSIFIER_TARGET_LABELS), 0.2, "random")
    # Force the head's weights non-negative. DenseNet's block-4 activations are post-ReLU, so a
    # non-negative head gives every class a non-negative Grad-CAM with some mass on it. With the
    # signs left random, a randomly initialised model produces an all-zero CAM for most classes,
    # peripheral mass is then 0/0 = NaN, and the fixture would be testing the undersized-class path
    # in every test instead of the wiring. Nothing about the real pipeline is changed by this: the
    # trained classifier this reference is really built from has no such degeneracy.
    model.classifier[1].weight.data.abs_()
    torch.save(model.state_dict(), out_dir / "model.pt")
    manifest = {
        "dataset": "ham10000",
        "role": "asism_auxiliary",
        "loss": "cross_entropy",
        "classes": list(CLASSIFIER_TARGET_LABELS),
        "split_namespace": NAMESPACE,
        "model": {
            "architecture": "densenet121",
            "pretrained_source": "random",
            "dropout_p": 0.2,
            "resolution": RESOLUTION,
        },
        "data": {
            "real_train_split": train_split,
            "selection_split": "classifier_val",
            "synthetic_manifest": None,
            "synthetic_images": 0,
        },
        "git_commit_hash": "fixture",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out_dir


def _overlay(root: Path) -> Path:
    """Lower the reference-size floor for a fixture-sized split, and nothing else."""
    path = root / "overlay.yaml"
    path.write_text(
        "ham_stage3:\n  signals:\n    explainability:\n      min_reference_size: 5\n",
        encoding="utf-8",
    )
    return path


def _build(root: Path, **kwargs):
    # No sys.modules purging: every path in the entry point is resolved from PROJECT_ROOT at call
    # time, and reloading the package would give the test a different exception CLASS than the one
    # the code raises, so `pytest.raises` would stop matching.
    from scripts.asism.ham10000_00_build_cam_reference import build

    return build(NAMESPACE, device="cpu", **kwargs)


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("cam-reference") as root:
        _workspace(root)
        _write_checkpoint(root)
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        yield root


def test_the_reference_is_built_per_class_from_exact_content_boxes(workspace):
    result = _build(workspace)
    payload = result["payload"]

    assert payload["reference_split"] == "gen_train"
    assert payload["reference_images_scored"] == PER_CLASS * len(CLASSES)
    assert payload["cam_model_id"].startswith("ham10000-classifier:")
    assert result["undersized"] == {}

    by_class = {}
    for entry in payload["references"]:
        by_class.setdefault(entry["diagnosis"], set()).add(entry["statistic"])
    assert set(by_class) == set(CLASSES)
    for statistics in by_class.values():
        assert "explainability_peripheral_mass" in statistics

    # Every reference image is a gen_train image, and each class's reference holds only its own.
    for entry in payload["references"]:
        assert all(entry["diagnosis"] in image_id for image_id in entry["image_ids"])
        assert len(entry["image_ids"]) == PER_CLASS


def test_the_reference_rows_are_kept_next_to_the_artifact_for_audit(workspace):
    import pandas as pd

    result = _build(workspace)
    rows_path = Path(result["path"]).with_name("explainability_reference_rows.csv")
    frame = pd.read_csv(rows_path)

    assert len(frame) == PER_CLASS * len(CLASSES)
    # The whole point of the exact box: not one row may have fallen back to the generic band.
    assert set(frame["explainability_content_box_source"]) == {"preprocessing_manifest"}
    assert frame["explainability_content_box"].notna().all()


def test_the_cams_are_taken_for_each_images_TRUE_class(workspace):
    """A reference built from the model's predicted class would describe only the cases it gets
    right — an optimistic reference that candidates are then scored against. With a random model,
    predictions are essentially never the true class, so the two choices are easy to tell apart."""
    import pandas as pd

    result = _build(workspace)
    frame = pd.read_csv(Path(result["path"]).with_name("explainability_reference_rows.csv"))

    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

    expected = {CLASSIFIER_TARGET_LABELS.index(name) for name in CLASSES}
    assert set(frame["class_index"]) == expected


def test_the_artifact_reloads_through_the_leakage_checks(workspace):
    from scripts.asism.ham10000_signals import load_final_eval_image_ids, references_from_artifact

    result = _build(workspace)
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    final_eval_ids = load_final_eval_image_ids(workspace / "data/ham10000/processed/splits" / NAMESPACE)

    references = references_from_artifact(payload, final_eval_ids, min_size=5)
    assert set(references) == set(CLASSES)
    assert references["mel"]["explainability_peripheral_mass"].scientific is True


def test_a_missing_content_box_file_stops_the_build_instead_of_using_the_generic_band(workspace):
    (workspace / "data/ham10000/processed/images" / NAMESPACE / "gen_train_content_boxes.csv").unlink()
    with pytest.raises(SystemExit, match="exact letterbox content box"):
        _build(workspace)


def test_a_single_missing_box_stops_the_build_rather_than_mixing_two_measurements(workspace):
    import pandas as pd

    path = workspace / "data/ham10000/processed/images" / NAMESPACE / "gen_train_content_boxes.csv"
    frame = pd.read_csv(path)
    frame.iloc[1:].to_csv(path, index=False)
    with pytest.raises(KeyError, match="refusing to fall back"):
        _build(workspace)


def test_a_cam_model_trained_on_the_reference_split_is_refused(workspace):
    """The collision the whole design exists around: the CheXpert convention trains the auxiliary
    classifier on gen_train, which here is the reference split."""
    from scripts.asism.ham10000_signals import ReferenceLeakageError

    _write_checkpoint(workspace, train_split="gen_train")
    with pytest.raises(ReferenceLeakageError, match="reference split"):
        _build(workspace)


def test_a_class_too_thin_to_calibrate_is_reported_not_padded(workspace):
    import pandas as pd

    path = workspace / "data/ham10000/processed/splits" / NAMESPACE / "gen_train.csv"
    frame = pd.read_csv(path)
    # Leave mel with two images: below the fixture's floor of five.
    thin = pd.concat([frame[frame["dx"] == "nv"], frame[frame["dx"] == "mel"].head(2)])
    thin.to_csv(path, index=False)

    result = _build(workspace)
    assert any(key.endswith("/mel") for key in result["undersized"])
    assert {entry["diagnosis"] for entry in result["payload"]["references"]} == {"nv"}


def test_a_missing_checkpoint_names_the_command_that_produces_it(workspace):
    checkpoint = workspace / "checkpoints/ham10000/asism_auxiliary" / NAMESPACE / "model.pt"
    checkpoint.unlink()
    with pytest.raises(SystemExit, match="train_auxiliary_classifier"):
        _build(workspace)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
"""The three classifier-dependent signals recomputed with the v2 auxiliary classifier.

`--config-key auxiliary_classifier_v2` on the Grad-CAM reference builder and the signal driver must:
read the v2 checkpoint, write under paths.aux_v2_outputs_dir only, run only uncertainty, agreement
and explainability, and leave every v1 artifact byte-for-byte where it was. The default (v1) must
behave exactly as before; test_ham10000_cam_reference_build.py and test_ham10000_compute_signals.py
still run on that default unchanged.

Randomly initialised models, as in those suites: the values are meaningless, the wiring is tested.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import hashlib
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from test_ham10000_cam_reference_build import NAMESPACE, _overlay, _workspace, _write_checkpoint  # noqa: E402
from test_ham10000_compute_signals import (  # noqa: E402
    N_CANDIDATES_PER_CLASS,
    _write_blur_calibration,
    _write_candidates,
)

V2 = "auxiliary_classifier_v2"
CLASSIFIER_SIGNALS = ["uncertainty", "agreement", "explainability"]


def _write_v2_checkpoint(root: Path) -> Path:
    """The v1 fixture checkpoint copied to the v2 directory with its head scaled, so the two files
    hash differently and a v1/v2 mix-up shows up as a cam_model_id mismatch."""
    import torch

    v1_dir = root / "checkpoints/ham10000/asism_auxiliary" / NAMESPACE
    v2_dir = root / "checkpoints/ham10000/asism_auxiliary_v2" / NAMESPACE
    v2_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(v1_dir, v2_dir)
    state = torch.load(v2_dir / "model.pt", map_location="cpu")
    state["classifier.1.weight"] = state["classifier.1.weight"] * 2.0
    torch.save(state, v2_dir / "model.pt")
    return v2_dir


def _model_id(checkpoint_dir: Path) -> str:
    digest = hashlib.sha256((checkpoint_dir / "model.pt").read_bytes()).hexdigest()
    return f"ham10000-classifier:{digest}"


def _v1_dir(root: Path) -> Path:
    return root / "outputs/ham10000/stage3" / NAMESPACE


def _v2_dir(root: Path) -> Path:
    return root / "outputs/ham10000/stage3_aux_v2" / NAMESPACE


def _build(config_key=None):
    from scripts.asism.ham10000_00_build_cam_reference import build

    return build(NAMESPACE, device="cpu", **({"config_key": config_key} if config_key else {}))


def _run(signals, config_key=None):
    from scripts.asism.ham10000_01_compute_signals import run

    return run(NAMESPACE, signals, device="cpu", **({"config_key": config_key} if config_key else {}))


def _provenance(directory: Path, name: str) -> dict:
    path = directory / "signals" / f"{name}_scores.provenance.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(directory: Path) -> dict:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("v2-signals") as root:
        _workspace(root)
        v1 = _write_checkpoint(root)
        v2 = _write_v2_checkpoint(root)
        _write_candidates(root)
        _write_blur_calibration(root)
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        root_info = {"root": root, "v1_id": _model_id(v1), "v2_id": _model_id(v2)}
        assert root_info["v1_id"] != root_info["v2_id"]
        yield root_info


# ==============================================================================================
# The committed config
# ==============================================================================================


def test_the_committed_v2_outputs_root_is_separate_from_v1():
    from omegaconf import OmegaConf

    config = OmegaConf.load(REPO / "configs/ham10000_stage3.yaml")
    paths = OmegaConf.to_container(config.paths, resolve=False)
    assert paths["outputs_dir"] == "${paths.project_root}/outputs/ham10000/stage3"
    assert paths["aux_v2_outputs_dir"] == "${paths.project_root}/outputs/ham10000/stage3_aux_v2"


# ==============================================================================================
# The Grad-CAM reference
# ==============================================================================================


def test_the_v2_reference_is_built_from_the_v2_model_under_the_v2_root(workspace):
    root = workspace["root"]
    result = _build(V2)

    reference = _v2_dir(root) / "explainability_reference.json"
    assert Path(result["path"]) == reference
    payload = json.loads(reference.read_text(encoding="utf-8"))
    assert payload["cam_model_id"] == workspace["v2_id"]
    assert payload["auxiliary_config_key"] == V2
    assert (_v2_dir(root) / "explainability_reference_rows.csv").is_file()
    # Nothing was written where v1 lives.
    assert not _v1_dir(root).exists() or not (_v1_dir(root) / "explainability_reference.json").exists()


def test_the_default_reference_is_still_v1_where_it_always_was(workspace):
    root = workspace["root"]
    result = _build()

    assert Path(result["path"]) == _v1_dir(root) / "explainability_reference.json"
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["cam_model_id"] == workspace["v1_id"]
    assert payload["auxiliary_config_key"] == "auxiliary_classifier"
    assert not _v2_dir(root).exists()


# ==============================================================================================
# The three signals
# ==============================================================================================


def test_v2_writes_the_three_signals_from_the_v2_model_under_the_v2_root(workspace):
    root = workspace["root"]
    _build(V2)
    result = _run(CLASSIFIER_SIGNALS, V2)

    assert Path(result["out_dir"]) == _v2_dir(root) / "signals"
    assert result["config_key"] == V2
    for name in CLASSIFIER_SIGNALS:
        provenance = _provenance(_v2_dir(root), name)
        assert provenance["cam_model_id"] == workspace["v2_id"], name
        assert provenance["auxiliary_config_key"] == V2, name
        assert provenance["n_candidates_scored"] == N_CANDIDATES_PER_CLASS * 2, name
    explainability = _provenance(_v2_dir(root), "explainability")
    assert Path(explainability["reference_artifact"]) == _v2_dir(root) / "explainability_reference.json"
    assert not (_v1_dir(root) / "signals").exists()


def test_the_v1_artifacts_are_byte_identical_after_a_full_v2_recompute(workspace):
    root = workspace["root"]
    _build()
    _run(["iqa", *CLASSIFIER_SIGNALS])
    before = _snapshot(_v1_dir(root))
    assert "signals/agreement_scores.parquet" in before

    _build(V2)
    _run(CLASSIFIER_SIGNALS, V2)

    assert _snapshot(_v1_dir(root)) == before
    assert _provenance(_v1_dir(root), "agreement")["cam_model_id"] == workspace["v1_id"]


@pytest.mark.parametrize("signals", [["iqa"], ["similarity"], ["agreement", "iqa"], ["all"]])
def test_v2_refuses_the_signals_that_do_not_depend_on_the_classifier(workspace, signals):
    from scripts.asism.ham10000_01_compute_signals import SIGNALS

    requested = list(SIGNALS) if signals == ["all"] else signals
    with pytest.raises(SystemExit, match="recomputes only"):
        _run(requested, V2)
    assert not _v2_dir(workspace["root"]).exists()


def test_v2_explainability_does_not_fall_back_to_the_v1_reference(workspace):
    _build()  # a v1 reference exists
    with pytest.raises(SystemExit, match="--config-key auxiliary_classifier_v2"):
        _run(["explainability"], V2)


def test_a_v1_reference_placed_in_the_v2_root_is_refused_by_model_hash(workspace):
    root = workspace["root"]
    _build()
    _v2_dir(root).mkdir(parents=True)
    shutil.copy(_v1_dir(root) / "explainability_reference.json", _v2_dir(root) / "explainability_reference.json")

    with pytest.raises(SystemExit, match="different CAM model"):
        _run(["explainability"], V2)


def test_an_unknown_config_key_is_refused(workspace):
    # v3 became a known key on 2026-10-02 (test_ham10000_v3_signals.py); an unlisted one is still refused.
    with pytest.raises(SystemExit, match="Unknown auxiliary classifier config key"):
        _run(["agreement"], "auxiliary_classifier_v9")
    with pytest.raises(SystemExit, match="Unknown auxiliary classifier config key"):
        _build("auxiliary_classifier_v9")


def test_a_v2_root_equal_to_the_v1_root_is_refused(workspace, monkeypatch):
    root = workspace["root"]
    overlay = root / "overlay_same_root.yaml"
    overlay.write_text(
        _overlay(root).read_text(encoding="utf-8")
        + f"  paths:\n    aux_v2_outputs_dir: {(root / 'outputs/ham10000/stage3').as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(overlay))

    with pytest.raises(SystemExit, match="would overwrite"):
        _build(V2)
    with pytest.raises(SystemExit, match="would overwrite"):
        _run(["agreement"], V2)

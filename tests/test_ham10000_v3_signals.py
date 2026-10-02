#!/usr/bin/env python3
"""The three classifier-dependent signals and their diagnostic, extended to the v3a classifier.

Amendment 2 of docs/ham10000_v2_signal_criteria.md, written before any v3 signal existed, fixes
what this covers:

- the v3 reference and signals are built from the v3 model, under their own root, and leave the
  v1 and v2 artifacts byte for byte where they were;
- the diagnostic applies the same questions and thresholds to v3, with v3's own outcomes
  (sufficient, v3b, or the next change chosen from the evidence);
- the default two-version analysis is unchanged.

The wiring tests use randomly initialised models, as in test_ham10000_v2_signals.py: the values
mean nothing, and only the wiring is tested.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from scripts.asism import ham10000_v2_signal_diagnostics as diag  # noqa: E402
from test_ham10000_cam_reference_build import NAMESPACE, _overlay, _workspace, _write_checkpoint  # noqa: E402
from test_ham10000_compute_signals import _write_blur_calibration, _write_candidates  # noqa: E402
from test_ham10000_v2_signal_diagnostics import N, _base, _write  # noqa: E402
from test_ham10000_v2_signals import (  # noqa: E402
    CLASSIFIER_SIGNALS,
    V2,
    _build,
    _model_id,
    _provenance,
    _run,
    _snapshot,
    _v1_dir,
    _v2_dir,
    _write_v2_checkpoint,
)

V3 = "auxiliary_classifier_v3"
V3_MODEL_ID = "ham10000-classifier:6849c456a9581c598a423b93b9c2cdbdbb79118fd6d31c72fe68eefcbe5f8bfd"


def _v3_dir(root: Path) -> Path:
    return root / "outputs/ham10000/stage3_aux_v3" / NAMESPACE


def _write_v3_checkpoint(root: Path) -> Path:
    """The v1 fixture checkpoint with its head scaled differently from v2's, so all three hash apart."""
    import torch

    v1_dir = root / "checkpoints/ham10000/asism_auxiliary" / NAMESPACE
    v3_dir = root / "checkpoints/ham10000/asism_auxiliary_v3" / NAMESPACE
    v3_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(v1_dir, v3_dir)
    state = torch.load(v3_dir / "model.pt", map_location="cpu")
    state["classifier.1.weight"] = state["classifier.1.weight"] * 3.0
    torch.save(state, v3_dir / "model.pt")
    return v3_dir


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("v3-signals") as root:
        _workspace(root)
        v1 = _write_checkpoint(root)
        v2 = _write_v2_checkpoint(root)
        v3 = _write_v3_checkpoint(root)
        _write_candidates(root)
        _write_blur_calibration(root)
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        ids = {"v1_id": _model_id(v1), "v2_id": _model_id(v2), "v3_id": _model_id(v3)}
        assert len(set(ids.values())) == 3
        yield {"root": root, **ids}


# ==============================================================================================
# The committed config and the amendment
# ==============================================================================================


def test_the_committed_v3_outputs_root_is_separate_from_v1_and_v2():
    from omegaconf import OmegaConf

    paths = OmegaConf.to_container(OmegaConf.load(REPO / "configs/ham10000_stage3.yaml").paths, resolve=False)
    assert paths["aux_v3_outputs_dir"] == "${paths.project_root}/outputs/ham10000/stage3_aux_v3"
    assert len({paths["outputs_dir"], paths["aux_v2_outputs_dir"], paths["aux_v3_outputs_dir"]}) == 3


def test_the_amendment_records_the_v3_model_and_recall_used_here():
    text = (REPO / "docs/ham10000_v2_signal_criteria.md").read_text(encoding="utf-8")
    assert "Amendment 2" in text
    assert V3_MODEL_ID in text
    for name, (correct, support) in diag.REAL_RECALL_COUNTS["v3"].items():
        assert f"{correct}/{support}" in text, name
    assert diag.EXPECTED_MODEL_IDS["v3"] == V3_MODEL_ID


def test_the_v3_real_recall_reproduces_its_acceptance_balanced_accuracy():
    recall = [k / n for k, n in diag.REAL_RECALL_COUNTS["v3"].values()]
    assert np.mean(recall) == pytest.approx(0.581769634426501, abs=1e-9)
    assert set(diag.REAL_RECALL_COUNTS["v3"]) == set(diag.CLASSES)


# ==============================================================================================
# Reference and signals
# ==============================================================================================


def test_the_v3_reference_and_signals_come_from_the_v3_model_under_the_v3_root(workspace):
    root = workspace["root"]
    reference = Path(_build(V3)["path"])
    assert reference == _v3_dir(root) / "explainability_reference.json"
    assert json.loads(reference.read_text(encoding="utf-8"))["cam_model_id"] == workspace["v3_id"]

    result = _run(CLASSIFIER_SIGNALS, V3)
    assert Path(result["out_dir"]) == _v3_dir(root) / "signals"
    for name in CLASSIFIER_SIGNALS:
        provenance = _provenance(_v3_dir(root), name)
        assert provenance["cam_model_id"] == workspace["v3_id"], name
        assert provenance["auxiliary_config_key"] == V3, name


def test_the_v1_and_v2_artifacts_are_byte_identical_after_a_v3_recompute(workspace):
    root = workspace["root"]
    _build()
    _run(["iqa", *CLASSIFIER_SIGNALS])
    _build(V2)
    _run(CLASSIFIER_SIGNALS, V2)
    before = {"v1": _snapshot(_v1_dir(root)), "v2": _snapshot(_v2_dir(root))}
    assert "signals/agreement_scores.parquet" in before["v2"]

    _build(V3)
    _run(CLASSIFIER_SIGNALS, V3)

    assert _snapshot(_v1_dir(root)) == before["v1"]
    assert _snapshot(_v2_dir(root)) == before["v2"]


@pytest.mark.parametrize("signals", [["iqa"], ["similarity"], ["agreement", "iqa"]])
def test_v3_refuses_the_signals_that_do_not_depend_on_the_classifier(workspace, signals):
    with pytest.raises(SystemExit, match="recomputes only"):
        _run(signals, V3)
    assert not _v3_dir(workspace["root"]).exists()


def test_a_v2_reference_placed_in_the_v3_root_is_refused_by_model_hash(workspace):
    root = workspace["root"]
    _build(V2)
    _v3_dir(root).mkdir(parents=True)
    shutil.copy(_v2_dir(root) / "explainability_reference.json", _v3_dir(root) / "explainability_reference.json")
    with pytest.raises(SystemExit, match="different CAM model"):
        _run(["explainability"], V3)


@pytest.mark.parametrize("shared_with", ["outputs/ham10000/stage3", "outputs/ham10000/stage3_aux_v2"])
def test_a_v3_root_shared_with_another_classifier_is_refused(workspace, monkeypatch, shared_with):
    root = workspace["root"]
    overlay = root / "overlay_shared_root.yaml"
    overlay.write_text(
        _overlay(root).read_text(encoding="utf-8")
        + f"  paths:\n    aux_v3_outputs_dir: {(root / shared_with).as_posix()}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(overlay))
    with pytest.raises(SystemExit, match="would overwrite"):
        _build(V3)
    with pytest.raises(SystemExit, match="would overwrite"):
        _run(["agreement"], V3)


# ==============================================================================================
# The diagnostic with v3
# ==============================================================================================


@pytest.fixture
def root():
    with fixture_workspace("v3-diagnostics") as path:
        yield path


def _analyse3(root: Path, v3=None, **kwargs) -> dict:
    frames = {"v1": _base("v1", 1), "v2": _base("v2", 2), "v3": _base("v3", 3) if v3 is None else v3}
    _write(root, frames, **kwargs)
    return diag.analyse(root, expected_n=N, versions=("v1", "v2", "v3"))


def test_the_default_analysis_is_still_the_two_version_one(root):
    _write(root, {"v1": _base("v1", 1), "v2": _base("v2", 2), "v3": _base("v3", 3)})
    report = diag.analyse(root, expected_n=N)
    assert list(report["results"]) == ["v1", "v2"]
    assert "decided_on" not in report
    assert not any(key.startswith("v3/") for key in report["input_sha256"])
    assert diag.render_markdown(report).startswith("# HAM10000 v2 signal diagnostics")


def test_adding_v3_does_not_change_the_v1_and_v2_results(root):
    _write(root, {"v1": _base("v1", 1), "v2": _base("v2", 2), "v3": _base("v3", 3)})
    two = diag.analyse(root, expected_n=N)
    three = diag.analyse(root, expected_n=N, versions=("v1", "v2", "v3"))
    # Compared as serialised JSON: v1's df real recall is 0, so its normalised match rate is NaN,
    # and NaN != NaN in a plain dict comparison.
    for version in ("v1", "v2"):
        assert json.dumps(three["results"][version], sort_keys=True) == json.dumps(two["results"][version], sort_keys=True)


def test_a_healthy_v3_is_sufficient_and_is_the_one_decided_on(root):
    report = _analyse3(root)
    assert report["decided_on"] == "v3"
    assert report["decision"] == {"verdict": "v3_sufficient",
                                  "reason": "Q1 sensible, Q2 not influential, Q4 discriminative in >= 4 classes",
                                  "notes": []}
    text = diag.render_markdown(report)
    assert "**Decision (v3): v3_sufficient**" in text
    assert "(v2 under the same rule, for reference only: v2_sufficient" in text
    assert "Reading aids for v3" in text


def test_v3_failing_q2_calls_for_the_next_change_not_for_a_v3(root):
    v3 = _base("v3", 3)
    misses = v3.index[(v3["dx"] == "mel") & ~v3["agreement_is_argmax_match"]][:3]
    v3.loc[misses, "agreement_predicted_diagnosis"] = "nv"
    report = _analyse3(root, v3=v3)
    assert report["decision"]["verdict"] == "next_change_from_evidence"
    assert report["decision"]["reason"] == "Q2 influential"
    assert any("written in the criteria document before it is tried" in note
               for note in report["decision"]["notes"])


def test_v3_passing_q1_q2_with_a_degenerate_q3_calls_for_v3b(root):
    v3 = _base("v3", 3)
    v3["uncertainty_mutual_information"] = np.linspace(0.0, 0.002, len(v3))
    report = _analyse3(root, v3=v3)
    assert report["results"]["v3"]["q3"]["degenerate"]
    assert report["decision"]["verdict"] == "v3b"


def test_v3_weak_explainability_is_still_sufficient_with_a_weak_signal_note(root):
    v3 = _base("v3", 3)
    for name in ("nv", "mel", "bkl", "bcc"):
        v3.loc[v3["dx"] == name, "explainability_peripheral_mass"] = 0.0
    report = _analyse3(root, v3=v3)
    assert report["decision"]["verdict"] == "v3_sufficient"
    assert any("weak signal" in note for note in report["decision"]["notes"])


def test_the_mel_reading_aids_count_what_amendment_2_names(root):
    v3 = _base("v3", 3)
    nv_rows = v3.index[v3["dx"] == "nv"][:2]
    v3.loc[nv_rows, "agreement_predicted_diagnosis"] = "mel"
    v3.loc[nv_rows, "agreement_is_argmax_match"] = False
    report = _analyse3(root, v3=v3)
    aids = report["results"]["v3"]["mel_reading_aids"]
    assert aids["synthetic_nv_predicted_mel"]["k"] == 2
    assert aids["synthetic_nv_predicted_mel"]["n"] == 10
    mismatches = report["results"]["v3"]["q2"]["mismatches_predicted_as"]
    assert aids["mel_share_of_mismatches"]["k"] == mismatches["mel"]
    assert aids["mel_share_of_mismatches"]["n"] == sum(mismatches.values())
    assert "mel_reading_aids" not in report["results"]["v2"]


def test_v3_artifacts_from_another_model_are_refused(root):
    ids = {**diag.EXPECTED_MODEL_IDS, "v3": diag.EXPECTED_MODEL_IDS["v2"]}
    with pytest.raises(SystemExit, match="Refusing to report on another model"):
        _analyse3(root, model_ids=ids)


def test_v3_on_different_candidates_is_refused(root):
    v3 = _base("v3", 3)
    v3.loc[0, "image_id"] = "syn_nv_99999"
    with pytest.raises(SystemExit, match="image_id set differs"):
        _analyse3(root, v3=v3)


@pytest.mark.parametrize("versions", [("v2", "v3"), ("v1", "v3"), ("v1",), ("v1", "v2", "v4")])
def test_other_version_lists_are_refused(root, versions):
    with pytest.raises(SystemExit, match="versions must be"):
        diag.analyse(root, expected_n=N, versions=versions)

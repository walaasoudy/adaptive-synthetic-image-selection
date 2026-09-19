#!/usr/bin/env python3
"""The Grad-CAM reference artifact: written once on a GPU run, read by everything afterwards.

The danger a file introduces is that loading it becomes a second constructor that skips the checks
the real one enforces. A JSON file can claim any split, any image ids and any CAM model, and the
code that reads it is running days later on another machine with nobody watching. These tests pin
the property that matters: a stored reference is re-verified on every load, not trusted because it
was verified once.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import copy
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.ham10000_signals import (  # noqa: E402
    REFERENCE_STATISTICS,
    ReferenceLeakageError,
    build_reference_distribution,
    calibrate_against_reference,
    reference_artifact_payload,
    references_from_artifact,
)

CAM_MODEL_ID = "ham10000-classifier:" + "a" * 64
FINAL_EVAL_IDS = frozenset({"ISIC_final_1", "ISIC_final_2"})


def _reference(statistic: str, diagnosis: str, n: int = 30, offset: float = 0.0, model_id: str = CAM_MODEL_ID):
    values = np.linspace(0.0, 0.3, n) + offset
    ids = [f"{diagnosis}_{statistic[:4]}_{index}" for index in range(n)]
    return build_reference_distribution(
        values, ids, "gen_train", diagnosis, statistic, FINAL_EVAL_IDS,
        cam_model_id=model_id, cam_model_trained=True,
    )


def _payload(diagnoses=("mel", "nv")):
    references = {
        (statistic, diagnosis): _reference(statistic, diagnosis)
        for diagnosis in diagnoses
        for statistic in REFERENCE_STATISTICS
    }
    return reference_artifact_payload(references, {"split_namespace": "ham-stratified-v1"})


def test_a_reference_survives_the_round_trip_with_its_values_and_provenance_intact():
    payload = _payload()
    loaded = references_from_artifact(payload, FINAL_EVAL_IDS)

    assert sorted(loaded) == ["mel", "nv"]
    assert sorted(loaded["mel"]) == sorted(REFERENCE_STATISTICS)
    reference = loaded["mel"]["explainability_peripheral_mass"]
    assert reference.split_name == "gen_train"
    assert reference.cam_model_id == CAM_MODEL_ID
    assert reference.scientific is True
    assert reference.size == 30

    # And it still calibrates identically to the in-memory original.
    original = _reference("explainability_peripheral_mass", "mel")
    assert calibrate_against_reference(0.12, reference) == pytest.approx(
        calibrate_against_reference(0.12, original)
    )


def test_a_stored_reference_that_overlaps_final_eval_is_REFUSED_AT_LOAD():
    """The check that cannot be done once and cached: the splits on disk can change after the
    artifact was written, and the run doing the calibrating is the one that has to be disjoint."""
    payload = _payload()
    payload["references"][0]["image_ids"][0] = "ISIC_final_2"
    with pytest.raises(ReferenceLeakageError, match="final_eval_heldout"):
        references_from_artifact(payload, FINAL_EVAL_IDS)


def test_a_stored_reference_naming_a_forbidden_split_is_refused_at_load():
    payload = _payload()
    payload["references"][0]["split_name"] = "final_eval_heldout"
    with pytest.raises(ReferenceLeakageError):
        references_from_artifact(payload, FINAL_EVAL_IDS)


def test_a_stored_reference_naming_an_unpermitted_split_is_refused_at_load():
    payload = _payload()
    payload["references"][0]["split_name"] = "classifier_train"
    with pytest.raises(ReferenceLeakageError, match="not a permitted reference split"):
        references_from_artifact(payload, FINAL_EVAL_IDS)


def test_the_consumers_minimum_size_applies_not_the_one_the_file_was_built_with():
    payload = _payload()
    with pytest.raises(ValueError, match="need >= 500"):
        references_from_artifact(payload, FINAL_EVAL_IDS, min_size=500)


def test_an_empty_final_eval_id_list_cannot_be_used_to_wave_a_reference_through():
    payload = _payload()
    with pytest.raises(ReferenceLeakageError, match="cannot be verified"):
        references_from_artifact(payload, frozenset())


def test_an_artifact_must_name_exactly_one_cam_model():
    """Calibrating one image against two models' attention is not a comparison; calibrate_
    explainability refuses it per image, and this refuses to let the file exist in the first place."""
    references = {
        ("explainability_peripheral_mass", "mel"): _reference("explainability_peripheral_mass", "mel"),
        ("explainability_peripheral_mass", "nv"): _reference(
            "explainability_peripheral_mass", "nv", model_id="ham10000-classifier:" + "b" * 64
        ),
    }
    with pytest.raises(ValueError, match="exactly one CAM model"):
        reference_artifact_payload(references)


def test_a_miskeyed_reference_is_refused_rather_than_written_under_the_wrong_class():
    references = {
        ("explainability_peripheral_mass", "nv"): _reference("explainability_peripheral_mass", "mel")
    }
    with pytest.raises(ValueError, match="holds"):
        reference_artifact_payload(references)


def test_a_foreign_artifact_is_not_read_as_a_reference():
    payload = _payload()
    payload["artifact"] = "something_else"
    with pytest.raises(ValueError, match="not a HAM10000 explainability reference"):
        references_from_artifact(payload, FINAL_EVAL_IDS)


def test_the_payload_is_json_serialisable_as_written():
    import json

    payload = _payload()
    restored = json.loads(json.dumps(payload))
    assert restored == copy.deepcopy(payload)
    loaded = references_from_artifact(restored, FINAL_EVAL_IDS)
    assert loaded["nv"]["explainability_focus_area"].size == 30


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

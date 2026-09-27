#!/usr/bin/env python3
"""The v1-vs-v2 signal diagnostic: the criteria are the document's, every question can pass and fail,
the decision table is applied row by row, and inputs it cannot vouch for are refused.

Hand-built artifacts with known values (10 candidates per class); nothing here reads real outputs.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from scripts.asism import ham10000_v2_signal_diagnostics as diag  # noqa: E402
from scripts.utils.manifest import sha256_file  # noqa: E402

PER_CLASS = 10
N = PER_CLASS * len(diag.CLASSES)
# Matches per class that follow the real recall ordering, all above chance (1.43 of 10).
GOOD_MATCHES = {"nv": 9, "mel": 5, "bkl": 6, "bcc": 7, "akiec": 5, "vasc": 6, "df": 4}


def _base(version: str, seed: int = 0) -> pd.DataFrame:
    """One classifier's joined rows in a healthy state: sensible agreement, no mel->nv, spread-out
    uncertainty, discriminative explainability, no redundancy."""
    rng = np.random.default_rng(seed)
    rows = []
    for name in diag.CLASSES:
        rival = "bkl" if name != "bkl" else "mel"
        for i in range(PER_CLASS):
            match = i < GOOD_MATCHES[name]
            rows.append({
                "image_id": f"syn_{name}_{i:05d}",
                "dx": name,
                "agreement_intended_diagnosis": name,
                "agreement_predicted_diagnosis": name if match else rival,
                "agreement_is_argmax_match": match,
                "agreement_score": float(rng.uniform()),
                "uncertainty_mutual_information": float(rng.uniform(0.0, 0.04)),
                "uncertainty_mean_std": float(rng.uniform(0.0, 0.05)),
                "explainability_calibrated_typicality": float(rng.uniform()),
                "explainability_peripheral_mass": float(rng.uniform()),
            })
    frame = pd.DataFrame(rows)
    return frame


def _write(root: Path, frames: dict[str, pd.DataFrame], model_ids: dict | None = None,
           bands: dict | None = None) -> Path:
    """Write the downloaded-artifact layout the script reads."""
    model_ids = model_ids or diag.EXPECTED_MODEL_IDS
    columns = {
        "uncertainty": ["uncertainty_mutual_information", "uncertainty_mean_std", "uncertainty_band"],
        "agreement": ["agreement_score", "agreement_intended_diagnosis", "agreement_predicted_diagnosis",
                      "agreement_is_argmax_match"],
        "explainability": ["explainability_calibrated_typicality", "explainability_peripheral_mass"],
    }
    for version, frame in frames.items():
        frame = frame.copy()
        mutual = frame["uncertainty_mutual_information"]
        frame["uncertainty_band"] = np.where(mutual <= 0.05, "low", np.where(mutual <= 0.20, "moderate", "extreme"))
        if bands and version in bands:
            frame["uncertainty_band"] = bands[version]
        directory = root / version
        directory.mkdir(parents=True, exist_ok=True)
        for signal, cols in columns.items():
            path = directory / f"{signal}_scores.parquet"
            frame[["image_id", *cols]].to_parquet(path, index=False)
            provenance = {"cam_model_id": model_ids[version], "parquet_sha256": sha256_file(path)}
            (directory / f"{signal}_scores.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    (root / "stage2").mkdir(exist_ok=True)
    frames["v2"][["image_id", "dx"]].to_csv(root / "stage2" / "all_candidates.csv", index=False)
    return root


@pytest.fixture
def root():
    with fixture_workspace("v2-diagnostics") as path:
        yield path


def _analyse(root: Path, v1: pd.DataFrame | None = None, v2: pd.DataFrame | None = None, **kwargs) -> dict:
    frames = {"v1": _base("v1", 1) if v1 is None else v1, "v2": _base("v2", 2) if v2 is None else v2}
    _write(root, frames, **kwargs)
    return diag.analyse(root, expected_n=N)


# ==============================================================================================
# The criteria are the document's
# ==============================================================================================


def test_the_criteria_constants_are_the_pre_registered_ones():
    assert diag.CRITERIA == {
        "q1_chance": 1.0 / 7.0,
        "q1_max_classes_at_or_below_chance": 1,
        "q1_min_spearman_with_real_recall": 0.5,
        "q2_max_mel_to_nv_share": 0.30,
        "q2_max_nv_share_of_mismatches": 0.50,
        "q3_degenerate_p90_below": 0.01,
        "q3_degenerate_iqr_below": 0.005,
        "q4_min_iqr": 0.15,
        "q4_max_tie_share": 0.20,
        "q4_min_discriminative_classes": 4,
        "q5_min_abs_spearman": 0.7,
        "q5_min_redundant_classes": 4,
    }
    assert (diag.UNCERTAINTY_LOW_BAND_MAX, diag.UNCERTAINTY_MODERATE_BAND_MAX) == (0.05, 0.20)


def test_the_criteria_document_states_the_same_thresholds():
    text = (REPO / "docs/ham10000_v2_signal_criteria.md").read_text(encoding="utf-8")
    for phrase in ["at most **1** class", "**≥ 0.5**", "**≥ 30%**", "**≥ 50%**", "P90 **< 0.01**",
                   "IQR **< 0.005**", "IQR is **≥ 0.15**", "tie share is **< 20%**", "|ρ| is **≥ 0.7**",
                   "**≥ 4** of the 7 classes", "Amendment 1"]:
        assert phrase in text, phrase


def test_the_real_recall_reproduces_each_classifiers_balanced_accuracy():
    def balanced(version):
        return np.mean([k / n for k, n in diag.REAL_RECALL_COUNTS[version].values()])

    assert balanced("v1") == pytest.approx(0.47827, abs=1e-4)
    assert balanced("v2") == pytest.approx(0.5998, abs=1e-4)
    assert set(diag.REAL_RECALL_COUNTS["v2"]) == set(diag.CLASSES)


# ==============================================================================================
# Small statistics
# ==============================================================================================


def test_wilson_interval_known_values():
    low, high = diag.wilson_interval(6, 15)
    assert (low, high) == pytest.approx((0.1982, 0.6425), abs=5e-4)
    assert diag.wilson_interval(0, 10)[0] == 0.0
    a, b = diag.wilson_interval(3, 10), diag.wilson_interval(7, 10)
    assert a[0] == pytest.approx(1 - b[1]) and a[1] == pytest.approx(1 - b[0])
    assert all(np.isnan(diag.wilson_interval(0, 0)))


def test_spearman_and_tie_share():
    assert diag.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert diag.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(diag.spearman([1, 2, 3], [5, 5, 5]))
    assert diag.tie_share([1.0, 1.0, 2.0, 3.0]) == pytest.approx(0.5)
    assert diag.tie_share([0.1, 0.2, 0.3]) == 0.0


# ==============================================================================================
# Each question passes and fails
# ==============================================================================================


def test_a_healthy_v2_is_sufficient(root):
    report = _analyse(root)
    v2 = report["results"]["v2"]
    assert v2["q1"]["sensible"] and not v2["q2"]["influential"]
    assert not v2["q3"]["degenerate"]
    assert v2["q4"]["n_discriminative_classes"] == 7
    assert not v2["q5"]["any_redundant"]
    assert report["decision"] == {"verdict": "v2_sufficient",
                                  "reason": "Q1 sensible, Q2 not influential, Q4 discriminative in >= 4 classes",
                                  "notes": []}


def test_q1_fails_when_two_classes_are_at_or_below_chance(root):
    v2 = _base("v2", 2)
    for name in ("df", "vasc"):
        rows = v2["dx"] == name
        v2.loc[rows, "agreement_is_argmax_match"] = False
        v2.loc[rows, "agreement_predicted_diagnosis"] = "bkl"
    report = _analyse(root, v2=v2)
    assert report["results"]["v2"]["q1"]["n_classes_at_or_below_chance"] == 2
    assert not report["results"]["v2"]["q1"]["sensible"]
    assert report["decision"]["verdict"] == "build_v3"
    assert report["decision"]["reason"] == "Q1 not sensible"


def test_q1_fails_when_the_synthetic_ordering_contradicts_the_real_recall(root):
    v2 = _base("v2", 2)
    # nv (highest real recall) matched least, df (lowest) matched most; everything above chance.
    reversed_matches = {"nv": 2, "mel": 7, "bkl": 5, "bcc": 3, "akiec": 7, "vasc": 4, "df": 9}
    for name, k in reversed_matches.items():
        index = v2.index[v2["dx"] == name]
        v2.loc[index, "agreement_is_argmax_match"] = [i < k for i in range(PER_CLASS)]
        v2.loc[index, "agreement_predicted_diagnosis"] = [name if i < k else ("bkl" if name != "bkl" else "mel")
                                                          for i in range(PER_CLASS)]
    q1 = _analyse(root, v2=v2)["results"]["v2"]["q1"]
    assert q1["n_classes_at_or_below_chance"] == 0
    assert q1["spearman_match_vs_real_recall"] < 0.5
    assert not q1["sensible"]


def test_q1_normalises_the_match_rate_by_the_real_recall(root):
    q1 = _analyse(root)["results"]["v2"]["q1"]
    mel = q1["per_class"]["mel"]
    assert mel["rate"] == pytest.approx(0.5)
    assert mel["real_recall"] == pytest.approx(69 / 149)
    assert mel["normalised_match_rate"] == pytest.approx(0.5 / (69 / 149))
    assert mel["ci95"] == pytest.approx(list(diag.wilson_interval(5, 10)))


def test_q2_mel_to_nv_at_thirty_percent_is_influential(root):
    v2 = _base("v2", 2)
    mel_misses = v2.index[(v2["dx"] == "mel") & ~v2["agreement_is_argmax_match"]][:3]
    v2.loc[mel_misses, "agreement_predicted_diagnosis"] = "nv"
    report = _analyse(root, v2=v2)
    q2 = report["results"]["v2"]["q2"]
    assert q2["mel_predicted_nv"]["rate"] == pytest.approx(0.30)
    assert q2["influential"]
    assert report["decision"] == {"verdict": "build_v3", "reason": "Q2 influential", "notes": []}


def test_q2_nv_as_the_sink_of_half_the_mismatches_is_influential(root):
    v2 = _base("v2", 2)
    misses = v2.index[~v2["agreement_is_argmax_match"] & (v2["dx"] != "mel") & (v2["dx"] != "nv")]
    v2.loc[misses, "agreement_predicted_diagnosis"] = "nv"
    q2 = _analyse(root, v2=v2)["results"]["v2"]["q2"]
    assert q2["mel_predicted_nv"]["rate"] == 0.0
    assert q2["nv_share_of_mismatches"]["rate"] >= 0.5
    assert q2["influential"]


def test_q3_degenerate_is_a_note_not_a_reason_for_v3(root):
    v2 = _base("v2", 2)
    v2["uncertainty_mutual_information"] = np.linspace(0.0, 0.002, len(v2))
    report = _analyse(root, v2=v2)
    assert report["results"]["v2"]["q3"]["degenerate"]
    assert report["decision"]["verdict"] == "v2_sufficient"
    assert any(note.startswith("Q3 degenerate") for note in report["decision"]["notes"])


def test_q4_weak_in_most_classes_keeps_v2_and_records_a_weak_signal(root):
    v2 = _base("v2", 2)
    for name in ("nv", "mel", "bkl", "bcc"):
        v2.loc[v2["dx"] == name, "explainability_peripheral_mass"] = 0.0  # every value tied
    report = _analyse(root, v2=v2)
    q4 = report["results"]["v2"]["q4"]
    assert q4["per_class"]["nv"]["tie_share_peripheral_mass"] == 1.0
    assert q4["n_discriminative_classes"] == 3
    assert report["decision"]["verdict"] == "v2_sufficient"
    assert report["decision"]["reason"].endswith("Q4 discriminative in < 4 classes")
    assert any("weak signal" in note for note in report["decision"]["notes"])
    assert q4["per_class"]["df"]["fragile"]


def test_q4_a_narrow_spread_is_not_discriminative(root):
    v2 = _base("v2", 2)
    rows = v2["dx"] == "bcc"
    v2.loc[rows, "explainability_calibrated_typicality"] = np.linspace(0.5, 0.6, PER_CLASS)
    bcc = _analyse(root, v2=v2)["results"]["v2"]["q4"]["per_class"]["bcc"]
    assert bcc["iqr_calibrated_typicality"] < 0.15
    assert not bcc["discriminative"]


def test_q5_a_restated_signal_is_redundant_but_does_not_change_the_verdict(root):
    v2 = _base("v2", 2)
    v2["agreement_score"] = v2["explainability_calibrated_typicality"] * 2.0
    report = _analyse(root, v2=v2)
    pair = report["results"]["v2"]["q5"]["pairs"]["agreement~explainability"]
    assert pair["pooled_spearman"] == pytest.approx(1.0)
    assert pair["n_classes_abs_rho_at_least_threshold"] == 7 and pair["redundant"]
    assert report["decision"]["verdict"] == "v2_sufficient"
    assert any(note.startswith("Q5 redundancy") for note in report["decision"]["notes"])


def test_both_q1_and_q2_failing_names_both(root):
    v2 = _base("v2", 2)
    for name in ("df", "vasc", "mel"):
        rows = v2["dx"] == name
        v2.loc[rows, "agreement_is_argmax_match"] = False
        v2.loc[rows, "agreement_predicted_diagnosis"] = "nv"
    report = _analyse(root, v2=v2)
    assert report["decision"]["verdict"] == "build_v3"
    assert report["decision"]["reason"] == "Q1 not sensible and Q2 influential"


def test_the_decision_is_v2s_and_v1_is_only_a_reference(root):
    v1 = _base("v1", 1)
    v1.loc[v1["dx"] == "mel", "agreement_predicted_diagnosis"] = "nv"
    v1.loc[v1["dx"] == "mel", "agreement_is_argmax_match"] = False
    report = _analyse(root, v1=v1)
    assert report["results"]["v1"]["rule_applied"]["verdict"] == "build_v3"
    assert report["decision"]["verdict"] == "v2_sufficient"


def test_the_report_renders_every_question(root):
    text = diag.render_markdown(_analyse(root))
    for heading in ["## Q1", "## Q2", "## Q3", "## Q4", "## Q5", "**Decision (v2): v2_sufficient**"]:
        assert heading in text
    assert "df (fragile)" in text


# ==============================================================================================
# Refusals
# ==============================================================================================


def test_another_models_artifacts_are_refused(root):
    ids = {"v1": diag.EXPECTED_MODEL_IDS["v1"], "v2": diag.EXPECTED_MODEL_IDS["v1"]}
    with pytest.raises(SystemExit, match="Refusing to report on another model"):
        _analyse(root, model_ids=ids)


def test_a_parquet_that_does_not_match_its_provenance_hash_is_refused(root):
    _write(root, {"v1": _base("v1", 1), "v2": _base("v2", 2)})
    path = root / "v2" / "agreement_scores.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "agreement_score"] = 0.123456
    frame.to_parquet(path, index=False)
    with pytest.raises(SystemExit, match="does not match its provenance hash"):
        diag.analyse(root, expected_n=N)


def test_a_wrong_candidate_count_is_refused(root):
    _write(root, {"v1": _base("v1", 1), "v2": _base("v2", 2)})
    with pytest.raises(SystemExit, match="expected 3168"):
        diag.analyse(root)


def test_v1_and_v2_on_different_candidates_are_refused(root):
    v1 = _base("v1", 1)
    v1.loc[0, "image_id"] = "syn_nv_99999"
    with pytest.raises(SystemExit, match="image_id set differs"):
        _analyse(root, v1=v1)


def test_an_intended_diagnosis_that_disagrees_with_the_pool_is_refused(root):
    v2 = _base("v2", 2)
    v2.loc[0, "agreement_intended_diagnosis"] = "mel"
    with pytest.raises(SystemExit, match="disagrees with the pool's dx"):
        _analyse(root, v2=v2)


def test_a_mutual_information_column_in_another_unit_stops_the_run(root):
    v2 = _base("v2", 2)
    # Bands written from the raw (nats) value, i.e. the column is not the one the bands describe.
    bands = np.where(v2["uncertainty_mutual_information"] * np.log(7) * 3 <= 0.05, "low", "moderate")
    with pytest.raises(SystemExit, match="does not reproduce uncertainty_band"):
        _analyse(root, v2=v2, bands={"v2": bands})

"""Known-answer tests for scripts/utils/metrics.py.

These use analytically-derived expected values, not fixtures blessed from the code's own output —
a test that only asserts "the code does what it currently does" would not catch a wrong AUROC.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils.metrics import (  # noqa: E402
    auroc,
    average_precision,
    benjamini_hochberg,
    expected_calibration_error,
    holm_bonferroni,
    macro_auroc_masked,
    paired_bootstrap_difference,
    patient_level_bootstrap,
    sensitivity_specificity_f1,
)


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 1.0


def test_auroc_perfect_inversion():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == 0.0


def test_auroc_all_ties_is_half():
    # Every score identical -> every pair is a tie -> AUROC exactly 0.5 via mid-ranks.
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    assert auroc(scores, labels) == 0.5


def test_auroc_known_value():
    # 2 positives (0.6, 0.4), 2 negatives (0.5, 0.3).
    # Pairs (pos,neg): (.6,.5)=1, (.6,.3)=1, (.4,.5)=0, (.4,.3)=1  -> 3/4 = 0.75
    scores = np.array([0.6, 0.4, 0.5, 0.3])
    labels = np.array([1, 1, 0, 0])
    assert auroc(scores, labels) == 0.75


def test_auroc_undefined_when_single_class():
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))
    assert np.isnan(auroc(np.array([0.1, 0.9]), np.array([0, 0])))


def test_average_precision_perfect():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert average_precision(scores, labels) == 1.0


def test_average_precision_known_value():
    # Ranked: 0.9(pos) 0.8(neg) 0.7(pos). Precision at hits: 1/1=1.0, 2/3=0.667. AP=(1+0.667)/2
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([1, 0, 1])
    assert abs(average_precision(scores, labels) - (1.0 + 2 / 3) / 2) < 1e-12


def test_sensitivity_specificity_f1():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0])
    sensitivity, specificity, f1 = sensitivity_specificity_f1(scores, labels, threshold=0.5)
    assert sensitivity == 0.5   # 1 of 2 positives caught
    assert specificity == 0.5   # 1 of 2 negatives correctly rejected
    assert abs(f1 - 0.5) < 1e-12


def test_ece_perfectly_calibrated_is_zero():
    # Predictions of exactly 0 and 1 matching outcomes -> no calibration gap.
    scores = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(scores, labels) == 0.0


def test_ece_maximally_miscalibrated_is_one():
    scores = np.array([1.0, 1.0])
    labels = np.array([0.0, 0.0])
    assert abs(expected_calibration_error(scores, labels) - 1.0) < 1e-12


def test_macro_auroc_masked_excludes_masked_positions():
    labels = ["Cardiomegaly", "Edema"]
    # Row 3's Cardiomegaly is masked; if the mask were ignored the AUROC would change.
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    targets = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    masks = np.array([[1, 1], [1, 1], [1, 1], [0, 1]], dtype=bool)

    result = macro_auroc_masked(
        probabilities, targets, masks, labels=labels, primary_labels=labels
    )
    assert result["per_label"]["Cardiomegaly"]["effective_n"] == 3
    assert result["per_label"]["Edema"]["effective_n"] == 4
    assert result["macro_auroc"] == 1.0


def test_macro_auroc_excludes_insufficient_support():
    labels = ["Cardiomegaly", "Edema"]
    probabilities = np.array([[0.9, 0.1], [0.8, 0.9]])
    targets = np.array([[1, 1], [0, 1]], dtype=np.float32)
    masks = np.array([[1, 1], [1, 1]], dtype=bool)

    # Edema has no negatives -> ineligible -> excluded from the macro rather than counted as 0.5.
    result = macro_auroc_masked(
        probabilities, targets, masks, labels=labels, primary_labels=labels
    )
    assert result["per_label"]["Edema"]["eligible"] is False
    assert result["n_labels_in_macro"] == 1
    assert result["n_labels_excluded"] == 1


def test_patient_level_bootstrap_resamples_patients_not_rows():
    # 2 patients, 5 rows. Patient A is over-represented by row count; patient-level resampling must
    # move all of a patient's rows together, so only 3 distinct means are reachable.
    patient_ids = np.array(["A", "A", "A", "B", "B"])
    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0])

    def metric(indices):
        return float(values[indices].mean())

    result = patient_level_bootstrap(patient_ids, metric, n_resamples=200, seed=7)
    assert result["n_patients"] == 2
    assert result["point_estimate"] == 0.6
    assert result["ci_lower"] >= 0.0 and result["ci_upper"] <= 1.0


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    n = 300
    patient_ids = np.arange(n).astype(str)
    labels = rng.integers(0, 2, size=n)
    strong = np.where(labels == 1, rng.uniform(0.6, 1.0, n), rng.uniform(0.0, 0.4, n))
    weak = rng.uniform(0.0, 1.0, n)

    result = paired_bootstrap_difference(
        patient_ids,
        lambda idx: auroc(strong[idx], labels[idx]),
        lambda idx: auroc(weak[idx], labels[idx]),
        n_resamples=200,
        seed=1,
    )
    assert result["observed_difference"] > 0.2
    assert result["ci_lower"] > 0.0     # CI excludes zero
    assert result["p_value"] < 0.05


def test_paired_bootstrap_no_difference_for_identical_scores():
    rng = np.random.default_rng(3)
    n = 200
    patient_ids = np.arange(n).astype(str)
    labels = rng.integers(0, 2, size=n)
    scores = rng.uniform(size=n)

    result = paired_bootstrap_difference(
        patient_ids,
        lambda idx: auroc(scores[idx], labels[idx]),
        lambda idx: auroc(scores[idx], labels[idx]),
        n_resamples=100,
        seed=2,
    )
    assert result["observed_difference"] == 0.0
    assert result["p_value"] == 1.0


def test_holm_bonferroni_known_sequence():
    # Classic worked example: p = .01, .02, .03, .04 at alpha .05, n=4.
    # Adjusted: .04, .06, .06, .06 -> only the first is rejected (step-down stops at the first fail).
    result = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}, alpha=0.05)
    assert abs(result["a"]["adjusted_p_value"] - 0.04) < 1e-12
    assert result["a"]["rejected"] is True
    assert result["b"]["rejected"] is False
    assert result["c"]["rejected"] is False
    assert result["d"]["rejected"] is False


def test_holm_bonferroni_is_monotonic():
    result = holm_bonferroni({"a": 0.001, "b": 0.5, "c": 0.02}, alpha=0.05)
    adjusted = sorted((entry["p_value"], entry["adjusted_p_value"]) for entry in result.values())
    values = [pair[1] for pair in adjusted]
    assert values == sorted(values), "adjusted p-values must be non-decreasing in raw p"


def test_benjamini_hochberg_rejects_more_than_holm():
    p_values = {f"t{i}": p for i, p in enumerate([0.001, 0.008, 0.02, 0.03, 0.04])}
    fdr = benjamini_hochberg(p_values, alpha=0.05)
    holm = holm_bonferroni(p_values, alpha=0.05)
    n_fdr = sum(1 for entry in fdr.values() if entry["rejected"])
    n_holm = sum(1 for entry in holm.values() if entry["rejected"])
    assert n_fdr >= n_holm, "FDR should be at least as permissive as Holm at the same alpha"


def test_corrections_label_their_family():
    holm = holm_bonferroni({"a": 0.01}, alpha=0.05)
    fdr = benjamini_hochberg({"a": 0.01}, alpha=0.05)
    assert holm["a"]["family"] == "confirmatory"
    assert fdr["a"]["family"] == "exploratory"


def test_corrections_handle_undefined_p_values():
    result = holm_bonferroni({"a": 0.01, "b": float("nan")}, alpha=0.05)
    assert result["b"]["rejected"] is False
    assert "note" in result["b"]


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

#!/usr/bin/env python3
"""The metrics Stage 5 adds after v1 (contract §12, decided 2026-10-04): macro average precision,
multi-class Brier score and top-label ECE with 15 equal-width bins.

Each metric is checked against a value worked out by hand. The rest is about where the metrics
appear: v1 reports exactly what it reported, every later protocol reports the three added ones, and
adding them does not touch the confirmatory family.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from test_ham10000_stage4_aggregate import SEEDS  # noqa: E402
from test_ham10000_stage5 import _prediction_frame, _reference, _write_predictions  # noqa: E402

from scripts.eval.ham10000_compare_conditions import (  # noqa: E402
    LOWER_IS_BETTER,
    PRIMARY_METRIC,
    compare,
    metric_functions,
    run,
)
from scripts.utils.ham10000_conditions import PROTOCOLS, V2_ADDED_METRICS, get_protocol  # noqa: E402
from scripts.utils.ham10000_metrics import (  # noqa: E402
    ECE_BINS,
    macro_average_precision,
    multiclass_brier_score,
    top_label_ece,
)

V1_METRICS = ["balanced_accuracy", "macro_f1", "macro_auroc_ovr", "accuracy"]


def test_brier_score_by_hand():
    certain = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    # first image certain and right: 0. second certain and wrong: (0-1)^2 + (1-0)^2 = 2.
    assert multiclass_brier_score(certain, np.array([0, 0])) == pytest.approx(1.0)
    uniform = np.full((4, 3), 1 / 3)
    # (1/3 - 1)^2 + 2 * (1/3)^2 = 2/3 for every image
    assert multiclass_brier_score(uniform, np.array([0, 1, 2, 0])) == pytest.approx(2 / 3)
    assert np.isnan(multiclass_brier_score(np.zeros((0, 3)), np.array([], dtype=int)))


def test_top_label_ece_by_hand():
    assert ECE_BINS == 15
    # four images at confidence 0.8, three of them right: |0.75 - 0.8| = 0.05, one bin
    same = np.tile([0.8, 0.1, 0.1], (4, 1))
    assert top_label_ece(same, np.array([0, 0, 0, 1])) == pytest.approx(0.05)
    # two bins of two images: confidence 0.9 both right (gap 0.1), confidence 0.6 one right (gap 0.1)
    two = np.array([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.6, 0.4, 0.0], [0.6, 0.4, 0.0]])
    assert top_label_ece(two, np.array([0, 0, 0, 1])) == pytest.approx(0.5 * 0.1 + 0.5 * 0.1)
    # certain and always right is calibrated
    assert top_label_ece(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0)
    # certain and always wrong is as miscalibrated as it gets
    assert top_label_ece(np.eye(3), np.array([1, 2, 0])) == pytest.approx(1.0)


def test_the_bin_count_changes_the_number_so_it_is_fixed():
    """Confidences 0.62 and 0.68 share a bin of width 0.1 and fall in different bins of width 1/15."""
    probabilities = np.array([[0.62, 0.38], [0.68, 0.32]])
    truth = np.array([0, 1])
    ten = top_label_ece(probabilities, truth, n_bins=10)        # one bin: |0.5 - 0.65| = 0.15
    fifteen = top_label_ece(probabilities, truth)                # 0.5*|1-0.62| + 0.5*|0-0.68| = 0.53
    assert ten == pytest.approx(0.15)
    assert fifteen == pytest.approx(0.53)


def test_macro_average_precision_by_hand():
    perfect = np.eye(3)[[0, 1, 2, 0]]
    assert macro_average_precision(perfect, np.array([0, 1, 2, 0]), 3) == pytest.approx(1.0)
    # class 0: scores 0.9 (neg), 0.8 (pos) -> AP 1/2. class 1: 0.1 (pos), 0.2 (neg) -> AP 1/2.
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])
    assert macro_average_precision(probabilities, np.array([1, 0]), 2) == pytest.approx(0.5)
    # a class that never occurs is left out of the mean, not scored as 0
    assert macro_average_precision(np.eye(3)[[0, 1]], np.array([0, 1]), 3) == pytest.approx(1.0)


def test_v1_reports_exactly_the_four_metrics_it_reported():
    assert list(metric_functions()) == V1_METRICS
    assert list(metric_functions(get_protocol("v1"))) == V1_METRICS
    assert get_protocol("v1").added_metrics == ()
    by_condition = {c: [_prediction_frame(c, seed) for seed in SEEDS] for c in ("A", "B", "C")}
    result = compare(by_condition, _reference(), 60, 42, 0.05, get_protocol("v1"))
    assert "added_metrics" not in result and "lower_is_better" not in result
    for condition in ("A", "B", "C"):
        assert not set(V2_ADDED_METRICS) & set(result["per_condition"][condition])
    assert not [name for name in result["exploratory"] if name.split(":")[1] in V2_ADDED_METRICS]


def test_every_protocol_after_v1_adds_the_three():
    for name, protocol in PROTOCOLS.items():
        if name != "v1":
            assert protocol.added_metrics == V2_ADDED_METRICS, name
            assert list(metric_functions(protocol)) == V1_METRICS + list(V2_ADDED_METRICS)
    assert set(LOWER_IS_BETTER) <= set(V2_ADDED_METRICS)


def test_the_learned_protocol_reports_them_and_the_confirmatory_family_is_untouched():
    protocol = get_protocol("asism_v2_learned")
    accuracies = {"A": 0.55, "B": 0.60, "C": 0.70, "D": 0.60}
    with fixture_workspace("v2-added-metrics") as root:
        _write_predictions(root, accuracies)
        result = run(root, n_resamples=60, protocol_name=protocol.name)
        table = pd.read_csv(result["table_path"])

    assert result["confirmatory_family"] == [f"C_vs_D:{PRIMARY_METRIC}"]
    assert list(result["confirmatory"]) == [f"C_vs_D:{PRIMARY_METRIC}"]
    assert result["added_metrics"] == list(V2_ADDED_METRICS)
    assert result["lower_is_better"] == ["brier_score", "ece_top_label"]
    for condition in protocol.conditions:
        for name in V2_ADDED_METRICS:
            entry = result["per_condition"][condition][name]
            assert entry["ci_lower"] <= entry["point_estimate"] <= entry["ci_upper"]
            assert {name, f"{name}_ci_lower", f"{name}_ci_upper"} <= set(table.columns)
    for name in V2_ADDED_METRICS:
        for pair in ("C_vs_D", "C_vs_B", "C_vs_A"):
            assert result["exploratory"][f"{pair}:{name}"]["family"] == "exploratory"


def test_a_better_condition_has_the_lower_brier_score():
    """The sign convention a reader needs: left minus right, and lower is better."""
    protocol = get_protocol("asism_v2_learned")
    by_condition = {c: [_prediction_frame(c, seed, accuracy=0.5) for seed in SEEDS] for c in ("A", "B", "D")}
    by_condition["C"] = [_prediction_frame("C", seed, perfect=True) for seed in SEEDS]
    result = compare(by_condition, _reference(), 60, 42, 0.05, protocol)
    assert result["exploratory"]["C_vs_D:brier_score"]["observed_difference"] < 0
    assert result["exploratory"]["C_vs_D:macro_average_precision"]["observed_difference"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

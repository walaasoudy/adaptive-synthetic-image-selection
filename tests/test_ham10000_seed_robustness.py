"""Seed-robustness analysis (protocol §5) and the ASISM v2 protocols and grid input checks."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from scripts.eval import ham10000_seed_robustness as rob
from scripts.followup import ham10000_asism_v2_stage4_grid as grid
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS
from scripts.utils.ham10000_conditions import get_protocol

K = len(CLASSIFIER_TARGET_LABELS)


def _frames(truth, accuracy, seeds, rng):
    frames = []
    for seed in seeds:
        predicted = np.where(rng.uniform(size=len(truth)) < accuracy, truth, rng.integers(0, K, len(truth)))
        probs = np.full((len(truth), K), 0.01)
        probs[np.arange(len(truth)), predicted] = 0.9
        frame = pd.DataFrame(probs, columns=[f"prob_{c}" for c in CLASSIFIER_TARGET_LABELS])
        frames.append(frame.assign(seed=seed))
    return frames


def _reference(n=140, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.repeat(np.arange(K), n // K)
    return pd.DataFrame({"true_class_index": truth, "lesion_id": [f"L{i // 2}" for i in range(len(truth))]}), rng


def _plain_ba(truth, predicted):
    return float(np.mean([(predicted[truth == c] == c).mean() for c in np.unique(truth)]))


def test_weighted_ba_with_unit_weights_is_balanced_accuracy():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, K, 300)
    predicted = np.where(rng.uniform(size=300) < 0.6, truth, rng.integers(0, K, 300))
    assert rob.balanced_accuracy_weighted(truth, predicted, np.ones(300)) == pytest.approx(_plain_ba(truth, predicted))


def test_weights_equal_expanded_rows():
    rng = np.random.default_rng(2)
    truth = rng.integers(0, K, 50)
    predicted = rng.integers(0, K, 50)
    weights = rng.integers(0, 4, 50)
    expanded_t, expanded_p = np.repeat(truth, weights), np.repeat(predicted, weights)
    assert rob.balanced_accuracy_weighted(truth, predicted, weights.astype(float)) == pytest.approx(
        _plain_ba(expanded_t, expanded_p))


def test_welch_matches_scipy():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.6, 0.05, 20), rng.normal(0.57, 0.07, 20)
    out = rob.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert out["p_value"] == pytest.approx(ref.pvalue)
    assert out["difference"] == pytest.approx(a.mean() - b.mean())
    assert out["ci95"][0] < out["difference"] < out["ci95"][1]


def test_bootstrap_is_deterministic_and_zero_for_identical_conditions():
    reference, rng = _reference()
    truth = reference["true_class_index"].to_numpy()
    lesions = reference["lesion_id"].to_numpy()
    left = _frames(truth, 0.6, [42, 43, 44], rng)
    one = rob.seed_lesion_bootstrap(left, left, truth, lesions, n_resamples=200)
    two = rob.seed_lesion_bootstrap(left, left, truth, lesions, n_resamples=200)
    assert one == two
    assert one["observed_difference"] == pytest.approx(0.0)
    assert one["ci95"][0] <= 0.0 <= one["ci95"][1]


def test_robustness_detects_a_clear_difference():
    reference, rng = _reference(n=700)
    truth = reference["true_class_index"].to_numpy()
    by_condition = {"C": _frames(truth, 0.8, range(42, 52), rng), "D": _frames(truth, 0.4, range(42, 52), rng)}
    out = rob.robustness(by_condition, reference, [("C", "D")], n_resamples=200)
    comparison = out["comparisons"]["C_vs_D"]
    assert comparison["welch"]["p_value"] < 1e-6
    assert comparison["seed_lesion_bootstrap"]["ci95"][0] > 0
    assert out["per_seed"]["C"]["seeds"] == list(range(42, 52))


# ---- protocols and grid -----------------------------------------------------------------------

def test_asism_v2_protocols():
    v2 = get_protocol("asism_v2")
    assert v2.conditions == ("A", "B", "C", "D")
    assert v2.confirmatory == (("C", "D"),)
    assert ("C", "D") in v2.equal_counts_required
    assert v2.selection_manifest.covers == ("C", "D")
    none = get_protocol("asism_v2_none")
    assert none.conditions == ("A", "B") and none.confirmatory == ()
    assert none.selection_manifest.covers == ()


def test_grid_refuses_a_file_the_selection_did_not_hash(tmp_path, monkeypatch):
    protocol = get_protocol("asism_v2")
    path = tmp_path / "c_selected.csv"
    path.write_text("image_id,image_path,dx\nx,/x.png,mel\n")
    monkeypatch.setattr(grid.stage4, "resolve_synthetic_manifest", lambda config, condition, seed: str(path))
    good = {"c_selected.csv": grid.sha256_file(path)}
    assert grid.verify_inputs(protocol, None, 42, "C", good) == good["c_selected.csv"]
    with pytest.raises(grid.GridError):
        grid.verify_inputs(protocol, None, 42, "C", {"c_selected.csv": "0" * 64})
    assert grid.verify_inputs(protocol, None, 42, "B", {}) is None

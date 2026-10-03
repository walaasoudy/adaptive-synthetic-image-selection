"""Reliability gate G1, ranker acceptance on unseen subsets, selection stability. Toy data only."""
import numpy as np
import pandas as pd
import pytest

from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.gates import acceptance, reliability_gate, selection_stability, stratified_spearman
from scripts.asism_v2.prereg import FROZEN

SEEDS = [1, 2, 3]
WEIGHTS = {"a": np.array([.9, .6, .3, .3]), "b": np.array([.4, -.5, .2, .1])}
PREREG = {**FROZEN, "prereg_sha256": fingerprint(FROZEN),
          "acceptance": {**FROZEN["acceptance"], "permutations": 500},
          "stopping": {**FROZEN["stopping"], "stability_fit_seeds": [42, 43]}}
FIT = {"seed": 42, "bootstrap": 20, "max_epochs": 300, "patience": 60, "learning_rate": 0.03,
       "weight_decay": 1e-5, "initial_lam": 0.02}


def _toy(signals_matter=True, noise=0.002):
    rng = np.random.default_rng(11)
    rows, ids = [], {}
    for role, n in (("train", 160), ("validation", 70), ("test", 70)):
        for dx in ("a", "b"):
            ids[role, dx] = [f"{role}-{dx}-{j}" for j in range(n)]
            rows += [{"image_id": i, "dx": dx, **dict(zip(SIGNALS, rng.normal(0, 1, 4)))} for i in ids[role, dx]]
    frame = pd.DataFrame(rows)
    x = frame.set_index("image_id")
    protocol = {"dataset": "synthetic_toy", "instrument_decision": "synthetic_toy_only", "recipe": "gates"}
    subsets, fit_rows, test_rows = {}, [], []
    for role, n_sets in (("train", 120), ("validation", 36), ("test", 36)):
        for k in range(n_sets):
            size = (8, 16, 32)[k % 3]
            n_a = max(1, min(size - 1, int(size * (0.25 + 0.5 * rng.random()))))
            members = list(rng.choice(ids[role, "a"], n_a, replace=False)) + \
                list(rng.choice(ids[role, "b"], size - n_a, replace=False))
            sid = f"{role}-{k}"
            subsets[sid] = {"role": role, "image_ids": members}
            weight = sum(1 + (WEIGHTS[x.loc[i, "dx"]] @ x.loc[i, list(SIGNALS)].to_numpy(float) if signals_matter else 0.0)
                         for i in members)
            truth = 0.5 + 0.04 * np.log1p(max(0.0, weight))
            for seed in SEEDS:
                row = {"subset_id": sid, "seed": seed, "members_sha256": fingerprint({"ids": sorted(members)}),
                       "protocol_sha256": fingerprint(protocol), "augmented_metric": float(truth + rng.normal(0, noise))}
                (test_rows if role == "test" else fit_rows).append(row)
    return frame, subsets, fit_rows, test_rows, protocol


def test_reliability_gate_separates_repeatable_from_noisy_labels():
    rng = np.random.default_rng(0)
    sizes = {f"s{k}": (10, 20)[k % 2] for k in range(40)}
    true = {sid: 0.6 + 0.02 * rng.normal() for sid in sizes}
    clean = {sid: list(true[sid] + rng.normal(0, 0.005, 5)) for sid in sizes}
    noisy = {sid: list(true[sid] + rng.normal(0, 0.2, 5)) for sid in sizes}
    passed, failed = reliability_gate(clean, sizes, 0.80), reliability_gate(noisy, sizes, 0.80)
    assert passed["passed"] and passed["reliability_of_mean"] > 0.95
    assert not failed["passed"] and failed["reliability_of_mean"] < 0.5
    assert set(passed["within_size_not_gating"]) == {"10", "20"}
    assert passed["icc_single_run"] > 0.8


def test_size_alone_can_pass_the_overall_gate_and_the_within_size_number_shows_it():
    """Labels that depend only on size: repeatable overall, nothing repeatable within a size."""
    rng = np.random.default_rng(1)
    sizes = {f"s{k}": (10, 100)[k % 2] for k in range(40)}
    values = {sid: list(0.6 + 0.001 * size + rng.normal(0, 0.01, 5)) for sid, size in sizes.items()}
    gate = reliability_gate(values, sizes, 0.80)
    assert gate["passed"]
    assert all(block["reliability_of_mean"] < 0.5 for block in gate["within_size_not_gating"].values())


def test_stratified_spearman_ignores_what_size_explains():
    rng = np.random.default_rng(2)
    strata = np.repeat([10, 20, 40], 12)
    noise = rng.normal(0, 1, 36)
    measured = strata * 10.0 + noise                    # size dominates the measured value
    follows = stratified_spearman(noise, measured, strata, 500, 42)
    size_only = stratified_spearman(strata + rng.normal(0, 1e-3, 36), measured, strata, 500, 42)
    assert follows["spearman_within_size"] > 0.99 and follows["one_sided_p"] < 0.01
    assert abs(size_only["spearman_within_size"]) < 0.4 and size_only["one_sided_p"] > 0.05


def test_a_ranker_that_predicts_unseen_subsets_is_accepted_and_the_test_is_read_once(tmp_path):
    frame, subsets, fit_rows, test_rows, protocol = _toy()
    out = tmp_path / "ranker_acceptance.json"
    result = acceptance(frame, subsets, fit_rows, test_rows, SEEDS, protocol, PREREG, FIT, out)
    assert result["accepted"] and result["a_correlation_passed"] and result["b_beats_size_and_class_only"]
    assert result["models"]["ranker"]["test_mse"] < result["models"]["similarity_only"]["test_mse"]
    assert set(result["models"]) == {"ranker", "size_and_class_only", "similarity_only", "equal_weight_composite"}
    assert out.is_file()
    with pytest.raises(ValueError, match="already read once"):
        acceptance(frame, subsets, fit_rows, test_rows, SEEDS, protocol, PREREG, FIT, out)


def test_when_the_signals_carry_no_utility_the_ranker_is_not_accepted(tmp_path):
    frame, subsets, fit_rows, test_rows, protocol = _toy(signals_matter=False)
    result = acceptance(frame, subsets, fit_rows, test_rows, SEEDS, protocol, PREREG, FIT, tmp_path / "a.json")
    assert not result["accepted"] and not result["a_correlation_passed"]


def test_test_outcomes_cannot_be_passed_as_fitting_measurements(tmp_path):
    frame, subsets, fit_rows, test_rows, protocol = _toy()
    with pytest.raises(ValueError, match="Test subset outcomes"):
        acceptance(frame, subsets, fit_rows + test_rows, test_rows, SEEDS, protocol, PREREG, FIT, tmp_path / "a.json")


def test_selection_stability_reports_each_fit_seed():
    frame, subsets, fit_rows, _, protocol = _toy()
    pool = frame[frame.image_id.str.startswith("test-")].reset_index(drop=True)
    report = selection_stability(frame, subsets, fit_rows, SEEDS, protocol, pool, PREREG, FIT)
    assert report["fit_seeds"] == [42, 43] and report["reported_selection_fit_seed"] == 42
    assert report["jaccard_with_the_reported_selection"][42] == 1.0
    assert report["total_min"] <= report["total_max"] <= len(pool)
    assert set(report["per_seed_counts"][43]) == {"a", "b"}

"""The per-image score as a network (contract §9, amendment of 2026-10-04). Toy data only.

What is checked: the network is the one that was approved (one hidden layer of 8 over the four
signals, one output per class, every image starting at weight 1); it learns a planted utility, including one a linear score cannot express;
the stopping rule, the saved file and the acceptance gate work with it unchanged; and the linear
score is still there as the reported baseline.
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.asism_v2 import prereg as prereg_module
from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS, TrainingStandardizer
from scripts.asism_v2.gates import acceptance
from scripts.asism_v2.persist import FORMAT, LINEAR_ONLY_FORMAT, load_fitted, save_fitted
from scripts.asism_v2.pipeline import (
    HIDDEN_UNITS,
    AdditiveUtilityRanker,
    NetworkUtilityRanker,
    _matrix,
    fit_ranker,
    new_ranker,
)
from scripts.asism_v2.prereg import FROZEN, PreregistrationError, fit_arguments, load_prereg
from scripts.asism_v2.stopping import progressive_select
from test_asism_v2_gates import FIT, PREREG
from test_asism_v2_gates import SEEDS as GATE_SEEDS
from test_asism_v2_gates import _toy as gates_toy
from test_asism_v2_pipeline import fitting_rows, small_scale_problem
from test_asism_v2_stopping import SEEDS as STOP_SEEDS
from test_asism_v2_stopping import _toy as stopping_toy

AMENDED = {"standardise_targets": True, "weight_decay": 0.0, "max_epochs": 5000, "patience": 200}
PROVENANCE = {"prereg_sha256": "p" * 64, "protocol_sha256": "q" * 64}


def planted_problem(weight_of):
    """small_scale_problem's design with the image weights given by `weight_of(signals, class)`."""
    rng = np.random.default_rng(78)
    rows, subsets, measurements, planted, pools = [], {}, [], {}, {}
    protocol = {"dataset": "synthetic_toy", "instrument_decision": "synthetic_toy_only",
                "recipe": "planted-network-utility", "metric": "toy_score"}
    for role, n in (("train", 300), ("validation", 150), ("test", 150)):
        for dx in ("a", "b"):
            pools[role, dx] = []
            for j in range(n):
                values = rng.normal(0, 1, len(SIGNALS))
                image_id = f"{role}-{dx}-{j}"
                rows.append({"image_id": image_id, "dx": dx, **dict(zip(SIGNALS, values))})
                planted[image_id] = float(weight_of(values, dx))
                pools[role, dx].append(image_id)
    for role, n_subsets in (("train", 120), ("validation", 40), ("test", 40)):
        for k in range(n_subsets):
            size = (20, 40, 80)[k % 3]
            n_a = int(size * (0.3 + 0.1 * (k % 5)))
            ids = list(rng.choice(pools[role, "a"], n_a, replace=False))
            ids += list(rng.choice(pools[role, "b"], size - n_a, replace=False))
            subsets[f"{role}-{k}"] = {"role": role, "image_ids": ids}
            value = 0.86 + 0.012 * np.log1p(sum(planted[i] for i in ids))
            for seed in (11, 12, 13):
                measurements.append({"subset_id": f"{role}-{k}", "seed": seed,
                                     "members_sha256": fingerprint({"ids": sorted(ids)}),
                                     "protocol_sha256": fingerprint(protocol), "augmented_metric": float(value)})
    return pd.DataFrame(rows), subsets, measurements, protocol, planted


def heldout_correlation(fitted, frame, planted):
    held = frame[frame.image_id.str.startswith("test-")]
    learned = fitted.score_frame(held)["image_weight"].to_numpy()
    return float(np.corrcoef(learned, held.image_id.map(planted).to_numpy())[0, 1])


def test_the_network_is_the_approved_one_and_every_image_starts_at_weight_one():
    assert HIDDEN_UNITS == 8
    assert FROZEN["ranker"] == {"architecture": "mlp", "hidden_units": 8, "activation": "tanh",
                                "outputs": "one_per_class"}
    model = new_ranker("mlp", 7)
    assert isinstance(model, NetworkUtilityRanker) and not hasattr(model, "weights")
    assert model.hidden.in_features == len(SIGNALS) and model.hidden.out_features == 8
    assert model.output.in_features == 8 and model.output.out_features == 7
    # hidden 4*8 + 8, outputs 7*8 + 7, class term 7, intercept, lam
    assert sum(p.numel() for p in model.parameters()) == 40 + 63 + 7 + 1 + 1
    x, classes = torch.randn(50, len(SIGNALS)), torch.randint(0, 7, (50,))
    with torch.no_grad():
        assert torch.equal(model.image_weight(x, classes), torch.ones(50))
    assert isinstance(new_ranker("linear", 7), AdditiveUtilityRanker)
    with pytest.raises(ValueError, match="architecture"):
        new_ranker("deep_sets", 7)


def test_the_prepared_forward_is_the_forward_and_a_resample_repeats_subsets():
    frame, subsets, _, _, _ = small_scale_problem()
    ids = sorted(subsets)[:25]
    batch = _matrix(subsets, ids, frame, TrainingStandardizer.fit(frame, list(frame.image_id)), ("a", "b"))
    torch.manual_seed(3)
    model = NetworkUtilityRanker(2)
    with torch.no_grad():
        model.output.weight.normal_(0, 0.3)
        model.class_mix.normal_(0, 0.3)
        prepared = model.prepare(*batch)
        assert torch.allclose(model(*batch), model.predict_prepared(prepared), atol=1e-5)
        assert torch.equal(model.prepared_sizes(prepared), batch[2].sum(1).to(torch.float32))
        rows = torch.tensor([3, 3, 0, 24])
        repeated = model.predict_prepared(model.resample(prepared, rows))
        assert torch.allclose(model(*[part[rows] for part in batch]), repeated, atol=1e-5)
    with pytest.raises(NotImplementedError):
        model.pool(*batch)


def test_the_network_recovers_a_planted_truth_at_the_scale_of_an_auroc_and_is_deterministic():
    frame, subsets, measurements, protocol, planted = small_scale_problem()
    rows = fitting_rows(measurements)
    first = fit_ranker(frame, subsets, rows, [11, 12, 13], protocol, architecture="mlp", **AMENDED)
    second = fit_ranker(frame, subsets, rows, [11, 12, 13], protocol, architecture="mlp", **AMENDED)
    assert first.history == second.history
    assert first.score_frame(frame).equals(second.score_frame(frame))
    assert first.history["architecture"] == {"kind": "mlp", "hidden_units": 8, "activation": "tanh",
                                             "outputs": "one_per_class"}
    assert first.history["epoch_limit_reached"] is False
    assert heldout_correlation(first, frame, planted) > 0.98
    assert float(first.model.log_count.detach()) == pytest.approx(0.012, abs=0.003)


def test_the_network_learns_a_utility_that_a_linear_score_cannot_express():
    """Planted weight 1 + 0.6 * (|similarity| - E|z|): useful at both ends of a signal, not in the
    middle. A linear score of the signals has nothing to fit there."""
    frame, subsets, measurements, protocol, planted = planted_problem(
        lambda values, dx: 1 + 0.6 * (abs(values[0]) - 0.798))
    rows = fitting_rows(measurements)
    network = fit_ranker(frame, subsets, rows, [11, 12, 13], protocol, architecture="mlp", **AMENDED)
    linear = fit_ranker(frame, subsets, rows, [11, 12, 13], protocol, architecture="linear", **AMENDED)
    assert heldout_correlation(network, frame, planted) > 0.9
    assert abs(heldout_correlation(linear, frame, planted)) < 0.3
    assert linear.history["architecture"] == {"kind": "linear"}


def test_restricting_the_signals_leaves_a_score_that_depends_on_the_class_only():
    frame, subsets, measurements, protocol, _ = small_scale_problem()
    fitted = fit_ranker(frame, subsets, fitting_rows(measurements), [11, 12, 13], protocol, architecture="mlp",
                        signal_mask=(False, False, False, False), max_epochs=200, patience=200,
                        standardise_targets=True, weight_decay=0.0)
    assert fitted.history["signals_used"] == []
    scored = fitted.score_frame(frame)
    assert all(scored[scored.dx == dx].image_weight.nunique() == 1 for dx in ("a", "b"))
    similarity_only = fit_ranker(frame, subsets, fitting_rows(measurements), [11, 12, 13], protocol,
                                 architecture="mlp", signal_mask=(True, False, False, False), max_epochs=200,
                                 patience=200, standardise_targets=True, weight_decay=0.0)
    changed = frame.assign(iqa_composite=frame.iqa_composite + 5.0)      # an unused signal
    assert similarity_only.score_frame(frame).image_weight.equals(similarity_only.score_frame(changed).image_weight)


@pytest.fixture(scope="module")
def network_and_pool():
    frame, subsets, measurements, protocol, pool = stopping_toy()
    fitted = fit_ranker(frame, subsets, measurements, STOP_SEEDS, protocol, max_epochs=400, patience=60,
                        bootstrap=20, architecture="mlp")
    return fitted, pool


def test_the_stopping_rule_reads_a_network_ranker_unchanged(network_and_pool):
    """Same toy as test_asism_v2_stopping: class "b" has only helpful images, class "a" has some
    with a negative weight."""
    fitted, pool = network_and_pool
    result = progressive_select(fitted, pool)
    assert result["bootstrap_models"] == 20
    counts, sizes = result["counts"], pool.dx.value_counts()
    assert 0 < counts["a"] < sizes["a"]
    assert counts["b"] > counts["a"]
    assert all(step["marginal_lower_bound"] > 0 for step in result["trajectory"])
    assert all(isinstance(member, NetworkUtilityRanker) for member in fitted.ensemble)


def test_a_saved_network_ranker_selects_identically_and_records_its_architecture(network_and_pool, tmp_path):
    fitted, pool = network_and_pool
    path = tmp_path / "ranker.json"
    save_fitted(path, fitted, PROVENANCE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == FORMAT == "asism_v2_fitted_ranking/2"
    assert payload["architecture"]["outputs"] == "one_per_class" and payload["architecture"]["hidden_units"] == 8
    loaded, _ = load_fitted(path, PROVENANCE)
    assert isinstance(loaded.model, NetworkUtilityRanker) and len(loaded.ensemble) == 20
    assert loaded.score_frame(pool).equals(fitted.score_frame(pool))
    before, after = progressive_select(fitted, pool), progressive_select(loaded, pool)
    assert before["counts"] == after["counts"] and before["trajectory"] == after["trajectory"]


def test_a_file_written_before_the_amendment_loads_as_the_linear_score(tmp_path):
    frame, subsets, measurements, protocol, pool = stopping_toy()
    fitted = fit_ranker(frame, subsets, measurements, STOP_SEEDS, protocol, max_epochs=60, patience=60)
    path = tmp_path / "ranker.json"
    save_fitted(path, fitted, PROVENANCE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("content_sha256", "architecture"):
        payload.pop(key)
    payload["format"] = LINEAR_ONLY_FORMAT
    old = tmp_path / "format1.json"
    old.write_text(json.dumps({**payload, "content_sha256": fingerprint(payload)}), encoding="utf-8")
    loaded, _ = load_fitted(old, PROVENANCE)
    assert type(loaded.model) is AdditiveUtilityRanker
    assert loaded.score_frame(pool).equals(fitted.score_frame(pool))


def test_acceptance_reports_the_linear_score_next_to_the_network_and_it_does_not_gate(tmp_path):
    frame, subsets, fit_rows, test_rows, protocol = gates_toy()
    result = acceptance(frame, subsets, fit_rows, test_rows, GATE_SEEDS, protocol, PREREG,
                        {**FIT, "architecture": "mlp", "hidden_units": 8}, tmp_path / "ranker_acceptance.json")
    assert set(result["models"]) == {"ranker", "size_and_class_only", "similarity_only", "linear_additive",
                                     "equal_weight_composite"}
    ranker, null = result["models"]["ranker"], result["models"]["size_and_class_only"]
    assert result["accepted"] == (result["a_correlation_passed"] and result["b_beats_size_and_class_only"])
    assert result["b_beats_size_and_class_only"] == (ranker["test_mse"] < null["test_mse"])
    assert result["accepted"]
    assert "linear_additive" in PREREG["acceptance"]["reported_baselines"]


def test_a_network_with_nothing_to_learn_is_not_accepted(tmp_path):
    frame, subsets, fit_rows, test_rows, protocol = gates_toy(signals_matter=False)
    result = acceptance(frame, subsets, fit_rows, test_rows, GATE_SEEDS, protocol, PREREG,
                        {**FIT, "architecture": "mlp", "hidden_units": 8}, tmp_path / "ranker_acceptance.json")
    assert not result["accepted"]


def test_the_approved_fit_arguments_name_the_network_and_a_changed_ranker_is_refused(monkeypatch):
    arguments = fit_arguments(load_prereg())
    assert arguments["architecture"] == "mlp" and arguments["hidden_units"] == 8
    from omegaconf import OmegaConf
    changed = copy.deepcopy(FROZEN)
    changed["ranker"]["hidden_units"] = 64
    monkeypatch.setattr(prereg_module, "load_named_config",
                        lambda *_: OmegaConf.create({**changed, "paths": {"outputs_dir": "x"}}))
    with pytest.raises(PreregistrationError, match="ranker"):
        load_prereg()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

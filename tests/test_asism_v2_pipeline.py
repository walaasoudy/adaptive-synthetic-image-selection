"""Scientific wiring tests use a known toy utility, never historical HAM labels."""
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.pipeline import AdditiveUtilityRanker, _matrix, fit_ranker


def toy_problem(fixed_mix=False):
    rng = np.random.default_rng(391)
    rows, subsets, measurements = [], {}, []
    weights = {"a": np.array([.04, .10, .02, .03]),
               "b": np.array([.04, -.10, .02, .03])}
    role_ids = {}
    for role, n in (("train", 80), ("validation", 35), ("test", 35)):
        role_ids[role] = {}
        for dx in ("a", "b"):
            ids = []
            for j in range(n):
                image_id = f"{role}-{dx}-{j}"
                values = rng.normal(0, 1, len(SIGNALS))
                rows.append({"image_id": image_id, "dx": dx, **dict(zip(SIGNALS, values))})
                ids.append(image_id)
            role_ids[role][dx] = ids
    frame = pd.DataFrame(rows)
    indexed = frame.set_index("image_id")
    protocol = {"dataset": "synthetic_toy", "instrument_decision": "synthetic_toy_only",
                "recipe": "known-additive-utility", "metric": "toy_score"}
    for role, n_subsets in (("train", 180), ("validation", 50), ("test", 50)):
        for k in range(n_subsets):
            # Vary class composition; retain enough within-class variation to
            # identify opposite IQA slopes rather than just an average effect.
            n_a = 4 if fixed_mix else 2 + k % 5
            ids = list(rng.choice(role_ids[role]["a"], n_a, replace=False))
            ids += list(rng.choice(role_ids[role]["b"], 8 - n_a, replace=False))
            sid = f"{role}-{k}"
            subsets[sid] = {"role": role, "image_ids": ids}
            utility = .55 + np.mean([weights[indexed.loc[i, "dx"]] @
                                      indexed.loc[i, list(SIGNALS)].to_numpy(dtype=float) for i in ids])
            for seed in (11, 12, 13):
                measurements.append({"subset_id": sid, "seed": seed,
                    "members_sha256": fingerprint({"ids": sorted(ids)}),
                    "protocol_sha256": fingerprint(protocol),
                    "augmented_metric": float(utility)})
    return frame, subsets, measurements, protocol, weights


def fitting_rows(rows):
    return [row for row in rows if not row["subset_id"].startswith("test-")]


def test_toy_ranker_recovers_known_contributions_and_is_deterministic():
    frame, subsets, measurements, protocol, weights = toy_problem()
    first = fit_ranker(frame, subsets, fitting_rows(measurements), [11, 12, 13], protocol,
                       max_epochs=350, patience=50)
    second = fit_ranker(frame, subsets, fitting_rows(measurements), [11, 12, 13], protocol,
                        max_epochs=350, patience=50)
    assert first.history == second.history
    actual = first.score_frame(frame)
    assert actual.equals(second.score_frame(frame))
    indexed = frame.set_index("image_id")
    test_scores = actual[actual.image_id.str.startswith("test-")]
    for dx in ("a", "b"):
        group = test_scores[test_scores.dx == dx]
        truth = [weights[dx] @ indexed.loc[i, list(SIGNALS)].to_numpy(dtype=float)
                 for i in group.image_id]
        assert pd.Series(truth).corr(group.ranking_score.reset_index(drop=True), method="spearman") > .88
    # Same other three signals, class-specific IQA perturbation gives opposite
    # directions. Zeroing IQA changes scores and is not a safety-only noop.
    probes = pd.DataFrame([{"image_id": f"probe-{dx}-{j}", "dx": dx,
                             **dict.fromkeys(SIGNALS, 0.), "iqa_composite": float(j)}
                            for dx in ("a", "b") for j in (0, 1)])
    perturbed = first.score_frame(probes)
    assert perturbed.ranking_score.iloc[1] > perturbed.ranking_score.iloc[0]
    assert perturbed.ranking_score.iloc[3] < perturbed.ranking_score.iloc[2]
    assert abs(perturbed.ranking_score.iloc[1] - perturbed.ranking_score.iloc[0]) > .02
    assert abs(perturbed.ranking_score.iloc[3] - perturbed.ranking_score.iloc[2]) > .02
    neutral = probes.copy()
    neutral["iqa_composite"] = 0.0
    neutral_scores = first.score_frame(neutral)
    assert neutral_scores.ranking_score.iloc[0] == neutral_scores.ranking_score.iloc[1]
    assert neutral_scores.ranking_score.iloc[2] == neutral_scores.ranking_score.iloc[3]


def test_rejected_real_instrument_and_leakage_fail():
    frame, subsets, rows, protocol, _ = toy_problem()
    with pytest.raises(ValueError, match="instrument"):
        fit_ranker(frame, subsets, fitting_rows(rows), [11, 12, 13],
                   {**protocol, "instrument_decision": "rejected"})
    overlap = {key: dict(value) for key, value in subsets.items()}
    overlap["validation-0"]["image_ids"] = overlap["train-0"]["image_ids"]
    with pytest.raises(ValueError, match="leakage"):
        fit_ranker(frame, overlap, fitting_rows(rows), [11, 12, 13], protocol)


def test_invalid_signals_and_stale_measurements_fail():
    frame, subsets, rows, protocol, _ = toy_problem()
    with pytest.raises(ValueError, match="signal"):
        fit_ranker(frame.drop(columns="iqa_composite"), subsets, fitting_rows(rows), [11, 12, 13], protocol)
    with pytest.raises(ValueError, match="Stale"):
        fit_ranker(frame, subsets, fitting_rows(rows), [11, 12, 13],
                   {**protocol, "recipe": "changed"})
    with pytest.raises(ValueError, match="provenance"):
        fit_ranker(frame, subsets, fitting_rows(rows), [11, 12, 13],
                   {**protocol, "instrument_decision": "accepted"})

    with pytest.raises(ValueError, match="Test subset outcomes"):
        fit_ranker(frame, subsets, rows, [11, 12, 13], protocol)


def test_every_score_carries_the_source_the_ranker_recorded():
    frame, subsets, measurements, protocol, _ = toy_problem()
    fitted = fit_ranker(frame, subsets, fitting_rows(measurements), [11, 12, 13], protocol,
                        max_epochs=50, patience=50)
    scored = fitted.score_frame(frame).set_index("image_id").score_source
    for sid, entry in subsets.items():
        expected = {"train": "train_fit", "validation": "validation_early_stopping",
                    "test": "heldout"}[entry["role"]]
        assert set(scored.loc[[str(i) for i in entry["image_ids"]]]) == {expected}, sid


def test_class_mix_is_frozen_only_when_every_subset_has_the_same_allocation():
    varied = toy_problem()
    fit = fit_ranker(*varied[:2], fitting_rows(varied[2]), [11, 12, 13], varied[3], max_epochs=30, patience=30)
    assert "class_mix" not in fit.history["frozen_unidentifiable"]
    fixed = toy_problem(fixed_mix=True)
    fit = fit_ranker(*fixed[:2], fitting_rows(fixed[2]), [11, 12, 13], fixed[3], max_epochs=30, patience=30)
    assert {"class_mix", "log_count"} <= set(fit.history["frozen_unidentifiable"])
    assert torch.count_nonzero(fit.model.class_mix) == 0


def test_identifiability_is_judged_on_train_subsets_only():
    """A second size or class allocation seen only outside the train subsets identifies nothing."""
    frame, subsets, measurements, protocol, _ = toy_problem(fixed_mix=True)
    rows = fitting_rows(measurements)
    changed = {sid: dict(entry) for sid, entry in subsets.items()}
    for sid in ("validation-0", "test-0"):
        changed[sid]["image_ids"] = list(changed[sid]["image_ids"])[:7]   # another size and mix
    members = sorted(map(str, changed["validation-0"]["image_ids"]))
    for row in rows:
        if row["subset_id"] == "validation-0":
            row["members_sha256"] = fingerprint({"ids": members})
    fit = fit_ranker(frame, changed, rows, [11, 12, 13], protocol, max_epochs=20, patience=20)
    assert {"class_mix", "log_count"} <= set(fit.history["frozen_unidentifiable"])
    assert fit.history["subset_sizes"] == [8]


def test_class_mix_is_frozen_when_sizes_vary_but_class_fractions_do_not():
    frame, subsets, measurements, protocol, _ = toy_problem(fixed_mix=True)
    rows = fitting_rows(measurements)
    changed = {sid: dict(entry) for sid, entry in subsets.items()}
    by_id = frame.set_index("image_id").dx
    for k, sid in enumerate(sid for sid in subsets if not sid.startswith("test-")):
        if k % 2:
            ids = list(map(str, changed[sid]["image_ids"]))
            half = [i for i in ids if by_id[i] == "a"][:2] + [i for i in ids if by_id[i] == "b"][:2]
            changed[sid]["image_ids"] = half                              # 2:2 instead of 4:4
            for row in rows:
                if row["subset_id"] == sid:
                    row["members_sha256"] = fingerprint({"ids": sorted(half)})
    fit = fit_ranker(frame, changed, rows, [11, 12, 13], protocol, max_epochs=20, patience=20)
    assert fit.history["frozen_unidentifiable"] == ["class_mix"]
    assert fit.history["subset_sizes"] == [4, 8]


def small_scale_problem():
    """A truth of the model's own form whose targets move by about 0.001 within a size, as a
    macro AUROC does. Noise-free, so a fit that reaches the minimum recovers the planted weights."""
    rng = np.random.default_rng(77)
    slopes = {"a": np.array([.5, 0, 0, -.3]), "b": np.array([-.5, .2, 0, 0])}
    rows, subsets, measurements, planted = [], {}, [], {}
    protocol = {"dataset": "synthetic_toy", "instrument_decision": "synthetic_toy_only",
                "recipe": "planted-sum-pooled-utility", "metric": "toy_score"}
    pools = {}
    for role, n in (("train", 300), ("validation", 150), ("test", 150)):
        for dx in ("a", "b"):
            pools[role, dx] = []
            for j in range(n):
                values = rng.normal(0, 1, len(SIGNALS))
                image_id = f"{role}-{dx}-{j}"
                rows.append({"image_id": image_id, "dx": dx, **dict(zip(SIGNALS, values))})
                planted[image_id] = 1 + slopes[dx] @ values
                pools[role, dx].append(image_id)
    for role, n_subsets in (("train", 120), ("validation", 40), ("test", 40)):
        for k in range(n_subsets):
            size = (20, 40, 80)[k % 3]
            n_a = int(size * (0.3 + 0.1 * (k % 5)))
            ids = list(rng.choice(pools[role, "a"], n_a, replace=False))
            ids += list(rng.choice(pools[role, "b"], size - n_a, replace=False))
            sid = f"{role}-{k}"
            subsets[sid] = {"role": role, "image_ids": ids}
            value = 0.86 + 0.012 * np.log1p(sum(planted[i] for i in ids))
            for seed in (11, 12, 13):
                measurements.append({"subset_id": sid, "seed": seed,
                                     "members_sha256": fingerprint({"ids": sorted(ids)}),
                                     "protocol_sha256": fingerprint(protocol), "augmented_metric": float(value)})
    return pd.DataFrame(rows), subsets, measurements, protocol, planted


def test_the_pooled_forward_is_the_forward():
    frame, subsets, _, _, _ = small_scale_problem()
    fitted_classes = ("a", "b")
    from scripts.asism_v2.features import TrainingStandardizer
    ids = sorted(subsets)[:25]
    normalizer = TrainingStandardizer.fit(frame, list(frame.image_id))
    batch = _matrix(subsets, ids, frame, normalizer, fitted_classes)
    torch.manual_seed(3)
    model = AdditiveUtilityRanker(2)
    with torch.no_grad():
        model.weights.normal_(0, 0.3)
        model.class_mix.normal_(0, 0.3)
    with torch.no_grad():
        assert torch.allclose(model(*batch), model.forward_pooled(*model.pool(*batch)), atol=1e-5)


def test_standardised_targets_recover_a_planted_truth_at_the_scale_of_an_auroc():
    frame, subsets, measurements, protocol, planted = small_scale_problem()
    truth = frame.image_id.map(planted).to_numpy()
    rows = fitting_rows(measurements)

    def recovered(**settings):
        fitted = fit_ranker(frame, subsets, rows, [11, 12, 13], protocol, **settings)
        learned = fitted.score_frame(frame)["image_weight"].to_numpy()
        return fitted, float(np.corrcoef(learned, truth)[0, 1])

    amended, good = recovered(standardise_targets=True, weight_decay=0.0, max_epochs=5000, patience=200)
    _, before = recovered(weight_decay=1e-5, max_epochs=600, patience=60)
    assert good > 0.99 and before < good - 0.1
    assert amended.history["epoch_limit_reached"] is False
    assert amended.history["fit_config"]["standardise_targets"] is True
    # the saved parameters are in the metric's own units: the planted intercept and size coefficient
    assert float(amended.model.log_count.detach()) == pytest.approx(0.012, abs=0.002)
    assert float(amended.model.intercept.detach()) == pytest.approx(0.86, abs=0.01)
    # and the recorded validation error is the error of those parameters, in those units
    ids = sorted(sid for sid, s in subsets.items() if s["role"] == "validation")
    batch = _matrix(subsets, ids, frame, amended.normalizer, amended.classes)
    target = {}
    for row in rows:
        target[row["subset_id"]] = row["augmented_metric"]
    with torch.no_grad():
        error = float(((amended.model(*batch) - torch.tensor([target[i] for i in ids])) ** 2).mean())
    assert error == pytest.approx(amended.history["validation_mse"], rel=0.05, abs=1e-9)


def test_targets_that_do_not_vary_cannot_be_standardised():
    frame, subsets, measurements, protocol, _ = small_scale_problem()
    flat = [{**row, "augmented_metric": 0.9} for row in fitting_rows(measurements)]
    with pytest.raises(ValueError, match="do not vary"):
        fit_ranker(frame, subsets, flat, [11, 12, 13], protocol, standardise_targets=True)

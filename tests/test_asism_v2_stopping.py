"""Contract §10 progressive selection with adaptive stopping, on a toy with a KNOWN set utility.

Toy truth: U(S) = 0.5 + 0.04 * log(1 + sum_{x in S} (1 + w_c . x)). Class "a" has strong signal
weights, so about a fifth of its images have a negative weight and should be left out; class "b" has
weak ones, every image has a positive weight and all of them should be taken.
Implementation correctness only: nothing here says anything about HAM10000.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.pipeline import fit_ranker
from scripts.asism_v2.stopping import LOWER_BOUND_QUANTILE, OFFER_ORDER, progressive_select

WEIGHTS = {"a": np.array([.9, .6, .3, .3]), "b": np.array([.1, .1, .05, .05])}
LAMBDA = 0.04
SEEDS = [11, 12, 13]


def _toy(sizes=tuple(range(2, 31, 2)), noise=0.002):
    rng = np.random.default_rng(7)
    rows, role_ids = [], {}
    for role, n in (("train", 120), ("validation", 50), ("test", 50), ("pool", 60)):
        role_ids[role] = {}
        for dx in ("a", "b"):
            role_ids[role][dx] = []
            for j in range(n):
                image_id = f"{role}-{dx}-{j}"
                rows.append({"image_id": image_id, "dx": dx, **dict(zip(SIGNALS, rng.normal(0, 1, 4)))})
                role_ids[role][dx].append(image_id)
    frame = pd.DataFrame(rows)
    x = frame.set_index("image_id")
    protocol = {"dataset": "synthetic_toy", "instrument_decision": "synthetic_toy_only", "recipe": "toy-§10"}
    subsets, measurements = {}, []
    for role, n_sets in (("train", 160), ("validation", 40), ("test", 20)):
        for k in range(n_sets):
            size = sizes[k % len(sizes)]
            n_a = max(1, min(size - 1, size // 2 + (k % 3) - 1))
            ids = list(rng.choice(role_ids[role]["a"], n_a, replace=False))
            ids += list(rng.choice(role_ids[role]["b"], size - n_a, replace=False))
            sid = f"{role}-{k}"
            subsets[sid] = {"role": role, "image_ids": ids}
            truth = 0.5 + LAMBDA * np.log1p(max(0.0, sum(
                1 + WEIGHTS[x.loc[i, "dx"]] @ x.loc[i, list(SIGNALS)].to_numpy(float) for i in ids)))
            if role == "test":
                continue
            for seed in SEEDS:
                measurements.append({"subset_id": sid, "seed": seed,
                                     "members_sha256": fingerprint({"ids": sorted(ids)}),
                                     "protocol_sha256": fingerprint(protocol),
                                     "augmented_metric": float(truth + rng.normal(0, noise))})
    pool = frame[frame.image_id.str.startswith("pool-")].reset_index(drop=True)
    return frame, subsets, measurements, protocol, pool


@pytest.fixture(scope="module")
def fitted_and_pool():
    frame, subsets, measurements, protocol, pool = _toy()
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=400, patience=60, bootstrap=30)
    return fitted, pool


def test_the_size_term_is_learned_when_sizes_vary(fitted_and_pool):
    fitted, _ = fitted_and_pool
    assert "log_count" not in fitted.history["frozen_unidentifiable"]
    assert abs(float(fitted.model.log_count.detach()) - LAMBDA) < 0.015


def test_selection_is_progressive_within_class_and_stops_on_the_lower_bound(fitted_and_pool):
    fitted, pool = fitted_and_pool
    result = progressive_select(fitted, pool)
    for dx in ("a", "b"):
        picked = [t for t in result["trajectory"] if t["dx"] == dx]
        # each class adds its candidates strictly in within-class rank order, from the top; the
        # rank is by the image's own lower bound (amendment 2026-10-03), not the point score
        assert [t["rank_in_class"] for t in picked] == list(range(len(picked)))
        bounds = [t["own_lower_bound"] for t in picked]
        assert bounds == sorted(bounds, reverse=True) and min(bounds) > 0
        assert result["counts"][dx] == len(picked)
    assert result["offer_order"] == OFFER_ORDER == "lower_bound"
    assert all(t["marginal_lower_bound"] > 0 for t in result["trajectory"])
    for dx, stop in result["stops"].items():
        assert stop["reason"] == "class exhausted" or stop["marginal_lower_bound"] <= 0, dx
    assert result["lower_bound_quantile"] == LOWER_BOUND_QUANTILE == 0.05
    assert result["selected"].groupby("dx").size().to_dict() == {k: v for k, v in result["counts"].items() if v}


def _true_weights(pool):
    return {r.image_id: 1 + WEIGHTS[r.dx] @ np.array([getattr(r, c) for c in SIGNALS]) for r in pool.itertuples()}


def test_counts_match_what_the_true_utility_implies(fitted_and_pool):
    """Under the true utility an image helps iff its weight is above 0, whatever else is selected."""
    fitted, pool = fitted_and_pool
    result = progressive_select(fitted, pool)
    true = _true_weights(pool)
    helpful = {dx: sum(true[i] > 0 for i in pool[pool.dx == dx].image_id) for dx in ("a", "b")}
    assert helpful["b"] == 60 and 40 <= helpful["a"] < 60
    assert result["counts"]["b"] == 60 and result["stops"]["b"]["reason"] == "class exhausted"
    # the lower bound is conservative: a few barely-helpful images may be left out, none added
    assert helpful["a"] - 6 <= result["counts"]["a"] <= helpful["a"], (result["counts"], helpful)
    harmful_selected = [i for i in result["selected"].image_id if true[i] < -0.1]
    assert not harmful_selected


def test_a_class_is_not_dropped_because_another_class_has_a_better_image(fitted_and_pool):
    """The retired mean-pooled form stopped a class for good when its best image was below the set
    mean. Each image is now judged on its own weight."""
    fitted, pool = fitted_and_pool
    result = progressive_select(fitted, pool)
    assert all(result["counts"][dx] > 30 for dx in ("a", "b"))


def test_selection_is_deterministic(fitted_and_pool):
    fitted, pool = fitted_and_pool
    first, second = progressive_select(fitted, pool), progressive_select(fitted, pool)
    assert first["trajectory"] == second["trajectory"] and first["counts"] == second["counts"]


def test_iqa_is_a_genuine_ranking_input(fitted_and_pool):
    fitted, pool = fitted_and_pool
    a = pool[pool.dx == "a"].copy()
    base = fitted.score_frame(a).ranking_score.to_numpy()
    shifted = a.copy()
    shifted["iqa_composite"] = shifted["iqa_composite"].to_numpy()[::-1]   # other signals fixed
    moved = fitted.score_frame(shifted).ranking_score.to_numpy()
    assert not np.allclose(base, moved)
    order_before = list(a.image_id.to_numpy()[np.argsort(-base)])
    order_after = list(a.image_id.to_numpy()[np.argsort(-moved)])
    assert order_before != order_after


def test_one_subset_size_is_refused_because_marginal_utility_is_not_identifiable():
    frame, subsets, measurements, protocol, pool = _toy(sizes=(8,))
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=40, patience=40, bootstrap=20)
    assert "log_count" in fitted.history["frozen_unidentifiable"]
    with pytest.raises(ValueError, match="not identifiable"):
        progressive_select(fitted, pool)


def test_too_few_bootstrap_models_cannot_give_a_confidence_bound():
    frame, subsets, measurements, protocol, pool = _toy()
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=40, patience=40, bootstrap=3)
    with pytest.raises(ValueError, match="bootstrap models"):
        progressive_select(fitted, pool)


def test_missing_signal_and_unknown_class_fail_closed(fitted_and_pool):
    fitted, pool = fitted_and_pool
    with pytest.raises(ValueError, match="signal"):
        progressive_select(fitted, pool.drop(columns="iqa_composite"))
    broken = pool.copy()
    broken.loc[0, "explainability_calibrated_typicality"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        progressive_select(fitted, broken)
    other = pool.copy()
    other.loc[0, "dx"] = "zzz"
    with pytest.raises(ValueError, match="Unknown diagnosis"):
        progressive_select(fitted, other)


def test_one_class_allocation_is_refused_because_classes_cannot_be_compared():
    """Sizes vary, so the size term is identified, but every subset is half "a" and half "b"."""
    frame, subsets, measurements, protocol, pool = _toy()
    by_id = frame.set_index("image_id").dx
    for sid, entry in subsets.items():
        ids = list(map(str, entry["image_ids"]))
        half = len(ids) // 2
        role = entry["role"]
        a = [i for i in ids if by_id[i] == "a"]
        b = [i for i in ids if by_id[i] == "b"]
        spare_a = [f"{role}-a-{j}" for j in range(50) if f"{role}-a-{j}" not in a]
        spare_b = [f"{role}-b-{j}" for j in range(50) if f"{role}-b-{j}" not in b]
        balanced = (a + spare_a)[:half] + (b + spare_b)[:len(ids) - half]
        entry["image_ids"] = balanced
        for row in measurements:
            if row["subset_id"] == sid:
                row["members_sha256"] = fingerprint({"ids": sorted(balanced)})
    fitted = fit_ranker(frame, subsets, measurements, SEEDS, protocol, max_epochs=40, patience=40, bootstrap=20)
    assert fitted.history["frozen_unidentifiable"] == ["class_mix"]
    with pytest.raises(ValueError, match="across classes is not identifiable"):
        progressive_select(fitted, pool)


def test_an_uncertain_top_scored_image_does_not_stop_its_class(fitted_and_pool):
    """The image the point model scores highest in a class is made uncertain: 4 of the 30 bootstrap
    models give it a negative weight. Offered in point-score order it would come first and stop the
    class at 0. Offered by its own lower bound it comes after the images that are shown to help."""
    import copy
    import dataclasses

    import torch

    fitted, pool = fitted_and_pool
    before = progressive_select(fitted, pool)
    scored = fitted.score_frame(pool)
    top = scored[scored.dx == "a"].sort_values("ranking_score", ascending=False).iloc[0]
    assert top.image_id in set(before["selected"].image_id)
    row = pool[pool.image_id == top.image_id]
    x = torch.tensor(fitted.normalizer.transform(row), dtype=torch.float32)[0]
    a = fitted.classes.index("a")
    ensemble = [copy.deepcopy(m) for m in fitted.ensemble]
    for member in ensemble[:4]:
        with torch.no_grad():
            weight = float(member.image_weight(x[None], torch.tensor([a])))
            member.weights[a] -= (weight + 1) * x / float(x @ x)          # this image's weight becomes -1
    after = progressive_select(dataclasses.replace(fitted, ensemble=tuple(ensemble)), pool)
    assert top.image_id not in set(after["selected"].image_id)
    assert after["counts"]["a"] >= 20
    assert all(t["marginal_lower_bound"] > 0 for t in after["trajectory"])

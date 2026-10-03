"""The designed utility subsets and the pre-registered configuration. No HAM10000 data is read."""
import copy

import numpy as np
import pandas as pd
import pytest

from scripts.asism_v2 import prereg as prereg_module
from scripts.asism_v2.contracts import fingerprint
from scripts.asism_v2.features import SIGNALS
from scripts.asism_v2.prereg import FROZEN, PreregistrationError, fit_arguments, load_prereg
from scripts.asism_v2.supervision import (ROLES, SupervisionError, assign_roles, build_plan, describe,
                                          fit_inputs, freeze_plan)

POOL_COUNTS = {"akiec": 486, "bcc": 400, "bkl": 273, "df": 885, "mel": 276, "nv": 112, "vasc": 736}


def _pool():
    rng = np.random.default_rng(5)
    rows = [{"image_id": f"{dx}-{j:04d}", "dx": dx, **dict(zip(SIGNALS, rng.normal(0, 1, 4)))}
            for dx, n in POOL_COUNTS.items() for j in range(n)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool_and_plan():
    pool = _pool()
    return pool, build_plan(pool, {**FROZEN, "prereg_sha256": fingerprint(FROZEN)})


def test_the_yaml_is_the_approved_configuration():
    loaded = load_prereg()
    assert {k: loaded[k] for k in FROZEN} == FROZEN
    assert loaded["prereg_sha256"] == fingerprint(FROZEN)
    assert fit_arguments(loaded) == {"seed": 42, "bootstrap": 200, "max_epochs": 5000, "patience": 200,
                                     "learning_rate": 0.03, "weight_decay": 0.0, "initial_lam": 0.02,
                                     "standardise_targets": True}


def test_a_changed_configuration_is_refused(monkeypatch):
    from omegaconf import OmegaConf
    changed = copy.deepcopy(FROZEN)
    changed["fit"]["bootstrap_models"] = 20
    monkeypatch.setattr(prereg_module, "load_named_config",
                        lambda *_: OmegaConf.create({**changed, "paths": {"outputs_dir": "x"}}))
    with pytest.raises(PreregistrationError, match="fit"):
        load_prereg()


def test_no_count_ratio_or_k_is_configured():
    def keys(node):
        return [k for key, value in node.items() for k in [key, *(keys(value) if isinstance(value, dict) else [])]]

    assert FROZEN["stopping"]["minimum_gain"] is None
    from scripts.asism_v2.stopping import LOWER_BOUND_QUANTILE, OFFER_ORDER
    assert FROZEN["stopping"]["offer_order"] == OFFER_ORDER
    assert FROZEN["stopping"]["lower_bound_quantile"] == LOWER_BOUND_QUANTILE
    for key in keys(FROZEN):
        for word in ("target", "ratio", "top_k", "fill_to", "accepted_per_class", "budget", "quota"):
            assert word not in key.replace("concentration", ""), key


def test_every_image_has_one_role_and_every_role_every_class():
    pool = _pool()
    roles = assign_roles(pool, FROZEN["supervision"]["role_fractions"], 42)
    assert set(roles) == set(pool.image_id) and set(roles.values()) == set(ROLES)
    by_role = pool.assign(role=pool.image_id.map(roles)).groupby(["role", "dx"]).size().unstack()
    assert (by_role > 0).all().all()
    assert abs(by_role.loc["train"].sum() / len(pool) - 0.6) < 0.01
    assert roles == assign_roles(pool, FROZEN["supervision"]["role_fractions"], 42)


def test_the_plan_matches_the_approved_design(pool_and_plan):
    pool, plan = pool_and_plan
    subsets = plan["subsets"]
    assert plan["runs_planned"] == 1000 == len(plan["cells"])
    assert {r: sum(s["role"] == r for s in subsets.values()) for r in ROLES} == {"train": 120, "validation": 40, "test": 40}
    assert {s["size"] for s in subsets.values() if s["role"] == "train"} == {125, 250, 500, 1000}
    assert {s["size"] for s in subsets.values() if s["role"] == "test"} == {125, 250, 500}
    train = [s for s in subsets.values() if s["role"] == "train"]
    assert sum(s["kind"] == "tilted" for s in train) == 60
    for size in (125, 250, 500, 1000):
        assert sum(s["kind"] == "tilted" for s in train if s["size"] == size) == 15
    assert {(s["tilt_signal"], s["tilt_direction"]) for s in train if s["kind"] == "tilted"} == \
        {(signal, direction) for signal in SIGNALS for direction in ("upper", "lower")}
    roles = plan["roles"]
    for subset in subsets.values():
        assert {roles[i] for i in subset["image_ids"]} == {subset["role"]}
        assert len(set(subset["image_ids"])) == subset["size"] == sum(subset["class_counts"].values())
    assert len({s["members_sha256"] for s in subsets.values()}) == len(subsets)


def test_tilted_subsets_differ_in_their_signal_and_random_ones_do_not(pool_and_plan):
    pool, plan = pool_and_plan
    indexed = pool.set_index("image_id")
    class_mean = pool.groupby("dx")[list(SIGNALS)].transform("mean").set_index(pool.image_id)
    centred = indexed[list(SIGNALS)] - class_mean
    shift = {"upper": [], "lower": [], "random": []}
    for subset in plan["subsets"].values():
        if subset["role"] != "train" or subset["size"] > 250:
            continue
        signal = subset["tilt_signal"] or SIGNALS[0]
        shift[subset["tilt_direction"] or "random"].append(centred.loc[subset["image_ids"], signal].mean())
    assert min(shift["upper"]) > 0.5 and max(shift["lower"]) < -0.5
    # a random subset of 125 has a standard error of about 0.09 on this mean
    assert np.mean(np.abs(shift["random"])) < 0.1 and max(map(abs, shift["random"])) < 0.35


def test_class_fractions_vary_and_respect_what_the_role_holds(pool_and_plan):
    _, plan = pool_and_plan
    train = [s for s in plan["subsets"].values() if s["role"] == "train"]
    nv_share = [s["class_counts"]["nv"] / s["size"] for s in train]
    assert max(nv_share) > 1.5 * min(nv_share)
    for subset in plan["subsets"].values():
        for dx, count in subset["class_counts"].items():
            assert count <= plan["role_counts"][subset["role"]][dx]


def test_the_plan_is_deterministic_frozen_once_and_fits_the_ranker_contract(pool_and_plan, tmp_path):
    pool, plan = pool_and_plan
    assert build_plan(pool, {**FROZEN, "prereg_sha256": fingerprint(FROZEN)}) == plan
    freeze_plan(tmp_path, plan)
    with pytest.raises(SupervisionError, match="never re-drawn"):
        freeze_plan(tmp_path, plan)
    inputs = fit_inputs(plan)
    assert set(inputs) == set(plan["subsets"]) and {v["role"] for v in inputs.values()} == set(ROLES)
    summary = describe(plan)
    assert summary["runs_planned"] == 1000 and summary["exposures_per_image"]["max"] >= 1


def test_a_class_too_small_for_three_roles_is_refused():
    pool = _pool()
    tiny = pd.concat([pool[pool.dx != "nv"], pool[pool.dx == "nv"].head(2)])
    with pytest.raises(SupervisionError, match="too few images"):
        build_plan(tiny, {**FROZEN, "prereg_sha256": fingerprint(FROZEN)})

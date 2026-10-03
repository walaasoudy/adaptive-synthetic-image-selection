"""Stage 4 for the learned selector: a count of all or none is a result, not a broken run."""
import pytest
from omegaconf import OmegaConf

from scripts.classify.ham10000_aggregate_conditions import (AggregationError, check_selection_outcome,
                                                            check_training_data_differs)
from scripts.utils.config import load_named_config
from scripts.utils.ham10000_conditions import PROTOCOLS, get_protocol

LEARNED, DEGENERATE = get_protocol("asism_v2_learned"), get_protocol("asism_v2_learned_all_or_none")


def _runs(counts: dict[str, int], ids: dict[str, str] | None = None) -> dict:
    return {(c, 42): {"manifest": {"data": {"synthetic_images": n, "real_train_images": 1641,
                                            "synthetic_manifest": f"/m/{c}.csv" if n else None,
                                            "synthetic_ids_sha256": (ids or {}).get(c)}}}
            for c, n in counts.items()}


def _selection(outcome: str, n: int) -> dict:
    protocol = "asism_v2_learned" if outcome == "subset" else "asism_v2_learned_all_or_none"
    return {"selection_outcome": outcome, "protocol": protocol, "n_selected_c": n,
            "per_class": {"nv": n}, "c_ids_sha256": "c" * 64 if outcome == "subset" else None}


def test_earlier_protocols_do_not_consult_a_selection_outcome():
    for name in ("v1", "v2", "asism_v2", "asism_v2_none"):
        assert PROTOCOLS[name].selection_outcomes == ()


def test_the_two_learned_protocols_cover_every_outcome_exactly_once():
    assert sorted(LEARNED.selection_outcomes + DEGENERATE.selection_outcomes) == ["all", "none", "subset"]
    assert LEARNED.confirmatory == (("C", "D"),) and DEGENERATE.conditions == ("A", "B")
    assert LEARNED.selection_manifest.filename == DEGENERATE.selection_manifest.filename


def test_a_subset_is_aggregated_as_a_b_c_d():
    runs = _runs({"A": 0, "B": 3168, "C": 900, "D": 900}, {"C": "c" * 64})
    check_training_data_differs(runs, LEARNED)
    report = check_selection_outcome(runs, LEARNED, _selection("subset", 900))
    assert report["selection_outcome"] == "subset" and report["n_selected_c"] == 900


@pytest.mark.parametrize("outcome,n", [("all", 3168), ("none", 0)])
def test_all_or_none_is_accepted_and_reported_not_raised(outcome, n):
    runs = _runs({"A": 0, "B": 3168})
    check_training_data_differs(runs, DEGENERATE)
    report = check_selection_outcome(runs, DEGENERATE, _selection(outcome, n))
    assert report["selection_outcome"] == outcome and report["n_selected_c"] == n
    assert ("C = B" in report["meaning"]) == (outcome == "all")


def test_the_wrong_protocol_for_the_outcome_names_the_right_one():
    with pytest.raises(AggregationError, match="--protocol asism_v2_learned_all_or_none"):
        check_selection_outcome(_runs({"A": 0, "B": 3168}), LEARNED, _selection("all", 3168))
    with pytest.raises(AggregationError, match="--protocol asism_v2_learned\\."):
        check_selection_outcome(_runs({"A": 0, "B": 3168}), DEGENERATE, _selection("subset", 900))


def test_c_trained_on_other_images_than_the_selection_is_refused():
    runs = _runs({"A": 0, "B": 3168, "C": 900, "D": 900}, {"C": "x" * 64})
    with pytest.raises(AggregationError, match="other than the ones"):
        check_selection_outcome(runs, LEARNED, _selection("subset", 900))


def test_the_stage4_recipe_is_the_frozen_one_and_only_the_paths_differ():
    learned = OmegaConf.to_container(load_named_config(LEARNED.stage4_config, "ham_stage4"))
    frozen = OmegaConf.to_container(load_named_config(get_protocol("asism_v2").stage4_config, "ham_stage4"))
    for key in ("model", "training", "seeds", "classes", "loss", "real_train_split", "selection_split"):
        assert learned[key] == frozen[key]
    assert learned["paths"]["results_dir"] != frozen["paths"]["results_dir"]
    assert learned["conditions"]["C"]["synthetic_manifest"].replace("\\", "/").endswith(
        "stage3_asism_v2_ranker/ham-stratified-v1/c_selected.csv")
    assert "{seed}" in learned["conditions"]["D"]["synthetic_manifest"]
    assert learned["conditions"]["B"] == frozen["conditions"]["B"]

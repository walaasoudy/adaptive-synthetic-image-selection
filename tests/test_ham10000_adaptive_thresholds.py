#!/usr/bin/env python3
"""Stage 3e — Adaptive Threshold Learning and the selection it publishes.

The threshold functions are pure, so they are tested on constructed score distributions, which is
what they take. The end-to-end tests run the real entry point over a laid-out PROJECT_ROOT with
ranking scores whose VALUES are arbitrary but whose SHAPE is deliberate: a class whose candidates
all score badly, a class with barely any candidates, a candidate that has become unsafe since it was
ranked. None of them asserts that a selection is good — what is under test is that the order of the
rules holds, and that what the selector cannot deliver is reported instead of hidden.
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
from test_ham10000_ranking_network import (  # noqa: E402
    ALL_SIGNALS,
    CLASSES,
    NAMESPACE,
    N,
    PER_CLASS,
    _dx,
    _ids,
    _write_pool,
)

LABELS = list(CLASSES)


# ==============================================================================================
# The pure threshold rules
# ==============================================================================================


def test_a_rarer_class_receives_a_more_lenient_percentile():
    """One global cut on a two-thirds-nv pool would admit almost nothing for df — which is the very
    imbalance the synthetic data exists to correct."""
    from scripts.asism.ham10000_thresholds import rarity_scaled_percentiles

    counts = {"nv": 4000, "mel": 800, "bkl": 700, "bcc": 350, "akiec": 220, "vasc": 100, "df": 80}
    percentiles = rarity_scaled_percentiles(counts, LABELS, base_percentile=50.0, min_percentile=10.0)

    assert percentiles["nv"] == pytest.approx(50.0)
    assert percentiles["df"] == pytest.approx(10.0)
    assert percentiles["df"] < percentiles["akiec"] < percentiles["mel"] < percentiles["nv"]


def test_the_classes_below_the_largest_are_spread_rather_than_collapsed_onto_the_minimum():
    """What the rule actually guarantees: the non-dominant classes receive DIFFERENT cuts ordered by
    their real rarity, not one shared cut at min_percentile.

    Deliberately not claimed here: that range normalisation differs meaningfully from dividing by
    the maximum. On HAM10000's real distribution the two agree to within 0.7 percentile points for
    every class, because nv is far enough above the rest that subtracting the minimum prevalence
    barely moves anything — see the note in configs/ham10000_stage3.yaml. The test below pins the
    property the policy has, not the one it was once described as having.
    """
    from scripts.asism.ham10000_thresholds import rarity_scaled_percentiles

    counts = {"nv": 10000, "mel": 300, "bkl": 250, "bcc": 200, "akiec": 150, "vasc": 100, "df": 50}
    percentiles = rarity_scaled_percentiles(counts, LABELS, 50.0, 10.0)

    rest = [percentiles[label] for label in LABELS if label != "nv"]
    assert max(rest) - min(rest) > 1.0  # they are spread, not collapsed onto min_percentile
    # and the ordering follows rarity: rarer class, more lenient (lower) percentile
    assert percentiles["df"] < percentiles["vasc"] < percentiles["akiec"] < percentiles["mel"]


def test_range_normalisation_is_not_claimed_to_separate_the_rare_classes_on_real_prevalences():
    """Guards the honest description: on the REAL HAM10000 class distribution the two candidate
    normalisations are interchangeable, so no write-up may present the choice as consequential."""
    from scripts.asism.ham10000_thresholds import rarity_scaled_percentiles

    real = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}
    total = sum(real.values())
    prevalence = {label: real[label] / total for label in LABELS}
    largest = max(prevalence.values())

    range_normalised = rarity_scaled_percentiles(real, LABELS, 50.0, 10.0)
    divided_by_max = {label: 10.0 + 40.0 * (prevalence[label] / largest) for label in LABELS}

    assert max(abs(range_normalised[label] - divided_by_max[label]) for label in LABELS) < 1.0
    # and the real consequence of the policy, stated plainly: nv strict, everyone else lenient
    assert range_normalised["nv"] == 50.0
    assert all(range_normalised[label] < 20.0 for label in LABELS if label != "nv")


def test_a_class_with_no_real_images_does_not_get_the_most_lenient_cut():
    """Missing information never buys leniency."""
    from scripts.asism.ham10000_thresholds import rarity_scaled_percentiles

    counts = {"nv": 1000, "mel": 500, "bkl": 400, "bcc": 300, "akiec": 200, "vasc": 100, "df": 0}
    percentiles = rarity_scaled_percentiles(counts, LABELS, 50.0, 10.0)
    assert percentiles["df"] == pytest.approx(50.0)
    assert percentiles["vasc"] == pytest.approx(10.0)


def test_the_budget_is_proportional_to_each_classes_real_count():
    from scripts.asism.ham10000_thresholds import target_counts

    counts = {"nv": 4000, "mel": 800, "df": 80}
    targets = target_counts(counts, ratio=0.5, minimum=50, maximum=1000, labels=["nv", "mel", "df"])

    assert targets["nv"] == 1000  # clipped by the cap
    assert targets["mel"] == 400
    assert targets["df"] == 50  # raised to the floor


def test_the_quality_floor_is_pooled_not_per_class():
    """A per-class floor would define "bad" relative to each class's own candidates, so the weakest
    class's mediocre images would clear a floor that its own mediocrity had set."""
    from scripts.asism.ham10000_thresholds import quality_floor

    good = np.linspace(0.6, 1.0, 100)
    bad = np.linspace(0.0, 0.2, 100)
    floor = quality_floor(np.concatenate([good, bad]), 25.0)

    assert floor > bad.max() * 0.5
    assert (bad >= floor).sum() < (good >= floor).sum()


def test_the_search_prefers_the_more_selective_of_two_equally_scoring_cuts():
    from scripts.asism.ham10000_thresholds import search_threshold

    scores = np.linspace(0.0, 1.0, 200)
    # A target of exactly half the pool: the objective is maximised near the median.
    threshold = search_threshold(scores, target_count=100, budget_weight=1.0)
    assert 0.4 <= threshold <= 0.6


def test_the_search_respects_the_budget_when_it_is_weighted():
    from scripts.asism.ham10000_thresholds import search_threshold

    scores = np.linspace(0.0, 1.0, 200)
    tight = search_threshold(scores, target_count=20, budget_weight=5.0)
    loose = search_threshold(scores, target_count=180, budget_weight=5.0)
    assert tight > loose


def test_the_class_floor_admits_a_starved_classes_BEST_remaining_candidates():
    """It runs after the quality floor, so the candidates it can reach are already the survivors of
    the absolute cut — it can never pull in a class's worst images."""
    from scripts.asism.ham10000_thresholds import enforce_class_floor

    scores = {"df": np.array([0.90, 0.85, 0.80, 0.75, 0.70])}
    adjusted, lowered = enforce_class_floor({"df": 0.95}, scores, minimum_per_class=3)

    assert adjusted["df"] == pytest.approx(0.80)
    assert (scores["df"] >= adjusted["df"]).sum() == 3
    assert lowered == {"df": 0}


def test_a_class_with_fewer_candidates_than_its_floor_reports_the_shortfall():
    from scripts.asism.ham10000_thresholds import enforce_class_floor

    scores = {"vasc": np.array([0.9, 0.8])}
    adjusted, lowered = enforce_class_floor({"vasc": 0.95}, scores, minimum_per_class=10)

    assert (scores["vasc"] >= adjusted["vasc"]).sum() == 2
    assert "vasc" in lowered


def test_min_max_normalisation_does_not_change_any_candidates_position():
    from scripts.asism.ham10000_thresholds import normalise_scores

    raw = np.array([-3.0, 0.5, 0.4, 7.2, 1.1])
    scaled, transform = normalise_scores(raw)

    assert list(np.argsort(raw)) == list(np.argsort(scaled))
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert transform["degenerate"] is False


def test_an_all_equal_score_column_is_flagged_degenerate_rather_than_thresholded():
    from scripts.asism.ham10000_thresholds import normalise_scores

    _, transform = normalise_scores(np.full(50, 0.3))
    assert transform["degenerate"] is True


def test_the_context_the_threshold_network_sees_excludes_the_candidates_themselves():
    """It decides a cut from a class's SITUATION. Which images are behind those scores is the
    ranking network's judgement, already made."""
    from scripts.asism.ham10000_thresholds import CONTEXT_FEATURES, class_context

    context = class_context(np.linspace(0, 1, 100), real_prevalence=0.1, target_count=30)
    assert context.shape == (len(CONTEXT_FEATURES),)
    assert context[0] == pytest.approx(0.1)
    assert context[-1] == pytest.approx(0.3)


# ==============================================================================================
# End to end
# ==============================================================================================


def _overlay(root: Path, extra: str = "") -> Path:
    path = root / "overlay.yaml"
    path.write_text(
        "ham_stage3:\n"
        "  selection:\n"
        "    min_accepted_per_class: 5\n"
        "    max_accepted_per_class: 40\n"
        "    network_policy:\n"
        "      epochs: 50\n" + extra,
        encoding="utf-8",
    )
    return path


def _write_splits(root: Path, counts: dict[str, int]) -> None:
    splits_dir = root / "data/ham10000/processed/splits" / NAMESPACE
    splits_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {"image_id": f"ISIC_{label}_{index:04d}", "lesion_id": f"HAM_{label}_{index}", "dx": label}
        for label, count in counts.items()
        for index in range(count)
    ]
    pd.DataFrame(rows).to_csv(splits_dir / "classifier_train.csv", index=False)


def _write_ranking_scores(root: Path, scores: np.ndarray | None = None) -> Path:
    """Arbitrary VALUES, deliberate SHAPE. The selector's job is to order and cut; these give it
    something to order. No assertion below treats the numbers as a result."""
    learned_dir = root / "outputs/ham10000/stage3" / NAMESPACE / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    if scores is None:
        scores = np.random.default_rng(5).normal(0, 1, N)
    path = learned_dir / "ranking_scores.parquet"
    pd.DataFrame({
        "image_id": _ids(), "dx": _dx(), "ranking_score": scores,
        "ranking_target": np.nan, "ranking_target_exposures": 0, "ranking_was_training_image": False,
    }).to_parquet(path, index=False)
    return path


REAL_COUNTS = {"nv": 400, "mel": 90, "bkl": 80, "bcc": 40, "akiec": 25, "vasc": 12, "df": 10}


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("thresholds") as root:
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        _write_pool(root)
        _write_splits(root, REAL_COUNTS)
        _write_ranking_scores(root)
        yield root


def _run(**kwargs):
    from scripts.asism.ham10000_05_adaptive_thresholds import run

    return run(NAMESPACE, **kwargs)


def test_the_selection_is_written_where_stage4_condition_C_reads_it(workspace):
    manifest = _run()

    selected = pd.read_csv(manifest["selected_path"])
    assert {"image_id", "image_path", "dx"} <= set(selected.columns)  # what Stage 4 requires
    assert len(selected) == manifest["n_selected"] > 0

    stage4_copy = Path(manifest["stage4_path"])
    assert stage4_copy.is_file()
    assert pd.read_csv(stage4_copy).equals(selected)
    # The un-namespaced copy names the namespace it came from, so a file left over from another
    # experiment is visible rather than silently training condition C on the wrong selection.
    provenance = json.loads(stage4_copy.with_suffix(".provenance.json").read_text(encoding="utf-8"))
    assert provenance["split_namespace"] == NAMESPACE


def test_both_policies_are_recorded_even_though_only_one_selects(workspace):
    """The comparison lives in the artifact rather than being assembled afterwards from whichever
    run happened to be kept."""
    manifest = _run(policy="percentile")

    assert manifest["threshold_policy"] == "percentile"
    assert set(manifest["thresholds_network"]) == set(LABELS)
    assert set(manifest["thresholds_percentile"]) == set(LABELS)
    assert set(manifest["selected_under_each_policy_before_floor"]) == {"network", "percentile"}
    assert manifest["thresholds_applied"]["nv"] == pytest.approx(manifest["thresholds_percentile"]["nv"])


def test_the_quality_floor_applies_before_any_class_gets_its_cut(workspace):
    """Rarity never buys admission for a bad image: nothing below the pooled floor is selected, and
    that includes the rarest class's candidates."""
    manifest = _run()
    selected = pd.read_csv(manifest["selected_path"])

    assert (selected["selection_score"] >= manifest["quality_floor_value"]).all()
    assert manifest["order"][:2] == ["safety", "quality_floor"]
    assert manifest["candidates_above_quality_floor"] < manifest["candidates_ranked"]


def test_a_weak_but_admissible_class_is_not_selected_down_to_nothing(workspace):
    """df's candidates are the weakest of those that clear the pooled quality floor, so a cut tuned
    on the pool would take almost none of them. The class floor lifts its threshold just enough —
    over the survivors of the floor, which is the whole point of the ordering."""
    rng = np.random.default_rng(6)
    scores = rng.normal(1.0, 0.3, N)
    df_positions = [index for index, label in enumerate(_dx()) if label == "df"]
    scores[df_positions] = rng.normal(0.75, 0.02, len(df_positions))
    _write_ranking_scores(workspace, scores)

    manifest = _run()

    assert manifest["selected_per_class"]["df"] >= 5  # the fixture's min_accepted_per_class
    assert manifest["classes_eliminated_by_the_quality_floor"] == {}


def test_a_class_with_nothing_above_the_quality_floor_selects_NOTHING_and_says_so(workspace):
    """The floor is absolute. If a class produced no candidate that clears it, selecting none of
    them is the floor working — but it is a different finding from missing a budget, and it says the
    generator produced nothing usable for that class."""
    rng = np.random.default_rng(6)
    scores = rng.normal(1.0, 0.2, N)
    df_positions = [index for index, label in enumerate(_dx()) if label == "df"]
    scores[df_positions] = rng.normal(-1.0, 0.05, len(df_positions))
    _write_ranking_scores(workspace, scores)

    manifest = _run()

    assert manifest["selected_per_class"]["df"] == 0
    assert manifest["classes_eliminated_by_the_quality_floor"] == {"df": PER_CLASS}
    assert "df" in manifest["classes_short_of_target"]


def test_a_class_that_cannot_reach_its_target_is_reported_not_papered_over(workspace):
    """A class short of its budget is a finding about the generator, and Stage 4 has to be read
    knowing it."""
    _write_splits(workspace, {**REAL_COUNTS, "nv": 4000})
    manifest = _run()

    assert manifest["target_counts"]["nv"] == 40  # the fixture cap
    assert isinstance(manifest["classes_short_of_target"], dict)
    for label, entry in manifest["classes_short_of_target"].items():
        assert entry["selected"] < entry["target"]


def test_no_class_exceeds_its_budget_cap(workspace):
    manifest = _run()
    for label, count in manifest["selected_per_class"].items():
        assert count <= 40


def test_a_candidate_that_became_unsafe_since_it_was_ranked_is_dropped(workspace):
    """The gate is re-run against the CURRENT artifacts. An image must not stay selected because it
    was safe when it was scored."""
    _write_pool(workspace, near_duplicates=PER_CLASS)  # every nv candidate is now a near-duplicate
    manifest = _run()
    selected = pd.read_csv(manifest["selected_path"])

    assert manifest["candidates_dropped_by_current_safety_gate"] == PER_CLASS
    assert "nv" not in set(selected["dx"])


def test_the_protected_split_cannot_be_used_to_set_thresholds(workspace):
    from scripts.asism.ham10000_05_adaptive_thresholds import real_class_counts

    with pytest.raises(SystemExit, match="protected split"):
        real_class_counts(workspace, "final_eval_heldout")


def test_missing_ranking_scores_name_the_command_that_produces_them(workspace):
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "learned/ranking_scores.parquet").unlink()
    with pytest.raises(SystemExit, match="ham10000_04_train_ranking_network"):
        _run()


def test_an_undifferentiated_ranking_column_refuses_to_select(workspace):
    """If every candidate scores the same there is no ordering, and a cut would be an artifact of
    floating-point noise rather than a decision."""
    _write_ranking_scores(workspace, np.full(N, 0.42))
    with pytest.raises(SystemExit, match="no ordering to threshold"):
        _run()


def test_the_network_policy_records_how_far_it_drifted_from_the_search_it_distilled(workspace):
    """The residual is what says whether a class's context explains the threshold it was given, or
    whether the network simply could not reproduce it."""
    manifest = _run(policy="network")
    diagnostics = manifest["network_diagnostics"]

    assert set(diagnostics["searched_thresholds"]) <= set(LABELS)
    assert set(diagnostics["distillation_residual"]) == set(diagnostics["searched_thresholds"])
    assert all(value >= 0 for value in diagnostics["distillation_residual"].values())
    assert diagnostics["context_features"][0] == "real_prevalence"


def test_the_distillation_is_also_scored_on_classes_the_network_never_saw(workspace):
    """The in-sample residual cannot distinguish a learned rule from memorisation.

    AdaptiveThresholdNetwork takes a per-class embedding and is fitted on one point per class, so it
    can drive the in-sample residual to ~0 through the embedding alone while the context vector
    contributes nothing. The leave-one-class-out residual is the measurement that separates the two,
    because a class predicted by a network that never saw it can only be answered from its context.
    This test pins that it is COMPUTED and REPORTED — not that it comes out low, which is an
    empirical result about the data and not something a test may assert in advance.
    """
    manifest = _run(policy="network")
    diagnostics = manifest["network_diagnostics"]

    held_out = diagnostics["distillation_residual_leave_one_class_out"]
    assert set(held_out) == set(diagnostics["distillation_residual"])
    assert all(value >= 0 for value in held_out.values())

    summary = diagnostics["residual_summary"]
    assert summary["in_sample_mean"] is not None and summary["leave_one_class_out_mean"] is not None
    assert "context" in summary["interpretation"]


def test_the_quality_floor_reports_how_much_of_each_class_it_removed(workspace):
    """A pooled floor need not remove the same share of every class, and a class that loses most of
    its candidates to it is a finding about the generator. The all-or-nothing elimination line fires
    only at exactly zero survivors, so it cannot show that."""
    manifest = _run()

    attrition = manifest["quality_floor_attrition_per_class"]
    assert attrition, "no class reported its attrition"
    for entry in attrition.values():
        assert entry["above_quality_floor"] <= entry["candidates"]
        assert 0.0 <= entry["removed_fraction"] <= 1.0


def test_an_empty_selection_is_refused_rather_than_written(workspace):
    """An empty asism_selected.csv would make condition C a second copy of condition A under a
    different name, and Stage 4 would train nine runs before anything noticed.

    Forced through the budget: with every class budgeted zero images, the frozen search maximises
    its objective by keeping nothing and the per-class floor has nothing to restore. That is a
    degenerate configuration on purpose — the point is the guard, not the route to it.
    """
    (workspace / "overlay.yaml").write_text(
        "ham_stage3:\n"
        "  selection:\n"
        "    min_accepted_per_class: 0\n"
        "    max_accepted_per_class: 0\n"
        "    network_policy:\n"
        "      epochs: 50\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="selection is empty"):
        _run()

    assert not (workspace / "outputs/ham10000/stage3/asism_selected.csv").exists(), (
        "an empty selection was written where Stage 4 condition C would read it"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

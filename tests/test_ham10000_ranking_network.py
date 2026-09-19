#!/usr/bin/env python3
"""Stage 3c/3d — the candidate pool, the subset design, and the Multi-Objective Ranking Network.

WHAT IS AND IS NOT FABRICATED HERE. The subset design and the pool are computed from constructed
score frames, which is what those functions take. The one place a measurement is invented is the
utility table in the end-to-end test: measuring it for real means one GPU training run per subset,
and the test exists to check the WIRING around that measurement — that the train and val images are
disjoint, that an image seen in too few subsets gets no target, that the manifest records how many
signals actually reached the model. Its numbers are arbitrary and no assertion treats them as
results. Every refusal test below runs against the real entry points.
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

NAMESPACE = "ham-ranking-v1"
CLASSES = ("nv", "mel", "bkl", "bcc", "akiec", "vasc", "df")
PER_CLASS = 60
N = PER_CLASS * len(CLASSES)

ALL_SIGNALS = ["similarity", "iqa", "uncertainty", "explainability", "agreement"]


def _ids() -> list[str]:
    return [f"SYN_{name}_{index:03d}" for name in CLASSES for index in range(PER_CLASS)]


def _dx() -> list[str]:
    return [name for name in CLASSES for _ in range(PER_CLASS)]


def _u(seed: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, N)


def _frames(*, near_duplicates: int = 0, invalid_iqa: int = 0) -> dict[str, pd.DataFrame]:
    ids = _ids()
    duplicate = np.zeros(N, dtype=bool)
    duplicate[:near_duplicates] = True
    valid = np.ones(N, dtype=bool)
    valid[N - invalid_iqa:] = False if invalid_iqa else valid[N - invalid_iqa:]
    return {
        "similarity": pd.DataFrame({
            "image_id": ids,
            "similarity_knn_mean": _u(1, 0.2, 0.9),
            "similarity_top1": _u(2, 0.3, 0.99),
            "similarity_topk_spread": _u(3, 0.0, 0.2),
            "novelty_is_near_duplicate": duplicate,
        }),
        "iqa": pd.DataFrame({
            "image_id": ids,
            "iqa_composite": _u(4),
            "iqa_sharpness": _u(5, 10, 200),
            "iqa_contrast_std": _u(6, 10, 60),
            "iqa_valid": valid,
        }),
        "uncertainty": pd.DataFrame({
            "image_id": ids,
            "uncertainty_mutual_information": _u(7, 0.0, 0.4),
            "uncertainty_predictive_entropy": _u(8, 0.1, 0.9),
        }),
        "explainability": pd.DataFrame({
            "image_id": ids,
            "explainability_calibrated_typicality": _u(9, 0.01, 0.99),
        }),
        "agreement": pd.DataFrame({
            "image_id": ids,
            "agreement_score": _u(10, 0.0, 1.0),
            "agreement_margin": _u(11, -0.5, 0.9),
        }),
    }


def _write_pool(root: Path, surviving: list[str] | None = None, **frame_kwargs) -> None:
    from scripts.asism.signals import write_score_artifact
    from scripts.utils.manifest import write_json

    stage3_dir = root / "outputs/ham10000/stage3" / NAMESPACE
    scores_dir = stage3_dir / "signals"
    scores_dir.mkdir(parents=True, exist_ok=True)
    for signal, frame in _frames(**frame_kwargs).items():
        write_score_artifact(
            frame, scores_dir / f"{signal}_scores.parquet", signal,
            {"dataset": "ham10000", "split_namespace": NAMESPACE, "candidates_csv_sha256": "a" * 64,
             "git_commit_hash": "fixture"},
        )
    write_json(stage3_dir / "gonogo_report.json", {
        "stage": "ham10000_asism_gonogo",
        "surviving_signals": ALL_SIGNALS if surviving is None else surviving,
        "ablation_only_signals": [], "excluded_signals": [],
    })

    manifest_dir = root / "outputs/ham10000/stage2" / NAMESPACE
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_id": _ids(), "dx": _dx(), "status": "accepted",
        "image_path": [str(manifest_dir / "full/candidates" / f"{name}.jpg") for name in _ids()],
    }).to_csv(manifest_dir / "all_candidates.csv", index=False)


def _overlay(root: Path, subset_sizes: str = "[10, 20, 25]", total_subsets: int = 40) -> Path:
    """Fixture-scale settings only.

    The real design's subset sizes cannot be drawn from a 420-image pool, and the feasibility gate
    says so correctly. The sizes here are SCALED to the fixture, not relaxed: the largest one still
    has to fit inside the smallest val-side quantile band, which is the rule under test.
    """
    path = root / "overlay.yaml"
    path.write_text(
        "ham_stage3:\n"
        "  learned_asism:\n"
        "    set_utility_network:\n"
        "      epochs: 3\n"
        "      early_stopping_patience: 3\n"
        "    ranking_network:\n"
        "      epochs: 3\n"
        "      min_training_images: 10\n"
        "    subset_design:\n"
        f"      subset_sizes: {subset_sizes}\n"
        f"      total_subsets: {total_subsets}\n",
        encoding="utf-8",
    )
    return path


def _stage3():
    from scripts.utils.config import load_named_config

    return load_named_config("ham10000_stage3.yaml", "ham_stage3")


def _pool(root: Path, **kwargs):
    from scripts.asism.ham10000_ranking import load_candidate_pool

    return load_candidate_pool(NAMESPACE, _stage3(), root / "outputs/ham10000/stage2", verbose=False, **kwargs)


@pytest.fixture
def workspace(monkeypatch):
    with fixture_workspace("ranking") as root:
        monkeypatch.setenv("PROJECT_ROOT", str(root))
        monkeypatch.setenv("THESIS_CONFIG_OVERLAY", str(_overlay(root)))
        _write_pool(root)
        yield root


# ==============================================================================================
# Which features the learned models are allowed to see
# ==============================================================================================


def test_a_signal_the_gate_excluded_cannot_reach_the_model_through_its_columns():
    """Implementation is not admission, and that rule has to bind the learned selector too — the
    gate's verdict is about evidence, not about which code path consumes it."""
    from scripts.asism.ham10000_ranking import active_feature_columns, contributing_signals

    configured = ["similarity_knn_mean", "iqa_composite", "agreement_score"]
    active = active_feature_columns(configured, ["similarity", "agreement"])

    assert active == ["similarity_knn_mean", "agreement_score"]
    assert contributing_signals(active) == ["agreement", "similarity"]


def test_a_configured_column_owned_by_no_signal_is_an_error_not_a_silent_drop():
    """Dropping it would shrink the model's input space with no trace in any manifest: the model
    would train on fewer features than the config claims, and nothing downstream could detect it."""
    from scripts.asism.ham10000_ranking import active_feature_columns

    with pytest.raises(ValueError, match="no registered signal owner"):
        active_feature_columns(["iqa_composite", "explainability_region_overlap"], ["iqa"])


def test_the_chexpert_uncertainty_and_explainability_columns_are_not_registered_here():
    """They exist in the CheXpert artifacts and measure something else. Reaching for one by name
    must fail loudly rather than resolve to nothing."""
    from scripts.asism.ham10000_ranking import FEATURE_COLUMNS_BY_SIGNAL

    owned = {column for columns in FEATURE_COLUMNS_BY_SIGNAL.values() for column in columns}
    assert "uncertainty_mean_std" not in owned
    assert "explainability_region_overlap" not in owned
    # And only the calibrated attention feature is admissible.
    assert FEATURE_COLUMNS_BY_SIGNAL["explainability"] == frozenset({"explainability_calibrated_typicality"})


def test_no_admitted_column_refuses_rather_than_training_on_nothing():
    from scripts.asism.ham10000_ranking import active_feature_columns

    with pytest.raises(ValueError, match="admitted no feature columns"):
        active_feature_columns(["iqa_composite"], ["agreement"])


# ==============================================================================================
# The candidate pool and its safety gate
# ==============================================================================================


def test_only_surviving_signals_columns_reach_the_pool(workspace):
    _write_pool(workspace, surviving=["iqa", "agreement"])
    frame, surviving, report = _pool(workspace)

    assert surviving == ["iqa", "agreement"]
    assert "iqa_composite" in frame.columns and "agreement_score" in frame.columns
    assert "similarity_knn_mean" not in frame.columns
    assert "explainability_calibrated_typicality" not in frame.columns
    # dx comes from the manifest, and the stratum is the class, because one candidate has one class.
    assert set(frame["dx"]) == set(CLASSES)
    assert (frame["__stratum"] == frame["dx"]).all()
    assert report["candidates_after_safety_gate"] == N


def test_the_safety_gate_runs_even_when_its_signals_did_not_survive_the_gate(workspace):
    """A memorised near-copy of a real patient's lesion must not become selectable because the
    similarity signal happened to be redundant with another one."""
    _write_pool(workspace, surviving=["agreement"], near_duplicates=5, invalid_iqa=3)
    frame, surviving, report = _pool(workspace)

    assert surviving == ["agreement"]
    assert report["candidates_after_safety_gate"] == N - 8
    assert report["safety_gate_removals"] == {"invalid_iqa_safety_gate": 3, "near_duplicate_safety_gate": 5}


def test_the_safety_switches_come_from_the_config_and_are_recorded_in_the_report(workspace):
    """Regression: both switches used to be plain Python defaults that no caller overrode, so
    `learned_asism.safety` in the config was decorative — turning a check off changed nothing and
    said nothing. A removal count of zero must not be readable as "nothing failed" when the truth is
    "nothing was checked", so the report states which checks actually ran.
    """
    _write_pool(workspace, near_duplicates=5, invalid_iqa=3)

    _, _, on = _pool(workspace)
    assert on["safety_checks_applied"] == {"reject_invalid_iqa": True, "reject_near_duplicates": True}
    assert on["candidates_after_safety_gate"] == N - 8

    (workspace / "overlay.yaml").write_text(
        "ham_stage3:\n"
        "  learned_asism:\n"
        "    safety:\n"
        "      reject_near_duplicates: false\n",
        encoding="utf-8",
    )
    _, _, off = _pool(workspace)
    assert off["safety_checks_applied"]["reject_near_duplicates"] is False
    # the config change reached the gate: the five near-duplicates are no longer removed
    assert off["candidates_after_safety_gate"] == N - 3
    assert "near_duplicate_safety_gate" not in off["safety_gate_removals"]


def test_the_safety_gate_fails_closed_on_a_candidate_it_has_no_row_for():
    """An image the gate cannot vouch for is not the same as an image the gate approved, and the
    difference matters most exactly when something upstream has gone wrong."""
    from scripts.asism.ham10000_ranking import unsafe_candidates

    iqa = pd.DataFrame({"image_id": ["a"], "iqa_valid": [True]})
    similarity = pd.DataFrame({"image_id": ["a", "b"], "novelty_is_near_duplicate": [False, False]})

    reasons = unsafe_candidates(["a", "b"], iqa, similarity)
    assert reasons == {"b": "invalid_iqa_safety_gate"}


def test_a_missing_safety_artifact_stops_the_pool_naming_the_command(workspace):
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals" / "similarity_scores.parquet").unlink()
    _write_pool(workspace, surviving=["iqa"])
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "signals" / "similarity_scores.parquet").unlink()

    with pytest.raises(SystemExit, match="SAFETY GATE"):
        _pool(workspace)


def test_a_pool_with_no_surviving_signal_refuses_to_train_anything(workspace):
    _write_pool(workspace, surviving=[])
    with pytest.raises(SystemExit, match="No signal survived"):
        _pool(workspace)


def test_a_missing_gonogo_report_names_the_gate(workspace):
    (workspace / "outputs/ham10000/stage3" / NAMESPACE / "gonogo_report.json").unlink()
    with pytest.raises(SystemExit, match="ham10000_02_gonogo"):
        _pool(workspace)


# ==============================================================================================
# Utility results
# ==============================================================================================


def _results(subset_ids, metric="balanced_accuracy"):
    return [
        {"subset_id": subset_id, f"real_only_{metric}": 0.50,
         f"augmented_{metric}": 0.50 + 0.01 * index, "seed": 42}
        for index, subset_id in enumerate(subset_ids)
    ]


def test_utility_delta_is_measured_against_the_real_only_baseline():
    from scripts.asism.ham10000_ranking import validate_utility_results

    subsets = [{"subset_id": "train_utility_0000"}, {"subset_id": "train_utility_0001"}]
    frame = validate_utility_results(subsets, _results([row["subset_id"] for row in subsets]))

    assert list(frame["utility_delta"].round(4)) == [0.0, 0.01]
    assert set(frame["utility_metric"]) == {"balanced_accuracy"}


def test_an_unmeasured_subset_is_an_error_rather_than_a_smaller_training_set():
    """The subsets that fail to measure are not a random sample of the design — one drawn from an
    extreme quantile band is exactly the one most likely to produce a degenerate proxy run — so
    dropping them would bias the utility model toward the easy middle of the pool."""
    from scripts.asism.ham10000_ranking import validate_utility_results

    subsets = [{"subset_id": "a"}, {"subset_id": "b"}]
    with pytest.raises(ValueError, match="incomplete"):
        validate_utility_results(subsets, _results(["a"]))


def test_results_measured_under_a_different_metric_are_refused():
    """A utility measured under balanced accuracy and compared under macro-F1 is not a comparison."""
    from scripts.asism.ham10000_ranking import validate_utility_results

    subsets = [{"subset_id": "a"}]
    with pytest.raises(ValueError, match="need columns"):
        validate_utility_results(subsets, _results(["a"], metric="macro_f1"), "balanced_accuracy")


# ==============================================================================================
# Feasibility and the build gate
# ==============================================================================================


def _run_phase(phase: str, **kwargs):
    from scripts.asism.ham10000_03_build_utility_subsets import run

    return run(NAMESPACE, phase, **kwargs)


def test_feasibility_reports_whether_the_design_can_be_drawn_without_replacement(workspace):
    report = _run_phase("feasibility")

    assert report["n_candidates"] == N
    assert report["train_pool_size"] + report["val_pool_size"] == N
    assert set(report["per_class_counts"]) == set(CLASSES)
    assert report["contributing_signals"] == sorted(ALL_SIGNALS)
    assert isinstance(report["feasible"], bool)


def test_a_design_whose_single_signal_band_is_too_small_is_reported_infeasible(workspace):
    """A single-signal subset draws entirely from one quantile band of one feature inside one role
    pool — the tightest constraint in the design, and the one that has to be checked."""
    _overlay(workspace, subset_sizes="[400]")
    report = _run_phase("feasibility")

    assert report["feasible"] is False
    assert any("without replacement" in failure for failure in report["failures"])


def test_build_refuses_until_a_matching_feasibility_report_exists(workspace):
    with pytest.raises(SystemExit, match="--phase feasibility"):
        _run_phase("build")


def test_build_refuses_a_feasibility_report_computed_for_a_different_design(workspace):
    """A config edited after a report was written cannot be built on that report's blessing."""
    _run_phase("feasibility")
    _overlay(workspace, total_subsets=32)
    with pytest.raises(SystemExit, match="different design"):
        _run_phase("build")


def test_build_refuses_a_failing_feasibility_report(workspace):
    _overlay(workspace, subset_sizes="[400]")
    _run_phase("feasibility")
    with pytest.raises(SystemExit, match="lists failures"):
        _run_phase("build")


def test_built_subsets_are_image_disjoint_between_the_two_roles(workspace):
    """Subsets share images, so a random split of SUBSETS would leak. The image pools are split
    first and each subset is drawn from exactly one of them."""
    from scripts.asism.learned import read_jsonl

    _run_phase("feasibility")
    manifest = _run_phase("build")
    records = read_jsonl(workspace / "outputs/ham10000/stage3" / NAMESPACE / "learned/utility_subsets.jsonl")

    train_ids = {i for row in records if row["role"] == "train" for i in row["image_ids"]}
    val_ids = {i for row in records if row["role"] == "val" for i in row["image_ids"]}
    assert train_ids and val_ids
    assert not (train_ids & val_ids)
    assert manifest["n_subsets"] == len(records)
    assert manifest["train_subsets"] + manifest["val_subsets"] == manifest["n_subsets"]
    assert all(len(set(row["image_ids"])) == row["size"] for row in records)


# ==============================================================================================
# Training the ranking network
# ==============================================================================================


def test_training_refuses_without_measured_utility(workspace):
    """There is no fallback that estimates utility from the signals — they are the model's own
    inputs, so a target derived from them would make the ranker fit its own features."""
    from scripts.asism.ham10000_04_train_ranking_network import run

    _run_phase("feasibility")
    _run_phase("build")
    with pytest.raises(SystemExit, match="--phase measure"):
        run(NAMESPACE, device="cpu")


def test_the_ranking_network_trains_end_to_end_and_records_what_fed_it(workspace):
    """A WIRING test. The utility table below is constructed, not measured, and nothing here asserts
    that the resulting scores are good — only that the pieces connect, that the held-out images are
    genuinely held out, and that the manifest states how many signals reached the model."""
    from scripts.asism.ham10000_04_train_ranking_network import run
    from scripts.asism.ham10000_ranking import write_jsonl
    from scripts.asism.learned import read_jsonl

    _run_phase("feasibility")
    _run_phase("build")

    learned_dir = workspace / "outputs/ham10000/stage3" / NAMESPACE / "learned"
    subsets = read_jsonl(learned_dir / "utility_subsets.jsonl")
    rng = np.random.default_rng(3)
    write_jsonl(learned_dir / "utility_results.jsonl", [
        {
            "subset_id": row["subset_id"], "role": row["role"], "size": row["size"], "seed": 42,
            "real_only_balanced_accuracy": 0.50,
            "augmented_balanced_accuracy": 0.50 + float(rng.normal(0.02, 0.01)),
            "utility_metric": "balanced_accuracy",
        }
        for row in subsets
    ])

    manifest = run(NAMESPACE, device="cpu")

    assert manifest["train_val_image_overlap"] == 0
    assert manifest["n_training_images"] >= 10
    assert manifest["contributing_signals"] == sorted(ALL_SIGNALS)
    assert manifest["is_multi_signal"] is True
    # The name says multi-objective; the manifest must not let it imply multi-task optimisation.
    assert manifest["optimisation"].startswith("single_objective")
    assert "NOT independent evidence" in manifest["pairwise_accuracy_measures"]

    scores = pd.read_parquet(manifest["scores_path"])
    assert len(scores) == N
    assert scores["ranking_score"].notna().all()
    # Every candidate is scored, including the ones no subset ever contained: selection has to rank
    # the whole pool, not only the images that happened to be measured.
    assert (~scores["ranking_was_training_image"]).any()
    assert scores.loc[~scores["ranking_was_training_image"], "ranking_target"].isna().all()

    assert (Path(manifest["checkpoint_dir"]) / "ranking_model.pt").is_file()
    assert (Path(manifest["checkpoint_dir"]) / "set_utility_model.pt").is_file()


def test_an_image_seen_in_too_few_subsets_gets_no_distilled_target(workspace):
    """A marginal contribution averaged over one or two subsets is noise, not a utility estimate."""
    from scripts.asism.ham10000_04_train_ranking_network import run
    from scripts.asism.ham10000_ranking import write_jsonl
    from scripts.asism.learned import read_jsonl

    _run_phase("feasibility")
    _run_phase("build")
    learned_dir = workspace / "outputs/ham10000/stage3" / NAMESPACE / "learned"
    subsets = read_jsonl(learned_dir / "utility_subsets.jsonl")
    write_jsonl(learned_dir / "utility_results.jsonl", [
        {"subset_id": row["subset_id"], "role": row["role"], "size": row["size"], "seed": 42,
         "real_only_balanced_accuracy": 0.5, "augmented_balanced_accuracy": 0.52,
         "utility_metric": "balanced_accuracy"}
        for row in subsets
    ])

    manifest = run(NAMESPACE, device="cpu")
    scores = pd.read_parquet(manifest["scores_path"])

    trained = scores[scores["ranking_was_training_image"]]
    assert (trained["ranking_target_exposures"] >= manifest["minimum_image_exposures"]).all()
    assert manifest["images_below_minimum_exposure"] >= 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

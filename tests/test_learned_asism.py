import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts.asism.learned import (
    active_feature_columns,
    aggregate_hard_proxy_best_threshold,
    apply_feature_frame,
    bootstrap_class_contexts,
    build_controlled_subsets,
    build_role_conditioned_subsets,
    class_aware_context_vector,
    choose_full_policy,
    compute_verified_context_counts,
    contributing_signals,
    critic_proxy_correlation_per_class,
    determine_per_class_official_method,
    diversify_verification_candidates,
    eligible_classes_for_official_training,
    filter_targets_to_eligible_classes,
    enforce_acceptance_criteria,
    evaluate_subset_design_feasibility,
    freeze_acceptance_criteria,
    hard_threshold_grid_search,
    held_out_generalization_metrics,
    pool_feasibility_report,
    real_class_support_context,
    resolve_critic_assisted_exploratory_targets,
    resolve_verified_only_targets,
    safe_feature_frame,
    scientific_status_for_namespace,
    split_image_pool,
    validate_utility_results,
    verify_built_subsets,
)
from scripts.asism.models import (
    AdaptiveThresholdNetwork,
    MultiObjectiveRankingNetwork,
    SetUtilityNetwork,
    soft_selection_gate,
)


def _load_script(name: str):
    """Load a scripts/asism/*.py module whose filename starts with a digit (not import-able)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "asism" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_set_utility_is_permutation_invariant():
    torch.manual_seed(1)
    model = SetUtilityNetwork(3, (8, 4), (3,))
    model.eval()
    x = torch.randn(1, 5, 3); mask = torch.ones(1, 5, dtype=torch.bool)
    assert torch.allclose(model(x, mask), model(x[:, [3, 1, 4, 0, 2]], mask), atol=1e-6)


def test_active_feature_columns_excludes_gonogo_rejected_and_ablation_only_signals():
    configured = [
        "similarity_knn_mean", "iqa_composite", "uncertainty_mean_std",
        "explainability_region_overlap", "agreement_score",
    ]
    # Explainability is excluded and uncertainty is ablation-only, so neither
    # may be required by the primary learned selector's merged feature frame.
    assert active_feature_columns(configured, ["similarity", "iqa", "agreement"]) == [
        "similarity_knn_mean", "iqa_composite", "agreement_score",
    ]


def test_soft_gate_and_threshold_shapes():
    model = AdaptiveThresholdNetwork(4, 8, embedding_dim=3, hidden=(5,))
    thresholds = model(torch.tensor([0, 1]), torch.randn(2, 8))
    gate = soft_selection_gate(torch.tensor([0.2, 0.9]), thresholds, 0.2)
    assert thresholds.shape == gate.shape == (2,)
    assert bool(((thresholds >= 0) & (thresholds <= 1)).all())


def test_controlled_subset_design_has_all_families():
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(90)],
                          "a": np.linspace(0, 1, 90), "b": np.linspace(1, 0, 90)})
    records = build_controlled_subsets(frame, ["a", "b"], 20, [12], .2, .3, .5, 42)
    assert len(records) == 20
    assert {row["design"]["type"] for row in records} == {"matched_random", "single_signal", "mixed"}
    assert all(len(set(row["image_ids"])) == row["size"] for row in records)


def test_build_controlled_subsets_rejects_inconsistent_fractions():
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(90)],
                          "a": np.linspace(0, 1, 90), "b": np.linspace(1, 0, 90)})
    with pytest.raises(ValueError):
        build_controlled_subsets(frame, ["a", "b"], 20, [12], .2, .3, .4, 42)  # sums to .9


def test_build_controlled_subsets_honors_quantile_bins():
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(200)],
                          "a": np.linspace(0, 1, 200), "b": np.linspace(1, 0, 200)})
    records = build_controlled_subsets(frame, ["a", "b"], 20, [12], 0.0, 1.0, 0.0, 42, quantile_bins=5)
    bands = {row["design"]["quantile_band"] for row in records if row["design"]["type"] == "single_signal"}
    assert bands, "expected at least one single_signal design"
    assert bands.issubset(set(range(5)))


def test_normalization_and_measured_utility_contract():
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [2.0, 2.0, 2.0]})
    normalized, _ = safe_feature_frame(frame, ["x", "y"])
    assert np.isfinite(normalized.to_numpy()).all()
    subsets = [{"subset_id": "s1"}]
    result = validate_utility_results(subsets, [{"subset_id": "s1", "real_only_macro_auroc": .7,
                                                "augmented_macro_auroc": .72, "fold": 0, "seed": 1}])
    assert abs(result.utility_delta.iloc[0] - .02) < 1e-9


def test_apply_feature_frame_reuses_frozen_stats_not_the_new_pool():
    """The real bug this guards against: re-fitting normalization at selection time silently drifts
    the ranker's inputs away from what it was trained on. apply_feature_frame must use the frozen
    fit stats regardless of what pool it is applied to."""
    pool_a = pd.DataFrame({
        "image_id": [f"a{i}" for i in range(50)],
        "x": np.linspace(0, 1, 50),
        "y": np.linspace(1, 2, 50),
    })
    _, stats = safe_feature_frame(pool_a, ["x", "y"])

    shared_row = {"image_id": "shared", "x": 0.42, "y": 1.7}
    rng = np.random.default_rng(0)
    pool_b = pd.concat([
        pd.DataFrame([shared_row]),
        pd.DataFrame({
            "image_id": [f"b{i}" for i in range(2000)],
            "x": rng.uniform(-5, 5, 2000),
            "y": rng.uniform(-5, 5, 2000),
        }),
    ], ignore_index=True)

    normalized_b = apply_feature_frame(pool_b, ["x", "y"], stats)
    shared_row_normalized = normalized_b.iloc[0]

    manual_x = (shared_row["x"] - stats["mean"]["x"]) / stats["std"]["x"]
    manual_y = (shared_row["y"] - stats["mean"]["y"]) / stats["std"]["y"]
    assert abs(shared_row_normalized["x"] - manual_x) < 1e-9
    assert abs(shared_row_normalized["y"] - manual_y) < 1e-9

    # Refitting on pool_b directly (the old, buggy behavior) would NOT match — proves
    # apply_feature_frame is actually reusing the frozen stats, not silently refitting.
    refit_on_b, _ = safe_feature_frame(pool_b, ["x", "y"])
    assert abs(refit_on_b.iloc[0]["x"] - shared_row_normalized["x"]) > 1e-3


def test_learned_ranking_score_matches_across_pools_through_full_network():
    """Production path end-to-end: apply_feature_frame -> MultiObjectiveRankingNetwork -> sigmoid
    (exactly what 06_learn_thresholds_select.py computes as `learned_ranking_score`). The same
    image with the same raw features, same frozen normalization stats, same ranking checkpoint,
    must get the exact same final score whether it sits in a 10-image pool or a 2000-image pool
    with a completely different score distribution."""
    pool_a = pd.DataFrame({
        "image_id": [f"a{i}" for i in range(50)],
        "x": np.linspace(0, 1, 50),
        "y": np.linspace(1, 2, 50),
    })
    _, stats = safe_feature_frame(pool_a, ["x", "y"])

    shared_row = {"image_id": "shared", "x": 0.42, "y": 1.7}
    rng = np.random.default_rng(0)
    small_pool = pd.concat([
        pd.DataFrame([shared_row]),
        pd.DataFrame({"image_id": [f"c{i}" for i in range(10)],
                     "x": rng.uniform(0, 1, 10), "y": rng.uniform(1, 2, 10)}),
    ], ignore_index=True)
    large_pool = pd.concat([
        pd.DataFrame([shared_row]),
        pd.DataFrame({"image_id": [f"b{i}" for i in range(2000)],
                     "x": rng.uniform(-5, 5, 2000), "y": rng.uniform(-5, 5, 2000)}),
    ], ignore_index=True)

    torch.manual_seed(7)
    ranker = MultiObjectiveRankingNetwork(input_dim=2, hidden=(4,), dropout=0.0)
    ranker.eval()

    def learned_ranking_score_for_shared_image(pool: pd.DataFrame) -> float:
        normalized = apply_feature_frame(pool, ["x", "y"], stats)
        with torch.no_grad():
            raw = ranker(torch.tensor(normalized.to_numpy(dtype=np.float32)))
        scores = 1.0 / (1.0 + np.exp(-raw.numpy()))
        position = int(np.flatnonzero(pool["image_id"].to_numpy() == "shared")[0])
        return float(scores[position])

    score_in_small_pool = learned_ranking_score_for_shared_image(small_pool)
    score_in_large_pool = learned_ranking_score_for_shared_image(large_pool)
    assert score_in_small_pool == pytest.approx(score_in_large_pool, abs=1e-9)


def test_real_class_support_context_uses_real_patient_prevalence():
    frame = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p3", "p4"],
        "Edema": [1.0, 1.0, 0.0, np.nan, -1.0],
    })
    context = real_class_support_context("dev", ["Edema"], frame=frame)
    assert context["namespace"] == "dev"
    assert context["scientific_status"] == "non-production / development-only"
    assert context["labels"]["Edema"]["positive_patients"] == 1  # p1
    assert context["labels"]["Edema"]["negative_patients"] == 1  # p2
    assert context["labels"]["Edema"]["real_prevalence"] == pytest.approx(0.5)


def test_real_class_support_context_handles_zero_support():
    frame = pd.DataFrame({"patient_id": ["p1", "p2"], "Fracture": [np.nan, -1.0]})
    context = real_class_support_context("dev", ["Fracture"], frame=frame)
    assert context["labels"]["Fracture"]["positive_patients"] == 0
    assert context["labels"]["Fracture"]["negative_patients"] == 0
    assert context["labels"]["Fracture"]["real_prevalence"] is None


def test_scientific_status_for_namespace_only_production_is_production():
    assert scientific_status_for_namespace("production") == "production"
    assert scientific_status_for_namespace("dev") == "non-production / development-only"
    assert scientific_status_for_namespace("dev-smoke-v1") == "non-production / development-only"


def _synthetic_candidate_fixture(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_id": [f"img{i:04d}" for i in range(n)],
        "iqa_composite": rng.uniform(0, 1, n),
        "similarity_knn_mean": rng.uniform(0, 1, n),
    })


def test_split_image_pool_is_disjoint_and_seeded():
    frame = _synthetic_candidate_fixture(300)
    train_a, val_a = split_image_pool(frame, val_fraction=0.2, seed=1)
    assert set(train_a["image_id"]).isdisjoint(set(val_a["image_id"]))
    assert len(train_a) + len(val_a) == len(frame)
    assert abs(len(val_a) / len(frame) - 0.2) < 0.02

    train_b, val_b = split_image_pool(frame, val_fraction=0.2, seed=1)
    assert set(train_a["image_id"]) == set(train_b["image_id"])  # deterministic given the seed


def test_split_image_pool_rejects_invalid_fraction_and_duplicates():
    frame = _synthetic_candidate_fixture(50)
    with pytest.raises(ValueError):
        split_image_pool(frame, val_fraction=0.0, seed=1)
    with pytest.raises(ValueError):
        split_image_pool(frame, val_fraction=1.0, seed=1)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        split_image_pool(duplicated, val_fraction=0.2, seed=1)


def test_build_role_conditioned_subsets_never_mixes_pools():
    # val pool is ~20% of 300 = ~60 images, split across 3 quantile bands (~20 each) by default —
    # sizes must fit within a single band since single_signal designs draw from one band, or
    # _sample now (correctly) raises on an insufficient pool instead of silently undersizing.
    frame = _synthetic_candidate_fixture(300)
    train_frame, val_frame = split_image_pool(frame, val_fraction=0.2, seed=3)
    records = build_role_conditioned_subsets(
        train_frame, val_frame, ["iqa_composite", "similarity_knn_mean"],
        train_total=6, val_total=4, sizes=[10, 15], random_fraction=0.5,
        single_fraction=0.5, mixed_fraction=0.0, seed=7,
    )
    train_ids = {image_id for r in records if r["role"] == "train" for image_id in r["image_ids"]}
    val_ids = {image_id for r in records if r["role"] == "val" for image_id in r["image_ids"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.issubset(set(train_frame["image_id"]))
    assert val_ids.issubset(set(val_frame["image_id"]))
    assert sum(1 for r in records if r["role"] == "train") == 6
    assert sum(1 for r in records if r["role"] == "val") == 4
    assert all(r["subset_id"].startswith(r["role"] + "_") for r in records)


_DEFAULT_THRESHOLDS = {
    "min_candidates_per_class": 20,
    "min_candidates_per_quantile_band": 20,
    "min_val_subsets": 10,
}


def test_pool_feasibility_report_flags_under_sized_val_pool_and_fails_closed():
    frame = _synthetic_candidate_fixture(100)
    intended = {row.image_id: {"Edema": 1} for _, row in frame.iloc[:30].iterrows()}
    report = pool_feasibility_report(
        frame, ["iqa_composite", "similarity_knn_mean"], intended, ["Edema", "Fracture"],
        subset_sizes=[50, 500], quantile_bins=3, val_fraction=0.2, seed=5,
        total_subsets=20, feasibility_thresholds=_DEFAULT_THRESHOLDS,
    )
    assert report["total_candidates"] == 100
    assert report["train_pool_size"] + report["val_pool_size"] == 100
    assert report["candidates_per_label"]["train"]["Edema"] + report["candidates_per_label"]["val"]["Edema"] == 30
    assert report["candidates_per_label"]["train"]["Fracture"] == 0
    assert report["val_pool_fraction_status"] == "provisional_default"
    assert report["train_val_overlap_count"] == 0

    # val pool is ~20 images: requesting a 500-image subset is NOT achievable without replacement,
    # and _sample never uses replacement — so this must surface as a hard failure, not a silent
    # smaller subset.
    val_500 = report["subset_size_achievability"]["val"]["500"]
    assert val_500["achievable_without_replacement"] is False
    assert val_500["achievable_size_if_smaller"] == report["val_pool_size"]
    # a 50-image subset from the ~80-image train pool IS achievable without replacement
    train_50 = report["subset_size_achievability"]["train"]["50"]
    assert train_50["achievable_without_replacement"] is True

    assert report["passed"] is False
    assert any("actual subset size below required" in reason for reason in report["failures"])
    # Fracture has zero candidates anywhere: must trip the min_candidates_per_class failure too.
    assert any("Fracture" in reason and "insufficient candidates per class" in reason
              for reason in report["failures"])


def test_production_scale_pool_supports_revised_subset_sizes_in_each_val_quantile_band():
    """~5k generated images leave ~1k validation images, or ~333 per quantile band.

    The revised largest subset (250) must therefore be feasible without replacement;
    the former 1000-image request was not.
    """
    frame = _synthetic_candidate_fixture(5000)
    intended = {str(image_id): {"Edema": 1} for image_id in frame["image_id"]}
    report = pool_feasibility_report(
        frame, ["iqa_composite", "similarity_knn_mean"], intended, ["Edema"],
        subset_sizes=[100, 200, 250], quantile_bins=3, val_fraction=0.2, seed=42,
        total_subsets=120, feasibility_thresholds=_DEFAULT_THRESHOLDS,
    )
    assert report["passed"] is True
    assert all("actual subset size below required" not in failure for failure in report["failures"])


def test_evaluate_subset_design_feasibility_flags_overlap_as_non_negotiable():
    report = {
        "train_val_overlap_count": 3,
        "candidates_per_label": {"train": {}, "val": {}},
        "candidates_per_quantile_band": {"train": {}, "val": {}},
        "subset_size_achievability": {"train": {}, "val": {}},
        "quantile_band_size_achievability": {"train": {}, "val": {}},
        "planned_val_subsets": 50,
    }
    failures = evaluate_subset_design_feasibility(report, _DEFAULT_THRESHOLDS)
    assert any("overlap" in reason for reason in failures)


def test_evaluate_subset_design_feasibility_passes_clean_report():
    report = {
        "train_val_overlap_count": 0,
        "candidates_per_label": {"train": {"Edema": 100}, "val": {"Edema": 30}},
        "candidates_per_quantile_band": {"train": {"x": {0: 40, 1: 40}}, "val": {"x": {0: 25, 1: 25}}},
        "subset_size_achievability": {
            "train": {"50": {"requested_size": 50, "pool_size": 200,
                             "achievable_without_replacement": True, "achievable_size_if_smaller": 50}},
            "val": {"50": {"requested_size": 50, "pool_size": 60,
                          "achievable_without_replacement": True, "achievable_size_if_smaller": 50}},
        },
        "quantile_band_size_achievability": {"train": {}, "val": {}},
        "planned_val_subsets": 20,
    }
    assert evaluate_subset_design_feasibility(report, _DEFAULT_THRESHOLDS) == []


def test_sample_raises_on_insufficient_pool_instead_of_silently_undersizing():
    """A single quantile band can be much smaller than the whole pool. single_signal/mixed designs
    drawing size=50 from a 5-image band must raise a clear invariant violation, never silently
    return a smaller subset (that must be caught by --phase feasibility before this is ever hit)."""
    small_frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(5)],
                                "a": np.linspace(0, 1, 5), "b": np.linspace(1, 0, 5)})
    # single_fraction=1.0 forces every design to draw from a single (small) quantile band.
    with pytest.raises(ValueError, match="Insufficient pool"):
        build_controlled_subsets(small_frame, ["a", "b"], 5, [50], 0.0, 1.0, 0.0, seed=1, quantile_bins=5)


def test_sample_mixed_design_backfills_partial_per_column_draws_but_still_enforces_total():
    """The 'mixed' design intentionally lets individual columns undershoot their quota (allow_partial)
    because it backfills from the whole frame afterward — this must still succeed when the frame as
    a whole has enough images, even though any single quantile band does not."""
    frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(300)],
                          "a": np.linspace(0, 1, 300), "b": np.linspace(1, 0, 300)})
    records = build_controlled_subsets(frame, ["a", "b"], 6, [80], 0.0, 0.0, 1.0, seed=2, quantile_bins=10)
    for record in records:
        assert record["design"]["type"] == "mixed"
        assert len(set(record["image_ids"])) == record["size"] == 80


def test_verify_built_subsets_detects_duplicate_images_within_a_subset():
    records = [{"subset_id": "train_0", "role": "train", "size": 2, "image_ids": ["i1", "i1"]}]
    with pytest.raises(SystemExit, match="duplicate image_id"):
        verify_built_subsets(records, train_total=1, val_total=0)


def test_verify_built_subsets_detects_size_mismatch():
    records = [{"subset_id": "train_0", "role": "train", "size": 5, "image_ids": ["i1", "i2"]}]
    with pytest.raises(SystemExit, match="declares size=5"):
        verify_built_subsets(records, train_total=1, val_total=0)


def test_verify_built_subsets_detects_train_val_overlap():
    records = [
        {"subset_id": "train_0", "role": "train", "size": 1, "image_ids": ["shared"]},
        {"subset_id": "val_0", "role": "val", "size": 1, "image_ids": ["shared"]},
    ]
    with pytest.raises(SystemExit, match="both a train-role and a val-role"):
        verify_built_subsets(records, train_total=1, val_total=1)


def test_verify_built_subsets_detects_wrong_subset_counts():
    records = [{"subset_id": "train_0", "role": "train", "size": 1, "image_ids": ["i1"]}]
    with pytest.raises(SystemExit, match="planned 2/1"):
        verify_built_subsets(records, train_total=2, val_total=1)


def test_verify_built_subsets_passes_clean_records():
    records = [
        {"subset_id": "train_0", "role": "train", "size": 2, "image_ids": ["i1", "i2"]},
        {"subset_id": "val_0", "role": "val", "size": 1, "image_ids": ["i3"]},
    ]
    verify_built_subsets(records, train_total=1, val_total=1)  # must not raise


def test_04_preflight_reports_missing_artifacts_not_a_raw_traceback():
    """A missing gonogo_report.json (or score/manifest artifacts) must produce an actionable
    SystemExit listing exactly what to run, never a bare FileNotFoundError traceback."""
    module = _load_script("04_build_utility_subsets.py")
    from scripts.utils.artifact_contracts import stage3_paths
    from scripts.utils.config import load_named_config

    cfg = load_named_config("stage3_asism.yaml", "stage3")
    fake_namespace = "unit-test-namespace-that-will-never-exist"
    for key, value in stage3_paths(cfg, fake_namespace).items():
        if key in cfg.paths:
            cfg.paths[key] = str(value)
    cfg.split_namespace = fake_namespace

    with pytest.raises(SystemExit) as excinfo:
        module.preflight_check_candidate_pool_inputs(cfg)
    message = str(excinfo.value)
    assert "UPSTREAM GATE" in message
    assert "01_compute_signals.py" in message
    assert "02_gonogo.py" in message
    assert "02_generate_synthetic_images.py" in message


def test_split_subsets_by_role_requires_role_field_and_splits_correctly():
    module = _load_script("05_train_learned_asism.py")
    subsets = [
        {"subset_id": "a", "image_ids": ["i1"], "role": "train"},
        {"subset_id": "b", "image_ids": ["i2"], "role": "val"},
        {"subset_id": "c", "image_ids": ["i3"], "role": "train"},
    ]
    train, val = module.split_subsets_by_role(subsets)
    assert [row["subset_id"] for row in train] == ["a", "c"]
    assert [row["subset_id"] for row in val] == ["b"]

    with pytest.raises(SystemExit):
        module.split_subsets_by_role([{"subset_id": "x", "image_ids": ["i1"]}])


def test_image_overlap_fraction():
    module = _load_script("05_train_learned_asism.py")
    disjoint_a = [{"image_ids": ["i1", "i2"]}]
    disjoint_b = [{"image_ids": ["i3", "i4"]}]
    assert module.image_overlap_fraction(disjoint_a, disjoint_b) == 0.0

    overlapping_a = [{"image_ids": ["i1", "i2"]}]
    overlapping_b = [{"image_ids": ["i2", "i3"]}]
    assert module.image_overlap_fraction(overlapping_a, overlapping_b) == pytest.approx(1 / 3)


def test_pairwise_ranking_accuracy_perfect_and_none_cases():
    module = _load_script("05_train_learned_asism.py")
    torch.manual_seed(0)

    class PerfectRanker(torch.nn.Module):
        def forward(self, x):
            return x[:, 0]

    features = {"a": np.array([0.1], dtype=np.float32), "b": np.array([0.5], dtype=np.float32),
               "c": np.array([0.9], dtype=np.float32)}
    targets = {"a": 0.0, "b": 1.0, "c": 2.0}  # perfectly correlated with feature order
    accuracy = module.pairwise_ranking_accuracy(PerfectRanker(), features, targets, torch.device("cpu"))
    assert accuracy == pytest.approx(1.0)

    assert module.pairwise_ranking_accuracy(PerfectRanker(), features, {"a": 1.0}, torch.device("cpu")) is None
    assert module.pairwise_ranking_accuracy(PerfectRanker(), features, {"a": 1.0, "b": 1.0}, torch.device("cpu")) is None


def test_train_set_model_reports_zero_overlap_and_validation_metrics_on_disjoint_fixture():
    module = _load_script("05_train_learned_asism.py")
    from scripts.asism.models import SetUtilityNetwork

    rng = np.random.default_rng(0)
    lookup = {f"img{i}": rng.normal(size=3).astype(np.float32) for i in range(40)}
    train_ids = [f"img{i}" for i in range(30)]
    val_ids = [f"img{i}" for i in range(30, 40)]

    def make_subsets(ids, prefix, n):
        rows = []
        for i in range(n):
            chosen = list(rng.choice(ids, size=6, replace=False))
            rows.append({"subset_id": f"{prefix}_{i}", "image_ids": chosen, "role": prefix})
        return rows

    train_subsets = make_subsets(train_ids, "train", 8)
    val_subsets = make_subsets(val_ids, "val", 4)
    utility_by_id = {row["subset_id"]: float(rng.uniform(-0.1, 0.1)) for row in train_subsets + val_subsets}

    config = OmegaConf.create({
        "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 5, "batch_size": 4,
        "utility_mse_weight": 1.0, "pairwise_ranking_weight": 0.0, "early_stopping_patience": 2,
    })
    torch.manual_seed(0)
    model = SetUtilityNetwork(3, (8, 4), (3,))
    validation = module.train_set_model(model, train_subsets, val_subsets, utility_by_id, lookup, config,
                                        torch.device("cpu"), min_val_subsets=4)
    assert validation["image_overlap_fraction"] == 0.0
    assert validation["n_train_subsets"] == 8
    assert validation["n_val_subsets"] == 4
    assert validation["mae"] is not None
    assert validation["spearman"] is not None
    assert validation["spearman_undefined_reason"] is None
    assert validation["early_stopped_at_epoch"] <= 5


def test_train_set_model_refuses_to_train_below_min_val_subsets():
    """Image-disjoint validation is a design invariant: too few val-role subsets must stop training
    with a clear message, never silently skip validation and keep going."""
    module = _load_script("05_train_learned_asism.py")
    from scripts.asism.models import SetUtilityNetwork

    rng = np.random.default_rng(1)
    lookup = {f"img{i}": rng.normal(size=3).astype(np.float32) for i in range(20)}
    train_subsets = [{"subset_id": f"train_{i}", "image_ids": [f"img{i}", f"img{i+1}"], "role": "train"}
                     for i in range(10)]
    val_subsets = [{"subset_id": "val_0", "image_ids": ["img18", "img19"], "role": "val"}]  # only 1
    utility_by_id = {row["subset_id"]: 0.01 for row in train_subsets + val_subsets}
    config = OmegaConf.create({
        "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 2, "batch_size": 4,
        "utility_mse_weight": 1.0, "pairwise_ranking_weight": 0.0, "early_stopping_patience": 2,
    })
    model = SetUtilityNetwork(3, (8, 4), (3,))
    with pytest.raises(SystemExit, match="val-role subsets"):
        module.train_set_model(model, train_subsets, val_subsets, utility_by_id, lookup, config,
                               torch.device("cpu"), min_val_subsets=4)


def test_train_set_model_refuses_to_train_with_zero_val_subsets():
    module = _load_script("05_train_learned_asism.py")
    from scripts.asism.models import SetUtilityNetwork

    train_subsets = [{"subset_id": "train_0", "image_ids": ["img0", "img1"], "role": "train"}]
    utility_by_id = {"train_0": 0.01}
    config = OmegaConf.create({
        "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 2, "batch_size": 4,
        "utility_mse_weight": 1.0, "pairwise_ranking_weight": 0.0, "early_stopping_patience": 2,
    })
    model = SetUtilityNetwork(3, (4,), (3,))
    with pytest.raises(SystemExit, match="val-role subsets"):
        module.train_set_model(model, train_subsets, [], utility_by_id,
                               {"img0": np.zeros(3, dtype=np.float32), "img1": np.zeros(3, dtype=np.float32)},
                               config, torch.device("cpu"), min_val_subsets=4)


def test_train_set_model_reports_undefined_spearman_as_none_with_reason():
    """Constant predictions/targets make Spearman correlation undefined (scipy returns NaN). NaN is
    not valid JSON and must never reach the frozen manifest — this must surface as None + an
    explicit machine-readable reason."""
    module = _load_script("05_train_learned_asism.py")
    from scripts.asism.models import SetUtilityNetwork

    rng = np.random.default_rng(2)
    lookup = {f"img{i}": rng.normal(size=3).astype(np.float32) for i in range(30)}
    train_ids = [f"img{i}" for i in range(20)]
    val_ids = [f"img{i}" for i in range(20, 30)]

    def make_subsets(ids, prefix, n):
        return [{"subset_id": f"{prefix}_{i}", "image_ids": list(rng.choice(ids, size=4, replace=False)),
                "role": prefix} for i in range(n)]

    train_subsets = make_subsets(train_ids, "train", 6)
    val_subsets = make_subsets(val_ids, "val", 4)
    # Constant measured utility for every val subset -> targets have zero variance -> Spearman is
    # mathematically undefined regardless of what the model predicts.
    utility_by_id = {row["subset_id"]: 0.0123 for row in train_subsets + val_subsets}
    config = OmegaConf.create({
        "learning_rate": 1e-3, "weight_decay": 0.0, "epochs": 3, "batch_size": 4,
        "utility_mse_weight": 1.0, "pairwise_ranking_weight": 0.0, "early_stopping_patience": 2,
    })
    torch.manual_seed(0)
    model = SetUtilityNetwork(3, (8, 4), (3,))
    validation = module.train_set_model(model, train_subsets, val_subsets, utility_by_id, lookup, config,
                                        torch.device("cpu"), min_val_subsets=4)
    assert validation["spearman"] is None
    assert validation["spearman_undefined_reason"] == "constant_predictions_or_targets"
    assert validation["mae"] is not None  # MAE is still well-defined even when Spearman isn't


def test_subset_design_config_hash_changes_with_relevant_fields_only():
    module = _load_script("04_build_utility_subsets.py")
    base = {
        "subset_sizes": [250, 500], "total_subsets": 120, "random_fraction": 0.2,
        "single_signal_fraction": 0.3, "mixed_fraction": 0.5, "quantile_bins": 3,
        "val_pool_fraction": 0.2, "feasibility_thresholds": dict(_DEFAULT_THRESHOLDS),
        "seed": 42, "minimum_image_exposures": 3,  # irrelevant fields present but not hashed
    }
    design_a = OmegaConf.create(base)
    design_b = OmegaConf.create(base)
    assert module.subset_design_config_hash(design_a) == module.subset_design_config_hash(design_b)

    changed = dict(base); changed["total_subsets"] = 200
    design_c = OmegaConf.create(changed)
    assert module.subset_design_config_hash(design_a) != module.subset_design_config_hash(design_c)


def test_compute_budget_gate_raises_when_over_budget_and_passes_when_within():
    module = _load_script("04b_evaluate_utility_subsets.py")
    cfg = OmegaConf.create({"learned_asism": {"compute_budget": {
        "max_gpu_hours": 1.0, "hours_per_proxy_run_estimate": 0.15,
    }}})

    over_budget = module.compute_budget_estimate(cfg, n_recipes=100)
    assert over_budget["within_budget"] is False
    with pytest.raises(SystemExit):
        module.enforce_compute_budget(cfg, n_recipes=100)

    within_budget = module.compute_budget_estimate(cfg, n_recipes=2)
    assert within_budget["within_budget"] is True
    module.enforce_compute_budget(cfg, n_recipes=2)  # must not raise


# ---------------------------------------------------------------------------
# Phase 2 — Adaptive Threshold Learning (code + fixtures + unit tests only, no GPU/proxy runs)
# ---------------------------------------------------------------------------

def test_class_aware_context_vector_shape_and_known_slots():
    vector = class_aware_context_vector(
        [0.1, 0.5, 0.9], {"real_prevalence": 0.3, "positive_patients": 50},
        {"target_synthetic_to_real_ratio": 1.5},
    )
    assert len(vector) == 10
    assert vector[7] == pytest.approx(0.3)   # real_prevalence slot
    assert vector[9] == pytest.approx(1.5)   # target_synthetic_to_real_ratio slot


def test_class_aware_context_vector_handles_empty_scores_and_missing_prevalence():
    vector = class_aware_context_vector([], {"real_prevalence": None}, {})
    assert len(vector) == 10
    assert all(np.isfinite(value) for value in vector)


def test_bootstrap_class_contexts_tags_honestly_and_stays_within_one_class_and_pool():
    frame = pd.DataFrame({"image_id": [f"img{i}" for i in range(50)],
                          "learned_ranking_score": np.linspace(0, 1, 50)})
    intended = {f"img{i}": {"Edema": 1} for i in range(30)}
    real_prevalence = {"real_prevalence": 0.2, "positive_patients": 100}
    budget_context = {"target_synthetic_to_real_ratio": 1.0, "min_selected_per_label": 10, "max_selected_per_label": 500}
    contexts = bootstrap_class_contexts(
        frame, class_id=3, label="Edema", intended_by_id=intended, score_column="learned_ranking_score",
        context_source="bootstrap_train_pool", n_replicates=5, min_fraction=0.3, max_fraction=0.8,
        ranking_checkpoint_hash="ckpt-rank", critic_checkpoint_hash="ckpt-critic",
        real_prevalence_context=real_prevalence, budget_context=budget_context, seed=1,
    )
    assert len(contexts) == 5
    edema_positive_ids = {f"img{i}" for i in range(30)}
    for context in contexts:
        assert context["context_source"] == "bootstrap_train_pool"
        assert context["independent_clinical_sample"] is False
        assert context["class_id"] == 3 and context["label"] == "Edema"
        assert set(context["image_ids"]).issubset(edema_positive_ids)
        assert len(set(context["image_ids"])) == len(context["image_ids"])  # no duplicate images
        assert len(context["context_features"]) == 10


def test_bootstrap_class_contexts_rejects_invalid_fraction_range():
    frame = pd.DataFrame({"image_id": ["a"], "score": [0.5]})
    with pytest.raises(ValueError):
        bootstrap_class_contexts(frame, 0, "L", {}, "score", "bootstrap_train_pool", 1, 0.8, 0.3,
                                 "h1", "h2", {}, {}, seed=1)


def test_hard_threshold_grid_search_is_deterministic_and_names_critic_field_explicitly():
    ranking_scores = {"a": 0.9, "b": 0.7, "c": 0.5, "d": 0.3}
    image_ids = list(ranking_scores)

    def critic_fn(selected):
        return len(selected) * 0.1

    t_grid = [0.2, 0.4, 0.6, 0.8]
    result_a = hard_threshold_grid_search(critic_fn, ranking_scores, image_ids, t_grid, 0.1, max_selected=10)
    result_b = hard_threshold_grid_search(critic_fn, ranking_scores, image_ids, t_grid, 0.1, max_selected=10)
    assert result_a == result_b
    assert all("critic_predicted_utility" in entry for entry in result_a)
    assert all("measured_utility" not in entry for entry in result_a)  # never conflated with a measured field


def test_hard_threshold_grid_search_rejects_all_empty_thresholds():
    with pytest.raises(ValueError, match="empty subset"):
        hard_threshold_grid_search(lambda selected: 1.0, {"a": 0.1}, ["a"], [0.99], 0.1, max_selected=10)


def test_hard_threshold_grid_search_respects_max_selected():
    ranking_scores = {f"i{i}": 1.0 for i in range(20)}

    def critic_fn(selected):
        return float(len(selected))

    result = hard_threshold_grid_search(critic_fn, ranking_scores, list(ranking_scores), [0.5], 0.0, max_selected=5)
    assert result[0]["selected_count"] == 5


def test_resolve_verified_only_targets_never_includes_critic_only_contexts():
    """The core separation: a context with zero proxy measurements must be entirely ABSENT from
    the official path's targets — never present with a critic-derived guess at any weight."""
    contexts = [
        {"context_id": "c1", "class_id": 0, "label": "Edema", "context_source": "bootstrap_train_pool"},
        {"context_id": "c2", "class_id": 0, "label": "Edema", "context_source": "bootstrap_train_pool"},
    ]
    measurements = [
        {"context_id": "c1", "candidate_threshold": 0.6, "measured_utility": 0.05},
        {"context_id": "c1", "candidate_threshold": 0.7, "measured_utility": 0.08},
        # c2 has NO measurements at all.
    ]
    targets = resolve_verified_only_targets(contexts, measurements)
    assert len(targets) == 1
    assert targets[0]["context_id"] == "c1"
    assert targets[0]["target_threshold"] == 0.7  # best AMONG VERIFIED, not a claimed global optimum
    assert targets[0]["proxy_best_among_verified_candidates"] == 0.7
    assert targets[0]["n_verified_candidates_in_context"] == 2
    assert targets[0]["target_source"] == "proxy_verified"
    assert "measured_best_threshold" not in targets[0]  # renamed away deliberately


def test_resolve_critic_assisted_exploratory_targets_never_touches_measurements():
    """The exploratory path is a completely separate function signature — it cannot even be called
    with a measurements list, so it structurally cannot leak proxy-verified data into itself, and
    its output must never be fed into official-path training."""
    contexts = [{"context_id": "c1", "class_id": 0, "label": "Edema", "context_source": "bootstrap_train_pool"}]
    evaluations = [
        {"context_id": "c1", "candidate_threshold": 0.6, "critic_predicted_utility": 0.05, "rank_within_context": 1},
        {"context_id": "c1", "candidate_threshold": 0.7, "critic_predicted_utility": 0.03, "rank_within_context": 2},
    ]
    targets = resolve_critic_assisted_exploratory_targets(contexts, evaluations)
    assert targets[0]["target_source"] == "critic_only"
    assert targets[0]["target_threshold"] == 0.6  # critic's own top grid pick
    assert targets[0]["supervision_weight"] == 1.0  # full weight WITHIN the exploratory path only


def test_determine_per_class_official_method_zero_verified_falls_back_to_baseline_not_hard():
    """The bug this guards against: claiming 'hard_class_specific_proxy_verified_thresholds' when
    zero proxy verification ever happened is a contradiction in terms. Zero verified must fall back
    to the honestly-named OLD baseline instead."""
    result = determine_per_class_official_method([], [], ["Edema"], min_verified_contexts_per_class=3,
                                                  acceptance_passed=True)
    assert result["Edema"] == "fixed_target_ratio_threshold_distillation_baseline_v1"


def test_determine_per_class_official_method_some_but_insufficient_verified():
    train_targets = [{"label": "Edema"}]  # only 1, below min_verified_contexts_per_class=3
    result = determine_per_class_official_method(train_targets, [], ["Edema"],
                                                  min_verified_contexts_per_class=3, acceptance_passed=True)
    assert result["Edema"] == "hard_proxy_best_among_verified"


def test_determine_per_class_official_method_requires_both_train_and_held_out_sufficiency():
    """Enough verified TRAIN contexts but insufficient verified HELD-OUT (image-disjoint validation)
    contexts must still fall back — 'enough evidence' means enough on both sides, not just train."""
    train_targets = [{"label": "Edema"}] * 5
    held_out_targets = [{"label": "Edema"}] * 1  # below min=3
    result = determine_per_class_official_method(train_targets, held_out_targets, ["Edema"],
                                                  min_verified_contexts_per_class=3, acceptance_passed=True)
    assert result["Edema"] == "hard_proxy_best_among_verified"


def test_determine_per_class_official_method_network_only_when_sufficient_and_accepted():
    train_targets = [{"label": "Edema"}] * 5
    held_out_targets = [{"label": "Edema"}] * 5
    passed = determine_per_class_official_method(train_targets, held_out_targets, ["Edema"], 3, acceptance_passed=True)
    assert passed["Edema"] == "adaptive_threshold_network"
    failed = determine_per_class_official_method(train_targets, held_out_targets, ["Edema"], 3, acceptance_passed=False)
    assert failed["Edema"] == "hard_proxy_best_among_verified"  # enough data, but acceptance criteria failed


def test_diversify_verification_candidates_includes_more_than_critic_top_k():
    evaluations = [
        {"candidate_threshold": t, "selected_count": count, "rank_within_context": rank}
        for rank, (t, count) in enumerate(
            [(0.9, 5), (0.8, 20), (0.7, 50), (0.5, 100), (0.3, 150), (0.1, 195)], start=1)
    ]
    diversified = diversify_verification_candidates(evaluations, top_k=1, quantile_fractions=[0.5], candidate_pool_size=200)
    reasons = {reason for entry in diversified for reason in entry["selection_reasons"]}
    assert "critic_top_k" in reasons
    assert "quantile_spaced" in reasons
    assert "boundary" in reasons
    assert "baseline_target_ratio" in reasons
    assert len(diversified) > 1  # never just the single critic top-1 pick


def test_diversify_verification_candidates_handles_empty_input():
    assert diversify_verification_candidates([], top_k=2, quantile_fractions=[0.5], candidate_pool_size=100) == []


def test_held_out_generalization_metrics_zero_when_predicted_equals_target():
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    metrics = held_out_generalization_metrics(0.5, 0.5, scores)
    assert metrics["absolute_threshold_error"] == 0.0
    assert metrics["selected_set_jaccard"] == 1.0


def test_held_out_generalization_metrics_detects_divergence():
    scores = {"a": 0.9, "b": 0.5, "c": 0.1}
    metrics = held_out_generalization_metrics(0.05, 0.5, scores)
    assert metrics["absolute_threshold_error"] == pytest.approx(0.45)
    assert metrics["selected_set_jaccard"] == pytest.approx(2 / 3)


def test_freeze_and_enforce_acceptance_criteria_pass_and_fail():
    criteria = {"critic_proxy_median_spearman_min": 0.3, "median_critic_predicted_utility_regret_max": 0.01,
               "threshold_stability_std_max": 0.05}
    frozen = freeze_acceptance_criteria(criteria)
    assert frozen["criteria_status"] == "provisional_pre_registered_defaults"
    assert frozen["criteria_frozen_at"]
    assert frozen["criteria_config_hash"]
    assert "critic_proxy_median_spearman_min" in frozen["criteria_definitions"]

    passing = enforce_acceptance_criteria(
        {"median_spearman": 0.5, "median_critic_predicted_utility_regret": 0.005, "threshold_stability_std": 0.02}, frozen)
    assert passing["passed"] is True
    assert passing["criteria_status"] == "provisional_pre_registered_defaults"

    failing = enforce_acceptance_criteria(
        {"median_spearman": 0.1, "median_critic_predicted_utility_regret": 0.005, "threshold_stability_std": 0.02}, frozen)
    assert failing["passed"] is False
    assert failing["per_criterion"]["critic_proxy_median_spearman_min"]["passed"] is False


def test_enforce_acceptance_criteria_none_evidence_fails_not_crashes():
    frozen = freeze_acceptance_criteria({"critic_proxy_median_spearman_min": 0.3})
    result = enforce_acceptance_criteria({"median_spearman": None}, frozen)
    assert result["passed"] is False


def test_critic_proxy_correlation_per_class_never_pools_as_the_primary_number():
    contexts = [
        {"context_id": "c1", "label": "Edema"}, {"context_id": "c2", "label": "Edema"},
        {"context_id": "c3", "label": "Edema"},
        {"context_id": "c4", "label": "Fracture"}, {"context_id": "c5", "label": "Fracture"},
    ]
    evaluations = [
        {"context_id": cid, "candidate_threshold": 0.5, "critic_predicted_utility": value}
        for cid, value in [("c1", 0.1), ("c2", 0.2), ("c3", 0.3), ("c4", 0.9), ("c5", 0.1)]
    ]
    measurements = [
        {"context_id": cid, "candidate_threshold": 0.5, "measured_utility": value}
        for cid, value in [("c1", 0.05), ("c2", 0.15), ("c3", 0.25), ("c4", 0.05), ("c5", 0.5)]
    ]
    result = critic_proxy_correlation_per_class(evaluations, measurements, contexts)
    assert set(result["per_class_spearman"]) <= {"Edema", "Fracture"}
    assert "pooled_spearman_diagnostic_only" in result
    assert result["median_spearman"] is not None or result["macro_spearman"] is not None


def test_critic_proxy_correlation_per_class_needs_two_points_per_class():
    contexts = [{"context_id": "c1", "label": "Edema"}]
    evaluations = [{"context_id": "c1", "candidate_threshold": 0.5, "critic_predicted_utility": 0.1}]
    measurements = [{"context_id": "c1", "candidate_threshold": 0.5, "measured_utility": 0.05}]
    result = critic_proxy_correlation_per_class(evaluations, measurements, contexts)
    assert result["per_class_spearman"]["Edema"] is None
    assert result["n_verified_by_class"]["Edema"] == 1


def test_train_threshold_network_zero_supervision_weight_means_no_learning():
    module = _load_script("08_train_threshold_network.py")
    torch.manual_seed(0)
    targets = [{"class_id": 0, "context_features": [0.1] * 4, "target_threshold": 0.9, "supervision_weight": 0.0}]
    model = AdaptiveThresholdNetwork(2, 4, embedding_dim=2, hidden=(4,))
    before = module.predict_thresholds(model, targets, torch.device("cpu"))
    module.train_threshold_network(model, targets, torch.device("cpu"), epochs=20, learning_rate=0.1)
    after = module.predict_thresholds(model, targets, torch.device("cpu"))
    assert before == pytest.approx(after)  # zero supervision_weight -> zero gradient -> no parameter change


def test_train_threshold_network_positive_weight_moves_prediction_toward_target():
    module = _load_script("08_train_threshold_network.py")
    torch.manual_seed(0)
    targets = [{"class_id": 0, "context_features": [0.1] * 4, "target_threshold": 0.9, "supervision_weight": 1.0}]
    model = AdaptiveThresholdNetwork(2, 4, embedding_dim=2, hidden=(4,))
    before = module.predict_thresholds(model, targets, torch.device("cpu"))[0]
    module.train_threshold_network(model, targets, torch.device("cpu"), epochs=200, learning_rate=0.05)
    after = module.predict_thresholds(model, targets, torch.device("cpu"))[0]
    assert abs(after - 0.9) < abs(before - 0.9)


def test_train_threshold_network_raises_on_empty_targets():
    module = _load_script("08_train_threshold_network.py")
    model = AdaptiveThresholdNetwork(2, 4)
    with pytest.raises(ValueError):
        module.train_threshold_network(model, [], torch.device("cpu"), epochs=5, learning_rate=0.01)


def test_threshold_stability_computes_per_class_std_excluding_singletons():
    module = _load_script("08_train_threshold_network.py")
    targets = [{"class_id": 0}, {"class_id": 0}, {"class_id": 1}]
    predictions = [0.5, 0.7, 0.3]
    result = module.threshold_stability(targets, predictions)
    assert 0 in result["per_class_std"]
    assert 1 not in result["per_class_std"]  # only one context for class 1 -> std undefined, excluded


def test_07_writes_frozen_artifacts_with_exclusive_create_mode():
    """threshold_contexts.jsonl / threshold_candidate_evaluations.jsonl / proxy_verification_plan.json
    must be write-once (07's own critic-only artifacts) — enforced at the open() call, not by
    convention."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "asism" / "07_build_threshold_contexts.py").read_text(encoding="utf-8")
    assert 'open(Path(cfg.paths.threshold_contexts), "x"' in source
    assert 'open(Path(cfg.paths.threshold_candidate_evaluations), "x"' in source
    assert 'open(Path(cfg.paths.proxy_verification_plan), "x"' in source


def test_07b_never_opens_threshold_candidate_evaluations_for_writing():
    """07b must never mutate 07's frozen critic-only artifact — it writes measurements to a
    completely separate, resumable (append) file instead."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "asism" / "07b_verify_thresholds_proxy.py").read_text(encoding="utf-8")
    assert 'open(Path(cfg.paths.threshold_candidate_evaluations)' not in source
    assert 'open(measurements_path, "a"' in source


def test_verification_compute_estimate_is_per_context_only_not_full_policy():
    """07's budget must NOT include full-policy runs — that has its own independent budget in 08b."""
    module = _load_script("07_build_threshold_contexts.py")
    cfg = OmegaConf.create({"learned_asism": {"threshold_network": {
        "verification_compute_budget": {"max_gpu_hours": 100.0, "hours_per_proxy_run_estimate": 0.1},
    }}})
    estimate = module.verification_compute_estimate(cfg, n_planned=10)
    assert "n_planned_full_policy_runs" not in estimate
    assert "total_planned_runs" not in estimate
    assert estimate["n_planned_threshold_evaluations"] == 10
    assert estimate["estimated_gpu_hours"] == pytest.approx(1.0)


def test_plan_full_policy_verification_is_plan_only_with_independent_budget():
    module = _load_script("08_train_threshold_network.py")
    full_policy_cfg = OmegaConf.create({
        "n_seeds": 3,
        "compute_budget": {"max_gpu_hours": 10.0, "hours_per_proxy_run_estimate": 0.15},
    })
    per_class_official_method = {"Edema": "adaptive_threshold_network", "Fracture": "hard_proxy_best_among_verified"}
    plan = module.plan_full_policy_verification(per_class_official_method, full_policy_cfg)
    assert "fixed_target_ratio_threshold_distillation_baseline_v1" in plan["policy_variants"]
    assert "literal_top_50_percent" in plan["policy_variants"]
    assert "adaptive_threshold_network" in plan["policy_variants"]
    assert plan["status"] == "plan_only_not_executed"
    assert plan["estimated_runs"] == len(plan["policy_variants"]) * 3
    assert plan["estimated_gpu_hours"] == pytest.approx(plan["estimated_runs"] * 0.15)
    assert "08b_verify_full_policy_proxy.py" in plan["next_step"]


def test_08b_compute_budget_is_independent_of_07b():
    """08b's estimate must use full_policy_verification.compute_budget, a config block entirely
    separate from 07b's verification_compute_budget."""
    module = _load_script("08b_verify_full_policy_proxy.py")
    cfg = OmegaConf.create({"learned_asism": {"threshold_network": {"full_policy_verification": {
        "compute_budget": {"max_gpu_hours": 1.0, "hours_per_proxy_run_estimate": 0.1},
    }}}})
    over_budget = module.compute_budget_estimate(cfg, n_variants=5, n_seeds=3)
    assert over_budget["within_budget"] is False
    with pytest.raises(SystemExit):
        module.enforce_compute_budget(cfg, n_variants=5, n_seeds=3)

    within_budget = module.compute_budget_estimate(cfg, n_variants=1, n_seeds=1)
    assert within_budget["within_budget"] is True
    module.enforce_compute_budget(cfg, n_variants=1, n_seeds=1)  # must not raise


def test_08b_refuses_real_training_without_confirmation_flag(monkeypatch):
    module = _load_script("08b_verify_full_policy_proxy.py")
    monkeypatch.setattr(sys, "argv", ["08b_verify_full_policy_proxy.py", "--phase", "run"])
    with pytest.raises(SystemExit, match="REFUSED"):
        module.main()


def test_08b_phase_estimate_does_not_require_confirmation_flag(monkeypatch):
    module = _load_script("08b_verify_full_policy_proxy.py")
    monkeypatch.setattr(sys, "argv", ["08b_verify_full_policy_proxy.py", "--phase", "estimate",
                                      "--namespace", "unit-test-namespace-that-will-never-exist"])
    with pytest.raises(SystemExit, match="UPSTREAM GATE"):
        module.main()


def test_08b_phase_run_with_confirmation_flag_reaches_the_next_real_gate(monkeypatch):
    module = _load_script("08b_verify_full_policy_proxy.py")
    monkeypatch.setattr(sys, "argv", [
        "08b_verify_full_policy_proxy.py", "--phase", "run", "--i-understand-this-trains-real-models",
        "--namespace", "unit-test-namespace-that-will-never-exist",
    ])
    with pytest.raises(SystemExit, match="UPSTREAM GATE"):
        module.main()


def test_compute_verified_context_counts_and_eligibility():
    module_targets = [
        {"label": "Edema"}, {"label": "Edema"}, {"label": "Edema"},
        {"label": "Fracture"},
    ]
    held_out_targets = [
        {"label": "Edema"}, {"label": "Edema"}, {"label": "Edema"},
        {"label": "Fracture"}, {"label": "Fracture"}, {"label": "Fracture"},
    ]
    train_counts, held_out_counts = compute_verified_context_counts(module_targets, held_out_targets, ["Edema", "Fracture"])
    assert train_counts == {"Edema": 3, "Fracture": 1}
    assert held_out_counts == {"Edema": 3, "Fracture": 3}
    eligible = eligible_classes_for_official_training(train_counts, held_out_counts, min_verified_contexts_per_class=3)
    assert eligible == {"Edema"}  # Fracture fails on train_count even though held_out is sufficient


def test_aggregate_hard_proxy_best_threshold_uses_train_only_never_held_out():
    verified_train = [
        {"label": "Edema", "target_threshold": 0.5, "context_source": "bootstrap_train_pool"},
        {"label": "Edema", "target_threshold": 0.7, "context_source": "bootstrap_train_pool"},
        {"label": "Edema", "target_threshold": 0.6, "context_source": "bootstrap_train_pool"},
    ]
    result = aggregate_hard_proxy_best_threshold(verified_train, "Edema")
    assert result == pytest.approx(0.6)  # median
    assert aggregate_hard_proxy_best_threshold(verified_train, "Fracture") is None


def test_determine_per_class_official_method_zero_train_falls_back_even_with_held_out_data():
    """The exact bug this guards against: held-out measurements existing must NEVER cause anything
    other than the fixed-ratio baseline when there is zero verified TRAIN evidence."""
    held_out_only = [{"label": "Edema"}] * 5  # plenty of held-out, but...
    result = determine_per_class_official_method([], held_out_only, ["Edema"],
                                                  min_verified_contexts_per_class=3, acceptance_passed=True)
    assert result["Edema"] == "fixed_target_ratio_threshold_distillation_baseline_v1"


def test_compute_official_acceptance_evidence_uses_utility_regret_not_threshold_error():
    module = _load_script("08_train_threshold_network.py")
    held_out_generalization = [
        {"absolute_threshold_error": 0.9, "critic_predicted_utility_regret": 0.01},
        {"absolute_threshold_error": 0.8, "critic_predicted_utility_regret": 0.03},
    ]
    correlation = {"median_spearman": 0.5}
    stability = {"median_std": 0.02}
    evidence = module.compute_official_acceptance_evidence(held_out_generalization, correlation, stability)
    assert evidence["median_critic_predicted_utility_regret"] == pytest.approx(0.02)


def test_compute_official_acceptance_evidence_ignores_infinite_regret():
    module = _load_script("08_train_threshold_network.py")
    held_out_generalization = [
        {"critic_predicted_utility_regret": float("inf")},
        {"critic_predicted_utility_regret": 0.02},
    ]
    evidence = module.compute_official_acceptance_evidence(held_out_generalization, {"median_spearman": None}, {"median_std": None})
    assert evidence["median_critic_predicted_utility_regret"] == pytest.approx(0.02)


def test_save_checkpoint_writes_file_and_returns_hash():
    module = _load_script("08_train_threshold_network.py")
    model = AdaptiveThresholdNetwork(2, 4, embedding_dim=2, hidden=(4,))
    tn_cfg = OmegaConf.create({"class_embedding_dim": 2, "hidden_dims": [4], "dropout": 0.1})
    path = Path(__file__).with_name("checkpoint_test_artifact.pt")
    try:
        checkpoint_hash = module.save_checkpoint(model, path, "verified_only_official", tn_cfg, context_dim=4)
        assert path.is_file()
        assert len(checkpoint_hash) == 64
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert loaded["path"] == "verified_only_official"
        assert loaded["context_dim"] == 4
    finally:
        path.unlink(missing_ok=True)


def test_official_targets_are_filtered_to_eligible_classes_only():
    targets = [{"label": "Edema", "context_id": "e"}, {"label": "Fracture", "context_id": "f"}]
    assert filter_targets_to_eligible_classes(targets, {"Edema"}) == [targets[0]]


def test_full_policy_choice_requires_complete_results_and_uses_frozen_tie_order():
    rows = [
        {"policy": "adaptive", "seed": 42, "measured_utility": 0.020},
        {"policy": "adaptive", "seed": 43, "measured_utility": 0.020},
        {"policy": "baseline", "seed": 42, "measured_utility": 0.019},
        {"policy": "baseline", "seed": 43, "measured_utility": 0.019},
    ]
    decision = choose_full_policy(rows, ["adaptive", "baseline"], [42, 43], 0.002, ["baseline", "adaptive"])
    assert decision["winning_policy"] == "baseline"
    with pytest.raises(ValueError, match="incomplete"):
        choose_full_policy(rows[:-1], ["adaptive", "baseline"], [42, 43], 0.002, ["baseline", "adaptive"])


def test_08b_requires_frozen_prevalence_context_and_rejects_empty_policy():
    module = _load_script("08b_verify_full_policy_proxy.py")
    contexts = [
        {"label": "Edema", "real_prevalence_context": {"real_prevalence": 0.2, "positive_patients": 10}},
        {"label": "Edema", "real_prevalence_context": {"real_prevalence": 0.2, "positive_patients": 10}},
    ]
    result = module.real_prevalence_contexts_from_threshold_contexts(contexts, ["Edema"])
    assert result["Edema"]["real_prevalence"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="missing frozen"):
        module.real_prevalence_contexts_from_threshold_contexts([], ["Edema"])
    with pytest.raises(ValueError, match="zero synthetic"):
        module.validate_policy_selection("adaptive", {"Edema": 0.5}, [], ["Edema"])


def test_stage4_routes_the_thesis_selected_synthetic_condition_to_learned_asism():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "classify" / "01_train_conditions.py").read_text(
        encoding="utf-8"
    )
    assert 'CONDITIONS = ["A", "B", "F"]' in source
    assert 'condition == "F"' in source and 'selector="adaptive"' in source
    assert 'conditions: [A, B, F]' in (Path(__file__).resolve().parents[1] / "configs" / "stage4_classifier.yaml").read_text(encoding="utf-8")


def test_finalizer_never_reads_final_eval_and_publishes_new_adaptive_artifacts():
    source = (Path(__file__).resolve().parents[1] / "scripts" / "asism" / "09_finalize_learned_selection.py").read_text(
        encoding="utf-8"
    )
    assert "load_split(" not in source
    assert "cfg.paths.adaptive_selected_manifest" in source
    assert "cfg.paths.adaptive_selection_manifest" in source
    assert "cfg.paths.learned_selected_manifest" in source  # read-only baseline input


def test_08_writes_two_separate_target_files_not_one_combined_file():
    """Verified and exploratory targets must be written to physically separate files — a loader
    that opens the wrong one gets a clean 'file not found', not silently-wrong rows from a shared
    file it forgot to filter."""
    source = (Path(__file__).resolve().parents[1] / "scripts" / "asism" / "08_train_threshold_network.py").read_text(encoding="utf-8")
    assert 'open(Path(cfg.paths.threshold_training_targets_verified), "x"' in source
    assert 'open(Path(cfg.paths.threshold_training_targets_exploratory), "x"' in source
    assert "cfg.paths.threshold_training_targets)" not in source  # the old single combined path is gone


def test_percentile_threshold_per_class_computes_real_values():
    module = _load_script("08b_verify_full_policy_proxy.py")
    ranking_scores = {"a": 0.1, "b": 0.5, "c": 0.9}
    intended = {"a": {"Edema": 1}, "b": {"Edema": 1}, "c": {"Edema": 1}}
    result = module.percentile_threshold_per_class(50.0, ranking_scores, intended, ["a", "b", "c"], ["Edema"])
    assert result["Edema"] == pytest.approx(0.5)


def test_07b_refuses_real_training_without_confirmation_flag(monkeypatch):
    """The core fail-closed guard: --phase run must refuse BEFORE any config/data/model I/O."""
    module = _load_script("07b_verify_thresholds_proxy.py")
    monkeypatch.setattr(sys, "argv", ["07b_verify_thresholds_proxy.py", "--phase", "run"])
    with pytest.raises(SystemExit, match="REFUSED"):
        module.main()


def test_07b_phase_estimate_does_not_require_confirmation_flag(monkeypatch):
    """--phase estimate never trains anything, so it must reach the (missing-plan) upstream gate
    rather than the REFUSED confirmation-flag message."""
    module = _load_script("07b_verify_thresholds_proxy.py")
    monkeypatch.setattr(sys, "argv", ["07b_verify_thresholds_proxy.py", "--phase", "estimate",
                                      "--namespace", "unit-test-namespace-that-will-never-exist"])
    with pytest.raises(SystemExit, match="UPSTREAM GATE"):
        module.main()


def test_07b_phase_run_with_confirmation_flag_reaches_the_next_real_gate(monkeypatch):
    """With the flag passed, the guard must NOT block — it should proceed to the next real check
    (the missing verification plan), never straight into training."""
    module = _load_script("07b_verify_thresholds_proxy.py")
    monkeypatch.setattr(sys, "argv", [
        "07b_verify_thresholds_proxy.py", "--phase", "run", "--i-understand-this-trains-real-models",
        "--namespace", "unit-test-namespace-that-will-never-exist",
    ])
    with pytest.raises(SystemExit, match="UPSTREAM GATE"):
        module.main()


def test_marginal_targets_are_size_normalized_across_subset_sizes():
    r"""The same image, with the same deviation from its subset mean, must get the same marginal
    target whether it sat in a small subset or a large one.

    Regression test for the 1/(n-1) size bias: SetUtilityNetwork pools by MEAN, so the raw
    leave-one-out difference U(S)-U(S\i) scales like 1/(n-1). Averaging raw differences across
    subsets of different sizes therefore encodes "which subset sizes did this image land in" into
    the ranking network's supervision. Without the (n-1) normalization in marginal_targets, the
    small-subset target here is ~(249/99)x the large-subset one and this test fails.
    """
    module = _load_script("05_train_learned_asism.py")
    device = torch.device("cpu")

    class LinearMeanPool(torch.nn.Module):
        """Exactly the pooling contract of SetUtilityNetwork, with an identity head so the
        expected value is analytic rather than an artefact of a random initialization."""

        def forward(self, features, mask):
            weights = mask.unsqueeze(-1).to(features.dtype)
            pooled = (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            return pooled[:, 0]

    model = LinearMeanPool()
    probe = "probe"
    lookup = {probe: np.array([1.0], dtype=np.float32)}
    # Peers all sit at 0.0, so the probe's deviation from the subset mean is identical in both
    # subsets; only the subset SIZE differs.
    small_ids, large_ids = [probe], [probe]
    for index in range(99):
        key = f"small_{index}"
        lookup[key] = np.array([0.0], dtype=np.float32)
        small_ids.append(key)
    for index in range(249):
        key = f"large_{index}"
        lookup[key] = np.array([0.0], dtype=np.float32)
        large_ids.append(key)

    small_only, _ = module.marginal_targets(
        model, [{"image_ids": small_ids}], lookup, device
    )
    large_only, _ = module.marginal_targets(
        model, [{"image_ids": large_ids}], lookup, device
    )

    # Analytic values. Raw U(S)-U(S\i) here is exactly 1/n, so after multiplying by (n-1) the
    # target is (n-1)/n: 0.990 at n=100 and 0.996 at n=250.
    assert small_only[probe] == pytest.approx(99 / 100, rel=1e-6)
    assert large_only[probe] == pytest.approx(249 / 250, rel=1e-6)

    # The point of the fix: the two sizes must now agree to well under 1%. Without the (n-1)
    # factor the raw targets are 0.01 vs 0.004 — a 2.5x size bias — and this assertion fails.
    assert small_only[probe] == pytest.approx(large_only[probe], rel=0.01)

    # Averaging across both subsets must land between them, not be dominated by the smaller one.
    combined, exposures = module.marginal_targets(
        model, [{"image_ids": small_ids}, {"image_ids": large_ids}], lookup, device
    )
    assert exposures[probe] == 2
    assert combined[probe] == pytest.approx((99 / 100 + 249 / 250) / 2, rel=1e-6)


def test_active_feature_columns_rejects_unmapped_column():
    configured = [
        "similarity_knn_mean", "iqa_composite", "agreement_score", "totally_unregistered_column",
    ]
    with pytest.raises(ValueError, match="not registered in FEATURE_COLUMNS_BY_SIGNAL"):
        active_feature_columns(configured, ["similarity", "iqa", "agreement"])


def test_contributing_signals_and_primary_vs_reduced_variant_threshold():
    """Mirrors 02_gonogo.py's >=3-signals-survive rule for the LEARNED selector (05's
    learned_variant_status). 3 signals is the primary/reduced_variant boundary: >=3 is primary,
    <3 is reduced_variant — verified on both sides so the boundary itself, not just its existence,
    is covered (the 01b CPU integration smoke only ever exercises the 4-signal case)."""
    configured = [
        "similarity_knn_mean", "similarity_top1", "similarity_topk_spread",
        "iqa_composite", "iqa_sharpness", "iqa_contrast_std",
        "uncertainty_mean_std", "explainability_region_overlap", "agreement_score",
    ]

    three_signals = active_feature_columns(configured, ["similarity", "iqa", "agreement"])
    assert contributing_signals(three_signals) == ["agreement", "iqa", "similarity"]
    status_at_three = (
        "primary" if len(contributing_signals(three_signals)) >= 3 else "reduced_variant"
    )
    assert status_at_three == "primary"

    two_signals = active_feature_columns(configured, ["similarity", "iqa"])
    assert contributing_signals(two_signals) == ["iqa", "similarity"]
    status_at_two = (
        "primary" if len(contributing_signals(two_signals)) >= 3 else "reduced_variant"
    )
    assert status_at_two == "reduced_variant"


def test_set_seed_makes_network_initialization_reproducible():
    """Regression test for the 2026-08-21 bug: 05/06/08's training scripts never called
    scripts.utils.seed.set_seed(), so torch's global RNG (weight init, dropout) was left
    unseeded — identical configs produced different models every run, and this was confirmed to
    occasionally degenerate 06_learn_thresholds_select.py into selecting ZERO images. This proves
    the mechanism the fix relies on: set_seed(same_seed) before construction must make two
    freshly-built networks start identical; without the fix (skip set_seed), they almost certainly
    would not."""
    from scripts.utils.seed import set_seed

    def fresh_weights(seed):
        set_seed(seed)
        set_utility = SetUtilityNetwork(9, (128, 64), (32,))
        ranker = MultiObjectiveRankingNetwork(9, (128, 64, 32), dropout=0.2)
        threshold = AdaptiveThresholdNetwork(11, 10, embedding_dim=16, hidden=(64, 32), dropout=0.1)
        return [p.detach().clone() for p in set_utility.parameters()], \
               [p.detach().clone() for p in ranker.parameters()], \
               [p.detach().clone() for p in threshold.parameters()]

    set_utility_a, ranker_a, threshold_a = fresh_weights(42)
    set_utility_b, ranker_b, threshold_b = fresh_weights(42)

    for a, b in zip(set_utility_a, set_utility_b):
        assert torch.equal(a, b)
    for a, b in zip(ranker_a, ranker_b):
        assert torch.equal(a, b)
    for a, b in zip(threshold_a, threshold_b):
        assert torch.equal(a, b)

    # Different seeds must NOT coincidentally match (sanity check that this test can actually fail).
    set_utility_c, _, _ = fresh_weights(43)
    assert any(not torch.equal(a, c) for a, c in zip(set_utility_a, set_utility_c))

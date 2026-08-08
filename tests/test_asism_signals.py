"""Behavioural tests for the ASISM signals that need no GPU (docs/stages2_to_5_plan.md §4).

These assert the SEMANTICS the methodology depends on — e.g. that a No-Finding recipe scores high
agreement exactly when disease probabilities are low, and that a memorized near-duplicate is
flagged rather than rewarded — not merely that the functions run.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.asism.signals import (  # noqa: E402
    compute_agreement_scores,
    compute_iqa_scores,
    compute_similarity_scores,
    compute_uncertainty_scores,
    expected_region_for,
    hierarchical_reference_indices,
    region_overlap_score,
)
from scripts.utils.config import CONFIGS_DIR  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from fixture_workspace import fixture_workspace

CONFIG = OmegaConf.load(CONFIGS_DIR / "stage3_asism.yaml")
OmegaConf.resolve(CONFIG)

def _write_image(workspace: Path, name: str, array: np.ndarray) -> Path:
    path = workspace / name
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path, format="JPEG", quality=95)
    return path


def _intended(positive_labels: list[str]) -> dict:
    return {label: (1 if label in positive_labels else 0) for label in PRIMARY_ENDPOINT_LABELS}


# ---------------------------------------------------------------- IQA (§4.2)

def test_iqa_flags_blank_image():
    blank = np.full((128, 128, 3), 128, dtype=np.uint8)
    with fixture_workspace("iqa-blank") as workspace:
        scores = compute_iqa_scores(_write_image(workspace, "blank.jpg", blank), CONFIG)
    assert bool(scores["iqa_valid"]) is True
    assert bool(scores["iqa_is_near_uniform"]) is True
    assert scores["iqa_composite"] < 0.6, "a blank image must be heavily penalized"


def test_iqa_rewards_structured_image():
    rng = np.random.default_rng(0)
    structured = rng.integers(0, 255, size=(128, 128, 1), dtype=np.uint8).repeat(3, axis=2)
    with fixture_workspace("iqa-structured") as workspace:
        scores = compute_iqa_scores(_write_image(workspace, "structured.jpg", structured), CONFIG)
    assert bool(scores["iqa_is_near_uniform"]) is False
    assert scores["iqa_composite"] > 0.7


def test_iqa_detects_colour_drift():
    # A real CXR is achromatic; strong per-channel divergence is a generation defect.
    rng = np.random.default_rng(1)
    coloured = rng.integers(0, 255, size=(128, 128, 3), dtype=np.uint8)
    with fixture_workspace("iqa-colour") as workspace:
        scores = compute_iqa_scores(_write_image(workspace, "coloured.jpg", coloured), CONFIG)
    assert scores["iqa_channel_spread"] > 2.0


def test_iqa_handles_unreadable_file_without_raising():
    with fixture_workspace("iqa-bad") as workspace:
        bad = workspace / "not_an_image.jpg"
        bad.write_bytes(b"definitely not a jpeg")
        scores = compute_iqa_scores(bad, CONFIG)
    assert bool(scores["iqa_valid"]) is False
    assert "iqa_error" in scores


# ---------------------------------------------------------------- Agreement (§4.5)

def test_agreement_high_when_intended_label_predicted():
    intended = _intended(["Cardiomegaly"])
    probabilities = np.full((1, 12), 0.05)
    probabilities[0, PRIMARY_ENDPOINT_LABELS.index("Cardiomegaly")] = 0.95
    result = compute_agreement_scores(probabilities, [intended], CONFIG)
    assert result.loc[0, "agreement_score"] > 0.9
    assert result.loc[0, "agreement_n_confident_unintended"] == 0


def test_agreement_low_when_intended_label_absent():
    intended = _intended(["Cardiomegaly"])
    probabilities = np.full((1, 12), 0.05)  # the intended label is NOT predicted
    result = compute_agreement_scores(probabilities, [intended], CONFIG)
    assert result.loc[0, "agreement_score"] < 0.2


def test_agreement_penalizes_confident_unintended_labels():
    intended = _intended(["Cardiomegaly"])
    probabilities = np.full((1, 12), 0.05)
    probabilities[0, PRIMARY_ENDPOINT_LABELS.index("Cardiomegaly")] = 0.95
    probabilities[0, PRIMARY_ENDPOINT_LABELS.index("Pneumothorax")] = 0.99  # unintended, confident

    penalized = compute_agreement_scores(probabilities, [intended], CONFIG).loc[0, "agreement_score"]

    clean = np.full((1, 12), 0.05)
    clean[0, PRIMARY_ENDPOINT_LABELS.index("Cardiomegaly")] = 0.95
    unpenalized = compute_agreement_scores(clean, [intended], CONFIG).loc[0, "agreement_score"]

    assert penalized < unpenalized


def test_agreement_no_finding_recipe_is_high_when_all_probabilities_low():
    """A No-Finding recipe is the all-zero intended vector: agreement must be HIGH exactly when the
    classifier sees no disease (§4.5)."""
    intended = _intended([])
    low = np.full((1, 12), 0.02)
    result = compute_agreement_scores(low, [intended], CONFIG)
    assert bool(result.loc[0, "agreement_is_no_finding_recipe"]) is True
    assert result.loc[0, "agreement_score"] > 0.9


def test_agreement_no_finding_recipe_is_low_when_disease_predicted():
    intended = _intended([])
    high = np.full((1, 12), 0.9)
    result = compute_agreement_scores(high, [intended], CONFIG)
    assert result.loc[0, "agreement_score"] < 0.2


# ---------------------------------------------------------------- Uncertainty (§4.3)

def test_uncertainty_bands_are_assigned_by_spread():
    std = np.array([[0.01] * 12, [0.10] * 12, [0.40] * 12])
    mean = np.full((3, 12), 0.5)
    result = compute_uncertainty_scores(std, mean, CONFIG)
    assert list(result["uncertainty_band"]) == ["low", "moderate", "extreme"]


def test_uncertainty_does_not_itself_penalize():
    """The signal reports bands; it must NOT encode 'higher = worse'. Moderate uncertainty may be
    the informative band, and that judgement belongs to the selection policy (§4.3/§4.8)."""
    std = np.array([[0.01] * 12, [0.10] * 12])
    mean = np.full((2, 12), 0.5)
    result = compute_uncertainty_scores(std, mean, CONFIG)
    assert "uncertainty_penalty" not in result.columns
    assert "uncertainty_score" not in result.columns


# ---------------------------------------------------------------- Similarity (§4.1)

def test_similarity_hierarchy_falls_back_when_exact_match_is_rare():
    references = [frozenset({"Cardiomegaly"})] * 3 + [frozenset({"Edema"})] * 40
    # Only 3 exact matches, below min_exact_match_references -> must not return tier1.
    indices, tier = hierarchical_reference_indices(
        _intended(["Cardiomegaly"]), references, min_exact=25
    )
    assert tier != "tier1_exact"
    assert len(indices) > 0, "fallback must always yield a usable reference pool"


def test_similarity_hierarchy_uses_exact_match_when_plentiful():
    references = [frozenset({"Cardiomegaly"})] * 40 + [frozenset({"Edema"})] * 5
    indices, tier = hierarchical_reference_indices(
        _intended(["Cardiomegaly"]), references, min_exact=25
    )
    assert tier == "tier1_exact"
    assert len(indices) == 40


def test_similarity_no_finding_recipe_falls_back_to_class_agnostic():
    references = [frozenset({"Cardiomegaly"})] * 30
    _, tier = hierarchical_reference_indices(_intended([]), references, min_exact=25)
    assert tier == "tier4_class_agnostic"


def test_similarity_flags_near_duplicate_as_memorization():
    """An image nearly identical to ONE reference must be flagged, not rewarded (§4.1)."""
    reference = np.eye(1, 16, 0).repeat(30, axis=0) + np.random.default_rng(0).normal(0, 0.3, (30, 16))
    exact_copy = reference[0:1].copy()

    result = compute_similarity_scores(
        exact_copy,
        reference,
        [frozenset()] * 30,
        [_intended([])],
        CONFIG,
    )
    assert result.loc[0, "similarity_top1"] > 0.99
    assert bool(result.loc[0, "novelty_is_near_duplicate"]) is True
    assert result.loc[0, "novelty_score"] == 0.0, "memorized images get no novelty credit"


def test_similarity_normal_image_is_not_flagged():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(40, 16))
    synthetic = rng.normal(size=(1, 16))
    result = compute_similarity_scores(
        synthetic, reference, [frozenset()] * 40, [_intended([])], CONFIG
    )
    assert bool(result.loc[0, "novelty_is_near_duplicate"]) is False


# ---------------------------------------------------------------- Explainability (§4.4)

def test_region_overlap_full_inside():
    cam = np.zeros((10, 10))
    cam[5:8, 5:8] = 1.0
    assert region_overlap_score(cam, (0.0, 0.0, 1.0, 1.0)) == 1.0


def test_region_overlap_fully_outside():
    cam = np.zeros((10, 10))
    cam[0:2, 0:2] = 1.0
    assert region_overlap_score(cam, (0.5, 0.5, 1.0, 1.0)) == 0.0


def test_region_overlap_undefined_for_empty_cam():
    assert np.isnan(region_overlap_score(np.zeros((10, 10)), (0.0, 0.0, 1.0, 1.0)))


def test_expected_region_is_pathology_specific():
    cardiomegaly = expected_region_for(_intended(["Cardiomegaly"]), CONFIG)
    effusion = expected_region_for(_intended(["Pleural Effusion"]), CONFIG)
    assert cardiomegaly != effusion, "different pathologies must have different expected regions"


def test_expected_region_no_finding_uses_baseline_box():
    assert expected_region_for(_intended([]), CONFIG) == tuple(
        CONFIG.signals.explainability.baseline_box
    )


def test_expected_region_multilabel_is_union():
    both = expected_region_for(_intended(["Cardiomegaly", "Pleural Effusion"]), CONFIG)
    cardiomegaly = expected_region_for(_intended(["Cardiomegaly"]), CONFIG)
    # The union must be at least as tall as either constituent region.
    assert both[3] >= cardiomegaly[3]


if __name__ == "__main__":
    import traceback

    tests = [(name, value) for name, value in sorted(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for name, function in tests:
        try:
            function()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

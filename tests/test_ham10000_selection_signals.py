#!/usr/bin/env python3
"""The three HAM10000 selection signals: similarity, uncertainty and agreement.

What these tests are really guarding against is not a crash. Every one of these signals has a
CheXpert implementation that RUNS on HAM10000 data and returns a well-formed frame — it just
computes a different quantity (a Jaccard tier that a mutually exclusive label set cannot express, an
average of per-label Bernoulli entropies over a softmax row, a penalty term treated as independent
evidence when the softmax makes it anything but). Nothing downstream would notice. So the tests
below pin the quantities themselves against hand-computed values, not merely the column names.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.ham10000_signals import (  # noqa: E402
    SIMILARITY_TIER_CLASS_AGNOSTIC,
    SIMILARITY_TIER_SAME_CLASS,
    compute_agreement_scores,
    compute_similarity_scores,
    compute_uncertainty_scores,
    single_label_reference_indices,
)
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402

N_CLASSES = len(CLASSIFIER_TARGET_LABELS)


def _config(**overrides):
    base = {
        "signals": {
            "similarity": {
                "k_neighbors": 3,
                "min_same_class_references": 2,
                "near_duplicate_similarity": 0.95,
                "near_duplicate_top1_gap": 0.02,
            },
            "uncertainty": {"low_band_max": 0.05, "moderate_band_max": 0.20},
            "agreement": {"rival_confidence_threshold": 0.5, "penalty_weight": 0.5},
        }
    }
    for signal, values in overrides.items():
        base["signals"][signal].update(values)
    return OmegaConf.create(base)


def _unit(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


# ==============================================================================================
# Similarity
# ==============================================================================================


def test_references_come_from_the_same_class_when_the_pool_is_large_enough():
    indices, tier = single_label_reference_indices("mel", ["nv", "mel", "nv", "mel"], min_same_class=2)
    assert tier == SIMILARITY_TIER_SAME_CLASS
    assert indices.tolist() == [1, 3]


def test_a_thin_same_class_pool_falls_back_and_SAYS_SO():
    """The fallback must be visible. A row scored against all classes that claims to be a
    same-class comparison is a silently weaker measurement, which is the failure this column
    exists to prevent."""
    indices, tier = single_label_reference_indices("df", ["nv", "mel", "df"], min_same_class=2)
    assert tier == SIMILARITY_TIER_CLASS_AGNOSTIC
    assert indices.tolist() == [0, 1, 2]


def test_similarity_is_measured_against_the_intended_class_not_the_whole_pool():
    """A synthetic 'mel' sitting far from every real mel but right on top of the nv cluster must
    score LOW. If the references were pooled across classes it would score high, and a wrong-class
    image would be selected for being realistic dermoscopy of the wrong lesion."""
    reference = _unit([[1, 0], [0.99, 0.14], [0.98, 0.2], [0, 1], [0.14, 0.99], [0.2, 0.98]])
    diagnoses = ["nv", "nv", "nv", "mel", "mel", "mel"]
    synthetic = _unit([[1, 0.02]])  # visually an nv

    frame = compute_similarity_scores(synthetic, reference, diagnoses, ["mel"], _config())

    assert frame.loc[0, "similarity_reference_tier"] == SIMILARITY_TIER_SAME_CLASS
    assert frame.loc[0, "similarity_n_references"] == 3
    assert frame.loc[0, "similarity_knn_mean"] < 0.5

    # The same image asked for as an nv is the same measurement against the other pool.
    as_nv = compute_similarity_scores(synthetic, reference, diagnoses, ["nv"], _config())
    assert as_nv.loc[0, "similarity_knn_mean"] > 0.95


def test_a_near_copy_of_one_real_image_is_flagged_and_earns_NO_novelty_credit():
    """The trap: a memorised image has the highest fidelity score in the pool. Novelty must not
    follow fidelity, or the selector ranks patient-data replication first."""
    reference = _unit([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    synthetic = _unit([[1, 0.01, 0.01]])  # essentially reference row 0

    frame = compute_similarity_scores(
        synthetic, reference, ["mel"] * 3, ["mel"], _config()
    )

    assert frame.loc[0, "similarity_top1"] >= 0.95
    assert bool(frame.loc[0, "novelty_is_near_duplicate"]) is True
    assert bool(frame.loc[0, "novelty_collapsed_onto_single_reference"]) is True
    assert frame.loc[0, "novelty_score"] == 0.0
    # Fidelity itself is untouched — the two readings stay separable downstream.
    assert frame.loc[0, "similarity_knn_mean"] > 0.0


def _references_at_cosines(cosines: list[float]) -> np.ndarray:
    """Unit vectors whose cosine against [1, 0, 0] is exactly each requested value.

    Built from the target cosine rather than eyeballed in 2D, where a small coordinate change moves
    the cosine far less than it looks like it should and a fixture drifts over the 0.95 line by
    accident.
    """
    rows = []
    for index, cosine in enumerate(cosines):
        perpendicular = float(np.sqrt(max(1.0 - cosine**2, 0.0)))
        # Alternate the perpendicular axis so the references are not collinear with each other.
        rows.append([cosine, perpendicular, 0.0] if index % 2 == 0 else [cosine, 0.0, perpendicular])
    return np.asarray(rows, dtype=np.float64)


SYNTHETIC_PROBE = np.asarray([[1.0, 0.0, 0.0]])


def test_an_image_typical_of_its_class_is_not_flagged_as_memorised():
    reference = _references_at_cosines([0.88, 0.86, 0.84])
    frame = compute_similarity_scores(
        SYNTHETIC_PROBE, reference, ["nv"] * 3, ["nv"], _config()
    )
    assert frame.loc[0, "similarity_top1"] == pytest.approx(0.88)
    assert bool(frame.loc[0, "novelty_is_near_duplicate"]) is False
    assert frame.loc[0, "novelty_score"] == pytest.approx(frame.loc[0, "similarity_knn_mean"])


def test_the_top1_gap_never_creates_a_flag_on_its_own():
    """A merely-similar image with an isolated nearest neighbour is not memorisation."""
    reference = _references_at_cosines([0.90, 0.80, 0.78])
    frame = compute_similarity_scores(
        SYNTHETIC_PROBE, reference, ["bcc"] * 3, ["bcc"], _config()
    )
    assert frame.loc[0, "similarity_top1"] == pytest.approx(0.90)
    assert frame.loc[0, "similarity_top1_gap"] == pytest.approx(0.90 - 0.79)
    assert bool(frame.loc[0, "novelty_is_near_duplicate"]) is False
    assert bool(frame.loc[0, "novelty_collapsed_onto_single_reference"]) is False


def test_k_is_capped_by_the_pool_and_reported():
    reference = _unit([[1, 0], [0.9, 0.4]])
    frame = compute_similarity_scores(
        _unit([[1, 0.1]]), reference, ["vasc", "vasc"], ["vasc"], _config()
    )
    assert frame.loc[0, "similarity_k_used"] == 2  # k_neighbors is 3


def test_misaligned_inputs_are_refused_rather_than_scored_against_the_wrong_rows():
    with pytest.raises(ValueError, match="positional"):
        compute_similarity_scores(
            _unit([[1, 0]]), _unit([[1, 0], [0, 1]]), ["nv"], ["nv"], _config()
        )
    with pytest.raises(ValueError, match="positional"):
        compute_similarity_scores(
            _unit([[1, 0], [0, 1]]), _unit([[1, 0]]), ["nv"], ["nv"], _config()
        )


def test_an_empty_reference_pool_is_an_error_not_a_zero_score():
    with pytest.raises(ValueError, match="nothing to measure against"):
        compute_similarity_scores(
            _unit([[1, 0]]), np.zeros((0, 2)), [], ["nv"], _config()
        )


# ==============================================================================================
# Uncertainty
# ==============================================================================================


def _passes(rows_per_pass) -> np.ndarray:
    return np.asarray(rows_per_pass, dtype=np.float64)


def test_a_confident_and_consistent_image_has_almost_no_uncertainty_of_either_kind():
    confident = [0.94] + [0.01] * (N_CLASSES - 1)
    frame = compute_uncertainty_scores(_passes([[confident], [confident]]), _config())
    assert frame.loc[0, "uncertainty_predictive_entropy"] < 0.2
    assert frame.loc[0, "uncertainty_mutual_information"] == pytest.approx(0.0, abs=1e-9)
    assert frame.loc[0, "uncertainty_band"] == "low"


def test_aleatoric_and_epistemic_are_SEPARATED_not_summed():
    """The whole reason the per-pass array is required. Both images below have an identical mean
    prediction and therefore an identical TOTAL uncertainty; they are completely different images
    for selection purposes, and only the decomposition tells them apart.

      ambiguous — every dropout mask agrees the image is 50/50 (a genuinely hard lesion)
      unsettled — the masks disagree outright (the model has no opinion; typically an artefact)
    """
    half = [0.5, 0.5] + [0.0] * (N_CLASSES - 2)
    first = [1.0, 0.0] + [0.0] * (N_CLASSES - 2)
    second = [0.0, 1.0] + [0.0] * (N_CLASSES - 2)

    ambiguous = compute_uncertainty_scores(_passes([[half], [half]]), _config())
    unsettled = compute_uncertainty_scores(_passes([[first], [second]]), _config())

    assert ambiguous.loc[0, "uncertainty_predictive_entropy"] == pytest.approx(
        unsettled.loc[0, "uncertainty_predictive_entropy"]
    )
    # ...and yet:
    assert ambiguous.loc[0, "uncertainty_mutual_information"] == pytest.approx(0.0, abs=1e-9)
    assert unsettled.loc[0, "uncertainty_mutual_information"] == pytest.approx(
        np.log(2) / np.log(N_CLASSES), abs=1e-9
    )
    assert ambiguous.loc[0, "uncertainty_expected_entropy"] > 0.0
    assert unsettled.loc[0, "uncertainty_expected_entropy"] == pytest.approx(0.0, abs=1e-9)

    assert ambiguous.loc[0, "uncertainty_band"] == "low"
    assert unsettled.loc[0, "uncertainty_band"] == "extreme"


def test_entropies_are_normalised_so_the_bands_read_as_fractions():
    uniform = [1.0 / N_CLASSES] * N_CLASSES
    frame = compute_uncertainty_scores(_passes([[uniform], [uniform]]), _config())
    assert frame.loc[0, "uncertainty_predictive_entropy"] == pytest.approx(1.0)


def test_mutual_information_never_goes_negative_from_float_noise():
    rng = np.random.default_rng(0)
    draws = rng.dirichlet(np.ones(N_CLASSES), size=(8, 25)).transpose(1, 0, 2)
    frame = compute_uncertainty_scores(draws, _config())
    assert (frame["uncertainty_mutual_information"] >= 0.0).all()
    assert (frame["uncertainty_predictive_entropy"] >= frame["uncertainty_expected_entropy"] - 1e-9).all()


def test_a_single_pass_is_refused_because_the_epistemic_term_would_be_zero_by_construction():
    confident = [0.94] + [0.01] * (N_CLASSES - 1)
    with pytest.raises(ValueError, match="constant, not a measurement"):
        compute_uncertainty_scores(_passes([[confident]]), _config())


def test_bands_must_be_ordered():
    confident = [0.94] + [0.01] * (N_CLASSES - 1)
    bad = _config(uncertainty={"low_band_max": 0.3, "moderate_band_max": 0.1})
    with pytest.raises(ValueError, match="low_band_max"):
        compute_uncertainty_scores(_passes([[confident], [confident]]), bad)


def test_a_two_dimensional_mean_array_is_refused():
    """Passing the (mean, std) output of predict_probabilities_mc_dropout here would silently
    measure nothing; the shape check is what makes that a failure instead."""
    with pytest.raises(ValueError, match="n_passes, n_images, n_classes"):
        compute_uncertainty_scores(np.full((3, N_CLASSES), 1.0 / N_CLASSES), _config())


# ==============================================================================================
# Agreement
# ==============================================================================================


def _row(mapping: dict[str, float]) -> list[float]:
    return [float(mapping.get(label, 0.0)) for label in CLASSIFIER_TARGET_LABELS]


def test_the_classifier_reading_back_the_intended_class_scores_high_and_unpenalised():
    probabilities = np.array([_row({"mel": 0.9, "nv": 0.1})])
    frame = compute_agreement_scores(probabilities, ["mel"], _config())
    assert frame.loc[0, "agreement_intended_prob"] == pytest.approx(0.9)
    assert bool(frame.loc[0, "agreement_is_argmax_match"]) is True
    assert bool(frame.loc[0, "agreement_penalty_applied"]) is False
    assert frame.loc[0, "agreement_score"] == pytest.approx(0.9)
    assert frame.loc[0, "agreement_margin"] == pytest.approx(0.8)


def test_confidently_the_wrong_lesion_is_penalised_harder_than_diffusely_unsure():
    """Both rows give the intended class 0.2. One is a model that cannot decide; the other says the
    image is a naevus. Only the second puts a mislabelled image into the training set, and the
    score has to be able to say so."""
    diffuse = _row({"mel": 0.2, "nv": 0.2, "bkl": 0.2, "bcc": 0.2, "akiec": 0.1, "vasc": 0.1})
    decided = _row({"mel": 0.2, "nv": 0.8})

    frame = compute_agreement_scores(np.array([diffuse, decided]), ["mel", "mel"], _config())

    assert frame.loc[0, "agreement_intended_prob"] == pytest.approx(
        frame.loc[1, "agreement_intended_prob"]
    )
    assert bool(frame.loc[0, "agreement_penalty_applied"]) is False
    assert bool(frame.loc[1, "agreement_penalty_applied"]) is True
    assert frame.loc[1, "agreement_score"] < frame.loc[0, "agreement_score"]
    assert frame.loc[1, "agreement_score"] == pytest.approx(0.2 - 0.5 * 0.8)
    assert frame.loc[1, "agreement_best_rival_diagnosis"] == "nv"
    assert frame.loc[1, "agreement_margin"] == pytest.approx(-0.6)


def test_the_best_rival_is_never_the_intended_class_itself():
    """Taking argmax over the full row would report the intended class as its own rival whenever the
    classifier agrees, and the penalty would fire on the images that are most correct."""
    frame = compute_agreement_scores(
        np.array([_row({"akiec": 0.7, "bkl": 0.3})]), ["akiec"], _config()
    )
    assert frame.loc[0, "agreement_best_rival_diagnosis"] == "bkl"
    assert frame.loc[0, "agreement_best_rival_prob"] == pytest.approx(0.3)
    assert bool(frame.loc[0, "agreement_penalty_applied"]) is False


def test_an_unrecognised_intended_diagnosis_is_refused_not_scored():
    with pytest.raises(ValueError, match="invalid HAM10000 dx"):
        compute_agreement_scores(np.array([_row({"nv": 1.0})]), ["melanoma"], _config())


def test_a_probability_matrix_with_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="softmax probabilities"):
        compute_agreement_scores(np.full((1, 11), 1 / 11), ["nv"], _config())


def test_row_counts_must_match():
    with pytest.raises(ValueError, match="intended diagnoses"):
        compute_agreement_scores(np.array([_row({"nv": 1.0})]), ["nv", "mel"], _config())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

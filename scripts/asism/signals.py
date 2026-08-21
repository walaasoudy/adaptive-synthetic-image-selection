"""The five ASISM signals (docs/stages2_to_5_plan.md §4.1-§4.5).

Each signal computes a per-image score set written to its own versioned parquet artifact. All five
are subject to the same Go/No-Go gate (§4.6) — implementation is not admission.

The IQA signal is deliberately pure NumPy/PIL: it is the one signal that needs no model and no GPU,
so it stays runnable and testable locally, which is what lets the whole selection/ranking layer be
developed off the pod.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS, intended_vector_to_labels

SCHEMA_VERSION = 2


def aggregate_pathology_overlaps(per_label: dict[str, float]) -> tuple[float, float]:
    """Frozen multi-label Grad-CAM rule: equal-weight mean, with minimum retained as a guardrail."""
    finite = [float(value) for value in per_label.values() if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(min(finite))


# ----------------------------------------------------------------------------------------------
# §4.2 Image Quality Assessment — no model, no GPU
# ----------------------------------------------------------------------------------------------

def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the Laplacian: the standard focus/blur measure. Low variance = few sharp edges."""
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(gray, 1, mode="edge")
    response = np.zeros_like(gray, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight != 0.0:
                response += weight * padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return float(response.var())


def compute_iqa_scores(image_path: Path, config) -> dict:
    """Multi-faceted technical quality (§4.2).

    Generic no-reference IQA is represented by the sharpness/contrast statistics; the domain checks
    (border artifact, clipping, near-uniformity, grayscale consistency, anatomical completeness
    proxy) are what the signal actually leans on. Whether the generic part adds anything beyond the
    domain part is settled by ablation, not assumed.
    """
    iqa_cfg = config.signals.iqa
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
            gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    except Exception as exc:
        return {"iqa_valid": False, "iqa_error": f"{type(exc).__name__}: {exc}"}

    height, width = gray.shape
    std = float(gray.std())
    mean = float(gray.mean())

    # Grayscale consistency: a real CXR is achromatic. Channel divergence indicates colour drift
    # from the RGB generation path.
    channel_spread = float(np.mean(np.std(array, axis=2)))

    # Clipping: mass piled at the extremes of the dynamic range.
    clipped_low = float(np.mean(gray <= 1.0))
    clipped_high = float(np.mean(gray >= 254.0))

    # Border artifact: letterbox residue or a dead frame around the anatomy.
    border_width = max(1, int(min(height, width) * 0.06))
    border = np.concatenate(
        [
            gray[:border_width, :].ravel(),
            gray[-border_width:, :].ravel(),
            gray[:, :border_width].ravel(),
            gray[:, -border_width:].ravel(),
        ]
    )
    border_uniform_fraction = float(np.mean(np.abs(border - border.mean()) < 2.0))

    # Anatomical completeness proxy: a plausible CXR has substantially more signal in the central
    # region than at the frame edge. A near-empty centre means the anatomy is missing or collapsed.
    centre = gray[
        int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75)
    ]
    centre_energy_ratio = float(centre.std() / (std + 1e-6))

    sharpness = _laplacian_variance(gray)

    # Histogram abnormality: distance from a smooth unimodal shape. A spiky/degenerate histogram
    # indicates posterization or a broken decode.
    histogram, _ = np.histogram(gray, bins=64, range=(0, 255))
    histogram = histogram / max(histogram.sum(), 1)
    histogram_entropy = float(-np.sum(histogram[histogram > 0] * np.log(histogram[histogram > 0])))

    is_near_uniform = std < float(iqa_cfg.blank_std_threshold)
    is_blurry = sharpness < float(iqa_cfg.laplacian_blur_threshold)
    is_low_contrast = std < float(iqa_cfg.low_contrast_std)
    has_border_artifact = border_uniform_fraction > float(iqa_cfg.border_uniform_fraction)

    # Composite in [0, 1]: 1.0 is clean. Each detected defect subtracts a fixed amount, so the
    # composite degrades gracefully rather than being dominated by one continuous term.
    penalties = (
        0.40 * float(is_near_uniform)
        + 0.20 * float(is_blurry)
        + 0.10 * float(is_low_contrast)
        + 0.10 * float(has_border_artifact)
        + 0.10 * float(clipped_low + clipped_high > 0.20)
        + 0.10 * float(channel_spread > 2.0)
    )
    composite = float(max(0.0, 1.0 - penalties))

    return {
        "iqa_valid": True,
        "iqa_composite": composite,
        "iqa_sharpness": sharpness,
        "iqa_contrast_std": std,
        "iqa_mean_intensity": mean,
        "iqa_channel_spread": channel_spread,
        "iqa_clipped_low_fraction": clipped_low,
        "iqa_clipped_high_fraction": clipped_high,
        "iqa_border_uniform_fraction": border_uniform_fraction,
        "iqa_centre_energy_ratio": centre_energy_ratio,
        "iqa_histogram_entropy": histogram_entropy,
        "iqa_is_near_uniform": bool(is_near_uniform),
        "iqa_is_blurry": bool(is_blurry),
        "iqa_is_low_contrast": bool(is_low_contrast),
        "iqa_has_border_artifact": bool(has_border_artifact),
    }


# ----------------------------------------------------------------------------------------------
# §4.5 Intended-label agreement
# ----------------------------------------------------------------------------------------------

def compute_agreement_scores(
    probabilities: np.ndarray,
    intended_vectors: list[dict],
    config,
) -> pd.DataFrame:
    """Proxy label-consistency between the generator's intent and the auxiliary classifier's read.

        agreement = mean P(intended positive labels)
                  - penalty_weight * mean P(confidently predicted UNintended labels)

    For a No Finding recipe the intended positive set is empty, so the first term is defined as
    (1 - mean probability over all 11 primary labels): agreement is high exactly when the classifier
    sees no disease, which is what that recipe asked for.

    NOT clinical proof. The generator and the classifier can share a bias and agree while both are
    wrong; this measures consistency between two models, nothing more (§4.5).
    """
    agreement_cfg = config.signals.agreement
    threshold = float(agreement_cfg.unintended_confidence_threshold)
    penalty_weight = float(agreement_cfg.penalty_weight)

    label_index = {label: position for position, label in enumerate(PRIMARY_ENDPOINT_LABELS)}
    rows = []

    for row_index, intended in enumerate(intended_vectors):
        probs = probabilities[row_index]
        intended_positive = intended_vector_to_labels(intended)
        unintended = [label for label in PRIMARY_ENDPOINT_LABELS if label not in intended_positive]

        if intended_positive:
            intended_mean = float(
                np.mean([probs[label_index[label]] for label in intended_positive])
            )
            is_no_finding = False
        else:
            # No Finding recipe: "intended" outcome is uniformly low disease probability.
            intended_mean = float(
                1.0 - np.mean([probs[label_index[label]] for label in PRIMARY_ENDPOINT_LABELS])
            )
            is_no_finding = True

        unintended_probs = [probs[label_index[label]] for label in unintended]
        confident_unintended = [p for p in unintended_probs if p >= threshold]
        unintended_penalty = float(np.mean(confident_unintended)) if confident_unintended else 0.0

        rows.append(
            {
                "agreement_score": intended_mean - penalty_weight * unintended_penalty,
                "agreement_intended_mean_prob": intended_mean,
                "agreement_unintended_penalty": unintended_penalty,
                "agreement_n_intended_positive": len(intended_positive),
                "agreement_n_confident_unintended": len(confident_unintended),
                "agreement_is_no_finding_recipe": is_no_finding,
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# §4.3 Uncertainty
# ----------------------------------------------------------------------------------------------

def compute_uncertainty_scores(
    probability_std: np.ndarray,
    probability_mean: np.ndarray,
    config,
) -> pd.DataFrame:
    """MC-Dropout predictive spread over the 11 primary labels, banded.

    Bands are reported, NOT converted into a penalty here: moderate uncertainty may be the most
    informative band (TSynD's premise, §4.3), so whether a band is good or bad is a selection-policy
    decision made later in §4.8, not baked into the signal.
    """
    uncertainty_cfg = config.signals.uncertainty
    low_max = float(uncertainty_cfg.low_band_max)
    moderate_max = float(uncertainty_cfg.moderate_band_max)

    mean_std = probability_std.mean(axis=1)
    max_std = probability_std.max(axis=1)
    # Predictive entropy of the mean probabilities: a second, complementary uncertainty view.
    clipped = np.clip(probability_mean, 1e-6, 1 - 1e-6)
    entropy = -(clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)).mean(axis=1)

    bands = np.where(mean_std <= low_max, "low", np.where(mean_std <= moderate_max, "moderate", "extreme"))

    return pd.DataFrame(
        {
            "uncertainty_mean_std": mean_std,
            "uncertainty_max_std": max_std,
            "uncertainty_entropy": entropy,
            "uncertainty_band": bands,
        }
    )


# ----------------------------------------------------------------------------------------------
# §4.1 Similarity
# ----------------------------------------------------------------------------------------------

def hierarchical_reference_indices(
    intended: dict,
    reference_label_vectors: list[frozenset],
    min_exact: int,
) -> tuple[np.ndarray, str]:
    """Pick real-reference rows for one synthetic image, walking the §4.1 hierarchy.

    Tier 1 exact intended-vector match -> Tier 2 nearest multilabel neighbourhood (Jaccard) ->
    Tier 3 pathology prototypes (any shared positive) -> Tier 4 class-agnostic realism fallback.
    The fallback is always available, so no image goes unscored for lack of an exact-match pool —
    which is precisely the failure mode a rigid exact-match requirement would cause for rare
    combinations.
    """
    intended_set = frozenset(intended_vector_to_labels(intended))

    exact = np.array(
        [i for i, combo in enumerate(reference_label_vectors) if combo == intended_set], dtype=int
    )
    if len(exact) >= min_exact:
        return exact, "tier1_exact"

    if intended_set:
        similarities = np.array(
            [
                len(intended_set & combo) / max(len(intended_set | combo), 1)
                for combo in reference_label_vectors
            ]
        )
        neighbourhood = np.where(similarities >= 0.5)[0]
        if len(neighbourhood) >= min_exact:
            return neighbourhood, "tier2_neighbourhood"

        prototype = np.array(
            [i for i, combo in enumerate(reference_label_vectors) if intended_set & combo], dtype=int
        )
        if len(prototype) >= min_exact:
            return prototype, "tier3_prototype"

    return np.arange(len(reference_label_vectors)), "tier4_class_agnostic"


def compute_similarity_scores(
    synthetic_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_label_vectors: list[frozenset],
    intended_vectors: list[dict],
    config,
) -> pd.DataFrame:
    """k-NN cosine similarity to real references, plus a SEPARATE memorization flag.

    Fidelity and memorization are reported as distinct columns on purpose: an image that is nearly
    identical to one specific training image is a memorization red flag, not a fidelity success, and
    collapsing the two would make the selector reward exactly what it should catch (§4.1).
    """
    similarity_cfg = config.signals.similarity
    k = int(similarity_cfg.k_neighbors)
    min_exact = int(similarity_cfg.min_exact_match_references)
    near_dup_similarity = float(similarity_cfg.near_duplicate_similarity)
    near_dup_spread = float(similarity_cfg.near_duplicate_spread)

    synthetic_norm = synthetic_embeddings / (
        np.linalg.norm(synthetic_embeddings, axis=1, keepdims=True) + 1e-8
    )
    reference_norm = reference_embeddings / (
        np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8
    )

    rows = []
    for row_index in range(len(synthetic_norm)):
        indices, tier = hierarchical_reference_indices(
            intended_vectors[row_index], reference_label_vectors, min_exact
        )
        similarities = reference_norm[indices] @ synthetic_norm[row_index]

        effective_k = min(k, len(similarities))
        top_k = np.sort(similarities)[-effective_k:][::-1]
        top1 = float(top_k[0])
        knn_mean = float(top_k.mean())
        # Gap between the single closest reference and the rest of the neighbourhood.
        spread = float(top_k[0] - top_k[-1]) if effective_k > 1 else 0.0
        # How far top1 stands out from the remaining neighbours specifically.
        rest_mean = float(top_k[1:].mean()) if effective_k > 1 else top1
        top1_gap = top1 - rest_mean

        # Memorization = the image is near-identical to SOME real training image. top1 crossing the
        # threshold is sufficient on its own; a large top1_gap additionally indicates it collapsed
        # onto ONE specific image rather than onto a cluster of near-identical real images.
        # (Requiring a SMALL spread here would be backwards: an image memorizing one reference is
        # by construction far from the others, so its spread is large.)
        is_near_duplicate = bool(top1 >= near_dup_similarity)
        collapsed_onto_single = bool(is_near_duplicate and top1_gap >= near_dup_spread)

        rows.append(
            {
                "similarity_knn_mean": knn_mean,
                "similarity_top1": top1,
                "similarity_topk_spread": spread,
                "similarity_reference_tier": tier,
                "similarity_n_references": int(len(indices)),
                "similarity_k_used": effective_k,
                "similarity_top1_gap": top1_gap,
                "novelty_is_near_duplicate": is_near_duplicate,
                "novelty_collapsed_onto_single_reference": collapsed_onto_single,
                # Novelty score: high when the image resembles the class without being a near-copy
                # of a specific training image. Memorized images get no novelty credit.
                "novelty_score": float(knn_mean * (1.0 - float(is_near_duplicate))),
            }
        )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# §4.4 Explainability
# ----------------------------------------------------------------------------------------------

def region_overlap_score(cam: np.ndarray, region: tuple[float, float, float, float]) -> float:
    """Fraction of total Grad-CAM activation mass falling inside the expected region.

    An explainability-PLAUSIBILITY signal, never proof of diagnostic correctness (§4.4). High
    overlap means the classifier looked where the pathology usually appears; it says nothing about
    whether the pathology is actually present.
    """
    height, width = cam.shape
    x0, y0, x1, y1 = region
    col0, col1 = int(x0 * width), int(x1 * width)
    row0, row1 = int(y0 * height), int(y1 * height)

    cam = np.maximum(cam, 0.0)
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")
    return float(cam[row0:row1, col0:col1].sum() / total)


def expected_region_for(intended: dict, config) -> tuple[float, float, float, float]:
    """Pathology-specific expected region, or the explicit baseline box for No Finding recipes."""
    explain_cfg = config.signals.explainability
    positives = intended_vector_to_labels(intended)
    if str(explain_cfg.region_mode) == "baseline" or not positives:
        return tuple(explain_cfg.baseline_box)

    regions = [
        tuple(explain_cfg.pathology_regions[label])
        for label in positives
        if label in explain_cfg.pathology_regions
    ]
    if not regions:
        return tuple(explain_cfg.baseline_box)

    # Union of the per-pathology regions: a multi-label recipe may legitimately activate anywhere
    # its constituent pathologies are expected.
    return (
        min(r[0] for r in regions),
        min(r[1] for r in regions),
        max(r[2] for r in regions),
        max(r[3] for r in regions),
    )


def write_score_artifact(
    frame: pd.DataFrame,
    path: Path,
    signal_name: str,
    provenance: dict,
) -> None:
    """Atomic, versioned parquet write with an embedded provenance block (§4.0)."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame.attrs["schema_version"] = SCHEMA_VERSION
    frame.attrs["signal"] = signal_name

    sidecar = path.with_suffix(".provenance.json")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_parquet(temp_name, index=False)
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)

    from scripts.utils.manifest import sha256_file, write_json
    write_json(sidecar, {
        "schema_version": SCHEMA_VERSION, "signal": signal_name, "n_rows": len(frame),
        "columns": list(frame.columns), "parquet_sha256": sha256_file(path), **provenance,
    })

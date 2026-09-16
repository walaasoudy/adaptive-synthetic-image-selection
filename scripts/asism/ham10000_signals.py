"""ASISM signals specialised for dermoscopy: colour-safe IQA and lesion-appropriate explainability.

Only the two signals whose CheXpert implementations encode radiograph-specific assumptions live
here. Similarity (DINOv2), uncertainty (MC dropout) and agreement are genuinely dataset-agnostic and
are still used from scripts/asism/signals.py unchanged.

scripts/asism/signals.py is not modified by this module. The CheXpert path keeps its behaviour
exactly; this is a parallel implementation selected by dataset, not a mutation of the shared one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# ==============================================================================================
# IQA — the same defect checks, minus the one that only makes sense for an achromatic image
# ==============================================================================================
#
# WHAT WAS REMOVED AND WHY
#   The CheXpert IQA penalises `channel_spread > 2.0`: a radiograph is achromatic, so per-channel
#   divergence means colour has leaked in from the RGB generation path, which is a defect. In
#   dermoscopy colour IS the diagnostic signal — pigment network hue, blue-white veil, vascular
#   patterns. Carrying that penalty over would fire on essentially every valid image and would
#   actively select AGAINST the most informative ones. It is removed, not re-tuned: there is no
#   threshold at which "this image is colourful" is evidence of a defect here.
#
#   `channel_saturation` is still COMPUTED and reported, because a near-zero value means a
#   dermoscopy frame arrived greyscale (a real pipeline failure worth seeing). It carries no
#   penalty weight — it is a diagnostic readout, not a quality judgement.
#
# WHAT IS MEASURED ON LUMINANCE, AND WHY THAT IS NOT THE SAME MISTAKE
#   Sharpness and contrast are computed on a luminance projection. That is a measurement convention
#   — focus and dynamic range are luminance properties, and the standard Laplacian-variance focus
#   measure is defined on a single channel. The IMAGE on disk stays RGB; nothing is stored or passed
#   downstream in greyscale. The bug this module exists to avoid was converting the DATA to
#   greyscale, not computing a scalar from its luminance.

CONTINUOUS_WEIGHT = 0.06


def _laplacian_variance(channel: np.ndarray) -> float:
    """Variance of the Laplacian: the standard focus/blur measure. Low variance = few sharp edges."""
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(channel, 1, mode="edge")
    response = np.zeros_like(channel, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy, dx]
            if weight != 0.0:
                response += weight * padded[dy : dy + channel.shape[0], dx : dx + channel.shape[1]]
    return float(response.var())


def compute_iqa_scores(image_path: Path, config) -> dict:
    """Technical quality for one dermoscopy image. Higher `iqa_composite` is better, range [0, 1].

    `config` is expected to expose signals.iqa with the same keys the CheXpert config uses
    (blank_std_threshold, laplacian_blur_threshold, low_contrast_std, border_uniform_fraction,
    clipping_percentile) so a single Stage 3 config shape serves both datasets.
    """
    iqa_cfg = config.signals.iqa
    try:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
            luminance = np.asarray(rgb.convert("L"), dtype=np.float32)
    except Exception as exc:
        return {"iqa_valid": False, "iqa_error": f"{type(exc).__name__}: {exc}"}

    height, width = luminance.shape
    std = float(luminance.std())
    mean = float(luminance.mean())

    # Reported, never penalised. Near zero means a dermoscopy frame arrived greyscale, which is a
    # pipeline failure worth being able to see in the artifact.
    channel_saturation = float(np.mean(array.max(axis=2) - array.min(axis=2)))

    clipped_low = float(np.mean(luminance <= 1.0))
    clipped_high = float(np.mean(luminance >= 254.0))

    border_width = max(1, int(min(height, width) * 0.06))
    border = np.concatenate(
        [
            luminance[:border_width, :].ravel(),
            luminance[-border_width:, :].ravel(),
            luminance[:, :border_width].ravel(),
            luminance[:, -border_width:].ravel(),
        ]
    )
    border_uniform_fraction = float(np.mean(np.abs(border - border.mean()) < 2.0))

    sharpness = _laplacian_variance(luminance)

    is_near_uniform = std < float(iqa_cfg.blank_std_threshold)
    is_blurry = sharpness < float(iqa_cfg.laplacian_blur_threshold)
    is_low_contrast = std < float(iqa_cfg.low_contrast_std)
    has_border_artifact = border_uniform_fraction > float(iqa_cfg.border_uniform_fraction)

    # Five defect flags, no colour penalty among them. Achievable penalty sums are
    # {0.0, 0.1, ..., 0.9}: ten discrete tiers, 0.1 apart.
    penalties = (
        0.40 * float(is_near_uniform)
        + 0.20 * float(is_blurry)
        + 0.10 * float(is_low_contrast)
        + 0.10 * float(has_border_artifact)
        + 0.10 * float(clipped_low + clipped_high > 0.20)
    )
    defect_tier = float(max(0.0, 1.0 - penalties))

    # Continuous refinement, for the same reason it was needed on the CheXpert side: a score taking
    # only ten distinct values cannot clear Go/No-Go's min_unique_values gate, so the signal would
    # be silently dropped from the candidate pool. sharpness_norm/contrast_norm reuse the SAME
    # frozen thresholds the discrete flags already use, and CONTINUOUS_WEIGHT sits below the 0.1 gap
    # between tiers, so a detected defect can never be out-ranked by sharpness alone — the
    # continuous term only orders images WITHIN one defect tier.
    sharpness_norm = float(np.tanh(sharpness / (2.0 * float(iqa_cfg.laplacian_blur_threshold))))
    contrast_norm = float(np.tanh(std / (2.0 * float(iqa_cfg.low_contrast_std))))
    continuous = 0.5 * (sharpness_norm + contrast_norm)
    composite = float(defect_tier * (1.0 - CONTINUOUS_WEIGHT) + CONTINUOUS_WEIGHT * continuous)

    return {
        "iqa_valid": True,
        "iqa_composite": composite,
        "iqa_sharpness": sharpness,
        "iqa_contrast_std": std,
        "iqa_mean_intensity": mean,
        "iqa_channel_saturation": channel_saturation,  # reported only
        "iqa_clipped_low_fraction": clipped_low,
        "iqa_clipped_high_fraction": clipped_high,
        "iqa_border_uniform_fraction": border_uniform_fraction,
        "iqa_is_near_uniform": bool(is_near_uniform),
        "iqa_is_blurry": bool(is_blurry),
        "iqa_is_low_contrast": bool(is_low_contrast),
        "iqa_has_border_artifact": bool(has_border_artifact),
    }


# ==============================================================================================
# Explainability — what replaces the anatomical boxes, and what it does NOT claim
# ==============================================================================================
#
# THE PROBLEM
#   CheXpert's explainability signal scores Grad-CAM mass falling inside a per-pathology anatomical
#   box: cardiomegaly belongs at the heart, pleural effusion at the costophrenic angles. Those boxes
#   encode real anatomical priors about WHERE a finding must appear in a standardised projection.
#
#   Dermoscopy has no such prior and HAM10000 ships NO lesion segmentation masks. There is no
#   ground truth for where the lesion is in any given frame, and inventing bounding boxes would be
#   fabricating clinical annotation that does not exist. So the question "did the model look at the
#   lesion?" is NOT answerable from this dataset, and this module does not pretend to answer it.
#
# WHAT IS ANSWERABLE, AND IS WHAT THIS SIGNAL ACTUALLY MEASURES
#   Two failure modes can be detected without knowing where the lesion is, because both are claims
#   about where attention is NOT, or about its shape:
#
#   1. PERIPHERAL ATTENTION (`explainability_peripheral_mass`, lower is better)
#      Dermoscopy frames carry known edge artifacts: the dark circular scope vignette, ruler
#      markings, gel bubbles at the rim — and, in this pipeline specifically, the letterbox padding
#      that 02_preprocess_images.py adds itself. Attention concentrated in the outer band is being
#      driven by something that is definitionally not the lesion, whatever the lesion happens to be.
#      This is a negative claim about a region known to be non-lesion, which needs no mask.
#      When the caller passes `content_box` (the un-padded region, computable exactly from the
#      source aspect ratio) the measure becomes exact rather than a generic band: mass landing on
#      our own padding is unambiguous evidence of an artifact-driven decision.
#
#   2. ATTENTION DISPERSION (`explainability_focus_area`, lower is better)
#      A classifier reading a discrete lesion produces compact attention. A classifier reading a
#      global property — overall colour cast, illumination, skin tone, a JPEG texture artifact —
#      produces attention smeared across the frame. Measured as the area of the smallest box holding
#      a fixed fraction of the Grad-CAM mass.
#
# THE LIMITATION THAT MAKES CALIBRATION NECESSARY, STATED PLAINLY
#   A genuinely large lesion filling the frame produces large focus_area and would look "diffuse" by
#   the raw statistic alone, so a fixed threshold on it would be arbitrary and wrong. That is why
#   `calibrate_against_reference` exists: a synthetic image is scored by how TYPICAL its attention
#   profile is against real images of the same class, where the label is known to be genuine. The
#   reference distribution, not an invented constant, is what makes the number mean something.
#
# WHAT THIS SIGNAL IS NOT
#   Not lesion localisation. Not evidence the diagnosis is correct. A model can attend compactly to
#   the wrong structure and score well here. It is a plausibility filter against two specific,
#   named artifact-driven failure modes, and it is reported as exactly that.


def peripheral_mass(
    cam: np.ndarray,
    border_fraction: float = 0.12,
    content_box: tuple[float, float, float, float] | None = None,
) -> float:
    """Fraction of Grad-CAM mass outside the image interior. Lower is better; range [0, 1].

    `content_box` as (x0, y0, x1, y1) fractions marks the real image content when letterbox padding
    was added; everything outside it is padding this pipeline created, so mass there is measured
    exactly. Without it, a symmetric border band of `border_fraction` is used as a generic stand-in
    for the scope vignette / ruler / rim region.
    """
    cam = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")

    height, width = cam.shape
    interior = np.zeros_like(cam, dtype=bool)
    if content_box is not None:
        x0, y0, x1, y1 = content_box
        interior[int(y0 * height) : max(int(y1 * height), 1), int(x0 * width) : max(int(x1 * width), 1)] = True
    else:
        band_y = max(1, int(round(height * border_fraction)))
        band_x = max(1, int(round(width * border_fraction)))
        interior[band_y : height - band_y, band_x : width - band_x] = True

    return float(1.0 - cam[interior].sum() / total)


def focus_area(cam: np.ndarray, mass_fraction: float = 0.80) -> float:
    """Area of the smallest axis-aligned box holding `mass_fraction` of the mass, as a fraction of
    the frame. Lower means more concentrated attention; range (0, 1].

    Rows and columns are grown independently outward from the centre of mass, each step taking
    whichever adjacent line carries more mass — the same construction the CheXpert path uses to
    derive a region, applied here to measure spread rather than to locate anything.
    """
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError(f"mass_fraction must be in (0, 1], got {mass_fraction}")
    cam = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")

    def span(profile: np.ndarray) -> int:
        profile = profile / profile.sum()
        centre = int(np.argmax(np.cumsum(profile) >= 0.5))
        low = high = centre
        captured = profile[centre]
        target = np.sqrt(mass_fraction)  # per-axis, so the 2-D box holds ~mass_fraction
        while captured < target and (low > 0 or high < len(profile) - 1):
            take_low = profile[low - 1] if low > 0 else -1.0
            take_high = profile[high + 1] if high < len(profile) - 1 else -1.0
            if take_low >= take_high:
                low -= 1
                captured += take_low
            else:
                high += 1
                captured += take_high
        return high - low + 1

    rows = span(cam.sum(axis=1))
    cols = span(cam.sum(axis=0))
    return float((rows / cam.shape[0]) * (cols / cam.shape[1]))


def compute_explainability_statistics(
    cam: np.ndarray,
    border_fraction: float = 0.12,
    mass_fraction: float = 0.80,
    content_box: tuple[float, float, float, float] | None = None,
) -> dict:
    """Both raw statistics plus an uncalibrated plausibility composite for one image.

    `explainability_plausibility` multiplies the two complements rather than averaging them, because
    the requirement is a conjunction: attention must be BOTH away from the frame edge AND compact.
    An image that is compact but sitting entirely on the padding should not be rescued by its
    compactness, and averaging would rescue it.
    """
    peripheral = peripheral_mass(cam, border_fraction=border_fraction, content_box=content_box)
    area = focus_area(cam, mass_fraction=mass_fraction)
    if np.isnan(peripheral) or np.isnan(area):
        plausibility = float("nan")
    else:
        plausibility = float((1.0 - peripheral) * (1.0 - area))
    return {
        "explainability_peripheral_mass": peripheral,
        "explainability_focus_area": area,
        "explainability_plausibility": plausibility,
    }


def calibrate_against_reference(value: float, reference_values: np.ndarray, higher_is_better: bool = True) -> float:
    """Typicality of `value` against a REAL-image reference distribution, in [0, 1].

    Returns the fraction of the reference distribution the value equals or beats, so the score is
    "how normal does this look compared with real images of this class" rather than a comparison
    against a constant nobody derived. With `higher_is_better=False` the direction is flipped, so a
    low peripheral mass or a small focus area scores high.

    This is what keeps the signal honest about the large-lesion case: if real images of a class
    genuinely produce diffuse attention, a synthetic image that also does is typical and scores
    well, instead of being penalised by a threshold that never fitted that class.
    """
    reference = np.asarray([v for v in np.asarray(reference_values, dtype=np.float64).ravel() if np.isfinite(v)])
    if reference.size == 0 or not np.isfinite(value):
        return float("nan")
    beaten = np.mean(reference <= value) if higher_is_better else np.mean(reference >= value)
    return float(beaten)


__all__ = [
    "CONTINUOUS_WEIGHT",
    "compute_iqa_scores",
    "peripheral_mass",
    "focus_area",
    "compute_explainability_statistics",
    "calibrate_against_reference",
]

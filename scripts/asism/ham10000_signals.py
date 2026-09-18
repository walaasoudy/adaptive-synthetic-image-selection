"""ASISM signals specialised for dermoscopy: colour-safe IQA and lesion-appropriate explainability.

Only the two signals whose CheXpert implementations encode radiograph-specific assumptions live
here. Similarity (DINOv2), uncertainty (MC dropout) and agreement are genuinely dataset-agnostic and
are still used from scripts/asism/signals.py unchanged.

scripts/asism/signals.py is not modified by this module. The CheXpert path keeps its behaviour
exactly; this is a parallel implementation selected by dataset, not a mutation of the shared one.
"""

from __future__ import annotations

from dataclasses import dataclass
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


# ---------------------------------------------------------------------------------------------
# Blur threshold calibration — from real gen_train images, never a constant
# ---------------------------------------------------------------------------------------------

BLUR_CALIBRATION_SOURCE_SPLIT = "gen_train"
BLUR_THRESHOLD_SOURCE_KEY = "laplacian_blur_threshold_source"


class UncalibratedIQAError(ValueError):
    """Raised when HAM10000 IQA is run without a gen_train-calibrated blur threshold."""


def measure_sharpness(image_path: Path) -> float:
    """Laplacian-variance sharpness on luminance — exactly the quantity compute_iqa_scores thresholds."""
    with Image.open(image_path) as image:
        luminance = np.asarray(image.convert("RGB").convert("L"), dtype=np.float32)
    return _laplacian_variance(luminance)


def calibrate_blur_threshold(
    sharpness_by_image: dict[str, float],
    split_name: str,
    split_image_ids,
    forbidden_image_ids,
    quantile: float,
    diagnosis_by_image: dict[str, str] | None = None,
) -> dict:
    """Blur threshold = lower `quantile` of real gen_train sharpness, with its evidence.

    Refuses (ReferenceLeakageError) when the source is not gen_train, when any measured image is not
    in the gen_train id list, or when any overlaps `forbidden_image_ids` (asism_tuning_heldout and
    final_eval_heldout). Returns the threshold together with the quantiles and the overall and
    per-class rejection rates it implies, so the choice is always reported with its consequence.
    """
    if split_name != BLUR_CALIBRATION_SOURCE_SPLIT:
        raise ReferenceLeakageError(f"blur threshold may only be calibrated on {BLUR_CALIBRATION_SOURCE_SPLIT}, not {split_name!r}")
    if not 0.0 < float(quantile) < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")
    forbidden = frozenset(str(item) for item in forbidden_image_ids)
    if not forbidden:
        raise ReferenceLeakageError("forbidden_image_ids is empty: cannot verify tuning/final-eval exclusion")
    allowed = frozenset(str(item) for item in split_image_ids)
    ids = [str(item) for item in sharpness_by_image]
    outside = sorted(set(ids) - allowed)
    if outside:
        raise ReferenceLeakageError(f"{len(outside)} measured image(s) are not in {split_name}, e.g. {outside[:3]}")
    leaked = sorted(set(ids) & forbidden)
    if leaked:
        raise ReferenceLeakageError(f"{len(leaked)} measured image(s) belong to a held-out split, e.g. {leaked[:3]}")

    values = np.asarray([sharpness_by_image[i] for i in ids], dtype=np.float64)
    if values.size < 100 or not np.all(np.isfinite(values)):
        raise ValueError(f"need >= 100 finite sharpness values, got {values.size}")
    threshold = float(np.quantile(values, float(quantile)))
    rejected = values < threshold

    per_class = {}
    if diagnosis_by_image is not None:
        labels = np.asarray([diagnosis_by_image[i] for i in ids])
        for label in sorted(set(labels)):
            mask = labels == label
            p5, p50, p95 = np.percentile(values[mask], [5, 50, 95])
            per_class[label] = {"n": int(mask.sum()), "rejection_rate": float(rejected[mask].mean()), "p5": p5, "median": p50, "p95": p95}

    probes = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95]
    return {
        "laplacian_blur_threshold": threshold,
        "rule": "lower_quantile",
        "quantile": float(quantile),
        "source_split": split_name,
        "n_images": int(values.size),
        "sharpness_quantiles": {str(q): float(np.quantile(values, q)) for q in probes},
        "overall_rejection_rate": float(rejected.mean()),
        "per_class": per_class,
    }


BORDER_REGIONS = ("canvas", "content_box")
BORDER_THRESHOLD_SOURCE_KEY = "border_uniform_fraction_source"
BORDER_BAND_FRACTION = 0.06


def border_uniform_fraction(luminance: np.ndarray, band_fraction: float = BORDER_BAND_FRACTION) -> float:
    """Fraction of the outer band's pixels within 2 grey levels of the band's mean luminance.

    High values mean a flat frame edge. Applied to the whole CANVAS, 60% of the band on a 600x450
    letterboxed image is padding this pipeline added, so the score mostly measures whether the
    image's edge brightness happens to equal the padding grey. Applied to the CONTENT box, it
    measures the real image's own edge.
    """
    height, width = luminance.shape
    band = max(1, int(min(height, width) * band_fraction))
    edge = np.concatenate([luminance[:band, :].ravel(), luminance[-band:, :].ravel(), luminance[:, :band].ravel(), luminance[:, -band:].ravel()])
    return float(np.mean(np.abs(edge - edge.mean()) < 2.0))


def border_region_luminance(luminance: np.ndarray, region: str, content_box=None) -> np.ndarray:
    if region == "canvas":
        return luminance
    if region != "content_box":
        raise ValueError(f"border_region must be one of {BORDER_REGIONS}; got {region!r}")
    if content_box is None:
        raise UncalibratedIQAError("border_region=content_box requires the image's persisted content_box; refusing the canvas fallback")
    from scripts.utils.ham10000_geometry import validate_content_box

    x0, y0, x1, y1 = validate_content_box(content_box)
    height, width = luminance.shape
    return luminance[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)]


def measure_border_uniformity(image_path: Path, region: str, content_box=None) -> float:
    """The border score compute_iqa_scores thresholds, for the given region."""
    with Image.open(image_path) as image:
        luminance = np.asarray(image.convert("RGB").convert("L"), dtype=np.float32)
    return border_uniform_fraction(border_region_luminance(luminance, region, content_box))


def calibrate_border_threshold(
    uniformity_by_image: dict[str, float],
    region: str,
    split_name: str,
    split_image_ids,
    forbidden_image_ids,
    quantile: float,
    diagnosis_by_image: dict[str, str] | None = None,
) -> dict:
    """Border threshold = UPPER `quantile` of real gen_train border uniformity — the blur rule's
    mirror image (a border defect is an unusually HIGH score). Same leakage refusals as the blur
    calibration; reports overall and per-class flag rates at the resulting threshold."""
    if region not in BORDER_REGIONS:
        raise ValueError(f"region must be one of {BORDER_REGIONS}")
    negated = {k: -float(v) for k, v in uniformity_by_image.items()}
    lower = calibrate_blur_threshold(negated, split_name, split_image_ids, forbidden_image_ids, 1.0 - float(quantile), diagnosis_by_image)
    values = np.asarray([float(v) for v in uniformity_by_image.values()])
    labels = np.asarray([diagnosis_by_image[k] for k in uniformity_by_image]) if diagnosis_by_image else None
    threshold = float(np.quantile(values, float(quantile)))
    flagged = values > threshold
    per_class = {}
    if labels is not None:
        for label in sorted(set(labels)):
            mask = labels == label
            per_class[label] = {"n": int(mask.sum()), "flag_rate": float(flagged[mask].mean()), "median": float(np.median(values[mask])), "p95": float(np.percentile(values[mask], 95))}
    return {
        "border_uniform_fraction": threshold,
        "rule": "upper_quantile",
        "quantile": float(quantile),
        "region": region,
        "source_split": lower["source_split"],
        "n_images": lower["n_images"],
        "uniformity_quantiles": {str(q): float(np.quantile(values, q)) for q in (0.5, 0.9, 0.95, 0.98, 0.99)},
        "overall_flag_rate": float(flagged.mean()),
        "per_class": per_class,
    }


def resolve_iqa_config(config, calibration: dict, border_calibration: dict | None = None):
    """A copy of the HAM10000 stage-3 config with the blur threshold filled from `calibration`.

    Refuses a calibration from another split or produced with a different quantile than the config
    prescribes, so a stale or hand-edited artifact cannot silently set the threshold.
    """
    from omegaconf import OmegaConf

    rule = config.signals.iqa.blur_calibration
    if calibration.get("source_split") != str(rule.source_split) or str(rule.source_split) != BLUR_CALIBRATION_SOURCE_SPLIT:
        raise ReferenceLeakageError(f"blur calibration source {calibration.get('source_split')!r} is not {BLUR_CALIBRATION_SOURCE_SPLIT}")
    if calibration.get("rule") != str(rule.rule) or abs(float(calibration.get("quantile", -1)) - float(rule.quantile)) > 1e-12:
        raise ValueError(f"calibration rule/quantile {calibration.get('rule')}/{calibration.get('quantile')} != config {rule.rule}/{rule.quantile}")
    resolved = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    resolved.signals.iqa.laplacian_blur_threshold = float(calibration["laplacian_blur_threshold"])
    resolved.signals.iqa[BLUR_THRESHOLD_SOURCE_KEY] = f"{calibration['source_split']}_{calibration['rule']}_{calibration['quantile']}"

    border_rule = config.signals.iqa.get("border_calibration")
    region = str(config.signals.iqa.get("border_region", "canvas"))
    if region not in BORDER_REGIONS:
        raise ValueError(f"signals.iqa.border_region must be one of {BORDER_REGIONS}; got {region!r}")
    if border_rule is not None and bool(border_rule.enabled):
        if border_calibration is None:
            raise UncalibratedIQAError("border_calibration.enabled=true but no border calibration artifact was supplied")
        expected = {"source_split": BLUR_CALIBRATION_SOURCE_SPLIT, "rule": str(border_rule.rule), "region": str(border_rule.region)}
        mismatch = {k: (border_calibration.get(k), v) for k, v in expected.items() if border_calibration.get(k) != v}
        if mismatch or abs(float(border_calibration.get("quantile", -1)) - float(border_rule.quantile)) > 1e-12:
            raise ReferenceLeakageError(f"border calibration does not match the configured rule: {mismatch or 'quantile'}")
        if region != str(border_rule.region):
            raise ValueError(f"border_region {region!r} != border_calibration.region {border_rule.region!r}")
        resolved.signals.iqa.border_uniform_fraction = float(border_calibration["border_uniform_fraction"])
        resolved.signals.iqa[BORDER_THRESHOLD_SOURCE_KEY] = f"{border_calibration['source_split']}_{border_calibration['rule']}_{border_calibration['quantile']}_{border_calibration['region']}"
    else:
        resolved.signals.iqa[BORDER_THRESHOLD_SOURCE_KEY] = "inherited_constant_uncalibrated"
    return resolved


def compute_iqa_scores(image_path: Path, config, content_box=None) -> dict:
    """Technical quality for one dermoscopy image. Higher `iqa_composite` is better, range [0, 1].

    `config` must be a HAM10000 stage-3 config RESOLVED by resolve_iqa_config: the blur threshold
    comes from the gen_train calibration and nowhere else. An unresolved config — including the
    CheXpert stage3_asism.yaml with its radiograph threshold of 40 — raises UncalibratedIQAError.
    """
    iqa_cfg = config.signals.iqa
    if iqa_cfg.get("laplacian_blur_threshold") is None or iqa_cfg.get(BLUR_THRESHOLD_SOURCE_KEY) is None:
        raise UncalibratedIQAError(
            "HAM10000 IQA requires a blur threshold calibrated on gen_train; resolve the config with "
            "resolve_iqa_config(config, calibration) first"
        )
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

    border_region = str(iqa_cfg.get("border_region", "canvas"))
    border_uniformity = border_uniform_fraction(border_region_luminance(luminance, border_region, content_box))

    sharpness = _laplacian_variance(luminance)

    is_near_uniform = std < float(iqa_cfg.blank_std_threshold)
    is_blurry = sharpness < float(iqa_cfg.laplacian_blur_threshold)
    is_low_contrast = std < float(iqa_cfg.low_contrast_std)
    has_border_artifact = border_uniformity > float(iqa_cfg.border_uniform_fraction)

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
        "iqa_border_uniform_fraction": border_uniformity,
        "iqa_border_region": border_region,
        "iqa_border_threshold_source": str(iqa_cfg.get(BORDER_THRESHOLD_SOURCE_KEY, "unresolved")),
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
# WHAT IS MEASURED
#   1. PERIPHERAL ATTENTION (`explainability_peripheral_mass`)
#      Fraction of Grad-CAM mass outside the image content. When the exact letterbox `content_box`
#      is supplied (persisted per image by 02_preprocess_images.py), "outside" means the padding
#      THIS PIPELINE added: mass there is attention on pixels that are definitionally not the
#      lesion, whatever and wherever the lesion is. Without a content box, a generic border band is
#      used as a stand-in for scope vignette / ruler / rim; that fallback is recorded as such and is
#      NOT selection-eligible (see the guard below).
#
#   2. ATTENTION DISPERSION (`explainability_focus_area`)
#      Area of a box holding a fixed fraction of the Grad-CAM mass. Reported as a diagnostic only.
#
# THE LARGE-LESION LIMITATION, STATED PLAINLY — AND WHAT WAS DONE ABOUT IT
#   focus_area is confounded with lesion size: a genuinely large lesion produces widely spread
#   attention, so a large lesion looks "diffuse". Removing that confound requires knowing lesion
#   size, which requires segmentation masks HAM10000 does not have. Comparing against real images
#   of the SAME CLASS does not remove it either: lesion size varies widely within a class, so a
#   large naevus is still atypical against a mostly-small naevus reference. No defensible
#   size-conditioning exists without masks, so none is invented here. Instead:
#     * focus_area is REPORTED but is NOT selection-eligible, raw or calibrated;
#     * the only selection-eligible explainability feature is the calibrated typicality of
#       peripheral mass measured with an exact content box. With the exact box, "outside" is our
#       padding, which a lesion cannot occupy, so a larger lesion does not by itself raise it. The
#       residual coupling is Grad-CAM's coarse grid (16x16 at 512 px for DenseNet121): attention on
#       a lesion touching the content edge can bleed into the adjacent padding cell. That residual
#       is stated, not hidden.
#
# WHAT THIS SIGNAL IS NOT
#   Not lesion localisation. Not evidence the diagnosis is correct. A model can attend inside the
#   content area to the wrong structure and score perfectly here. It is a plausibility filter
#   against one specific, named failure mode — a decision driven by padding — and is reported as
#   exactly that.

MIN_REFERENCE_SIZE = 20

# Where real-image reference statistics may come from.
#
#   gen_train — PERMITTED. It is the pool the generator imitates, so "is this synthetic image's
#       attention typical of real images?" is asked against exactly the images it was modelled on.
#       It is disjoint from classifier_train, so the Grad-CAM model has not been fitted to these
#       images and their attention is not the optimistic, memorised attention of training data.
#   final_eval_heldout — FORBIDDEN, always. It is touched by Stage 5 only.
#   everything else — not permitted: classifier_train biases the reference (the CAM model trained
#       on it); classifier_val selects checkpoints; asism_tuning_heldout is the Go/No-Go evidence,
#       and calibrating on it would reuse that evidence to build the thing it evaluates.
PERMITTED_REFERENCE_SPLITS = frozenset({"gen_train"})
FORBIDDEN_REFERENCE_SPLITS = frozenset({"final_eval_heldout"})

EXACT_CONTENT_BOX_SOURCE = "preprocessing_manifest"
GENERIC_BAND_SOURCE = "generic_border_band"

# The ONLY explainability column a Stage 3 selection model may consume.
SELECTION_ELIGIBLE_EXPLAINABILITY_FEATURES = frozenset({"explainability_calibrated_typicality"})


class ReferenceLeakageError(ValueError):
    """Raised when calibration reference data comes from a split that must not be used for it."""


class UncalibratedExplainabilityError(ValueError):
    """Raised when a raw / uncalibrated / non-exact explainability value is offered for selection."""


def _axis_coverage(n_cells: int, low: float, high: float) -> np.ndarray:
    """Fraction of each of n_cells equal-width cells on [0, 1] that lies inside [low, high)."""
    edges = np.arange(n_cells + 1, dtype=np.float64) / n_cells
    overlap = np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
    return np.clip(overlap * n_cells, 0.0, 1.0)


def content_weights(shape: tuple[int, int], content_box: tuple[float, float, float, float]) -> np.ndarray:
    """Per-cell fraction of a CAM grid that lies inside `content_box`, in [0, 1].

    A CAM cell straddling the content edge is SPLIT by area, not assigned wholesale to one side.
    Truncating the box edge with int() instead — the previous implementation — counted a cell that
    was 87.5% padding as entirely content on a 7x7 grid, reporting 0.0 peripheral mass for
    attention sitting almost wholly on padding. Area weighting makes the result exact for a CAM
    that is piecewise-constant per cell, which is what an upsampled-free Grad-CAM is.
    """
    from scripts.utils.ham10000_geometry import validate_content_box

    x0, y0, x1, y1 = validate_content_box(content_box)
    height, width = shape
    return np.outer(_axis_coverage(height, y0, y1), _axis_coverage(width, x0, x1))


def peripheral_mass(
    cam: np.ndarray,
    border_fraction: float = 0.12,
    content_box: tuple[float, float, float, float] | None = None,
) -> float:
    """Fraction of Grad-CAM mass outside the image content. Lower means less attention on padding
    (or on the generic border band); range [0, 1].

    `content_box` (x0, y0, x1, y1), normalised and half-open, is the exact letterbox layout. Without
    it, a symmetric band of `border_fraction` is treated as "outside" — a generic stand-in only.
    """
    cam = np.maximum(np.asarray(cam, dtype=np.float64), 0.0)
    total = float(cam.sum())
    if total <= 0.0:
        return float("nan")
    box = content_box if content_box is not None else (border_fraction, border_fraction, 1.0 - border_fraction, 1.0 - border_fraction)
    inside = float((cam * content_weights(cam.shape, box)).sum())
    return float(min(1.0, max(0.0, 1.0 - inside / total)))


def focus_area(cam: np.ndarray, mass_fraction: float = 0.80) -> float:
    """Approximate dispersion: area of a box grown to hold about `mass_fraction` of the mass, as a
    fraction of the frame. Lower means more concentrated attention; range (0, 1].

    Rows and columns are grown independently and greedily outward from the median line, each to
    sqrt(mass_fraction) of its marginal mass. This is NOT the smallest such box, and the 2-D mass
    inside is only approximately `mass_fraction` (exact for separable CAMs; higher for multi-blob
    CAMs, e.g. ~0.90 for two diagonal blobs). It is a dispersion diagnostic, not a region estimate.
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
        target = np.sqrt(mass_fraction)
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
    """RAW, UNCALIBRATED statistics for one CAM. None of these keys is selection-eligible.

    `explainability_raw_plausibility_uncalibrated` multiplies the two complements (a conjunction:
    away from padding AND compact). It is kept as a human-readable diagnostic. Its name says what it
    is, and `assert_selection_features_allowed` rejects it, because it carries the focus_area
    lesion-size confound and has no reference distribution behind it.
    """
    peripheral = peripheral_mass(cam, border_fraction=border_fraction, content_box=content_box)
    area = focus_area(cam, mass_fraction=mass_fraction)
    if np.isnan(peripheral) or np.isnan(area):
        raw_plausibility = float("nan")
    else:
        raw_plausibility = float((1.0 - peripheral) * (1.0 - area))
    return {
        "explainability_peripheral_mass": peripheral,
        "explainability_focus_area": area,
        "explainability_raw_plausibility_uncalibrated": raw_plausibility,
        "explainability_content_box_source": EXACT_CONTENT_BOX_SOURCE if content_box is not None else GENERIC_BAND_SOURCE,
    }


@dataclass(frozen=True)
class ReferenceDistribution:
    """Real-image values of one statistic, for one class, from one PERMITTED split.

    Construct only through `build_reference_distribution`, which is where the leakage checks live.
    """

    statistic: str
    diagnosis: str
    split_name: str
    values: tuple[float, ...]
    image_ids: tuple[str, ...]
    cam_model_id: str
    # True only when the CAMs came from the TRAINED HAM10000 classifier. A reference built from an
    # untrained (e.g. random-init plumbing) model can exercise the code but must never calibrate a
    # selection feature; selection_explainability_column refuses it.
    scientific: bool

    @property
    def size(self) -> int:
        return len(self.values)


def load_final_eval_image_ids(split_dir: Path) -> frozenset[str]:
    """The image_ids of final_eval_heldout, read from the frozen split CSV — ids only, no pixels."""
    import pandas as pd

    path = Path(split_dir) / "final_eval_heldout.csv"
    if not path.is_file():
        raise FileNotFoundError(f"cannot prove calibration is disjoint from final eval: {path} is missing")
    return frozenset(pd.read_csv(path, usecols=["image_id"])["image_id"].astype(str))


def build_reference_distribution(
    values,
    image_ids,
    split_name: str,
    diagnosis: str,
    statistic: str,
    final_eval_image_ids,
    *,
    cam_model_id: str,
    cam_model_trained: bool,
    min_size: int = MIN_REFERENCE_SIZE,
) -> ReferenceDistribution:
    """Build a calibration reference, refusing anything that could leak evaluation data into it.

    Two independent checks, because either alone can be defeated by a mislabelled argument:
      * by NAME: split_name must be permitted and never final_eval_heldout;
      * by CONTENT: no image_id may appear in final_eval_heldout's id list. `final_eval_image_ids`
        is required and must be non-empty — an empty set would make the check vacuous.
    """
    if split_name in FORBIDDEN_REFERENCE_SPLITS:
        raise ReferenceLeakageError(f"{split_name!r} must never be used for calibration/reference statistics")
    if split_name not in PERMITTED_REFERENCE_SPLITS:
        raise ReferenceLeakageError(
            f"{split_name!r} is not a permitted reference split; permitted: {sorted(PERMITTED_REFERENCE_SPLITS)}"
        )
    final_eval = frozenset(str(item) for item in final_eval_image_ids)
    if not final_eval:
        raise ReferenceLeakageError("final_eval_image_ids is empty: disjointness from final eval cannot be verified")

    ids = tuple(str(item) for item in image_ids)
    array = np.asarray(values, dtype=np.float64).ravel()
    if len(ids) != array.size:
        raise ValueError(f"{array.size} values but {len(ids)} image_ids")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate image_ids in reference")
    leaked = sorted(set(ids) & final_eval)
    if leaked:
        raise ReferenceLeakageError(
            f"{len(leaked)} reference image(s) belong to final_eval_heldout (e.g. {leaked[:3]}); "
            "final eval must never enter calibration"
        )

    finite = np.isfinite(array)
    kept_values = tuple(float(v) for v in array[finite])
    kept_ids = tuple(image_id for image_id, ok in zip(ids, finite) if ok)
    if len(kept_values) < min_size:
        raise ValueError(
            f"reference for {statistic}/{diagnosis} has {len(kept_values)} finite values; need >= {min_size}"
        )
    if not str(cam_model_id).strip():
        raise ValueError("cam_model_id is required so every reference names the model its CAMs came from")
    return ReferenceDistribution(
        statistic, str(diagnosis), split_name, kept_values, kept_ids, str(cam_model_id), bool(cam_model_trained)
    )


def cam_model_identity(run_manifest: dict, model_path: Path, reference_split: str = "gen_train") -> dict:
    """Validate the classifier whose Grad-CAMs will build a calibration reference; return its identity.

    This is the ONLY way to obtain `cam_model_id` / `cam_model_trained=True` for a scientific
    reference. It refuses a model that:
      * is not a HAM10000 softmax + CrossEntropy classifier;
      * saw ANY synthetic image (the reference describes real attention);
      * was trained or selected on the reference split itself — its attention on those images is the
        memorised attention of its own training data, and the reference would be optimistic.
    The CheXpert auxiliary classifier convention trains on gen_train, which is exactly the reference
    split here, so that convention cannot be copied unchanged (docs/ham10000_gradcam_calibration.md).
    """
    import hashlib

    data = run_manifest.get("data", {})
    problems = []
    if run_manifest.get("dataset") != "ham10000" or run_manifest.get("loss") != "cross_entropy":
        problems.append("not a HAM10000 cross_entropy classifier")
    if int(data.get("synthetic_images", -1)) != 0:
        problems.append(f"trained with {data.get('synthetic_images')} synthetic images; the CAM model must be real-only")
    used = {data.get("real_train_split"), data.get("selection_split")}
    if reference_split in used:
        problems.append(f"trained/selected on the reference split {reference_split!r}")
    if "final_eval_heldout" in used:
        problems.append("touched final_eval_heldout")
    if problems:
        raise ReferenceLeakageError("CAM model unsuitable for a calibration reference: " + "; ".join(problems))
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"CAM model checkpoint not found: {model_path}")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return {
        "cam_model_id": f"ham10000-classifier:{digest}",
        "cam_model_trained": True,
        "trained_on_split": data.get("real_train_split"),
        "selected_on_split": data.get("selection_split"),
        "reference_split": reference_split,
    }


def tie_report(values) -> dict:
    """How discrete a reference distribution is — the check to run on the TRAINED classifier's CAMs
    before trusting the two-sided p-value's resolution.

    Heavy ties (e.g. many CAMs with exactly zero peripheral mass) make mid-rank typicality coarse:
    every tied value gets the same score. This reports the size of that effect; it changes nothing.
    """
    array = np.asarray([v for v in np.asarray(values, dtype=np.float64).ravel() if np.isfinite(v)])
    if array.size == 0:
        return {"n": 0}
    _, counts = np.unique(array, return_counts=True)
    return {
        "n": int(array.size),
        "unique_values": int(counts.size),
        "unique_fraction": float(counts.size / array.size),
        "largest_tie_group_fraction": float(counts.max() / array.size),
        "exact_zero_fraction": float(np.mean(array == 0.0)),
        "near_zero_fraction_lt_1e-3": float(np.mean(np.abs(array) < 1e-3)),
    }


def calibrate_against_reference(value: float, reference: ReferenceDistribution) -> float:
    """Two-sided TYPICALITY of `value` against a real-image reference, in (0, 1].

    Formulation — a two-sided conformal p-value with mid-rank tie handling:

        F = (#{r < v} + 0.5 * #{r == v} + 0.5) / (n + 1)
        typicality = min(1, 2 * min(F, 1 - F))

    F is the smoothed mid-rank position of v among the n reference values. The score is 1 at the
    reference median and falls toward 1/(n+1) at EITHER extreme, so a value beyond every real image
    — in either direction — is scored as atypical rather than as "better than real". The +0.5 and
    n+1 terms are the standard conformal smoothing: the score is never exactly 0, and under
    exchangeability with the reference it is (conservatively) uniformly distributed, so it can be
    read as a p-value for "this value came from the real distribution". No fixed threshold enters.

    The previous one-sided percentile gave 1.0 to any value past the best real image, rewarding
    exactly the out-of-distribution extremes a typicality score exists to catch.
    """
    if not isinstance(reference, ReferenceDistribution):
        raise TypeError(
            "calibrate_against_reference requires a ReferenceDistribution from build_reference_distribution; "
            "raw arrays are refused because they bypass the reference-split leakage checks"
        )
    if value is None or not np.isfinite(value):
        return float("nan")
    reference_values = np.asarray(reference.values, dtype=np.float64)
    n = reference_values.size
    below = float(np.sum(reference_values < value))
    equal = float(np.sum(reference_values == value))
    position = (below + 0.5 * equal + 0.5) / (n + 1)
    return float(min(1.0, 2.0 * min(position, 1.0 - position)))


def calibrate_explainability(raw: dict, diagnosis: str, references: dict[str, ReferenceDistribution]) -> dict:
    """Calibrated explainability for one image; the only producer of a selection-eligible feature.

    `references` maps statistic name -> ReferenceDistribution for this image's intended class.
    `explainability_calibrated_typicality` is the typicality of peripheral mass, and is NaN unless
    the raw statistic was measured with the exact content box. Focus-area typicality is reported
    alongside but, for the lesion-size reason documented above, is not part of the eligible feature.
    """
    peripheral_reference = references.get("explainability_peripheral_mass")
    if peripheral_reference is None:
        raise ValueError("a peripheral_mass reference distribution is required")
    for name, reference in references.items():
        if reference.statistic != name:
            raise ValueError(f"reference keyed {name!r} holds statistic {reference.statistic!r}")
        if reference.cam_model_id != peripheral_reference.cam_model_id:
            raise ValueError("all references for one image must come from the same CAM model")
        if reference.diagnosis != str(diagnosis):
            raise ValueError(f"reference for {name} is class {reference.diagnosis!r}, image is {diagnosis!r}")

    exact = raw.get("explainability_content_box_source") == EXACT_CONTENT_BOX_SOURCE
    peripheral_typicality = calibrate_against_reference(raw["explainability_peripheral_mass"], peripheral_reference)
    focus_reference = references.get("explainability_focus_area")
    focus_typicality = (
        calibrate_against_reference(raw["explainability_focus_area"], focus_reference) if focus_reference else float("nan")
    )
    return {
        "explainability_peripheral_typicality": peripheral_typicality,
        "explainability_focus_typicality_diagnostic_only": focus_typicality,
        "explainability_calibrated_typicality": peripheral_typicality if exact else float("nan"),
        "explainability_calibrated": True,
        "explainability_content_box_exact": bool(exact),
        "explainability_reference_split": peripheral_reference.split_name,
        "explainability_reference_diagnosis": peripheral_reference.diagnosis,
        "explainability_reference_n": peripheral_reference.size,
        "explainability_reference_cam_model": peripheral_reference.cam_model_id,
        "explainability_reference_scientific": bool(peripheral_reference.scientific),
    }


def assert_selection_features_allowed(feature_names) -> None:
    """Stage 3 guard: refuse any explainability feature that is raw, uncalibrated or diagnostic-only.

    Any column starting with `explainability_` must be in SELECTION_ELIGIBLE_EXPLAINABILITY_FEATURES.
    Non-explainability features pass through untouched.
    """
    rejected = sorted(
        name
        for name in feature_names
        if str(name).startswith("explainability_") and name not in SELECTION_ELIGIBLE_EXPLAINABILITY_FEATURES
    )
    if rejected:
        raise UncalibratedExplainabilityError(
            f"explainability feature(s) not eligible for selection: {rejected}. Only "
            f"{sorted(SELECTION_ELIGIBLE_EXPLAINABILITY_FEATURES)} may be used, and only as produced by "
            "calibrate_explainability against a permitted reference split"
        )


def selection_explainability_column(frame):
    """Return the selection-eligible explainability column from a scores frame, verifying provenance.

    Refuses a frame whose rows were not calibrated, were calibrated against a non-permitted split,
    or lack the eligible column entirely — so the raw composite can never be substituted silently.
    """
    required = {
        "explainability_calibrated_typicality",
        "explainability_calibrated",
        "explainability_reference_split",
        "explainability_reference_scientific",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise UncalibratedExplainabilityError(f"scores frame is missing calibrated explainability columns: {missing}")
    if not bool(frame["explainability_calibrated"].astype(bool).all()):
        raise UncalibratedExplainabilityError("some rows were not calibrated")
    if not bool(frame["explainability_reference_scientific"].astype(bool).all()):
        raise UncalibratedExplainabilityError(
            "some rows were calibrated against a reference from an untrained CAM model; rebuild the "
            "reference with the trained HAM10000 classifier on gen_train before selection"
        )
    splits = set(frame["explainability_reference_split"].astype(str))
    if not splits <= PERMITTED_REFERENCE_SPLITS:
        raise ReferenceLeakageError(f"rows calibrated against non-permitted split(s): {sorted(splits - PERMITTED_REFERENCE_SPLITS)}")
    return frame["explainability_calibrated_typicality"]


__all__ = [
    "CONTINUOUS_WEIGHT",
    "MIN_REFERENCE_SIZE",
    "PERMITTED_REFERENCE_SPLITS",
    "FORBIDDEN_REFERENCE_SPLITS",
    "EXACT_CONTENT_BOX_SOURCE",
    "GENERIC_BAND_SOURCE",
    "SELECTION_ELIGIBLE_EXPLAINABILITY_FEATURES",
    "ReferenceLeakageError",
    "UncalibratedExplainabilityError",
    "ReferenceDistribution",
    "compute_iqa_scores",
    "UncalibratedIQAError",
    "measure_sharpness",
    "calibrate_blur_threshold",
    "BORDER_REGIONS",
    "border_uniform_fraction",
    "measure_border_uniformity",
    "calibrate_border_threshold",
    "resolve_iqa_config",
    "content_weights",
    "peripheral_mass",
    "focus_area",
    "compute_explainability_statistics",
    "load_final_eval_image_ids",
    "build_reference_distribution",
    "calibrate_against_reference",
    "cam_model_identity",
    "tie_report",
    "calibrate_explainability",
    "assert_selection_features_allowed",
    "selection_explainability_column",
]

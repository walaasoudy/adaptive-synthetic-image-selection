"""ASISM signals specialised for dermoscopy: colour-safe IQA, lesion-appropriate explainability,
and the single-label forms of similarity, uncertainty and agreement.

WHY ALL FIVE LIVE HERE. The first two were ported because their CheXpert implementations encode
radiograph-specific assumptions (achromatic images, anatomical region priors). The other three are
here for a different reason, discovered when they were about to be reused as-is: they are not
dataset-agnostic either. Every one of them is written against CheXpert's ELEVEN INDEPENDENT BINARY
labels — reference selection walks a Jaccard neighbourhood over label SETS, uncertainty averages a
per-label Bernoulli entropy, agreement averages probabilities over an intended positive SET and
penalises "unintended" labels that a sigmoid head can raise independently. HAM10000 has ONE
mutually exclusive class per image under a softmax. Running the CheXpert versions on a softmax row
would not crash; it would silently compute a different quantity (see each function's docstring for
the specific one). So they are reimplemented, not reused.

scripts/asism/signals.py is not modified by this module. The CheXpert path keeps its behaviour
exactly; this is a parallel implementation selected by dataset, not a mutation of the shared one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
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


# ----------------------------------------------------------------------------------------------
# Persisting a reference — and rebuilding it through the same guards, never around them
# ----------------------------------------------------------------------------------------------
#
# The reference is computed on a GPU run and consumed by later runs, so it has to survive as a file.
# The risk that creates is that a loader becomes a second, unguarded constructor: a JSON file could
# name any split, carry any image ids and claim any cam_model_id. `references_from_artifact`
# therefore does not reconstruct ReferenceDistribution objects directly — it calls
# `build_reference_distribution` again, with the CURRENT final_eval_heldout id list, so a stored
# reference is re-checked for leakage on every load rather than trusted because it was checked once.

REFERENCE_STATISTICS = ("explainability_peripheral_mass", "explainability_focus_area")


def reference_artifact_payload(references: dict, extra: dict | None = None) -> dict:
    """Serialise {(statistic, diagnosis): ReferenceDistribution} into a JSON-safe artifact."""
    entries = []
    model_ids = set()
    for (statistic, diagnosis), reference in sorted(references.items()):
        if reference.statistic != statistic or reference.diagnosis != diagnosis:
            raise ValueError(
                f"reference keyed ({statistic}, {diagnosis}) holds "
                f"({reference.statistic}, {reference.diagnosis})"
            )
        model_ids.add(reference.cam_model_id)
        entries.append(
            {
                "statistic": reference.statistic,
                "diagnosis": reference.diagnosis,
                "split_name": reference.split_name,
                "scientific": bool(reference.scientific),
                "n": reference.size,
                "values": [float(value) for value in reference.values],
                "image_ids": list(reference.image_ids),
            }
        )
    if len(model_ids) != 1:
        raise ValueError(
            f"a reference artifact must name exactly one CAM model, got {sorted(model_ids)}"
        )
    return {
        "artifact": "ham10000_explainability_reference",
        "schema_version": 1,
        "cam_model_id": model_ids.pop(),
        "statistics": list(REFERENCE_STATISTICS),
        "references": entries,
        **(extra or {}),
    }


def references_from_artifact(
    payload: dict, final_eval_image_ids, min_size: int = MIN_REFERENCE_SIZE
) -> dict:
    """{diagnosis: {statistic: ReferenceDistribution}} — re-verified, not merely deserialised.

    Every entry goes back through `build_reference_distribution`, so a stored reference that names a
    forbidden split, overlaps final_eval_heldout, or falls below `min_size` fails HERE, at load
    time, rather than silently calibrating a selection feature.

    `min_size` is an argument rather than whatever the build run happened to use, because the
    consumer is the one that has to live with a thin reference: a smoke run may lower it
    deliberately, and a production run must not inherit a lowered value from the file it is reading.
    """
    if payload.get("artifact") != "ham10000_explainability_reference":
        raise ValueError(f"not a HAM10000 explainability reference artifact: {payload.get('artifact')!r}")
    cam_model_id = str(payload["cam_model_id"])

    references: dict[str, dict] = {}
    for entry in payload["references"]:
        reference = build_reference_distribution(
            entry["values"],
            entry["image_ids"],
            str(entry["split_name"]),
            str(entry["diagnosis"]),
            str(entry["statistic"]),
            final_eval_image_ids,
            cam_model_id=cam_model_id,
            cam_model_trained=bool(entry["scientific"]),
            min_size=int(min_size),
        )
        references.setdefault(reference.diagnosis, {})[reference.statistic] = reference
    return references


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


# ==============================================================================================
# Similarity — realism against real images OF THE SAME LESION CLASS, and memorisation kept apart
# ==============================================================================================
#
# WHAT THE CHEXPERT VERSION DOES AND WHY IT DOES NOT TRANSFER
#   `signals.hierarchical_reference_indices` picks a synthetic image's real references by walking a
#   four-tier hierarchy over label SETS: exact set match -> Jaccard >= 0.5 neighbourhood -> any
#   shared positive -> everything. Tiers 2 and 3 exist because CheXpert labels CO-OCCUR: an image
#   intending {Cardiomegaly, Edema} has a meaningful partial match in {Cardiomegaly, Effusion}.
#
#   HAM10000's seven classes are MUTUALLY EXCLUSIVE. Every intended set has exactly one element, so
#   Jaccard against another single-element set is 1.0 for the same class and 0.0 for any other, and
#   "shares a positive" is identical to "is the same class". Tiers 1, 2 and 3 therefore collapse
#   onto one another, and running the CheXpert code would report a tier name ("tier2_neighbourhood")
#   that describes a partial-overlap relationship this dataset cannot express. Two honest tiers
#   remain, and they are named for what they are.
#
# WHY A CLASS-AGNOSTIC FALLBACK STILL EXISTS
#   Not for the rare classes as such — every HAM10000 class has enough real gen_train images to
#   reference. It exists so an unusable per-class pool (a class whose preprocessed reference images
#   are missing on this machine) produces a SCORED row carrying its tier, rather than a silent gap
#   in the candidate table that later joins would turn into a dropped candidate. The tier column is
#   what makes the weaker comparison visible; a tier-2 row is a measurement against "real
#   dermoscopy in general", not against the class, and must be read as such.

SIMILARITY_TIER_SAME_CLASS = "tier1_same_class"
SIMILARITY_TIER_CLASS_AGNOSTIC = "tier2_class_agnostic"


def single_label_reference_indices(
    diagnosis: str,
    reference_diagnoses: list[str],
    min_same_class: int,
) -> tuple[np.ndarray, str]:
    """Real-reference rows for one synthetic image: same class when there are enough, else all.

    Returns (indices, tier). `min_same_class` is a floor on the POOL, not on k: a top-k mean taken
    over a three-image pool is dominated by whichever three images happen to be present.
    """
    same = np.array(
        [index for index, value in enumerate(reference_diagnoses) if value == diagnosis], dtype=int
    )
    if len(same) >= int(min_same_class):
        return same, SIMILARITY_TIER_SAME_CLASS
    return np.arange(len(reference_diagnoses), dtype=int), SIMILARITY_TIER_CLASS_AGNOSTIC


def compute_similarity_scores(
    synthetic_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_diagnoses: list[str],
    intended_diagnoses: list[str],
    config,
) -> pd.DataFrame:
    """k-NN cosine similarity to real images of the same class, plus a SEPARATE memorisation flag.

    Fidelity and memorisation stay in different columns on purpose. An image nearly identical to one
    specific real training image is a privacy and novelty FAILURE, not a fidelity success; a selector
    fed a single blended score would rank exactly that image highest. Here the fidelity columns are
    free to be high while `novelty_is_near_duplicate` marks the image, and the selection policy
    decides what to do about it.

    Cosine similarity on DINOv2 embeddings measures VISUAL agreement — colour, pigment texture,
    structure. It is not a diagnostic judgement: a synthetic image can sit close to real melanomas
    and still show a dermoscopic pattern no melanoma has.
    """
    similarity_cfg = config.signals.similarity
    k = int(similarity_cfg.k_neighbors)
    min_same_class = int(similarity_cfg.min_same_class_references)
    near_dup_similarity = float(similarity_cfg.near_duplicate_similarity)
    near_dup_gap = float(similarity_cfg.near_duplicate_top1_gap)

    if len(reference_embeddings) != len(reference_diagnoses):
        raise ValueError(
            f"{len(reference_embeddings)} reference embeddings but {len(reference_diagnoses)} "
            "reference diagnoses; the two are positional and must describe the same images"
        )
    if len(synthetic_embeddings) != len(intended_diagnoses):
        raise ValueError(
            f"{len(synthetic_embeddings)} synthetic embeddings but {len(intended_diagnoses)} "
            "intended diagnoses; the two are positional and must describe the same images"
        )
    if len(reference_embeddings) == 0:
        raise ValueError("no real reference embeddings; similarity has nothing to measure against")

    synthetic_norm = synthetic_embeddings / (
        np.linalg.norm(synthetic_embeddings, axis=1, keepdims=True) + 1e-8
    )
    reference_norm = reference_embeddings / (
        np.linalg.norm(reference_embeddings, axis=1, keepdims=True) + 1e-8
    )

    rows = []
    for row_index, diagnosis in enumerate(intended_diagnoses):
        indices, tier = single_label_reference_indices(
            diagnosis, reference_diagnoses, min_same_class
        )
        similarities = reference_norm[indices] @ synthetic_norm[row_index]

        effective_k = min(k, len(similarities))
        top_k = np.sort(similarities)[-effective_k:][::-1]
        top1 = float(top_k[0])
        knn_mean = float(top_k.mean())
        spread = float(top_k[0] - top_k[-1]) if effective_k > 1 else 0.0
        rest_mean = float(top_k[1:].mean()) if effective_k > 1 else top1
        top1_gap = top1 - rest_mean

        # Crossing the similarity threshold is sufficient for the flag on its own. The gap is a
        # SECOND, narrower reading: a large gap means the image collapsed onto ONE real image rather
        # than onto a cluster of near-identical real ones — the difference between reproducing an
        # individual patient's lesion and reproducing a common appearance.
        is_near_duplicate = bool(top1 >= near_dup_similarity)
        collapsed_onto_single = bool(is_near_duplicate and top1_gap >= near_dup_gap)

        rows.append(
            {
                "similarity_knn_mean": knn_mean,
                "similarity_top1": top1,
                "similarity_topk_spread": spread,
                "similarity_top1_gap": top1_gap,
                "similarity_reference_tier": tier,
                "similarity_reference_diagnosis": diagnosis,
                "similarity_n_references": int(len(indices)),
                "similarity_k_used": int(effective_k),
                "novelty_is_near_duplicate": is_near_duplicate,
                "novelty_collapsed_onto_single_reference": collapsed_onto_single,
                # High only when the image resembles its class WITHOUT copying a specific real
                # image. A memorised image earns no novelty credit at all.
                "novelty_score": float(knn_mean * (1.0 - float(is_near_duplicate))),
            }
        )

    return pd.DataFrame(rows)


# ==============================================================================================
# Uncertainty — the softmax decomposition, not an average of per-label Bernoulli entropies
# ==============================================================================================
#
# WHAT THE CHEXPERT VERSION MEASURES
#   `signals.compute_uncertainty_scores` bands on the mean MC-Dropout standard deviation across
#   labels and reports a per-label Bernoulli entropy averaged over the eleven labels. Both are
#   correct for eleven INDEPENDENT sigmoid outputs.
#
# WHY THAT NUMBER IS THE WRONG ONE HERE
#   A softmax row is a single distribution over seven mutually exclusive classes; its components are
#   constrained to sum to 1 and are strongly negatively correlated. Treating each as an independent
#   Bernoulli and averaging their entropies does not estimate the uncertainty of the prediction — it
#   is dominated by the many near-zero components, and on this dataset by whichever class is
#   frequent (nv is ~66% of HAM10000), so it would rank images by their class more than by how
#   unsure the model is.
#
# WHAT IS MEASURED INSTEAD — and why the split matters for SELECTING synthetic images
#   Over MC-Dropout passes p = 1..P with softmax rows q_p:
#       predictive entropy  H[mean_p q_p]            TOTAL uncertainty
#       expected entropy    mean_p H[q_p]            ALEATORIC — ambiguity in the image itself
#       mutual information  predictive - expected    EPISTEMIC — disagreement BETWEEN dropout masks
#   Only the last is what dropout sampling actually adds, and only it separates the two cases that
#   matter to this stage: an image the model finds genuinely ambiguous (high aleatoric — plausibly a
#   hard, informative example) from an image the model has no settled opinion about (high epistemic —
#   typically off-distribution, i.e. a generation artefact).
#
#   All three are normalised by log(7) so they land in [0, 1] and the band thresholds read as
#   fractions of maximum uncertainty rather than as nats.
#
# BANDS ARE REPORTED, NOT PENALISED
#   Whether moderate uncertainty is good or bad is a SELECTION-POLICY question (TSynD's premise is
#   that the moderate band is the most informative), decided downstream together with the other
#   signals. The band column exists so that decision can be made and audited; nothing here turns it
#   into a score.

UNCERTAINTY_BANDS = ("low", "moderate", "extreme")


def _entropy(distributions: np.ndarray) -> np.ndarray:
    """Shannon entropy in nats along the last axis, with 0 log 0 := 0."""
    clipped = np.clip(distributions, 1e-12, 1.0)
    return -(distributions * np.log(clipped)).sum(axis=-1)


def compute_uncertainty_scores(probability_passes: np.ndarray, config) -> pd.DataFrame:
    """Uncertainty columns from the raw (n_passes, n_images, n_classes) MC-Dropout softmax passes.

    The per-pass array is required rather than the (mean, std) pair, because the aleatoric/epistemic
    split cannot be recovered from a mean and a standard deviation: mean_p H[q_p] needs each pass's
    own distribution. `ham10000_classifier.predict_probability_passes` returns exactly this array.
    """
    passes = np.asarray(probability_passes, dtype=np.float64)
    if passes.ndim != 3:
        raise ValueError(
            f"expected a (n_passes, n_images, n_classes) array, got shape {passes.shape}"
        )
    n_passes, _, n_classes = passes.shape
    if n_passes < 2:
        raise ValueError(
            f"n_passes={n_passes}: with one pass the epistemic term is identically zero by "
            "construction and the signal would be a constant, not a measurement"
        )

    uncertainty_cfg = config.signals.uncertainty
    low_max = float(uncertainty_cfg.low_band_max)
    moderate_max = float(uncertainty_cfg.moderate_band_max)
    if not 0.0 < low_max < moderate_max:
        raise ValueError(
            f"uncertainty bands must satisfy 0 < low_band_max ({low_max}) < moderate_band_max "
            f"({moderate_max}); both are fractions of log({n_classes})"
        )

    mean_probabilities = passes.mean(axis=0)
    scale = float(np.log(n_classes))

    predictive_entropy = _entropy(mean_probabilities) / scale
    expected_entropy = _entropy(passes).mean(axis=0) / scale
    # Non-negative in exact arithmetic (Jensen); the clip removes float noise around zero only.
    mutual_information = np.clip(predictive_entropy - expected_entropy, 0.0, None)

    standard_deviation = passes.std(axis=0)

    bands = np.where(
        mutual_information <= low_max,
        "low",
        np.where(mutual_information <= moderate_max, "moderate", "extreme"),
    )

    return pd.DataFrame(
        {
            "uncertainty_predictive_entropy": predictive_entropy,
            "uncertainty_expected_entropy": expected_entropy,
            "uncertainty_mutual_information": mutual_information,
            "uncertainty_mean_std": standard_deviation.mean(axis=1),
            "uncertainty_max_std": standard_deviation.max(axis=1),
            "uncertainty_band": bands,
            "uncertainty_n_passes": np.full(len(mean_probabilities), n_passes, dtype=int),
        }
    )


# ==============================================================================================
# Agreement — does the auxiliary classifier read back the class the generator was asked for?
# ==============================================================================================
#
# WHAT IS DIFFERENT FROM THE CHEXPERT VERSION
#   There, "unintended labels the classifier is confident about" is INDEPENDENT evidence: sigmoid
#   outputs are unconstrained, so a recipe can score a high intended probability AND a high
#   unintended one, and the penalty catches exactly that. Under a softmax the probabilities sum to
#   1, so a confident rival class MATHEMATICALLY implies a low intended probability — the penalty is
#   not new evidence and cannot be added as though it were.
#
#   It is kept, with a different justification, because it separates two failure modes that share
#   the same low intended probability:
#     * diffuse — probability spread over several classes: the classifier is unsure;
#     * decided — one rival class holds most of the mass: the image reads as a DIFFERENT lesion.
#   The second is the one that puts a mislabelled image into training data, so it is penalised
#   further. `agreement_margin` (intended minus best rival) carries the same distinction as a
#   continuous quantity, for a downstream policy that prefers not to use a threshold at all.
#
# WHAT THIS IS NOT
#   Not clinical verification. The generator and the classifier were fitted to the same real images
#   and can share a bias — agreeing while both are wrong. It measures consistency between two
#   models, and the auxiliary classifier is itself weakest on the rare classes this stage cares most
#   about.


def compute_agreement_scores(
    probabilities: np.ndarray,
    intended_diagnoses: list[str],
    config,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Label-consistency between each recipe's intended class and the auxiliary classifier's read.

        agreement = P(intended) - penalty_weight * P(best rival)   [when that rival is confident]
                  = P(intended)                                    [otherwise]
    """
    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis

    labels = list(labels) if labels is not None else list(CLASSIFIER_TARGET_LABELS)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError(
            f"expected (n_images, {len(labels)}) softmax probabilities, got shape {probabilities.shape}"
        )
    if len(probabilities) != len(intended_diagnoses):
        raise ValueError(
            f"{len(probabilities)} probability rows but {len(intended_diagnoses)} intended diagnoses"
        )

    agreement_cfg = config.signals.agreement
    threshold = float(agreement_cfg.rival_confidence_threshold)
    penalty_weight = float(agreement_cfg.penalty_weight)

    label_index = {label: position for position, label in enumerate(labels)}
    rows = []

    for row_index, raw_diagnosis in enumerate(intended_diagnoses):
        # An unrecognised dx is a broken recipe row, not an unlabelled one: scoring it would silently
        # compare against an arbitrary column. normalize_diagnosis raises instead.
        diagnosis = normalize_diagnosis(raw_diagnosis)
        probability_row = probabilities[row_index]
        intended_position = label_index[diagnosis]

        intended_probability = float(probability_row[intended_position])
        rival_probabilities = np.delete(probability_row, intended_position)
        rival_position = int(np.argmax(rival_probabilities))
        best_rival_probability = float(rival_probabilities[rival_position])
        best_rival = [label for label in labels if label != diagnosis][rival_position]

        predicted = labels[int(np.argmax(probability_row))]
        penalty_applied = bool(best_rival_probability >= threshold)
        penalty = best_rival_probability if penalty_applied else 0.0

        rows.append(
            {
                "agreement_score": intended_probability - penalty_weight * penalty,
                "agreement_intended_diagnosis": diagnosis,
                "agreement_intended_prob": intended_probability,
                "agreement_predicted_diagnosis": predicted,
                "agreement_is_argmax_match": bool(predicted == diagnosis),
                "agreement_best_rival_diagnosis": best_rival,
                "agreement_best_rival_prob": best_rival_probability,
                "agreement_margin": intended_probability - best_rival_probability,
                "agreement_penalty_applied": penalty_applied,
            }
        )

    return pd.DataFrame(rows)


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
    "REFERENCE_STATISTICS",
    "reference_artifact_payload",
    "references_from_artifact",
    "tie_report",
    "calibrate_explainability",
    "assert_selection_features_allowed",
    "selection_explainability_column",
    "SIMILARITY_TIER_SAME_CLASS",
    "SIMILARITY_TIER_CLASS_AGNOSTIC",
    "single_label_reference_indices",
    "compute_similarity_scores",
    "UNCERTAINTY_BANDS",
    "compute_uncertainty_scores",
    "compute_agreement_scores",
]

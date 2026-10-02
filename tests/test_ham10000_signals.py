#!/usr/bin/env python3
"""HAM10000 signals, metrics, recipes and RGB preprocessing.

The assertions that matter here are the ones that would have caught the four defects the audit
found: a greyscale conversion, a colour penalty, chest anatomy in the explainability path, and
label combinations that cannot exist.
"""

from __future__ import annotations

import importlib.util
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.ham10000_signals import (  # noqa: E402
    CONTINUOUS_WEIGHT,
    MIN_REFERENCE_SIZE,
    ReferenceLeakageError,
    UncalibratedExplainabilityError,
    assert_selection_features_allowed,
    build_reference_distribution,
    calibrate_against_reference,
    calibrate_explainability,
    compute_explainability_statistics,
    compute_iqa_scores,
    UncalibratedIQAError,
    calibrate_blur_threshold,
    focus_area,
    load_final_eval_image_ids,
    measure_sharpness,
    resolve_iqa_config,
    peripheral_mass,
    selection_explainability_column,
)
from scripts.generate.ham10000_recipes import build_recipes, class_counts, quotas_from_gen_train, rarity_quotas  # noqa: E402
from scripts.utils.config import CONFIGS_DIR  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, DIAGNOSIS_CLASSES  # noqa: E402
from scripts.utils.ham10000_metrics import (  # noqa: E402
    balanced_accuracy,
    confusion_matrix,
    full_metric_suite,
    macro_f1,
    per_class_precision_recall_f1,
)
from fixture_workspace import fixture_workspace

HAM_STAGE3 = OmegaConf.load(CONFIGS_DIR / "ham10000_stage3.yaml")
FIXTURE_CALIBRATION = {"source_split": "gen_train", "rule": "lower_quantile", "quantile": 0.02, "laplacian_blur_threshold": 20.0}
CONFIG = resolve_iqa_config(HAM_STAGE3, FIXTURE_CALIBRATION)

# The committed config measures the border on the CONTENT BOX, so compute_iqa_scores now requires
# each image's box and refuses to invent one. The fixtures below are square, unpadded arrays written
# straight to disk: their true content box is the whole canvas, which is what this constant says. It
# is not a way of switching the region off — on these images the content box and the canvas are the
# same region, so the numbers are identical to what they were before the region changed, and the
# tests keep exercising the committed configuration rather than a private one.
UNPADDED_BOX = (0.0, 0.0, 1.0, 1.0)

# The inherited canvas behaviour, kept ONLY for the tests whose subject is the difference between
# the two regions. Nothing in the pipeline reads it.
CANVAS_CONFIG = resolve_iqa_config(
    OmegaConf.merge(HAM_STAGE3, {"signals": {"iqa": {"border_region": "canvas"}}}), FIXTURE_CALIBRATION
)


def _load_preprocess():
    spec = importlib.util.spec_from_file_location(
        "ham_preprocess", REPO / "scripts" / "data" / "ham10000" / "02_preprocess_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _colourful_lesion(size: int = 128, seed: int = 0) -> np.ndarray:
    """A synthetic dermoscopy-like frame: a saturated brown-pink lesion on lighter skin, with real
    texture so it is neither blank nor blurry."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * (size / 5) ** 2)))
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = 200 - 90 * blob   # R
    image[..., 1] = 170 - 120 * blob  # G
    image[..., 2] = 160 - 60 * blob   # B  -> strongly non-grey
    image += rng.normal(0, 12, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _write(workspace: Path, name: str, array: np.ndarray) -> Path:
    path = workspace / name
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95)
    return path


# ---------------------------------------------------------------- RGB preprocessing

def test_preprocessing_keeps_rgb_and_never_converts_to_greyscale():
    """The defect this whole module exists to prevent: colour is the diagnostic signal in
    dermoscopy, and converting to 'L' deletes it."""
    module = _load_preprocess()
    with fixture_workspace("ham-rgb") as workspace:
        source = Image.fromarray(_colourful_lesion(), mode="RGB")
        out = module.letterbox_resize_rgb(source, 64, (128, 128, 128))
        assert out.mode == "RGB"
        array = np.asarray(out, dtype=np.float32)
        saturation = float(np.mean(array.max(axis=2) - array.min(axis=2)))
        assert saturation > 5.0, "colour was flattened; the output is effectively greyscale"


def test_preprocessing_letterboxes_instead_of_cropping():
    """A centre-crop would cut a lesion that runs to the frame edge. Padding must be used instead,
    so the full source content survives at the correct aspect ratio."""
    module = _load_preprocess()
    # A wide 120x40 source: letterboxing must keep all 120 columns, scaled, and pad vertically.
    source = Image.fromarray(np.full((40, 120, 3), 200, dtype=np.uint8), mode="RGB")
    out = module.letterbox_resize_rgb(source, 60, (128, 128, 128))
    assert out.size == (60, 60)

    array = np.asarray(out)
    content_rows = np.where(np.any(np.abs(array.astype(int) - 128) > 10, axis=(1, 2)))[0]
    # Content occupies a horizontal band; the rest is padding, and the aspect ratio is preserved.
    assert len(content_rows) == 20, f"expected a 20-row content band, got {len(content_rows)}"
    assert np.all(array[0] == 128), "top rows must be padding, not cropped content"


def test_preprocessing_output_validator_rejects_a_greyscale_file():
    """The resume path validates existing outputs; it must treat a greyscale file as invalid so an
    older or mis-wired run's output is never silently reused."""
    module = _load_preprocess()
    with fixture_workspace("ham-grey-reject") as workspace:
        grey = workspace / "grey.jpg"
        Image.fromarray(np.full((64, 64), 120, dtype=np.uint8), mode="L").save(grey, format="JPEG")
        valid, reason = module.validate_processed_jpeg(grey, 64)
        assert valid is False
        assert reason is not None and "mode" in reason


# ---------------------------------------------------------------- IQA

def test_iqa_does_not_penalise_a_colourful_image():
    """The CheXpert IQA subtracts 0.10 for channel divergence because a radiograph is achromatic.
    Carrying that over would fire on every valid dermoscopy image."""
    with fixture_workspace("ham-iqa-colour") as workspace:
        scores = compute_iqa_scores(_write(workspace, "lesion.jpg", _colourful_lesion()), CONFIG, content_box=UNPADDED_BOX)
    assert scores["iqa_valid"] is True
    assert scores["iqa_channel_saturation"] > 5.0, "fixture must actually be colourful"
    assert scores["iqa_composite"] > 0.8, "a clean colourful lesion must not be penalised"


def test_iqa_saturation_is_reported_but_carries_no_penalty():
    """Two images identical except for saturation must score the same: saturation is a readout, not
    a quality judgement."""
    with fixture_workspace("ham-iqa-saturation") as workspace:
        rng = np.random.default_rng(3)
        base = rng.integers(60, 200, size=(128, 128), dtype=np.uint8)
        grey_like = np.stack([base, base, base], axis=2)
        coloured = grey_like.copy()
        coloured[..., 0] = np.clip(coloured[..., 0].astype(int) + 40, 0, 255)  # add a red cast

        grey_scores = compute_iqa_scores(_write(workspace, "grey.jpg", grey_like), CONFIG, content_box=UNPADDED_BOX)
        colour_scores = compute_iqa_scores(_write(workspace, "colour.jpg", coloured), CONFIG, content_box=UNPADDED_BOX)

    assert colour_scores["iqa_channel_saturation"] > grey_scores["iqa_channel_saturation"]
    assert colour_scores["iqa_composite"] >= grey_scores["iqa_composite"] - 0.02, (
        "adding colour must not reduce the quality score"
    )


def test_iqa_still_catches_real_defects():
    with fixture_workspace("ham-iqa-defects") as workspace:
        blank = compute_iqa_scores(
            _write(workspace, "blank.jpg", np.full((128, 128, 3), 130, dtype=np.uint8)), CONFIG,
            content_box=UNPADDED_BOX,
        )
    assert blank["iqa_is_near_uniform"] is True
    assert blank["iqa_composite"] < 0.6, "a blank frame must still be heavily penalised"


def test_iqa_composite_is_continuous_enough_for_the_gonogo_gate():
    """Five binary flags alone give ten distinct values, below Go/No-Go's min_unique_values=20, so
    the signal would be dropped from the pool. The continuous term fixes that without letting a
    defect-free image be out-ranked by a defective one."""
    with fixture_workspace("ham-iqa-unique") as workspace:
        composites = [
            compute_iqa_scores(_write(workspace, f"img{i}.jpg", _colourful_lesion(seed=i)), CONFIG, content_box=UNPADDED_BOX)["iqa_composite"]
            for i in range(24)
        ]
    assert len(set(composites)) >= 20
    # Ordering guarantee: the continuous weight stays under the 0.1 gap between defect tiers.
    assert (1.0 - CONTINUOUS_WEIGHT) > 0.9 * (1.0 - CONTINUOUS_WEIGHT) + CONTINUOUS_WEIGHT


# ---------------------------------------------------------------- IQA blur calibration

def _sharpness_fixture(n: int = 400):
    rng = np.random.default_rng(0)
    ids = [f"ISIC_G{i:05d}" for i in range(n)]
    values = dict(zip(ids, rng.lognormal(mean=4.0, sigma=0.6, size=n)))
    diagnosis = {image_id: ("nv" if i % 4 else "df") for i, image_id in enumerate(ids)}
    return ids, values, diagnosis


def test_iqa_refuses_the_inherited_chexpert_blur_threshold():
    """Regression: HAM10000 IQA must not run on stage3_asism.yaml's radiograph threshold of 40."""
    chexpert = OmegaConf.load(CONFIGS_DIR / "stage3_asism.yaml")
    OmegaConf.resolve(chexpert)
    for config in (chexpert, HAM_STAGE3):
        with fixture_workspace("ham-iqa-uncalibrated") as workspace:
            path = _write(workspace, "lesion.jpg", _colourful_lesion())
            try:
                compute_iqa_scores(path, config)
            except UncalibratedIQAError:
                continue
        raise AssertionError("IQA ran without a gen_train blur calibration")


def test_the_ham_config_carries_no_blur_constant():
    assert HAM_STAGE3.signals.iqa.laplacian_blur_threshold is None
    assert HAM_STAGE3.signals.iqa.blur_calibration.source_split == "gen_train"


def test_blur_threshold_is_the_configured_lower_quantile_and_reports_its_rejection_rate():
    ids, values, diagnosis = _sharpness_fixture()
    result = calibrate_blur_threshold(values, "gen_train", ids, {"ISIC_FINAL"}, 0.02, diagnosis)
    assert abs(result["laplacian_blur_threshold"] - float(np.quantile(list(values.values()), 0.02))) < 1e-9
    assert abs(result["overall_rejection_rate"] - 0.02) <= 1.0 / len(ids)
    assert set(result["per_class"]) == {"nv", "df"}
    assert result["n_images"] == len(ids)


def test_blur_calibration_FAILS_on_any_split_other_than_gen_train():
    ids, values, _ = _sharpness_fixture()
    for split in ("asism_tuning_heldout", "final_eval_heldout", "classifier_val"):
        try:
            calibrate_blur_threshold(values, split, ids, {"ISIC_FINAL"}, 0.02)
        except ReferenceLeakageError:
            continue
        raise AssertionError(f"blur threshold calibrated on {split}")


def test_blur_calibration_FAILS_when_a_held_out_image_is_measured():
    ids, values, _ = _sharpness_fixture()
    try:
        calibrate_blur_threshold(values, "gen_train", ids, {ids[5]}, 0.02)
    except ReferenceLeakageError:
        pass
    else:
        raise AssertionError("an asism_tuning/final_eval image informed the blur threshold")
    try:
        calibrate_blur_threshold({**values, "ISIC_OUTSIDE": 1.0}, "gen_train", ids, {"ISIC_FINAL"}, 0.02)
    except ReferenceLeakageError:
        return
    raise AssertionError("an image outside the gen_train id list informed the blur threshold")


def test_resolved_config_refuses_a_calibration_that_does_not_match_the_rule():
    for bad in (
        {**FIXTURE_CALIBRATION, "source_split": "final_eval_heldout"},
        {**FIXTURE_CALIBRATION, "quantile": 0.10},
        {**FIXTURE_CALIBRATION, "rule": "fixed_constant"},
    ):
        try:
            resolve_iqa_config(HAM_STAGE3, bad)
        except (ReferenceLeakageError, ValueError):
            continue
        raise AssertionError(f"accepted calibration {bad}")


# ---------------------------------------------------------------- IQA border artifact

def _letterboxed_edge_matching_padding(size: int = 128) -> tuple[np.ndarray, tuple]:
    """A textured image whose LEFT/RIGHT content edges have luminance ~128 (the pad grey), letterboxed
    with grey bars: exactly the case the inherited canvas rule mistakes for a border artifact."""
    rng = np.random.default_rng(1)
    canvas = np.full((size, size, 3), 128, dtype=np.uint8)
    top, bottom = size // 8, size - size // 8
    content = rng.integers(60, 230, size=(bottom - top, size, 3)).astype(np.uint8)
    edge = max(1, int(size * 0.06)) + 2
    content[:, :edge] = 128
    content[:, -edge:] = 128
    canvas[top:bottom] = content
    return canvas, (0.0, top / size, 1.0, bottom / size)


def test_border_score_on_the_canvas_is_driven_by_our_own_padding_and_on_the_content_box_is_not():
    from scripts.asism.ham10000_signals import border_region_luminance, border_uniform_fraction

    canvas, box = _letterboxed_edge_matching_padding()
    luminance = np.asarray(Image.fromarray(canvas).convert("L"), dtype=np.float32)
    on_canvas = border_uniform_fraction(border_region_luminance(luminance, "canvas"))
    on_content = border_uniform_fraction(border_region_luminance(luminance, "content_box", box))
    assert on_canvas > 0.30 > on_content, (on_canvas, on_content)


def test_content_box_border_region_refuses_to_fall_back_to_the_canvas():
    from scripts.asism.ham10000_signals import border_region_luminance

    _expect_raises(UncalibratedIQAError, border_region_luminance, np.zeros((16, 16), dtype=np.float32), "content_box", None)
    _expect_raises(ValueError, border_region_luminance, np.zeros((16, 16), dtype=np.float32), "whole_image", None)


def _expect_raises(error, function, *args):
    try:
        function(*args)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def test_border_calibration_is_an_upper_quantile_from_gen_train_only():
    from scripts.asism.ham10000_signals import calibrate_border_threshold

    ids = [f"ISIC_G{i:04d}" for i in range(500)]
    values = dict(zip(ids, np.random.default_rng(2).beta(2, 12, size=500)))
    diagnosis = {i: ("mel" if n % 5 == 0 else "nv") for n, i in enumerate(ids)}
    result = calibrate_border_threshold(values, "content_box", "gen_train", ids, {"ISIC_FINAL"}, 0.98, diagnosis)
    assert abs(result["border_uniform_fraction"] - float(np.quantile(list(values.values()), 0.98))) < 1e-12
    assert abs(result["overall_flag_rate"] - 0.02) <= 2.0 / len(ids)
    assert result["rule"] == "upper_quantile" and result["region"] == "content_box" and set(result["per_class"]) == {"mel", "nv"}
    _expect_raises(ReferenceLeakageError, calibrate_border_threshold, values, "content_box", "final_eval_heldout", ids, {"ISIC_FINAL"}, 0.98)
    _expect_raises(ReferenceLeakageError, calibrate_border_threshold, values, "content_box", "gen_train", ids, {ids[3]}, 0.98)


def test_the_committed_border_region_is_the_content_box_with_calibration_still_disabled():
    """The frozen methodology choice, pinned so it cannot drift back silently.

    The region and the CUTOFF are two separate decisions: the region moved to the content box
    because on the canvas the score is dominated by padding this pipeline added, while the threshold
    is still the inherited 0.30 constant and is labelled as uncalibrated in every artifact row.
    """
    resolved = resolve_iqa_config(HAM_STAGE3, FIXTURE_CALIBRATION)
    assert HAM_STAGE3.signals.iqa.border_calibration.enabled is False
    assert resolved.signals.iqa.border_uniform_fraction == 0.30 and resolved.signals.iqa.border_region == "content_box"
    assert resolved.signals.iqa.border_uniform_fraction_source == "inherited_constant_uncalibrated"


def test_enabled_border_calibration_requires_a_matching_artifact():
    enabled = OmegaConf.merge(HAM_STAGE3, {"signals": {"iqa": {"border_region": "content_box", "border_calibration": {"enabled": True}}}})
    artifact = {"source_split": "gen_train", "rule": "upper_quantile", "quantile": 0.98, "region": "content_box", "border_uniform_fraction": 0.33}
    _expect_raises(UncalibratedIQAError, resolve_iqa_config, enabled, FIXTURE_CALIBRATION)
    _expect_raises(ReferenceLeakageError, resolve_iqa_config, enabled, FIXTURE_CALIBRATION, {**artifact, "region": "canvas"})
    _expect_raises(ReferenceLeakageError, resolve_iqa_config, enabled, FIXTURE_CALIBRATION, {**artifact, "source_split": "asism_tuning_heldout"})
    canvas_region = OmegaConf.merge(enabled, {"signals": {"iqa": {"border_region": "canvas"}}})
    _expect_raises(ValueError, resolve_iqa_config, canvas_region, FIXTURE_CALIBRATION, artifact)
    resolved = resolve_iqa_config(enabled, FIXTURE_CALIBRATION, artifact)
    assert resolved.signals.iqa.border_uniform_fraction == 0.33
    assert resolved.signals.iqa.border_uniform_fraction_source == "gen_train_upper_quantile_0.98_content_box"


def test_iqa_reports_a_numeric_border_score_and_its_region():
    """Regression: the refactor briefly returned the helper FUNCTION under iqa_border_uniform_fraction."""
    canvas, box = _letterboxed_edge_matching_padding()
    content_config = CONFIG  # the committed config already measures on the content box
    with fixture_workspace("ham-iqa-border") as workspace:
        path = _write(workspace, "letterboxed.jpg", canvas)
        on_canvas = compute_iqa_scores(path, CANVAS_CONFIG)
        on_content = compute_iqa_scores(path, content_config, content_box=box)
        _expect_raises(UncalibratedIQAError, compute_iqa_scores, path, content_config)
    assert isinstance(on_canvas["iqa_border_uniform_fraction"], float) and on_canvas["iqa_border_region"] == "canvas"
    assert on_canvas["iqa_has_border_artifact"] is True and on_content["iqa_has_border_artifact"] is False
    assert on_content["iqa_border_threshold_source"] == "inherited_constant_uncalibrated"


def test_measure_sharpness_is_the_value_compute_iqa_thresholds():
    with fixture_workspace("ham-sharpness") as workspace:
        path = _write(workspace, "lesion.jpg", _colourful_lesion())
        assert measure_sharpness(path) == compute_iqa_scores(path, CONFIG, content_box=UNPADDED_BOX)["iqa_sharpness"]


# ---------------------------------------------------------------- explainability

def _cam(kind: str, size: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    if kind == "centred":
        return np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * 3.0**2)))
    if kind == "diffuse":
        return np.ones((size, size))
    if kind == "edge":
        cam = np.zeros((size, size))
        cam[:3, :] = cam[-3:, :] = cam[:, :3] = cam[:, -3:] = 1.0
        return cam
    raise ValueError(kind)


def test_peripheral_mass_flags_edge_driven_attention():
    """The signal's one safe positive claim: attention on the frame edge is attention on scope
    vignette, ruler, rim — or this pipeline's own letterbox padding — none of which is the lesion."""
    assert peripheral_mass(_cam("edge")) > 0.9
    assert peripheral_mass(_cam("centred")) < 0.05


def test_peripheral_mass_is_exact_when_the_content_box_is_known():
    """With letterbox padding the non-lesion region is known exactly, not approximated by a band."""
    cam = np.zeros((32, 32))
    cam[:8, :] = 1.0  # all mass in the top quarter
    # Content occupies the middle half vertically: the top quarter is padding we added ourselves.
    assert peripheral_mass(cam, content_box=(0.0, 0.25, 1.0, 0.75)) == 1.0
    assert peripheral_mass(cam, content_box=(0.0, 0.0, 1.0, 1.0)) == 0.0


def test_focus_area_separates_compact_from_diffuse_attention():
    """A classifier reading a discrete lesion attends compactly; one reading a global colour cast
    smears across the frame."""
    assert focus_area(_cam("centred")) < 0.2
    assert focus_area(_cam("diffuse")) > 0.7


def test_plausibility_requires_both_conditions_not_just_one():
    """Compact attention sitting entirely on the padding must not be rescued by its compactness —
    which is why the composite multiplies rather than averages."""
    compact_on_edge = np.zeros((32, 32))
    compact_on_edge[0:3, 0:3] = 1.0  # tight, but in the corner

    corner = compute_explainability_statistics(compact_on_edge)
    centred = compute_explainability_statistics(_cam("centred"))

    assert corner["explainability_focus_area"] < 0.2, "fixture is compact"
    assert corner["explainability_peripheral_mass"] > 0.9, "fixture is on the edge"
    raw = "explainability_raw_plausibility_uncalibrated"
    assert corner[raw] < 0.2 < centred[raw]


# ---------------------------------------------------------------- content box: exact geometry

def test_content_box_matches_the_real_letterbox_pixels_for_every_aspect_ratio():
    """Regression: the persisted box must be the layout actually applied, pixel for pixel."""
    module = _load_preprocess()
    for width, height in [(600, 450), (450, 600), (600, 600), (601, 449), (1024, 768), (37, 91)]:
        source = Image.fromarray(np.full((height, width, 3), 200, dtype=np.uint8), mode="RGB")
        canvas = np.asarray(module.letterbox_resize_rgb(source, 512, (128, 128, 128))).astype(int)
        content = np.any(canvas != 128, axis=2)
        rows, cols = np.where(content.any(axis=1))[0], np.where(content.any(axis=0))[0]
        measured = (cols[0] / 512, rows[0] / 512, (cols[-1] + 1) / 512, (rows[-1] + 1) / 512)
        row = module.content_box_row("x", width, height, 512)
        assert (row["x0"], row["y0"], row["x1"], row["y1"]) == measured, (width, height, measured)


def test_preprocessing_persists_a_content_box_for_every_kept_image_including_resumed_ones():
    """Regression: content_box must be computed at preprocessing time AND written, and a resumed
    run (outputs already valid) must still write the box for the skipped images."""
    from omegaconf import OmegaConf as _OC

    from scripts.utils.ham10000_geometry import load_content_boxes

    module = _load_preprocess()
    data_cfg = _OC.create(
        {"resolution": 64, "pad_colour": [128, 128, 128], "jpeg_quality": 95, "min_source_resolution": 10}
    )
    with fixture_workspace("ham-content-box") as workspace:
        raw = workspace / "raw"
        (raw / "part1").mkdir(parents=True)
        sizes = {"ISIC_A": (120, 90), "ISIC_B": (90, 120), "ISIC_C": (100, 100)}
        for image_id, (width, height) in sizes.items():
            Image.fromarray(_colourful_lesion(size=max(width, height))[:height, :width]).save(raw / "part1" / f"{image_id}.jpg")
        frame = pd.DataFrame({"image_id": list(sizes)})
        out_root = workspace / "out"

        for attempt in ("fresh", "resumed"):
            counts = module.process_split("gen_train", frame, raw, out_root, ["part1"], data_cfg, "image_id")
            if attempt == "resumed":
                assert counts["skipped_existing"] == 3, counts
            boxes = load_content_boxes(out_root / "gen_train_content_boxes.csv")
            assert set(boxes) == set(sizes), attempt
            assert boxes["ISIC_A"] == (0.0, 0.125, 1.0, 0.875)
            assert boxes["ISIC_B"] == (0.125, 0.0, 0.875, 1.0)
            assert boxes["ISIC_C"] == (0.0, 0.0, 1.0, 1.0)


def test_preprocessing_manifest_names_the_content_box_table():
    module = _load_preprocess()
    stage1 = OmegaConf.load(CONFIGS_DIR / "ham10000_stage1.yaml")
    dataset = OmegaConf.load(CONFIGS_DIR / "dataset_ham10000.yaml")
    manifest = module.expected_manifest(stage1, dataset, "ns", "gen_train", "sha")
    assert manifest["content_box_file"] == "gen_train_content_boxes.csv"


# ---------------------------------------------------------------- one geometry contract: real + synthetic

STAGE1 = OmegaConf.load(CONFIGS_DIR / "ham10000_stage1.yaml")


def _textured(width: int, height: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(40, 220, size=(height, width, 3), dtype=np.uint8), mode="RGB")


def test_real_and_synthetic_paths_produce_the_same_valid_content_box():
    """A real 600x450 source and a synthetic image at the contracted 768x576 generation size must
    come out with the identical, valid, exact box — through the same letterbox function."""
    from scripts.utils.ham10000_geometry import standardize_generated_image, validate_content_box

    module = _load_preprocess()
    resolution, pad = int(STAGE1.data.resolution), tuple(STAGE1.data.pad_colour)
    real_row = module.content_box_row("real", 600, 450, resolution)
    synthetic_canvas, synthetic_row = standardize_generated_image(
        _textured(*STAGE1.geometry.generation_size), "synthetic", tuple(STAGE1.geometry.generation_size), resolution, pad
    )
    box = lambda row: (row["x0"], row["y0"], row["x1"], row["y1"])  # noqa: E731
    assert box(real_row) == box(synthetic_row) == (0.0, 0.125, 1.0, 0.875)
    assert validate_content_box(box(synthetic_row)) == box(synthetic_row)
    assert synthetic_canvas.size == (resolution, resolution) and synthetic_canvas.mode == "RGB"


def test_generation_size_matches_the_real_source_aspect_ratio():
    width, height = STAGE1.geometry.generation_size
    assert width * 3 == height * 4 and width % 8 == 0 and height % 8 == 0
    assert list(STAGE1.geometry.source_aspect) == [4, 3]


def test_synthetic_image_with_generator_drawn_padding_is_refused_not_guessed():
    from scripts.utils.ham10000_geometry import UnknownPaddingError, standardize_generated_image

    width, height = STAGE1.geometry.generation_size
    padded = np.asarray(_textured(width, height)).copy()
    padded[: height // 8] = 128  # grey bar the generator drew itself
    try:
        standardize_generated_image(Image.fromarray(padded), "bars", (width, height), 512, (128, 128, 128))
    except UnknownPaddingError:
        return
    raise AssertionError("a generated image with its own padding was given a content box")


def test_synthetic_image_at_the_wrong_size_is_refused():
    from scripts.utils.ham10000_geometry import UnknownPaddingError, standardize_generated_image

    for size in ((512, 512), (576, 768)):
        try:
            standardize_generated_image(_textured(*size), "square", tuple(STAGE1.geometry.generation_size), 512, (128, 128, 128))
        except UnknownPaddingError:
            continue
        raise AssertionError(f"accepted a generated image of size {size}")


def test_synthetic_directory_writes_a_box_table_that_feeds_explainability_and_logs_rejections():
    from scripts.asism.ham10000_explainability import explainability_rows
    from scripts.utils.ham10000_geometry import load_content_boxes

    spec = importlib.util.spec_from_file_location("ham_std", REPO / "scripts" / "generate" / "ham10000_standardize_generated.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    width, height = STAGE1.geometry.generation_size
    with fixture_workspace("ham-synthetic-geometry") as workspace:
        raw = workspace / "raw"
        raw.mkdir()
        _textured(width, height, 1).save(raw / "syn_ok.png")
        _textured(512, 512, 2).save(raw / "syn_square.png")
        result = module.standardize_directory(raw, workspace / "out", STAGE1)
        boxes = load_content_boxes(workspace / "out" / "content_boxes.csv")
        rejected = pd.read_csv(workspace / "out" / "rejected_geometry.csv")

    assert result == {"standardized": 1, "rejected": 1}
    assert set(boxes) == {"syn_ok"} and list(rejected["image_id"]) == ["syn_square"]
    cam = np.zeros((16, 16))
    cam[:2, :] = 1.0  # all attention on the padding the standardisation added
    row = explainability_rows([{"image_id": "syn_ok", "class_index": 6}], lambda _r: cam, boxes)[0]
    assert row["explainability_content_box_source"] == "preprocessing_manifest"
    assert row["explainability_peripheral_mass"] == 1.0


def test_preprocessing_delegates_to_the_shared_geometry_function():
    source = (REPO / "scripts" / "data" / "ham10000" / "02_preprocess_images.py").read_text(encoding="utf-8")
    assert "letterbox_with_content_box(" in source
    assert ".paste(" not in source, "real preprocessing must not keep a private letterbox implementation"


# ---------------------------------------------------------------- content box: CAM-grid rounding

def test_peripheral_mass_splits_a_straddling_cam_cell_by_area_instead_of_truncating():
    """Regression: int() truncation counted a 7x7 cell that is 87.5% padding as pure content and
    reported 0.0 for attention sitting almost wholly on padding."""
    cam = np.zeros((7, 7))
    cam[0, :] = 1.0  # row 0 spans [0, 1/7); the content starts at 0.125, so 0.875 of it is padding
    assert abs(peripheral_mass(cam, content_box=(0.0, 0.125, 1.0, 0.875)) - 0.875) < 1e-9


def test_peripheral_mass_is_exact_on_aligned_grids():
    cam = np.zeros((16, 16))
    cam[:2, :] = 1.0  # 600x450 at 512 px -> content rows 2..13 on a 16x16 DenseNet grid
    assert peripheral_mass(cam, content_box=(0.0, 0.125, 1.0, 0.875)) == 1.0
    cam = np.zeros((16, 16))
    cam[2:14, :] = 1.0
    assert peripheral_mass(cam, content_box=(0.0, 0.125, 1.0, 0.875)) == 0.0


def test_peripheral_mass_rejects_an_invalid_content_box():
    for bad in [(0.5, 0.0, 0.4, 1.0), (0.0, 0.0, 1.2, 1.0), (0.0, 0.0, 1.0)]:
        try:
            peripheral_mass(np.ones((8, 8)), content_box=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid content_box {bad}")


# ---------------------------------------------------------------- content box: reaches Grad-CAM stats

def test_content_box_flows_from_the_table_into_explainability_rows():
    """Regression: the box must actually reach peripheral_mass. The CAM puts all mass in rows 2-3 (cols 2-13) of
    16, which is padding for a (0, .25, 1, .75) box but interior for the generic band — so the two
    paths give 1.0 vs 0.0 and a silently dropped box cannot pass."""
    from scripts.asism.ham10000_explainability import explainability_rows

    cam = np.zeros((16, 16))
    cam[2:4, 2:14] = 1.0  # clear of the generic band on the left/right edges too
    records = [{"image_id": "ISIC_X", "class_index": 3}]
    exact = explainability_rows(records, lambda _r: cam, {"ISIC_X": (0.0, 0.25, 1.0, 0.75)})[0]
    generic = explainability_rows(records, lambda _r: cam, None)[0]

    assert exact["explainability_peripheral_mass"] == 1.0
    assert exact["explainability_content_box_source"] == "preprocessing_manifest"
    assert exact["explainability_content_box"] == "[0.0, 0.25, 1.0, 0.75]"
    assert abs(generic["explainability_peripheral_mass"]) < 1e-9
    assert generic["explainability_content_box_source"] == "generic_border_band"


def test_explainability_rows_refuse_a_missing_box_instead_of_falling_back():
    from scripts.asism.ham10000_explainability import explainability_rows

    try:
        explainability_rows([{"image_id": "ISIC_MISSING", "class_index": 0}], lambda _r: np.ones((4, 4)), {"ISIC_OTHER": (0, 0, 1, 1)})
    except KeyError:
        return
    raise AssertionError("a missing content_box silently fell back to the generic band")


def test_gradcam_on_a_real_densenet_feeds_the_box_through():
    """End-to-end on an actual DenseNet121 (random init, CPU): CAM comes back at feature-map
    resolution and the row carries the exact box."""
    try:
        import torch
    except ImportError:
        return  # torch-less environment: covered by the injected-CAM test above
    from scripts.asism.ham10000_explainability import explainability_rows, gradcam
    from scripts.utils.classifier import build_model

    model = build_model(len(CLASSIFIER_TARGET_LABELS), 0.0, "random", seed=0).eval()
    image = torch.randn(3, 64, 64)
    cam = gradcam(model, image, class_index=1)
    assert cam.shape == (2, 2)
    row = explainability_rows([{"image_id": "I", "class_index": 1}], lambda _r: cam, {"I": (0.0, 0.125, 1.0, 0.875)})[0]
    assert row["explainability_content_box_source"] == "preprocessing_manifest"
    assert row["explainability_cam_shape"] == "[2, 2]"


# ---------------------------------------------------------------- calibration: two-sided typicality

def _reference(values, split="gen_train", diagnosis="nv", statistic="explainability_peripheral_mass", ids=None, trained=True):
    ids = ids if ids is not None else [f"ISIC_R{i:04d}" for i in range(len(values))]
    return build_reference_distribution(
        values, ids, split, diagnosis, statistic, final_eval_image_ids={"ISIC_FINAL_0001"},
        cam_model_id="ham10000-classifier-fixture", cam_model_trained=trained,
    )


def test_typicality_does_not_reward_extremes_on_either_side():
    """Regression: the one-sided percentile gave 1.0 to any value beyond the best real image."""
    reference = _reference(np.linspace(0.0, 0.2, 41))
    median = calibrate_against_reference(0.1, reference)
    far_low = calibrate_against_reference(-5.0, reference)
    far_high = calibrate_against_reference(5.0, reference)
    assert median == 1.0
    assert abs(far_low - 1.0 / 42) < 1e-12 and abs(far_high - 1.0 / 42) < 1e-12, (far_low, far_high)
    assert far_low < 0.05 and far_high < 0.05


def test_typicality_is_symmetric_and_monotone_away_from_the_median():
    reference = _reference(np.arange(1, 22, dtype=float))  # 1..21, median 11
    scores = [calibrate_against_reference(v, reference) for v in (11, 8, 5, 2)]
    assert scores == sorted(scores, reverse=True)
    for offset in (1, 4, 9):
        assert abs(calibrate_against_reference(11 - offset, reference) - calibrate_against_reference(11 + offset, reference)) < 1e-12


def test_typicality_is_approximately_uniform_for_values_drawn_from_the_reference_distribution():
    """The defensibility claim: a conformal p-value is ~Uniform(0,1) under exchangeability."""
    rng = np.random.default_rng(0)
    reference = _reference(rng.normal(size=199))
    scores = np.array([calibrate_against_reference(v, reference) for v in rng.normal(size=4000)])
    for q in (0.1, 0.25, 0.5):
        assert abs(np.mean(scores <= q) - q) < 0.03, (q, np.mean(scores <= q))


def test_calibration_refuses_raw_arrays():
    try:
        calibrate_against_reference(0.1, np.array([0.1] * 30))
    except TypeError:
        return
    raise AssertionError("a raw array bypassed the reference-split checks")


# ---------------------------------------------------------------- calibration: reference split

def test_calibration_FAILS_when_final_eval_is_named_as_the_reference_split():
    try:
        _reference(np.linspace(0, 1, 30), split="final_eval_heldout")
    except ReferenceLeakageError:
        return
    raise AssertionError("final_eval_heldout was accepted as a calibration reference")


def test_calibration_FAILS_when_final_eval_images_are_smuggled_in_under_a_permitted_name():
    """Name checks alone are defeatable by a mislabelled argument; the ids are checked too."""
    ids = [f"ISIC_R{i:04d}" for i in range(29)] + ["ISIC_FINAL_0001"]
    try:
        _reference(np.linspace(0, 1, 30), split="gen_train", ids=ids)
    except ReferenceLeakageError:
        return
    raise AssertionError("a final-eval image entered the reference under the gen_train label")


def test_calibration_FAILS_for_every_non_permitted_split():
    for split in ("classifier_train", "classifier_val", "asism_tuning_heldout", "gen_val"):
        try:
            _reference(np.linspace(0, 1, 30), split=split)
        except ReferenceLeakageError:
            continue
        raise AssertionError(f"{split} was accepted as a reference split")


def test_calibration_FAILS_without_a_final_eval_id_list_to_check_against():
    try:
        build_reference_distribution(np.linspace(0, 1, 30), [f"I{i}" for i in range(30)], "gen_train", "nv", "explainability_peripheral_mass", final_eval_image_ids=set(), cam_model_id="m", cam_model_trained=True)
    except ReferenceLeakageError:
        return
    raise AssertionError("an empty final-eval id list made the disjointness check vacuous")


def test_calibration_refuses_a_reference_that_is_too_small():
    try:
        _reference(np.linspace(0, 1, MIN_REFERENCE_SIZE - 1))
    except ValueError:
        return
    raise AssertionError("an undersized reference was accepted")


def test_final_eval_ids_are_read_from_the_frozen_split_file():
    with fixture_workspace("ham-final-eval-ids") as workspace:
        pd.DataFrame({"image_id": ["ISIC_1", "ISIC_2"], "lesion_id": ["L1", "L2"]}).to_csv(workspace / "final_eval_heldout.csv", index=False)
        assert load_final_eval_image_ids(workspace) == frozenset({"ISIC_1", "ISIC_2"})
        try:
            load_final_eval_image_ids(workspace / "missing")
        except FileNotFoundError:
            return
    raise AssertionError("a missing final-eval split file did not fail closed")


# ---------------------------------------------------------------- large lesions / selection guard

def test_a_larger_lesion_is_not_penalised_by_the_selection_eligible_feature():
    """Two CAMs entirely inside the content area, one compact and one spread over a large lesion.
    Their focus_area differs a lot; the eligible feature (peripheral typicality, exact box) must not."""
    box = (0.0, 0.125, 1.0, 0.875)
    small, large = np.zeros((16, 16)), np.zeros((16, 16))
    small[7:9, 7:9] = 1.0
    large[2:14, 1:15] = 1.0
    references = {"explainability_peripheral_mass": _reference(np.zeros(30))}
    rows = {}
    for name, cam in (("small", small), ("large", large)):
        raw = compute_explainability_statistics(cam, content_box=box)
        rows[name] = (raw, calibrate_explainability(raw, "nv", references))
    assert rows["large"][0]["explainability_focus_area"] > 5 * rows["small"][0]["explainability_focus_area"]
    assert rows["large"][1]["explainability_calibrated_typicality"] == rows["small"][1]["explainability_calibrated_typicality"]


def test_generic_band_measurements_never_yield_an_eligible_value():
    raw = compute_explainability_statistics(np.ones((16, 16)))  # no content box
    calibrated = calibrate_explainability(raw, "nv", {"explainability_peripheral_mass": _reference(np.linspace(0, 1, 30))})
    assert np.isnan(calibrated["explainability_calibrated_typicality"])
    assert calibrated["explainability_content_box_exact"] is False


def test_calibration_refuses_a_reference_from_another_class():
    raw = compute_explainability_statistics(np.ones((16, 16)), content_box=(0, 0, 1, 1))
    try:
        calibrate_explainability(raw, "mel", {"explainability_peripheral_mass": _reference(np.linspace(0, 1, 30), diagnosis="nv")})
    except ValueError:
        return
    raise AssertionError("a naevus reference calibrated a melanoma image")


def test_a_reference_from_an_untrained_cam_model_is_marked_non_scientific():
    """Random-init CAMs may exercise the plumbing but must never calibrate a selection feature."""
    raw = compute_explainability_statistics(np.ones((16, 16)), content_box=(0, 0.125, 1, 0.875))
    calibrated = calibrate_explainability(raw, "nv", {"explainability_peripheral_mass": _reference(np.linspace(0, 1, 30), trained=False)})
    assert calibrated["explainability_reference_scientific"] is False
    frame = pd.DataFrame([calibrated])
    try:
        selection_explainability_column(frame)
    except UncalibratedExplainabilityError:
        return
    raise AssertionError("a random-init CAM reference reached selection")


def _run_manifest(**data):
    base = {"real_train_split": "classifier_train", "selection_split": "classifier_val", "synthetic_images": 0}
    return {"dataset": "ham10000", "loss": "cross_entropy", "data": {**base, **data}}


def test_cam_model_identity_accepts_only_a_real_only_classifier_not_trained_on_the_reference_split():
    from scripts.asism.ham10000_signals import cam_model_identity

    with fixture_workspace("ham-cam-model") as workspace:
        model = workspace / "model.pt"
        model.write_bytes(b"fixture-state-dict")
        identity = cam_model_identity(_run_manifest(), model)
        assert identity["cam_model_trained"] is True and identity["cam_model_id"].startswith("ham10000-classifier:")
        for bad in (
            _run_manifest(real_train_split="gen_train"),            # CheXpert aux convention: trained on the reference split
            _run_manifest(selection_split="gen_train"),
            _run_manifest(synthetic_images=120),                    # condition B/C model
            _run_manifest(selection_split="final_eval_heldout"),
            {**_run_manifest(), "loss": "bce"},
        ):
            _expect_raises(ReferenceLeakageError, cam_model_identity, bad, model)
        _expect_raises(FileNotFoundError, cam_model_identity, _run_manifest(), workspace / "missing.pt")


def test_build_reference_requires_an_explicit_cam_model_provenance():
    try:
        build_reference_distribution(np.linspace(0, 1, 30), [f"I{i}" for i in range(30)], "gen_train", "nv", "explainability_peripheral_mass", {"ISIC_F"})
    except TypeError:
        return
    raise AssertionError("a reference was built without naming its CAM model")


def test_tie_report_measures_how_coarse_the_reference_is():
    from scripts.asism.ham10000_signals import tie_report

    report = tie_report([0.0] * 12 + list(np.linspace(0.1, 0.5, 8)))
    assert report["n"] == 20 and report["exact_zero_fraction"] == 0.6
    assert report["largest_tie_group_fraction"] == 0.6 and report["unique_values"] == 9


def test_selection_guard_rejects_every_raw_or_diagnostic_explainability_feature():
    for feature in (
        "explainability_raw_plausibility_uncalibrated",
        "explainability_peripheral_mass",
        "explainability_focus_area",
        "explainability_focus_typicality_diagnostic_only",
        "explainability_plausibility",
    ):
        try:
            assert_selection_features_allowed(["iqa_composite", feature])
        except UncalibratedExplainabilityError:
            continue
        raise AssertionError(f"{feature} was allowed as a selection feature")
    assert_selection_features_allowed(["iqa_composite", "similarity_score", "explainability_calibrated_typicality"])


def test_selection_column_requires_calibrated_rows_from_a_permitted_split():
    good = pd.DataFrame({"explainability_calibrated_typicality": [0.5], "explainability_calibrated": [True], "explainability_reference_split": ["gen_train"], "explainability_reference_scientific": [True]})
    assert list(selection_explainability_column(good)) == [0.5]

    raw_only = pd.DataFrame({"explainability_raw_plausibility_uncalibrated": [0.9]})
    bad_split = good.assign(explainability_reference_split=["final_eval_heldout"])
    untrained = good.assign(explainability_reference_scientific=[False])
    for frame, error in ((raw_only, UncalibratedExplainabilityError), (bad_split, ReferenceLeakageError), (untrained, UncalibratedExplainabilityError)):
        try:
            selection_explainability_column(frame)
        except error:
            continue
        raise AssertionError(f"selection_explainability_column accepted {list(frame.columns)}")


def test_explainability_path_contains_no_chest_anatomy():
    """Regression guard: no CheXpert pathology name or anatomical box may reach this module."""
    source = (REPO / "scripts" / "asism" / "ham10000_signals.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in ("Cardiomegaly", "Edema", "Pneumothorax", "pathology_regions", "baseline_box"):
        assert forbidden not in code, f"chest-anatomy reference {forbidden!r} leaked into the code path"


# ---------------------------------------------------------------- Stage 2 recipes

def test_every_recipe_carries_exactly_one_diagnosis():
    """HAM10000's classes are mutually exclusive: a multi-diagnosis recipe would ask the generator
    to synthesise something that cannot exist."""
    recipes = build_recipes({label: 2 for label in DIAGNOSIS_CLASSES})
    assert len(recipes) == 2 * len(DIAGNOSIS_CLASSES)
    for recipe in recipes:
        assert sum(recipe["intended_label_vector"].values()) == 1
        assert recipe["prompt"].startswith("A dermoscopic image of")


def test_quotas_oversample_the_rare_classes():
    counts = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}
    quotas = rarity_quotas(counts)
    assert quotas["df"] > quotas["nv"]
    assert quotas["vasc"] > quotas["mel"]
    assert all(100 <= value <= 1500 for value in quotas.values()), "quotas must stay inside the clip"


def test_rarity_exponent_zero_disables_oversampling():
    counts = {"nv": 6705, "mel": 1113, "bkl": 1099, "bcc": 514, "akiec": 327, "vasc": 142, "df": 115}
    quotas = rarity_quotas(counts, rarity_exponent=0.0)
    assert len(set(quotas.values())) == 1, "exponent 0 must give every class the same quota"


def test_generation_quotas_are_derived_from_gen_train_only():
    """Regression: the smoke test derived quotas from the full metadata, including final eval.
    Here every other split has a wildly different class mix, so reading any of them changes the
    result."""
    with fixture_workspace("ham-quota-gen-train") as workspace:
        mix = {"nv": 400, "mel": 60, "bkl": 55, "bcc": 30, "akiec": 20, "vasc": 9, "df": 7}
        diagnoses = [label for label, count in mix.items() for _ in range(count)]
        gen_train = pd.DataFrame({"image_id": [f"G{i}" for i in range(len(diagnoses))], "lesion_id": [f"L{i}" for i in range(len(diagnoses))], "dx": diagnoses})
        gen_train.to_csv(workspace / "gen_train.csv", index=False)
        poison = pd.DataFrame({"image_id": [f"F{i}" for i in range(50)], "lesion_id": [f"M{i}" for i in range(50)], "dx": ["df"] * 50})
        for name in ("gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"):
            poison.to_csv(workspace / f"{name}.csv", index=False)

        quotas, counts = quotas_from_gen_train(workspace)
    assert counts == mix
    assert quotas == rarity_quotas(mix)


def test_generation_quotas_refuse_a_class_with_no_real_images():
    """Regression: rarity_quotas maps a zero-count class to max_per_class, i.e. 1500 images of a
    class the generator never saw. Deriving quotas from a split must fail closed instead."""
    with fixture_workspace("ham-quota-zero") as workspace:
        pd.DataFrame({"image_id": ["A", "B"], "lesion_id": ["L1", "L2"], "dx": ["nv", "mel"]}).to_csv(workspace / "gen_train.csv", index=False)
        try:
            quotas_from_gen_train(workspace)
        except ValueError:
            return
    raise AssertionError("a class absent from gen_train received a generation quota")


def test_generation_quotas_fail_closed_without_gen_train():
    with fixture_workspace("ham-quota-missing") as workspace:
        try:
            quotas_from_gen_train(workspace)
        except FileNotFoundError:
            return
    raise AssertionError("quotas were derived without a gen_train split file")


def test_class_counts_uses_normalised_diagnoses():
    frame = pd.DataFrame({"dx": ["MEL", " mel ", "nv"]})
    counts = class_counts(frame)
    assert counts["mel"] == 2 and counts["nv"] == 1


# ---------------------------------------------------------------- multi-class metrics

def test_balanced_accuracy_exposes_a_majority_class_predictor():
    """The reason plain accuracy is not enough on HAM10000: predicting `nv` for everything scores
    ~0.67 accuracy and has learned nothing."""
    n_classes = len(CLASSIFIER_TARGET_LABELS)
    y_true = np.array([0] * 67 + list(range(1, n_classes)) * 5)
    y_pred = np.zeros_like(y_true)  # always predicts class 0 (nv)
    matrix = confusion_matrix(y_true, y_pred, n_classes)

    accuracy = float((y_pred == y_true).mean())
    assert accuracy > 0.6, "the degenerate model does look good on plain accuracy"
    assert balanced_accuracy(matrix) < 0.2, "balanced accuracy must expose it"
    assert macro_f1(matrix) < 0.2


def test_per_class_metrics_distinguish_undefined_from_zero():
    """A class with no true instances has an UNDEFINED recall, not a recall of zero; averaging zero
    would silently drag the macro score down."""
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 0] = 5   # class 0 present and correct
    matrix[1, 1] = 3   # class 1 present and correct
    per_class = per_class_precision_recall_f1(matrix)
    assert np.isfinite(per_class["recall"][0]) and np.isfinite(per_class["recall"][1])
    assert np.isnan(per_class["recall"][2]), "absent class must be NaN, not 0.0"


def test_full_suite_uses_argmax_and_reports_the_confusion_matrix():
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    logits = rng.normal(size=(len(y_true), len(CLASSIFIER_TARGET_LABELS)))
    logits[np.arange(len(y_true)), y_true] += 5.0
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    suite = full_metric_suite(probabilities, y_true)
    assert suite["accuracy"] == 1.0, "a strongly separated fixture must be classified perfectly"
    assert suite["balanced_accuracy"] == 1.0
    assert np.array(suite["confusion_matrix"]).sum() == len(y_true)
    assert set(suite["per_class"]) == set(CLASSIFIER_TARGET_LABELS)
    assert all(key in suite["per_class"]["mel"] for key in ("precision", "recall", "f1", "auroc_ovr", "support"))


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


def test_iqa_composite_is_reproducible_from_its_own_reported_fields():
    """The artifact must be auditable: every flag and the composite follow from the reported
    measurements and the resolved thresholds, with no hidden term."""
    iqa = CONFIG.signals.iqa
    with fixture_workspace("ham-iqa-reconstruct") as workspace:
        images = [_colourful_lesion(seed=i) for i in range(6)] + [np.full((128, 128, 3), 130, dtype=np.uint8)]
        rows = [compute_iqa_scores(_write(workspace, f"r{i}.jpg", img), CONFIG, content_box=UNPADDED_BOX)
                for i, img in enumerate(images)]
    for row in rows:
        assert row["iqa_valid"] is True
        assert row["iqa_is_near_uniform"] == (row["iqa_contrast_std"] < float(iqa.blank_std_threshold))
        assert row["iqa_is_blurry"] == (row["iqa_sharpness"] < float(iqa.laplacian_blur_threshold))
        assert row["iqa_is_low_contrast"] == (row["iqa_contrast_std"] < float(iqa.low_contrast_std))
        assert row["iqa_has_border_artifact"] == (row["iqa_border_uniform_fraction"] > float(iqa.border_uniform_fraction))
        clipped = row["iqa_clipped_low_fraction"] + row["iqa_clipped_high_fraction"] > 0.20
        penalties = (0.40 * row["iqa_is_near_uniform"] + 0.20 * row["iqa_is_blurry"] + 0.10 * row["iqa_is_low_contrast"]
                     + 0.10 * row["iqa_has_border_artifact"] + 0.10 * clipped)
        continuous = 0.5 * (np.tanh(row["iqa_sharpness"] / (2 * float(iqa.laplacian_blur_threshold)))
                            + np.tanh(row["iqa_contrast_std"] / (2 * float(iqa.low_contrast_std))))
        expected = max(0.0, 1.0 - penalties) * (1.0 - CONTINUOUS_WEIGHT) + CONTINUOUS_WEIGHT * continuous
        assert row["iqa_composite"] == pytest.approx(expected, abs=1e-12)
        assert 0.0 <= row["iqa_composite"] <= 1.0

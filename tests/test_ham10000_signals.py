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
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.ham10000_signals import (  # noqa: E402
    CONTINUOUS_WEIGHT,
    calibrate_against_reference,
    compute_explainability_statistics,
    compute_iqa_scores,
    focus_area,
    peripheral_mass,
)
from scripts.generate.ham10000_recipes import build_recipes, class_counts, rarity_quotas  # noqa: E402
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

CONFIG = OmegaConf.load(CONFIGS_DIR / "stage3_asism.yaml")
OmegaConf.resolve(CONFIG)


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
        scores = compute_iqa_scores(_write(workspace, "lesion.jpg", _colourful_lesion()), CONFIG)
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

        grey_scores = compute_iqa_scores(_write(workspace, "grey.jpg", grey_like), CONFIG)
        colour_scores = compute_iqa_scores(_write(workspace, "colour.jpg", coloured), CONFIG)

    assert colour_scores["iqa_channel_saturation"] > grey_scores["iqa_channel_saturation"]
    assert colour_scores["iqa_composite"] >= grey_scores["iqa_composite"] - 0.02, (
        "adding colour must not reduce the quality score"
    )


def test_iqa_still_catches_real_defects():
    with fixture_workspace("ham-iqa-defects") as workspace:
        blank = compute_iqa_scores(
            _write(workspace, "blank.jpg", np.full((128, 128, 3), 130, dtype=np.uint8)), CONFIG
        )
    assert blank["iqa_is_near_uniform"] is True
    assert blank["iqa_composite"] < 0.6, "a blank frame must still be heavily penalised"


def test_iqa_composite_is_continuous_enough_for_the_gonogo_gate():
    """Five binary flags alone give ten distinct values, below Go/No-Go's min_unique_values=20, so
    the signal would be dropped from the pool. The continuous term fixes that without letting a
    defect-free image be out-ranked by a defective one."""
    with fixture_workspace("ham-iqa-unique") as workspace:
        composites = [
            compute_iqa_scores(_write(workspace, f"img{i}.jpg", _colourful_lesion(seed=i)), CONFIG)["iqa_composite"]
            for i in range(24)
        ]
    assert len(set(composites)) >= 20
    # Ordering guarantee: the continuous weight stays under the 0.1 gap between defect tiers.
    assert (1.0 - CONTINUOUS_WEIGHT) > 0.9 * (1.0 - CONTINUOUS_WEIGHT) + CONTINUOUS_WEIGHT


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
    assert corner["explainability_plausibility"] < 0.2 < centred["explainability_plausibility"]


def test_calibration_scores_typical_values_above_outliers():
    """What makes the signal meaningful rather than an invented threshold: a synthetic image is
    judged against how REAL images of the same class actually behave."""
    reference = np.array([0.05, 0.07, 0.06, 0.08, 0.05, 0.09])  # real peripheral-mass values
    typical = calibrate_against_reference(0.06, reference, higher_is_better=False)
    outlier = calibrate_against_reference(0.80, reference, higher_is_better=False)
    assert typical > outlier
    assert 0.0 <= outlier <= 1.0 and 0.0 <= typical <= 1.0


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

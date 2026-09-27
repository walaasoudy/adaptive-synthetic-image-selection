#!/usr/bin/env python3
"""HAM10000 ASISM auxiliary classifier: real-only, never gen_train, and usable as a CAM reference.

The guards here exist because this one model's identity is what every calibrated explainability row
downstream will name. A model that quietly saw synthetic images, or the reference split itself,
would not fail loudly later — it would produce a plausible, wrong reference.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.asism.ham10000_signals import ReferenceLeakageError, cam_model_identity  # noqa: E402
from scripts.utils.config import CONFIGS_DIR  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from fixture_workspace import fixture_workspace

ENTRY = REPO / "scripts" / "classify" / "ham10000_train_auxiliary_classifier.py"
NAMESPACE = "fixture-ns"


def _module():
    spec = importlib.util.spec_from_file_location("ham_aux_clf", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_auxiliary():
    return OmegaConf.load(CONFIGS_DIR / "ham10000_stage3.yaml").auxiliary_classifier


def _committed_v2():
    return OmegaConf.load(CONFIGS_DIR / "ham10000_stage3.yaml").auxiliary_classifier_v2


def _fixture(workspace: Path, block=None, **overrides):
    """A tiny real-shaped split pair plus the committed config pointed at it, with a 1-step budget."""
    splits, images = workspace / "splits" / NAMESPACE, workspace / "images" / NAMESPACE
    rng = np.random.default_rng(0)
    for split, per_class in (("classifier_train", 2), ("classifier_val", 1)):
        (images / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for label in CLASSIFIER_TARGET_LABELS:
            for index in range(per_class):
                image_id = f"ISIC_{split[:4]}_{label}_{index}"
                Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(images / split / f"{image_id}.jpg")
                rows.append({"image_id": image_id, "lesion_id": f"L_{image_id}", "dx": label})
        splits.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(splits / f"{split}.csv", index=False)

    auxiliary = OmegaConf.merge(
        block if block is not None else _committed_auxiliary(),
        {
            "checkpoint_dir": str(workspace / "checkpoints"),
            "pretrained_source": "random",
            "resolution": 64,
            "max_steps": 1,
            "batch_size": 4,
        },
        overrides,
    )
    return auxiliary, workspace / "splits", workspace / "images"


# --------------------------------------------------------------------------------------------
# The committed configuration
# --------------------------------------------------------------------------------------------

def test_committed_config_trains_on_classifier_train_and_never_on_the_cam_reference_split():
    auxiliary = _committed_auxiliary()
    module = _module()
    assert auxiliary.train_split == "classifier_train" and auxiliary.selection_split == "classifier_val"
    assert module.CAM_REFERENCE_SPLIT == "gen_train"
    assert auxiliary.train_split not in module.FORBIDDEN_AUXILIARY_SPLITS
    assert auxiliary.architecture == "densenet121" and auxiliary.class_weighting == "none"
    # 512 canvas -> 16x16 last dense block, so the content box (0.125/0.875) lands on cell edges.
    assert int(auxiliary.resolution) == 512 and int(auxiliary.resolution) % 32 == 0
    module.validate_config(auxiliary)


@pytest.mark.parametrize("key", ["train_split", "selection_split"])
@pytest.mark.parametrize("split", ["gen_train", "asism_tuning_heldout", "final_eval_heldout"])
def test_refuses_the_cam_reference_split_and_every_held_out_split(key, split):
    module = _module()
    auxiliary = OmegaConf.merge(_committed_auxiliary(), {key: split})
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(auxiliary)


def test_refuses_measuring_on_the_split_it_trained_on():
    module = _module()
    auxiliary = OmegaConf.merge(_committed_auxiliary(), {"selection_split": "classifier_train"})
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(auxiliary)


def test_refuses_an_architecture_without_the_gradcam_target_layer():
    module = _module()
    auxiliary = OmegaConf.merge(_committed_auxiliary(), {"architecture": "resnet50"})
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(auxiliary)


def test_refuses_class_weighting_on_the_reference_model():
    module = _module()
    auxiliary = OmegaConf.merge(_committed_auxiliary(), {"class_weighting": "inverse_frequency"})
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(auxiliary)


# --------------------------------------------------------------------------------------------
# No synthetic data can reach this model
# --------------------------------------------------------------------------------------------

def test_entry_point_has_no_code_path_that_reads_synthetic_images():
    code = "\n".join(line.split("#")[0] for line in ENTRY.read_text(encoding="utf-8").splitlines())
    for forbidden in ("synthetic_records", "all_candidates", "asism_selected", "stage2", "conditions"):
        assert forbidden not in code, forbidden
    for forbidden in ("binary_cross_entropy", "BCEWithLogits", "stage4_classifier.yaml"):
        assert forbidden not in code, forbidden
    # The CheXpert auxiliary script is named in the docstring on purpose — as the convention this
    # one deliberately departs from. What must not happen is importing or executing it.
    imports = [line for line in code.splitlines() if line.lstrip().startswith(("import ", "from "))]
    for line in imports:
        assert "00_train_auxiliary_classifier" not in line and "scripts.utils.classifier import train_classifier" not in line


def test_provenance_reports_zero_synthetic_images():
    module = _module()
    with fixture_workspace("ham-aux-records") as workspace:
        auxiliary, splits_root, images_root = _fixture(workspace)
        _, _, provenance = module.build_records(auxiliary, splits_root, images_root, NAMESPACE)
    assert provenance["synthetic_images"] == 0 and provenance["synthetic_manifest"] is None
    assert provenance["real_train_images"] == 14 and provenance["selection_images"] == 7
    assert provenance["selection_role"] == "measured_only_never_selected_on"


# --------------------------------------------------------------------------------------------
# One real run, and the identity it mints
# --------------------------------------------------------------------------------------------

def test_one_step_run_writes_its_artifacts_and_a_cam_usable_manifest():
    module = _module()
    with fixture_workspace("ham-aux-run") as workspace:
        auxiliary, splits_root, images_root = _fixture(workspace)
        result = module.run(auxiliary, splits_root, images_root, NAMESPACE, device="cpu")
        out = Path(result["out_dir"])
        model_path = out / "model.pt"
        assert model_path.is_file() and (out / "selection_metrics.json").is_file() and (out / "run_manifest.json").is_file()

        manifest = result["manifest"]
        assert manifest["dataset"] == "ham10000" and manifest["loss"] == "cross_entropy"
        assert manifest["role"] == "asism_auxiliary" and manifest["prediction_activation"] == "softmax"
        assert manifest["data"]["synthetic_images"] == 0
        assert manifest["data"]["real_train_split"] == "classifier_train"
        assert manifest["data"]["selection_split"] == "classifier_val"
        assert manifest["final_eval_heldout_read"] is False
        assert manifest["checkpoint_selection"] == "none_fixed_step_budget"
        assert manifest["class_weighting"] == "none" and manifest["class_weights"] is None
        assert list(manifest["classes"]) == list(CLASSIFIER_TARGET_LABELS)

        # The identity is the sha256 of the checkpoint actually written, not a recomputed guess.
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        assert manifest["cam_model_id"] == f"ham10000-classifier:{digest}"
        assert manifest["cam_model_trained"] is True and manifest["reference_split"] == "gen_train"

        # The guard Stage 3 will apply accepts the manifest this entry point wrote.
        identity = cam_model_identity(manifest, model_path, reference_split="gen_train")
        assert identity["cam_model_id"] == manifest["cam_model_id"]

        assert set(result["metrics"]["per_class"]) == set(CLASSIFIER_TARGET_LABELS)


def test_the_trained_model_exposes_the_layer_gradcam_hooks():
    from scripts.utils.classifier import build_model

    model = build_model(len(CLASSIFIER_TARGET_LABELS), 0.2, "random", seed=42)
    assert hasattr(model.features, "denseblock4"), "ham10000_explainability.gradcam hooks this layer by name"


@pytest.mark.parametrize(
    "damage",
    [
        {"data": {"synthetic_images": 5}},
        {"data": {"real_train_split": "gen_train"}},
        {"data": {"selection_split": "gen_train"}},
        {"data": {"real_train_split": "final_eval_heldout"}},
        {"loss": "bce"},
        {"dataset": "chexpert"},
    ],
)
def test_cam_model_identity_rejects_a_manifest_this_entry_point_would_never_write(damage):
    manifest = {
        "dataset": "ham10000",
        "loss": "cross_entropy",
        "data": {"synthetic_images": 0, "real_train_split": "classifier_train", "selection_split": "classifier_val"},
    }
    broken = {**manifest, **{k: ({**manifest[k], **v} if isinstance(v, dict) else v) for k, v in damage.items()}}
    with pytest.raises(ReferenceLeakageError):
        cam_model_identity(broken, ENTRY, reference_split="gen_train")


# --------------------------------------------------------------------------------------------
# v2: a separate, class-balanced model with orientation flips. v1 must not move.
# --------------------------------------------------------------------------------------------

def test_the_v1_block_is_still_the_plain_model_every_v1_artifact_names():
    auxiliary = _committed_auxiliary()
    assert auxiliary.class_weighting == "none"
    assert "augmentation" not in auxiliary and "acceptance" not in auxiliary
    assert str(auxiliary.checkpoint_dir).endswith("checkpoints/ham10000/asism_auxiliary")
    assert (int(auxiliary.seed), int(auxiliary.max_steps), int(auxiliary.resolution)) == (42, 3000, 512)
    _module().validate_config(auxiliary, "auxiliary_classifier")


def test_the_v1_block_refuses_augmentation():
    module = _module()
    auxiliary = OmegaConf.merge(_committed_auxiliary(), {"augmentation": "flips"})
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(auxiliary, "auxiliary_classifier")


def test_the_v2_block_differs_from_v1_only_in_weighting_flips_acceptance_and_directory():
    v1, v2 = _committed_auxiliary(), _committed_v2()
    _module().validate_config(v2, "auxiliary_classifier_v2")
    assert v2.class_weighting == "inverse_frequency" and v2.augmentation == "flips"
    assert str(v2.checkpoint_dir) != str(v1.checkpoint_dir)
    assert str(v2.checkpoint_dir).endswith("checkpoints/ham10000/asism_auxiliary_v2")
    assert float(v2.acceptance.min_balanced_accuracy_exclusive) == 0.478
    assert bool(v2.acceptance.require_nonzero_recall_every_class) is True
    for key in set(v1) - {"checkpoint_dir", "class_weighting"}:
        assert v2[key] == v1[key], key
    assert set(v2) - set(v1) == {"augmentation", "acceptance"}


@pytest.mark.parametrize(
    "override",
    [
        {"class_weighting": "sqrt_inverse_frequency"},
        {"augmentation": "rotate90"},
        {"augmentation": "flips_and_rotate90"},
    ],
)
def test_the_v2_block_refuses_unknown_weighting_or_any_rotation(override):
    module = _module()
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(OmegaConf.merge(_committed_v2(), override), "auxiliary_classifier_v2")


def test_the_v2_block_refuses_to_train_without_its_acceptance_criteria():
    module = _module()
    v2 = OmegaConf.to_container(_committed_v2())
    del v2["acceptance"]["require_nonzero_recall_every_class"]
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(OmegaConf.create(v2), "auxiliary_classifier_v2")


def test_an_unknown_config_key_is_refused():
    module = _module()
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(_committed_v2(), "auxiliary_classifier_v3")


@pytest.mark.parametrize("key", ["train_split", "selection_split"])
@pytest.mark.parametrize("split", ["gen_train", "asism_tuning_heldout", "final_eval_heldout"])
def test_the_v2_block_refuses_the_same_splits_v1_does(key, split):
    module = _module()
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(OmegaConf.merge(_committed_v2(), {key: split}), "auxiliary_classifier_v2")


# --- the dataset flips ------------------------------------------------------------------------

def _one_image_dataset(workspace: Path, augment: bool):
    from scripts.utils.ham10000_classifier import LesionRecordDataset

    rng = np.random.default_rng(3)
    path = workspace / "lesion.jpg"
    Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(path)
    return LesionRecordDataset([{"image_path": str(path), "class_index": 4}], 32, augment=augment), path


def test_without_augment_the_transform_is_the_old_one_and_draws_no_random_numbers():
    import torch

    with fixture_workspace("ham-aux-noaug") as workspace:
        dataset, path = _one_image_dataset(workspace, augment=False)
        state = torch.get_rng_state()
        item = dataset[0]
        assert torch.equal(state, torch.get_rng_state())
        with Image.open(path) as image:
            expected = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
        assert torch.equal(item["image"], (expected - 0.5) / 0.5)
        assert item["target"] == 4


def test_with_augment_every_output_is_a_flip_of_the_image_and_the_label_never_changes():
    import torch
    from scripts.utils.ham10000_classifier import LesionRecordDataset

    with fixture_workspace("ham-aux-aug") as workspace:
        dataset, _ = _one_image_dataset(workspace, augment=True)
        base = LesionRecordDataset(dataset.records, 32)[0]["image"]
        variants = {
            "identity": base,
            "horizontal": torch.flip(base, dims=[2]),
            "vertical": torch.flip(base, dims=[1]),
            "both": torch.flip(base, dims=[1, 2]),
        }
        torch.manual_seed(0)
        seen = set()
        for _ in range(64):
            item = dataset[0]
            assert item["target"] == 4
            matches = [name for name, tensor in variants.items() if torch.equal(item["image"], tensor)]
            assert matches, "an augmented image must be one of the four flips of the original"
            seen.add(matches[0])
        # 64 draws at p = 0.5 per flip: missing any of the four has probability below 1e-7.
        assert seen == set(variants)


# --- v2 end to end, and the acceptance rule ---------------------------------------------------

def test_one_step_v2_run_uses_balanced_weights_and_flips_and_records_its_acceptance(monkeypatch):
    from scripts.utils.ham10000_classifier import class_weights_from_records

    module = _module()
    captured = {}
    original = module.train_classifier

    def spy(train_records, *args, **kwargs):
        captured["records"], captured["kwargs"] = train_records, kwargs
        return original(train_records, *args, **kwargs)

    monkeypatch.setattr(module, "train_classifier", spy)
    with fixture_workspace("ham-aux-v2-run") as workspace:
        auxiliary, splits_root, images_root = _fixture(workspace, block=_committed_v2())
        result = module.run(
            auxiliary, splits_root, images_root, NAMESPACE, device="cpu", config_key="auxiliary_classifier_v2"
        )
        manifest = result["manifest"]
        model_path = Path(result["out_dir"]) / "model.pt"

        expected = class_weights_from_records(captured["records"])
        assert np.allclose(captured["kwargs"]["class_weights"], expected)
        assert captured["kwargs"]["augment"] is True
        assert manifest["config_key"] == "auxiliary_classifier_v2"
        assert manifest["class_weighting"] == "inverse_frequency" and manifest["augmentation"] == "flips"
        assert np.allclose(manifest["class_weights"], expected) and len(manifest["class_weights"]) == 7
        assert manifest["data"]["synthetic_images"] == 0 and manifest["final_eval_heldout_read"] is False

        acceptance = manifest["acceptance"]
        assert acceptance["evaluated_once_on"] == "classifier_val"
        assert acceptance["passed"] == (acceptance["criterion_a"]["passed"] and acceptance["criterion_b"]["passed"])
        assert set(acceptance["criterion_b"]["per_class_recall"]) == set(CLASSIFIER_TARGET_LABELS)

        identity = cam_model_identity(manifest, model_path, reference_split="gen_train")
        assert identity["cam_model_id"] == manifest["cam_model_id"]


def test_the_v1_run_still_trains_unweighted_and_unaugmented_with_no_acceptance(monkeypatch):
    module = _module()
    captured = {}
    original = module.train_classifier

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "train_classifier", spy)
    with fixture_workspace("ham-aux-v1-run") as workspace:
        auxiliary, splits_root, images_root = _fixture(workspace)
        manifest = module.run(auxiliary, splits_root, images_root, NAMESPACE, device="cpu")["manifest"]
    assert captured["class_weights"] is None and captured["augment"] is False
    assert manifest["config_key"] == "auxiliary_classifier" and manifest["augmentation"] == "none"
    assert "acceptance" not in manifest


def _metrics(balanced: float, recalls: dict) -> dict:
    return {"balanced_accuracy": balanced, "per_class": {label: {"recall": value} for label, value in recalls.items()}}


_ACCEPTANCE = {"min_balanced_accuracy_exclusive": 0.478, "require_nonzero_recall_every_class": True}
_ALL_POSITIVE = {label: 0.5 for label in CLASSIFIER_TARGET_LABELS}


def test_acceptance_passes_only_when_both_criteria_hold():
    verdict = _module().evaluate_acceptance(_metrics(0.55, _ALL_POSITIVE), _ACCEPTANCE)
    assert verdict["criterion_a"]["passed"] and verdict["criterion_b"]["passed"] and verdict["passed"]


@pytest.mark.parametrize("balanced", [0.478, 0.40])
def test_acceptance_fails_at_or_below_the_threshold(balanced):
    verdict = _module().evaluate_acceptance(_metrics(balanced, _ALL_POSITIVE), _ACCEPTANCE)
    assert not verdict["criterion_a"]["passed"] and not verdict["passed"]


@pytest.mark.parametrize("bad", [0.0, float("nan")])
def test_acceptance_fails_when_any_class_recall_is_zero_or_unmeasurable(bad):
    recalls = {**_ALL_POSITIVE, "df": bad}
    verdict = _module().evaluate_acceptance(_metrics(0.60, recalls), _ACCEPTANCE)
    assert verdict["criterion_a"]["passed"]
    assert not verdict["criterion_b"]["passed"] and verdict["criterion_b"]["classes_failing"] == ["df"]
    assert not verdict["passed"]

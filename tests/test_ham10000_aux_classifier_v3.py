#!/usr/bin/env python3
"""The v3a auxiliary classifier: v2 plus a training step chosen by lesion-grouped 5-fold CV on
classifier_train. The folds never split a lesion, the step rule is the pre-registered one, scoring
during training does not change the training, classifier_val never reaches the choice, and the
final model stops at the chosen step.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace  # noqa: E402
from scripts.utils.config import CONFIGS_DIR  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from test_ham10000_aux_classifier import NAMESPACE, _committed_auxiliary, _committed_v2, _module  # noqa: E402

V3 = "auxiliary_classifier_v3"


def _committed_v3():
    return OmegaConf.load(CONFIGS_DIR / "ham10000_stage3.yaml").auxiliary_classifier_v3


def _records(per_class: dict[int, list[int]]) -> list[dict]:
    """per_class: class_index -> list of lesion sizes."""
    records = []
    for class_index, sizes in per_class.items():
        for lesion, size in enumerate(sizes):
            for image in range(size):
                records.append({"image_id": f"I_{class_index}_{lesion}_{image}", "class_index": class_index,
                                "lesion_id": f"L_{class_index}_{lesion}"})
    return records


def _fixture(workspace: Path, per_class_train: int = 5, max_steps: int = 4, every: int = 2):
    splits, images = workspace / "splits" / NAMESPACE, workspace / "images" / NAMESPACE
    rng = np.random.default_rng(0)
    for split, per_class in (("classifier_train", per_class_train), ("classifier_val", 1)):
        (images / split).mkdir(parents=True, exist_ok=True)
        rows = []
        for label in CLASSIFIER_TARGET_LABELS:
            for index in range(per_class):
                image_id = f"ISIC_{split}_{label}_{index}"
                Image.fromarray(rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)).save(images / split / f"{image_id}.jpg")
                rows.append({"image_id": image_id, "lesion_id": f"L_{split}_{label}_{index // 2}", "dx": label})
        splits.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(splits / f"{split}.csv", index=False)
    auxiliary = OmegaConf.merge(_committed_v3(), {
        "checkpoint_dir": str(workspace / "checkpoints"), "pretrained_source": "random",
        "resolution": 32, "max_steps": max_steps, "batch_size": 4,
        "checkpoint_selection": {"eval_every_steps": every},
    })
    return auxiliary, workspace / "splits", workspace / "images"


@pytest.fixture
def small_selection(monkeypatch):
    """The fixture trains for a few steps, so the pre-registered 250-step spacing is shrunk to 2 in
    the validator's copy as well; every other selection constant stays the committed one."""
    module = _module()
    monkeypatch.setitem(module.V3_SELECTION, "eval_every_steps", 2)
    return module


# ==============================================================================================
# The committed configuration
# ==============================================================================================


def test_the_v3_block_is_v2_plus_the_pre_registered_selection_and_its_own_directory():
    v2, v3 = _committed_v2(), _committed_v3()
    module = _module()
    module.validate_config(v3, V3)
    for key in set(v2) - {"checkpoint_dir"}:
        assert v3[key] == v2[key], key
    assert set(v3) - set(v2) == {"checkpoint_selection"}
    assert str(v3.checkpoint_dir).endswith("checkpoints/ham10000/asism_auxiliary_v3")
    assert OmegaConf.to_container(v3.checkpoint_selection) == module.V3_SELECTION == {
        "method": "lesion_grouped_stratified_kfold", "n_folds": 5, "fold_seed": 42,
        "eval_every_steps": 250, "tie_tolerance": 0.005, "tie_breaker": "pooled_oof_nll",
    }
    assert len({str(_committed_auxiliary().checkpoint_dir), str(v2.checkpoint_dir), str(v3.checkpoint_dir)}) == 3


@pytest.mark.parametrize("override", [
    {"checkpoint_selection": {"n_folds": 3}},
    {"checkpoint_selection": {"tie_tolerance": 0.01}},
    {"checkpoint_selection": {"eval_every_steps": 200}},
    {"max_steps": 3100},
])
def test_the_v3_block_refuses_any_other_selection(override):
    module = _module()
    with pytest.raises(module.AuxiliaryClassifierConfigError):
        module.validate_config(OmegaConf.merge(_committed_v3(), override), V3)


def test_the_v3_block_refuses_the_held_out_splits():
    module = _module()
    for split in ("gen_train", "asism_tuning_heldout", "final_eval_heldout"):
        with pytest.raises(module.AuxiliaryClassifierConfigError):
            module.validate_config(OmegaConf.merge(_committed_v3(), {"train_split": split}), V3)


# ==============================================================================================
# Folds and the step rule
# ==============================================================================================


def test_folds_never_split_a_lesion_and_spread_each_class():
    module = _module()
    records = _records({0: [3, 1, 1, 2, 1, 1, 2, 1, 1, 1], 5: [1] * 21, 6: [2, 1, 1, 3, 1, 1, 2]})
    folds = module.lesion_grouped_folds(records, 5, 42)
    by_lesion = {}
    for record, fold in zip(records, folds):
        by_lesion.setdefault(record["lesion_id"], set()).add(int(fold))
    assert all(len(f) == 1 for f in by_lesion.values())
    for class_index in (0, 5, 6):
        counts = np.bincount(folds[[r["class_index"] == class_index for r in records]], minlength=5)
        assert counts.max() - counts.min() <= 3, (class_index, counts)
    counts_vasc = np.bincount(folds[[r["class_index"] == 5 for r in records]], minlength=5)
    assert counts_vasc.tolist() == [5, 4, 4, 4, 4]
    assert np.array_equal(folds, module.lesion_grouped_folds(records, 5, 42))


def test_a_lesion_with_two_diagnoses_is_refused():
    module = _module()
    records = _records({0: [1], 1: [1]})
    records[1]["lesion_id"] = records[0]["lesion_id"]
    with pytest.raises(module.AuxiliaryClassifierConfigError, match="more than one diagnosis"):
        module.lesion_grouped_folds(records, 5, 42)


def test_a_record_without_a_lesion_is_its_own_lesion():
    module = _module()
    records = _records({0: [1] * 10})
    for record in records:
        record["lesion_id"] = None
    folds = module.lesion_grouped_folds(records, 5, 42)
    assert np.bincount(folds, minlength=5).tolist() == [2, 2, 2, 2, 2]


def test_the_pooled_table_is_balanced_accuracy_and_nll():
    module = _module()
    truth = np.array([0, 0, 1, 1])
    p = np.array([[[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]]])
    row = module.pooled_step_table(p, truth, [250])[0]
    assert row["balanced_accuracy"] == pytest.approx((0.5 + 1.0) / 2)
    assert row["nll"] == pytest.approx(-np.mean(np.log([0.9, 0.4, 0.8, 0.7])))


def test_the_best_step_wins_and_a_near_tie_goes_to_the_lower_nll():
    module = _module()
    table = [{"step": 250, "balanced_accuracy": 0.60, "nll": 1.0},
             {"step": 500, "balanced_accuracy": 0.64, "nll": 0.9},
             {"step": 750, "balanced_accuracy": 0.636, "nll": 0.7},   # 0.004 below: tied
             {"step": 1000, "balanced_accuracy": 0.635, "nll": 0.5}]  # exactly 0.005 below: not tied
    chosen = module.select_step(table, 0.005)
    assert chosen["tied_steps"] == [500, 750]
    assert chosen["step"] == 750
    assert module.select_step(table[:2], 0.005)["step"] == 500


def test_an_exact_nll_tie_goes_to_the_earlier_step():
    module = _module()
    table = [{"step": 500, "balanced_accuracy": 0.6, "nll": 0.8}, {"step": 250, "balanced_accuracy": 0.6, "nll": 0.8}]
    assert module.select_step(table, 0.005)["step"] == 250


# ==============================================================================================
# Training
# ==============================================================================================


def _tiny_train(records, callback: bool, steps: int = 4):
    from scripts.utils.classifier import TrainingBudget
    from scripts.utils.ham10000_classifier import predict_probabilities, train_classifier

    budget = TrainingBudget(max_steps=steps, batch_size=4, seed=42, num_workers=0)
    seen = []

    def score(step, model):
        seen.append(step)
        predict_probabilities(model, records[:3], 32, device="cpu")
        import torch
        torch.rand(100)  # anything the callback draws must not reach the training

    kwargs = {"checkpoint_every": 2, "on_checkpoint": score} if callback else {}
    model, history = train_classifier(records, budget, 0.2, "random", 32, device="cpu", augment=True, **kwargs)
    return model, history, seen


def test_scoring_during_training_does_not_change_the_training():
    import torch

    with fixture_workspace("v3-rng") as workspace:
        _fixture(workspace)
        from scripts.utils.ham10000_classifier import records_from_split

        split = workspace / "splits" / NAMESPACE / "classifier_train.csv"
        records = records_from_split(pd.read_csv(split), workspace / "images" / NAMESPACE / "classifier_train")
        plain, plain_history, _ = _tiny_train(records, callback=False, steps=6)
        scored, scored_history, seen = _tiny_train(records, callback=True, steps=6)
    assert seen == [2, 4, 6]
    assert [h["loss"] for h in plain_history] == pytest.approx([h["loss"] for h in scored_history])
    for key, value in plain.state_dict().items():
        assert torch.equal(value, scored.state_dict()[key]), key


def test_checkpoint_every_and_on_checkpoint_come_together():
    from scripts.utils.classifier import TrainingBudget
    from scripts.utils.ham10000_classifier import train_classifier

    with pytest.raises(ValueError, match="together"):
        train_classifier([], TrainingBudget(max_steps=1), 0.2, "random", 32, checkpoint_every=2)


def test_a_v3_run_selects_on_classifier_train_only_and_stops_at_the_chosen_step(small_selection, monkeypatch):
    module = small_selection
    calls = []
    original_train, original_predict = module.train_classifier, module.predict_probabilities

    def spy_train(records, budget, *args, **kwargs):
        calls.append(("train", {r["image_id"] for r in records}, budget.max_steps))
        return original_train(records, budget, *args, **kwargs)

    def spy_predict(model, records, *args, **kwargs):
        calls.append(("predict", {r["image_id"] for r in records}, None))
        return original_predict(model, records, *args, **kwargs)

    monkeypatch.setattr(module, "train_classifier", spy_train)
    monkeypatch.setattr(module, "predict_probabilities", spy_predict)
    with fixture_workspace("v3-run") as workspace:
        auxiliary, splits_root, images_root = _fixture(workspace, max_steps=4, every=2)
        result = module.run(auxiliary, splits_root, images_root, NAMESPACE, device="cpu", config_key=V3)
        out_dir = Path(result["out_dir"])
        cv = json.loads((out_dir / "cv_selection.json").read_text(encoding="utf-8"))
        with np.load(out_dir / "oof_probabilities.npz") as data:
            oof = {key: data[key] for key in data.files}
        train_frame = pd.read_csv(splits_root / NAMESPACE / "classifier_train.csv")
        train_ids = set(train_frame["image_id"])
        val_ids = set(pd.read_csv(splits_root / NAMESPACE / "classifier_val.csv")["image_id"])

    fold_trains = [c for c in calls if c[0] == "train"][:5]
    final_train = [c for c in calls if c[0] == "train"][5]
    assert len([c for c in calls if c[0] == "train"]) == 6
    # Every fold model and every out-of-fold score stays inside classifier_train.
    for kind, ids, _ in calls[: len(calls) - 1]:
        assert ids <= train_ids, kind
    assert all(max_steps == 4 for _, _, max_steps in fold_trains)
    # The one read of classifier_val is the acceptance, after the final model exists.
    assert calls[-1][0] == "predict" and calls[-1][1] == val_ids
    assert final_train[1] == train_ids and final_train[2] == cv["step"]

    assert cv["steps"] == [2, 4] and [row["step"] for row in cv["table"]] == [2, 4]
    assert oof["probabilities"].shape == (2, len(train_ids), 7) and not np.isnan(oof["probabilities"]).any()
    manifest = result["manifest"]
    assert manifest["config_key"] == V3
    selection = manifest["checkpoint_selection"]
    assert selection["selected_step"] == cv["step"] and selection["classifier_val_used_for_selection"] is False
    assert selection["step_rescaled_for_full_data"] is False
    assert manifest["acceptance"]["evaluated_once_on"] == "classifier_val"
    assert manifest["budget"]["max_steps"] == cv["step"]
    folds = np.array(cv["folds"])
    lesion_of = dict(zip(train_frame["image_id"], train_frame["lesion_id"]))
    by_lesion = {}
    for image_id, fold in zip(cv["image_ids"], folds):
        by_lesion.setdefault(lesion_of[image_id], set()).add(int(fold))
    assert all(len(f) == 1 for f in by_lesion.values())


def test_the_v2_run_is_unchanged_by_v3(monkeypatch):
    module = _module()
    captured = {}
    original = module.train_classifier

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "train_classifier", spy)
    from test_ham10000_aux_classifier import _fixture as v2_fixture

    with fixture_workspace("v3-v2-unchanged") as workspace:
        auxiliary, splits_root, images_root = v2_fixture(workspace, block=_committed_v2())
        manifest = module.run(auxiliary, splits_root, images_root, NAMESPACE, device="cpu",
                              config_key="auxiliary_classifier_v2")["manifest"]
        assert not (Path(auxiliary.checkpoint_dir) / NAMESPACE / "cv_selection.json").exists()
    assert "checkpoint_every" not in captured and "on_checkpoint" not in captured
    assert manifest["checkpoint_selection"] == "none_fixed_step_budget"

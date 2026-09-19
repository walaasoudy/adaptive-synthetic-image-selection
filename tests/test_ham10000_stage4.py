#!/usr/bin/env python3
"""HAM10000 Stage 4 entry point: CrossEntropy only, never the CheXpert BCE path, never final eval."""

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

from scripts.utils.config import CONFIGS_DIR  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from fixture_workspace import fixture_workspace

ENTRY = REPO / "scripts" / "classify" / "ham10000_train_conditions.py"


def _module():
    spec = importlib.util.spec_from_file_location("ham_stage4", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workspace_config(workspace: Path, **overrides):
    """The committed config with paths pointed at a tiny real-shaped fixture and a 1-step budget."""
    config = OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml")
    namespace = "fixture-ns"
    splits, images = workspace / "splits" / namespace, workspace / "images" / namespace
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

    synthetic_dir = workspace / "synthetic"
    synthetic_dir.mkdir(exist_ok=True)
    synthetic_rows = []
    for label in ("df", "vasc"):
        path = synthetic_dir / f"syn_{label}.jpg"
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
        synthetic_rows.append({"image_id": path.stem, "image_path": str(path), "dx": label})
    pd.DataFrame(synthetic_rows).to_csv(workspace / "all_candidates.csv", index=False)

    merged = OmegaConf.merge(
        OmegaConf.to_container(config, resolve=False),
        {
            "paths": {
                "project_root": str(workspace),
                "splits_root": str(workspace / "splits"),
                "images_root": str(workspace / "images"),
                "results_dir": str(workspace / "results"),
                "checkpoints_dir": str(workspace / "checkpoints"),
            },
            "split_namespace": namespace,
            "conditions": {"B": {"synthetic_manifest": str(workspace / "all_candidates.csv")}, "C": {"synthetic_manifest": None}},
            "model": {"pretrained_source": "random", "resolution": 64},
            "training": {"max_steps": 1, "batch_size": 4},
        },
        overrides,
    )
    OmegaConf.resolve(merged)
    return merged


def test_committed_config_is_cross_entropy_with_the_ham10000_classes():
    config = OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml")
    assert config.loss == "cross_entropy"
    assert list(config.classes) == list(CLASSIFIER_TARGET_LABELS)
    assert config.selection_split == "classifier_val" and config.real_train_split == "classifier_train"
    _module().validate_config(config)


def test_entry_point_never_touches_the_chexpert_bce_trainer():
    source = ENTRY.read_text(encoding="utf-8")
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    for forbidden in ("binary_cross_entropy", "BCEWithLogits", "sigmoid", "01_train_conditions", "stage4_classifier.yaml"):
        assert forbidden not in code, forbidden
    assert "from scripts.utils.classifier import TrainingBudget" in source
    assert "import train_classifier" not in source.replace("ham10000_classifier import (", "")


def test_one_step_run_uses_cross_entropy_softmax_and_writes_its_artifacts():
    import torch

    module = _module()
    seen = {"ce": 0, "bce": 0, "sigmoid": 0}
    original_ce = torch.nn.CrossEntropyLoss.forward
    original_bce = torch.nn.functional.binary_cross_entropy_with_logits
    original_sigmoid = torch.sigmoid

    def ce_spy(self, logits, target):
        seen["ce"] += 1
        assert target.dtype == torch.int64 and target.dim() == 1 and logits.shape[1] == len(CLASSIFIER_TARGET_LABELS)
        return original_ce(self, logits, target)

    torch.nn.CrossEntropyLoss.forward = ce_spy
    torch.nn.functional.binary_cross_entropy_with_logits = lambda *a, **k: seen.__setitem__("bce", seen["bce"] + 1) or original_bce(*a, **k)
    torch.sigmoid = lambda *a, **k: seen.__setitem__("sigmoid", seen["sigmoid"] + 1) or original_sigmoid(*a, **k)
    try:
        with fixture_workspace("ham-stage4-run") as workspace:
            result = module.run(_workspace_config(workspace), "B", seed=42, device="cpu")
            out = Path(result["out_dir"])
            assert (out / "model.pt").is_file() and (out / "selection_metrics.json").is_file() and (out / "run_manifest.json").is_file()
            manifest = result["manifest"]
    finally:
        torch.nn.CrossEntropyLoss.forward = original_ce
        torch.nn.functional.binary_cross_entropy_with_logits = original_bce
        torch.sigmoid = original_sigmoid

    assert seen["ce"] >= 1 and seen["bce"] == 0 and seen["sigmoid"] == 0, seen
    assert manifest["loss"] == "cross_entropy" and manifest["prediction_activation"] == "softmax"
    assert manifest["class_weighting"] == "none" and manifest["class_weights"] is None
    assert manifest["data"]["synthetic_images"] == 2 and manifest["data"]["real_train_images"] == 14
    assert manifest["data"]["selection_split"] == "classifier_val" and manifest["final_eval_heldout_read"] is False
    assert set(result["metrics"]["per_class"]) == set(CLASSIFIER_TARGET_LABELS)


def _records(real_counts: dict, synthetic_counts: dict) -> list:
    index = {label: i for i, label in enumerate(CLASSIFIER_TARGET_LABELS)}
    records = [{"class_index": index[l], "synthetic": False} for l, n in real_counts.items() for _ in range(n)]
    return records + [{"class_index": index[l], "synthetic": True} for l, n in synthetic_counts.items() for _ in range(n)]


def test_class_weighting_none_passes_no_weights():
    module = _module()
    weights, record = module.resolve_class_weights(OmegaConf.create({"class_weighting": "none", "class_weight_basis": "real_train_split"}), _records({"nv": 5}, {}))
    assert weights is None and record["class_weights"] is None


def test_real_basis_gives_conditions_A_and_B_the_identical_weight_vector():
    """The fairness rule: with inverse_frequency on the real split, adding synthetic images must not
    change the weights, otherwise weighting silently undoes the rebalancing being measured."""
    module = _module()
    real = {"nv": 1085, "mel": 186, "bkl": 188, "bcc": 87, "akiec": 49, "vasc": 21, "df": 25}
    synthetic = {"nv": 112, "mel": 276, "bkl": 273, "bcc": 400, "akiec": 486, "vasc": 736, "df": 885}
    frozen = OmegaConf.create({"class_weighting": "inverse_frequency", "class_weight_basis": "real_train_split"})
    per_condition = OmegaConf.create({"class_weighting": "inverse_frequency", "class_weight_basis": "condition_training_set"})

    a, _ = module.resolve_class_weights(frozen, _records(real, {}))
    b, record_b = module.resolve_class_weights(frozen, _records(real, synthetic))
    assert np.allclose(a, b) and record_b["class_weight_source_images"] == sum(real.values())
    df, nv = CLASSIFIER_TARGET_LABELS.index("df"), CLASSIFIER_TARGET_LABELS.index("nv")
    assert a[df] / a[nv] > 40

    b_shifting, _ = module.resolve_class_weights(per_condition, _records(real, synthetic))
    assert b_shifting[df] / b_shifting[nv] < 2, "the per-condition basis is the confounded one"


def test_entry_point_refuses_an_unknown_class_weight_basis():
    module = _module()
    config = OmegaConf.merge(OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml"), {"training": {"class_weight_basis": "per_batch"}})
    try:
        module.validate_config(config)
    except SystemExit:
        return
    raise AssertionError("unknown class_weight_basis accepted")


def test_entry_point_refuses_a_non_cross_entropy_loss():
    module = _module()
    config = OmegaConf.merge(OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml"), {"loss": "bce"})
    try:
        module.validate_config(config)
    except SystemExit:
        return
    raise AssertionError("HAM10000 Stage 4 accepted loss=bce")


def test_entry_point_refuses_final_eval_as_selection_or_training_split():
    module = _module()
    for key in ("selection_split", "real_train_split"):
        config = OmegaConf.merge(OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml"), {key: "final_eval_heldout"})
        try:
            module.validate_config(config)
        except SystemExit:
            continue
        raise AssertionError(f"Stage 4 accepted {key}=final_eval_heldout")


def test_entry_point_refuses_a_reordered_class_list():
    module = _module()
    config = OmegaConf.merge(OmegaConf.load(CONFIGS_DIR / "ham10000_stage4.yaml"), {"classes": list(reversed(CLASSIFIER_TARGET_LABELS))})
    try:
        module.validate_config(config)
    except SystemExit:
        return
    raise AssertionError("a reordered class list was accepted")


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

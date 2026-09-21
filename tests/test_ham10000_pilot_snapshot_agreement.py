#!/usr/bin/env python3
"""Pilot snapshot comparison: refuses unfair comparisons, and measures without choosing.

The snapshots may differ in exactly one thing — the LoRA. Every test that damages some other input
checks that the comparison is refused rather than computed, because a table built on mismatched
recipes, seeds, code or classifier would look exactly as convincing as a fair one.
"""

from __future__ import annotations

import importlib.util
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_metrics import full_metric_suite  # noqa: E402

ENTRY = REPO / "scripts" / "eval" / "ham10000_pilot_snapshot_agreement.py"
NAMESPACE = "fixture-ns"
SNAPSHOTS = ("step_2000", "step_4000", "step_8000", "final")
CODE = "code-identity-fixture"


def _module():
    spec = importlib.util.spec_from_file_location("ham_pilot_snapshot_agreement", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _module()


def _write_snapshot(root: Path, name: str, step: int, ema: bool, per_class: int = 5, damage=None) -> Path:
    stage2_root = root / "cmp" / name
    pilot = stage2_root / NAMESPACE / "pilot"
    (pilot / "raw").mkdir(parents=True)
    (pilot / "candidates").mkdir(parents=True)
    lora_dir = root / "lora" / name
    lora_dir.mkdir(parents=True)
    (lora_dir / "metadata.json").write_text(json.dumps({"step": step, "ema_applied": ema}), encoding="utf-8")
    provenance = {
        "schema_version": 1, "dataset": "ham10000", "split_namespace": NAMESPACE, "split_manifest_hash": "split",
        "source_split": "gen_train", "recipes_csv_sha256": "recipes", "recipes_manifest_sha256": "recipes-manifest",
        "generation_config_sha256": "gen", "geometry_sha256": "geo", "code_identity_sha256": CODE,
        "lora_train_image_ids_sha256": "train-ids", "lora_weights_dir": str(lora_dir),
        "lora_checkpoint_sha256": f"lora-{name}", "lora_step": step, "checkpoint_config_sha256": f"ckpt-{name}",
    }
    rows = []
    for class_index, label in enumerate(CLASSIFIER_TARGET_LABELS):
        for k in range(per_class):
            image_id = f"syn_{label}_{k:05d}"
            seed = 1000 * class_index + k
            array = np.random.default_rng(seed + step).integers(0, 255, (24, 32, 3), dtype=np.uint8)
            Image.fromarray(array).save(pilot / "raw" / f"{image_id}.jpg")
            Image.fromarray(array).resize((32, 32)).save(pilot / "candidates" / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "dx": label, "class_index": class_index, "seed": seed,
                         "prompt": f"a {label}", "intended_label_vector": {l: int(l == label) for l in CLASSIFIER_TARGET_LABELS},
                         "context_image_id": f"ISIC_{seed}", "status": "accepted",
                         "raw_image_path": str(pilot / "raw" / f"{image_id}.jpg"),
                         "image_path": str(pilot / "candidates" / f"{image_id}.jpg"), **provenance})
    approval = {"pilot_completed": True, "checks": {"passed": True}, "split_provenance": provenance, "approval": {"approved": False}}
    if damage:
        damage(rows, approval, provenance)
    (pilot / "generation_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (pilot / "pilot_approval_manifest.json").write_text(json.dumps(approval), encoding="utf-8")
    return stage2_root


def _snapshots(root: Path, damage_for: dict | None = None) -> dict[str, Path]:
    steps = {"step_2000": (2000, False), "step_4000": (4000, False), "step_8000": (8000, False), "final": (8000, True)}
    return {n: _write_snapshot(root, n, *steps[n], damage=(damage_for or {}).get(n)) for n in SNAPSHOTS}


class FakePredictor:
    """Deterministic stand-in: 'knows' a class when the image id says so, except akiec, which it reads as mel."""

    def __init__(self, recorded=None, remeasured=None):
        truth = np.repeat(np.arange(7), 3)
        probs = np.full((21, 7), 0.02)
        probs[np.arange(21), truth] = 0.88
        self._val = full_metric_suite(probs, truth)
        self._recorded = recorded if recorded is not None else self._val
        self._remeasured = remeasured if remeasured is not None else self._val
        self.calls = []

    def predict(self, records):
        self.calls.append(len(records))
        out = np.full((len(records), 7), 0.05)
        for i, record in enumerate(records):
            target = CLASSIFIER_TARGET_LABELS.index("mel") if record["dx"] == "akiec" else int(record["class_index"])
            out[i, target] = 0.7
        return out / out.sum(axis=1, keepdims=True)

    def recorded_val_metrics(self):
        return self._recorded

    def remeasured_val_metrics(self):
        return self._remeasured

    def describe(self):
        return {"cam_model_id": "fixture-model", "trained_on": "classifier_train", "synthetic_images": 0}


def test_wilson_interval_matches_the_textbook_values():
    low, high = M.wilson_interval(3, 5)
    assert low == pytest.approx(0.2307, abs=1e-3) and high == pytest.approx(0.8824, abs=1e-3)
    assert M.wilson_interval(0, 5)[0] == 0.0 and M.wilson_interval(5, 5)[1] == 1.0


def test_a_fair_comparison_writes_every_output_and_measures_per_class(tmp_path):
    predictor = FakePredictor()
    result = M.run(NAMESPACE, _snapshots(tmp_path), tmp_path / "eval", predictor, CODE)
    out = Path(result["out_dir"])
    for name in ("snapshot_class_agreement.csv", "snapshot_class_mean_probability.csv", "per_image_predictions.csv",
                 "paired_flips.csv", "paired_flips_per_recipe.csv", "confusion_by_snapshot.json", "classifier_val_recall.json",
                 "fairness_check.json", "report.md", "review_sheet.jpg", "evaluation_manifest.json"):
        assert (out / name).is_file(), name
    agreement = result["agreement"].set_index(["snapshot", "class"])
    assert agreement.loc[("final", "akiec"), "k"] == 0 and agreement.loc[("final", "vasc"), "k"] == 5
    assert agreement.loc[("step_2000", "ALL"), "n"] == 35
    assert predictor.calls == [35, 35, 35, 35], "one classifier call per snapshot, every image"
    confusion = json.loads((out / "confusion_by_snapshot.json").read_text())["final"]["rows_true_cols_predicted"]
    assert confusion[CLASSIFIER_TARGET_LABELS.index("akiec")][CLASSIFIER_TARGET_LABELS.index("mel")] == 5
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert manifest["selection_rule_applied"] is None and manifest["pilot_recipes_excluded"] is False
    assert json.loads((out / "fairness_check.json").read_text())["snapshots"]["final"]["ema_applied"] is True


def test_per_image_predictions_keep_every_probability(tmp_path):
    result = M.run(NAMESPACE, _snapshots(tmp_path), tmp_path / "eval", FakePredictor(), CODE)
    import pandas as pd

    table = pd.read_csv(Path(result["out_dir"]) / "per_image_predictions.csv")
    assert len(table) == 140
    assert {f"p_{l}" for l in CLASSIFIER_TARGET_LABELS} <= set(table.columns)
    assert np.allclose(table[[f"p_{l}" for l in CLASSIFIER_TARGET_LABELS]].sum(axis=1), 1.0)


def test_paired_flips_count_each_direction():
    import pandas as pd

    rows = []
    for name, verdicts in (("a", [True, True, False, False]), ("b", [True, False, True, False])):
        for i, v in enumerate(verdicts):
            rows.append({"snapshot": name, "image_id": f"syn_nv_{i}", "dx": "nv", "predicted": "nv" if v else "mel", "correct": v})
    per_recipe, pairs = M.paired_flips(pd.DataFrame(rows), ["a", "b"])
    assert pairs.iloc[0][["both_correct", "only_a_correct", "only_b_correct", "both_wrong"]].tolist() == [1, 1, 1, 1]
    assert per_recipe["verdict_changes"].tolist() == [False, True, True, False]


def _set_row(key, value, index=0):
    def damage(rows, approval, provenance):
        rows[index][key] = value
    return damage


def _set_provenance(key, value):
    def damage(rows, approval, provenance):
        provenance[key] = value
        for row in rows:
            row[key] = value
    return damage


@pytest.mark.parametrize("damage, message", [
    (_set_row("seed", 999999), "seed differs"),
    (_set_row("prompt", "something else"), "prompt differs"),
    (_set_provenance("recipes_csv_sha256", "other-recipes"), "recipes_csv_sha256"),
    (_set_provenance("generation_config_sha256", "other-gen"), "generation_config_sha256"),
    (_set_provenance("code_identity_sha256", "other-code"), "code_identity_sha256"),
    (_set_provenance("lora_train_image_ids_sha256", "other-run"), "lora_train_image_ids_sha256"),
    (_set_row("status", "geometry_rejected"), "not 'accepted'"),
    (_set_row("lora_step", 1), "provenance lora_step differs"),
])
def test_any_difference_other_than_the_lora_refuses_the_comparison(tmp_path, damage, message):
    with pytest.raises(M.ComparisonRefused, match=message):
        M.run(NAMESPACE, _snapshots(tmp_path, {"step_4000": damage}), tmp_path / "eval", FakePredictor(), None)
    assert not (tmp_path / "eval").exists(), "a refused comparison writes nothing"


def test_two_snapshots_with_the_same_lora_are_refused(tmp_path):
    same = _set_provenance("lora_checkpoint_sha256", "lora-final")
    with pytest.raises(M.ComparisonRefused, match="lora_checkpoint_sha256 must differ"):
        M.run(NAMESPACE, _snapshots(tmp_path, {"step_8000": same}), tmp_path / "eval", FakePredictor(), CODE)


def test_a_missing_recipe_is_refused(tmp_path):
    def drop(rows, approval, provenance):
        rows.pop()
    with pytest.raises(M.ComparisonRefused, match="same recipe ids"):
        M.run(NAMESPACE, _snapshots(tmp_path, {"final": drop}), tmp_path / "eval", FakePredictor(), CODE)


def test_a_pilot_that_failed_its_checks_is_refused(tmp_path):
    def failed(rows, approval, provenance):
        approval["checks"]["passed"] = False
    with pytest.raises(M.ComparisonRefused, match="automatic checks did not pass"):
        M.run(NAMESPACE, _snapshots(tmp_path, {"step_2000": failed}), tmp_path / "eval", FakePredictor(), CODE)


def test_pilots_from_another_source_tree_are_refused(tmp_path):
    with pytest.raises(M.ComparisonRefused, match="different source tree"):
        M.run(NAMESPACE, _snapshots(tmp_path), tmp_path / "eval", FakePredictor(), "code-now")


def test_a_classifier_that_does_not_reproduce_its_recorded_metrics_is_refused(tmp_path):
    other = full_metric_suite(np.eye(7)[np.repeat(np.arange(7), 3)][::-1], np.repeat(np.arange(7), 3))
    with pytest.raises(M.ComparisonRefused, match="does not reproduce"):
        M.run(NAMESPACE, _snapshots(tmp_path), tmp_path / "eval", FakePredictor(remeasured=other), CODE)


def test_an_existing_result_is_never_overwritten(tmp_path):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "report.md").write_text("earlier", encoding="utf-8")
    with pytest.raises(M.ComparisonRefused, match="not empty"):
        M.run(NAMESPACE, _snapshots(tmp_path), tmp_path / "eval", FakePredictor(), CODE)


def test_the_script_contains_no_selection_rule():
    source = ENTRY.read_text(encoding="utf-8")
    assert "selection_rule_applied\": None" in source
    for forbidden in ("best_snapshot", "recommended_snapshot", "argmax(agreement", "idxmax"):
        assert forbidden not in source


def test_snapshot_arguments_keep_their_order_and_refuse_duplicates():
    assert list(M.parse_snapshots(["b=/x", "a=/y"])) == ["b", "a"]
    with pytest.raises(SystemExit):
        M.parse_snapshots(["a=/x", "a=/y"])
    with pytest.raises(SystemExit):
        M.parse_snapshots(["no-root"])

"""CPU-only positive and negative producer/consumer round trips."""
from __future__ import annotations

import importlib.util
import json
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fixture_workspace import fixture_workspace
from scripts.asism.signals import write_score_artifact
from scripts.utils.artifact_contracts import ArtifactContractError, config_sha256, require_score_artifact, stage2_paths
from scripts.utils.classifier import TrainingBudget, train_classifier
from scripts.utils.labels import CLASSIFIER_TARGET_LABELS
from scripts.utils.manifest import sha256_file, write_json
from scripts.utils.splits import validate_final_eval_context


def load_script(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def tiny_model(num_labels, _dropout, _source, seed=42):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, num_labels))


def records(workspace: Path, count: int = 6):
    result = []
    for index in range(count):
        path = workspace / f"image-{index}.png"
        array = np.full((16, 16, 3), 30 + index * 20, dtype=np.uint8)
        Image.fromarray(array).save(path)
        result.append({
            "image_id": f"i{index}", "patient_id": f"p{index}", "image_path": str(path),
            "labels": {label: index % 2 for label in CLASSIFIER_TARGET_LABELS}, "is_synthetic": False,
        })
    return result


def test_auxiliary_best_checkpoint_roundtrip_and_refusal():
    import torch
    import scripts.utils.classifier as classifier
    module = load_script("scripts/asism/01_compute_signals.py", "signal_loader_roundtrip")
    original = classifier.build_model
    classifier.build_model = tiny_model
    try:
        with fixture_workspace("aux-roundtrip") as workspace:
            resume = workspace / "aux.pt"
            best = workspace / "aux.best.pt"
            provenance = {"split_namespace": "dev-smoke-v1", "split_manifest_hash": "split", "code_identity_sha256": "code"}
            torch.save({"checkpoint_schema_version": 2, "checkpoint_role": "best_validation_model",
                        "model_state": tiny_model(14, 0.0, "none").state_dict(), "provenance": provenance,
                        "model_config": {"dropout_p": 0.0, "pretrained_source": "none", "resolution": 16}}, best)
            write_json(resume.with_suffix(".manifest.json"), {"best_checkpoint_sha256": sha256_file(best)})
            module.auxiliary_checkpoint_path = lambda _config, _namespace: resume
            model, resolution = module.load_auxiliary_classifier(SimpleNamespace(), "dev-smoke-v1", provenance)
            assert resolution == 16 and model.training is False
            bad = {**provenance, "split_manifest_hash": "different"}
            try:
                module.load_auxiliary_classifier(SimpleNamespace(), "dev-smoke-v1", bad)
            except SystemExit as exc:
                assert "provenance mismatch" in str(exc)
            else:
                raise AssertionError("incompatible auxiliary checkpoint was accepted")
    finally:
        classifier.build_model = original


def test_classifier_resume_and_best_checkpoint_persistence():
    import torch
    import scripts.utils.classifier as classifier
    original = classifier.build_model
    classifier.build_model = tiny_model
    try:
        with fixture_workspace("classifier-resume") as workspace:
            dataset = records(workspace)
            checkpoint = workspace / "resume.pt"
            metadata = {"split_manifest_hash": "split", "split_namespace": "dev-smoke-v1", "config_hash": "cfg"}
            first = TrainingBudget(1, 2, 1e-3, 0.0, 1, 7)
            _, accounting1, _ = train_classifier(dataset, dataset, first, 0.0, "none", 16,
                                                   checkpoint_path=checkpoint, checkpoint_metadata=metadata)
            best = Path(accounting1.extra["best_checkpoint_path"])
            assert checkpoint.is_file() and best.is_file()
            second = TrainingBudget(2, 2, 1e-3, 0.0, 1, 7)
            model, accounting2, _ = train_classifier(dataset, dataset, second, 0.0, "none", 16,
                                                       checkpoint_path=checkpoint, resume_from=checkpoint,
                                                       checkpoint_metadata=metadata)
            assert accounting2.optimizer_steps == 2
            payload = torch.load(best, map_location="cpu", weights_only=False)
            assert payload["checkpoint_role"] == "best_validation_model"
            model.load_state_dict(payload["model_state"])
            try:
                train_classifier(dataset, dataset, second, 0.0, "none", 16, checkpoint_path=checkpoint,
                                 resume_from=checkpoint, checkpoint_metadata={**metadata, "config_hash": "wrong"})
            except ValueError as exc:
                assert "incompatible classifier checkpoint provenance" in str(exc)
            else:
                raise AssertionError("incompatible resume was accepted")
    finally:
        classifier.build_model = original


def test_stage2_generation_manifest_to_stage3_contract():
    producer = load_script("scripts/generate/02_generate_synthetic_images.py", "generation_producer_roundtrip")
    consumer = load_script("scripts/asism/01_compute_signals.py", "generation_consumer_roundtrip")
    with fixture_workspace("generation-roundtrip") as workspace:
        lora = workspace / "lora"; lora.mkdir(); (lora / "weights.safetensors").write_bytes(b"fixture")
        write_json(lora / "metadata.json", {"split_namespace": "dev-smoke-v1", "split_manifest_hash": "split"})
        config = OmegaConf.create({"paths": {"synthetic_root": str(workspace / "synthetic")},
                                   "recipes": {"seed": 1}, "generation": {"seed": 2},
                                   "checkpoint": {"lora_weights_dir": str(lora)}})
        paths = stage2_paths(config, "dev-smoke-v1"); paths["root"].mkdir(parents=True)
        pd.DataFrame({"recipe_id": ["syn-0", "syn-1"]}).to_csv(paths["recipes_path"], index=False)
        producer.namespace_identity = lambda _namespace: {"split_namespace": "dev-smoke-v1", "namespace_class": "dev",
                                                           "split_manifest_hash": "split", "split_manifest_version": 2,
                                                           "split_manifest_file_sha256": "file", "split_frozen": False}
        producer.current_code_identity_hash = lambda: "code"
        write_json(paths["recipes_manifest"], {"schema_version": 2, "split_namespace": "dev-smoke-v1",
                   "namespace_class": "dev", "split_manifest_hash": "split",
                   "recipe_config_sha256": config_sha256(config.recipes),
                   "recipes_csv_sha256": sha256_file(paths["recipes_path"]), "code_identity_sha256": "code"})
        keys = producer.validate_generation_inputs(config, "dev-smoke-v1", paths)
        completion = {**keys, "num_rows": 2}
        frame = pd.DataFrame([{"image_id": f"syn-{i}", **keys} for i in range(2)])
        consumer.validate_generation_manifest_rows(frame, completion)
        frame.loc[1, "split_manifest_hash"] = "stale"
        try:
            consumer.validate_generation_manifest_rows(frame, completion)
        except SystemExit as exc:
            assert "provenance mismatch" in str(exc)
        else:
            raise AssertionError("stale generation row reached Stage 3")


def test_score_sidecar_gonogo_and_tuning_roundtrip():
    gonogo = load_script("scripts/asism/02_gonogo.py", "gonogo_roundtrip")
    tuning = load_script("scripts/asism/03_tune_freeze_select.py", "tuning_roundtrip")
    with fixture_workspace("score-roundtrip") as workspace:
        provenance = {"split_namespace": "dev-smoke-v1", "split_manifest_hash": "split", "code_identity_sha256": "code"}
        frame = pd.DataFrame({"image_id": [f"i{i}" for i in range(25)], "iqa_composite": np.linspace(0, 1, 25)})
        path = workspace / "iqa_scores.parquet"
        write_score_artifact(frame, path, "iqa", provenance)
        require_score_artifact(path, "iqa", provenance)
        checks = {"technical_validity": gonogo.check_technical_validity(frame, "iqa"),
                  "numerical_stability": {"passed": True}, "missing_rate": {"passed": True},
                  "reproducibility": {"passed": True}, "directionality": {"passed": True},
                  "redundancy": {"passed": True}, "usefulness": {"passed": True}}
        assert gonogo.decide(checks)[0] == "include"
        cfg = SimpleNamespace(paths=SimpleNamespace(scores_dir=str(workspace)), _expected_score_provenance=provenance)
        merged = tuning.load_merged_scores(cfg, ["iqa"])
        assert merged["image_id"].tolist() == frame["image_id"].tolist()
        sidecar = path.with_suffix(".provenance.json")
        altered = json.loads(sidecar.read_text(encoding="utf-8")); altered["split_manifest_hash"] = "wrong"
        sidecar.write_text(json.dumps(altered), encoding="utf-8")
        try:
            tuning.load_merged_scores(cfg, ["iqa"])
        except ArtifactContractError:
            pass
        else:
            raise AssertionError("tuning accepted stale score provenance")


def test_stage5_registration_resume_and_second_run_refusal():
    module = load_script("scripts/eval/stage5_evaluate.py", "stage5_registration_roundtrip")
    with fixture_workspace("stage5-registration") as workspace:
        common = {"namespace": "dev-smoke-v1", "split_manifest_hash": "split", "asism_manifest_hash": "asism",
                  "protocol_manifest_hash": "protocol", "checkpoint_hashes": {"A": "hash"},
                  "threshold_policy_hash": "threshold", "code_git_version": "git", "code_identity_sha256": "code",
                  "registered_at_utc": "fixed", "status": "registered"}
        first_path = workspace / "run-1" / "final_eval_registration.json"
        first = {**common, "final_eval_run_id": "run-1"}
        module.register_final_evaluation(first_path, first)
        validate_final_eval_context(first_path, "run-1", "dev-smoke-v1")
        assert module.register_final_evaluation(first_path, first)["final_eval_run_id"] == "run-1"
        prediction_path = first_path.parent / "predictions" / "A.parquet"
        prediction_frame = pd.DataFrame({"image_id": ["i1", "i2"], "patient_id": ["p1", "p2"],
                                         **{label: [0.1, 0.9] for label in CLASSIFIER_TARGET_LABELS}})
        prediction_provenance = {"checkpoint_hash": "checkpoint", "split_hash": "split",
                                 "protocol_hash": "protocol", "threshold_hash": "threshold", "code_version": "git"}
        module.publish_predictions_atomic(prediction_path, prediction_frame, prediction_provenance,
                                          ["i1", "i2"], ["p1", "p2"])
        module.validate_completed_predictions(prediction_path, prediction_provenance, ["i1", "i2"], ["p1", "p2"])
        try:
            module.validate_completed_predictions(prediction_path, {**prediction_provenance, "checkpoint_hash": "wrong"},
                                                  ["i1", "i2"], ["p1", "p2"])
        except ValueError as exc:
            assert "provenance mismatch" in str(exc)
        else:
            raise AssertionError("incompatible Stage 5 technical resume was accepted")
        module.mark_final_outcome_access(first_path)
        second_path = workspace / "run-2" / "final_eval_registration.json"
        try:
            module.register_final_evaluation(second_path, {**common, "final_eval_run_id": "run-2"})
        except SystemExit as exc:
            assert "already accessed" in str(exc)
        else:
            raise AssertionError("second methodological run was accepted")


if __name__ == "__main__":
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = failed = 0
    for name, function in tests:
        try:
            function(); print(f"  PASS  {name}"); passed += 1
        except Exception:
            print(f"  FAIL  {name}"); traceback.print_exc(); failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(tests)} total")
    raise SystemExit(1 if failed else 0)

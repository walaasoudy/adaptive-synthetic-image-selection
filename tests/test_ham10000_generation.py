#!/usr/bin/env python3
"""HAM10000 Stage 2: recipes from gen_train only, LoRA/recipe contamination guards, deterministic
seeded generation, geometry standardisation, manifest consistency, resume/idempotency and the pilot
gate — end to end through main() with a fake diffusion pipeline (no GPU, no downloads)."""

from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.generate import ham10000_generate_synthetic_images as gen  # noqa: E402
from scripts.generate.ham10000_recipes import derive_image_seed  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, DIAGNOSIS_PHRASE  # noqa: E402
from fixture_workspace import fixture_workspace

SPLITS = ["gen_train", "gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"]
NAMESPACE = "fixture-ns"


class FakePipeline:
    """Deterministic stand-in for SDXL: each image depends only on its generator's seed. Prompts
    for dermatofibroma come back with a flat bar on top, i.e. generator-drawn padding."""

    def __init__(self):
        self.calls = 0
        self.images_made = 0

    def __call__(self, prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, generator):
        self.calls += 1
        images = []
        for text, g in zip(prompt, generator):
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=g))
            array = np.random.default_rng(seed).integers(20, 235, (height, width, 3), dtype=np.uint8)
            if DIAGNOSIS_PHRASE["df"] in text:
                array[: height // 6] = 128
            images.append(Image.fromarray(array))
        self.images_made += len(images)
        return type("Result", (), {"images": images})()


def _write_fixture(workspace: Path, lora_overrides: dict | None = None) -> dict:
    rng = np.random.default_rng(0)
    split_dir = workspace / "splits" / NAMESPACE
    split_dir.mkdir(parents=True)
    for split in SPLITS:
        per_class = 3 if split == "gen_train" else 1
        rows = [
            {"image_id": f"ISIC_{split}_{label}_{i}", "lesion_id": f"L_{split}_{label}_{i}", "dx": label,
             "age": int(rng.integers(20, 80)), "sex": ["male", "female"][i % 2], "localization": ["back", "face", "unknown"][i % 3]}
            for label in CLASSIFIER_TARGET_LABELS for i in range(per_class)
        ]
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)
    (split_dir / "split_manifest_v2.json").write_text(json.dumps({"manifest_hash": "fixture-split-hash"}), encoding="utf-8")

    lora_dir = workspace / "lora" / "final"
    lora_dir.mkdir(parents=True)
    (lora_dir / "pytorch_lora_weights.safetensors").write_bytes(b"fixture-weights")
    metadata = {
        "dataset": "ham10000", "split_namespace": NAMESPACE, "split_manifest_hash": "fixture-split-hash", "train_split": "gen_train",
        "base_model_id": "stabilityai/stable-diffusion-xl-base-1.0", "base_model_revision": "462165984030d82259a11f4367a4eed129e94a7b",
        "lora_training_size": [64, 48], "train_image_ids_sha256": "fixture-ids", "step": 10,
    }
    metadata.update(lora_overrides or {})
    (lora_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    overlay = {
        "ham_stage1": {"geometry": {"generation_size": [64, 48], "lora_training_size": [64, 48]}, "data": {"resolution": 64}},
        "ham_stage2": {
            "paths": {"stage2_root": str(workspace / "stage2"), "splits_root": str(workspace / "splits")},
            "split_namespace": NAMESPACE,
            "checkpoint": {"lora_weights_dir": str(lora_dir)},
            "generation": {"width": 64, "height": 48, "batch_size": 3, "num_inference_steps": 1},
            "recipes": {"base_quota": 3, "min_per_class": 2, "max_per_class": 4},
            "pilot": {"num_images_per_class": 1, "checks": {"max_geometry_rejected_fraction": 0.2}},
        },
    }
    overlay_path = workspace / "overlay.yaml"
    overlay_path.write_text(OmegaConf.to_yaml(OmegaConf.create(overlay)), encoding="utf-8")
    return {"split_dir": split_dir, "lora_dir": lora_dir, "overlay": overlay_path}


@contextmanager
def _overlay(path: Path):
    previous = os.environ.get("THESIS_CONFIG_OVERLAY")
    os.environ["THESIS_CONFIG_OVERLAY"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("THESIS_CONFIG_OVERLAY", None)
        else:
            os.environ["THESIS_CONFIG_OVERLAY"] = previous


def _configs():
    from scripts.utils.config import load_named_config

    cfg = gen.load_config()
    return cfg, load_named_config("ham10000_stage1.yaml", "ham_stage1")


def _expect(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{getattr(function, '__name__', function)} did not raise {error.__name__}")


# ---------------------------------------------------------------- config

def test_committed_config_is_consistent_with_the_geometry_contract():
    from scripts.utils.config import load_named_config

    gen.validate_config(gen.load_config(), load_named_config("ham10000_stage1.yaml", "ham_stage1"))


def test_config_refuses_a_non_gen_train_recipe_source_or_mismatched_size():
    from scripts.utils.config import load_named_config

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    _expect(gen.GenerationContractError, gen.validate_config, gen.load_config(["recipes.source_split=final_eval_heldout"]), stage1)
    _expect(gen.GenerationContractError, gen.validate_config, gen.load_config(["generation.width=1024"]), stage1)


# ---------------------------------------------------------------- recipes

def test_recipes_are_single_class_from_gen_train_only_and_deterministic():
    with fixture_workspace("ham-gen-recipes") as workspace:
        fixture = _write_fixture(workspace)
        with _overlay(fixture["overlay"]):
            cfg, stage1 = _configs()
            first, manifest = gen.build_recipes(cfg, stage1, NAMESPACE, fixture["split_dir"])
            second, _ = gen.build_recipes(cfg, stage1, NAMESPACE, fixture["split_dir"])
        gen_train = pd.read_csv(fixture["split_dir"] / "gen_train.csv")
    pd.testing.assert_frame_equal(first, second)
    assert first["recipe_id"].str.startswith("syn_").all() and not first["recipe_id"].duplicated().any()
    lookup = gen_train.set_index("image_id")["dx"]
    for row in first.itertuples():
        assert json.loads(row.intended_label_vector) == {label: int(label == row.dx) for label in CLASSIFIER_TARGET_LABELS}
        assert lookup[row.context_image_id] == row.dx, "context must come from a gen_train image of the same class"
        assert DIAGNOSIS_PHRASE[row.dx] in row.prompt and row.source_split == "gen_train"
        assert row.seed == derive_image_seed(int(cfg.generation.seed), row.recipe_id)
    assert manifest["source_split"] == "gen_train" and set(manifest["forbidden_splits_checked"]) == set(SPLITS) - {"gen_train"}


def test_recipes_FAIL_when_gen_train_contains_a_final_eval_image():
    with fixture_workspace("ham-gen-recipes-leak") as workspace:
        fixture = _write_fixture(workspace)
        split_dir = fixture["split_dir"]
        leaked = pd.concat([pd.read_csv(split_dir / "gen_train.csv"), pd.read_csv(split_dir / "final_eval_heldout.csv").iloc[[0]]])
        leaked.to_csv(split_dir / "gen_train.csv", index=False)
        with _overlay(fixture["overlay"]):
            cfg, stage1 = _configs()
            _expect(SystemExit, gen.build_recipes, cfg, stage1, NAMESPACE, split_dir)


# ---------------------------------------------------------------- input validation / contamination

def _build(fixture) -> None:
    with _overlay(fixture["overlay"]):
        assert gen.main(["--build-recipes"]) == 0


def test_generation_refuses_a_lora_not_trained_on_gen_train_or_on_another_split_run():
    for bad in ({"train_split": "classifier_train"}, {"split_manifest_hash": "other"}, {"dataset": "chexpert"}, {"lora_training_size": [768, 768]}):
        with fixture_workspace("ham-gen-bad-lora") as workspace:
            fixture = _write_fixture(workspace, lora_overrides=bad)
            _build(fixture)
            with _overlay(fixture["overlay"]):
                cfg, stage1 = _configs()
                _expect(gen.GenerationContractError, gen.validate_generation_inputs, cfg, stage1, NAMESPACE, fixture["split_dir"], gen.stage2_paths(cfg, NAMESPACE))


def test_generation_refuses_without_an_explicit_lora_checkpoint():
    with fixture_workspace("ham-gen-no-lora") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        with _overlay(fixture["overlay"]):
            cfg = gen.load_config(["checkpoint.lora_weights_dir=null"])
            _, stage1 = _configs()
            _expect(gen.GenerationContractError, gen.validate_generation_inputs, cfg, stage1, NAMESPACE, fixture["split_dir"], gen.stage2_paths(cfg, NAMESPACE))


def test_generation_refuses_tampered_recipes_and_real_id_collisions():
    with fixture_workspace("ham-gen-tamper") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        with _overlay(fixture["overlay"]):
            cfg, stage1 = _configs()
            paths = gen.stage2_paths(cfg, NAMESPACE)
            recipes = pd.read_csv(paths["recipes_path"])
            original = recipes.copy()
            recipes.loc[0, "dx"] = "mel"
            recipes.to_csv(paths["recipes_path"], index=False)
            _expect(gen.GenerationContractError, gen.validate_generation_inputs, cfg, stage1, NAMESPACE, fixture["split_dir"], paths)

            # Collision: rename a recipe to a real final-eval image id and re-sign the manifest, so
            # only the id-collision guard (not the tamper hash) can catch it.
            from scripts.utils.manifest import sha256_file

            collided = original.copy()
            collided.loc[0, "recipe_id"] = pd.read_csv(fixture["split_dir"] / "final_eval_heldout.csv")["image_id"].iloc[0]
            collided.to_csv(paths["recipes_path"], index=False)
            manifest = json.loads(paths["recipes_manifest"].read_text(encoding="utf-8"))
            manifest["recipes_csv_sha256"] = sha256_file(paths["recipes_path"])
            paths["recipes_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            _expect(gen.GenerationContractError, gen.validate_generation_inputs, cfg, stage1, NAMESPACE, fixture["split_dir"], paths)


def test_recipes_are_immutable_once_written():
    with fixture_workspace("ham-gen-immutable") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        with _overlay(fixture["overlay"]):
            _expect(gen.GenerationContractError, gen.main, ["--build-recipes"])


# ---------------------------------------------------------------- end-to-end: pilot gate, generation, resume

def test_full_generation_end_to_end_with_gate_validation_and_idempotent_resume():
    with fixture_workspace("ham-gen-e2e") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        fake = FakePipeline()
        factory = lambda _c, _s: fake  # noqa: E731
        with _overlay(fixture["overlay"]):
            _expect(SystemExit, gen.main, ["--mode", "full"], factory)  # no pilot yet
            assert gen.main(["--mode", "pilot"], factory) == 0
            _expect(SystemExit, gen.main, ["--mode", "full"], factory)  # pilot not approved
            assert gen.main(["--approve-pilot", "--reviewer", "fixture"]) == 0
            assert gen.main(["--mode", "full"], factory) == 0

            cfg, stage1 = _configs()
            paths = gen.stage2_paths(cfg, NAMESPACE)
            recipes = pd.read_csv(paths["recipes_path"])
            rows = [json.loads(line) for line in paths["full_manifest"].read_text(encoding="utf-8").splitlines()]
            completion = json.loads(paths["generation_completion"].read_text(encoding="utf-8"))
            candidates = pd.read_csv(paths["all_candidates_csv"])
            boxes = pd.read_csv(paths["full_candidates"] / "content_boxes.csv")

            assert len(rows) == len(recipes) and completion["status"] == "complete"
            rejected = [r for r in rows if r["status"] == "geometry_rejected"]
            accepted = [r for r in rows if r["status"] == "accepted"]
            assert {r["dx"] for r in rejected} == {"df"}, "generator-drawn padding must be refused, not boxed"
            assert set(candidates["image_id"]) == {r["image_id"] for r in accepted} == set(boxes["image_id"])
            assert "df" not in set(candidates["dx"])
            for row in accepted:
                assert row["source_split"] == "gen_train" and row["lora_checkpoint_sha256"] and row["code_identity_sha256"]
                assert row["content_box"] == [0.0, 0.125, 1.0, 0.875]
                with Image.open(row["image_path"]) as image:
                    assert image.size == (64, 64) and image.mode == "RGB"

            made = fake.images_made
            assert gen.main(["--mode", "full"], factory) == 0
            assert fake.images_made == made, "a complete rerun must generate nothing"
            assert len(paths["full_manifest"].read_text(encoding="utf-8").splitlines()) == len(recipes), "no duplicate rows"

            victim = accepted[0]
            Path(victim["image_path"]).unlink()
            assert gen.main(["--mode", "full"], factory) == 0
            assert fake.images_made == made + 1, "only the missing image is regenerated"
            with Image.open(victim["raw_image_path"]) as image:
                regenerated = np.asarray(image)

            consumer = _stage4_module().synthetic_records(paths["all_candidates_csv"])
            assert len(consumer) == len(candidates) and all(r["synthetic"] for r in consumer)
        again = _independent_run(workspace / "second")
        with Image.open(Path(again) / f"{victim['image_id']}.jpg") as image:
            assert np.array_equal(np.asarray(image), regenerated), "same seed must reproduce the same image"


def _independent_run(workspace: Path) -> Path:
    workspace.mkdir()
    fixture = _write_fixture(workspace)
    _build(fixture)
    fake = FakePipeline()
    with _overlay(fixture["overlay"]):
        cfg, stage1 = _configs()
        paths = gen.stage2_paths(cfg, NAMESPACE)
        recipes, provenance = gen.validate_generation_inputs(cfg, stage1, NAMESPACE, fixture["split_dir"], paths)
        gen.run_generation(recipes, cfg, stage1, paths, "full", provenance, lambda _c, _s: fake)
    return paths["full_raw"]


def _stage4_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ham_stage4", REPO / "scripts" / "classify" / "ham10000_train_conditions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_refuses_unregistered_images_and_stale_manifest_provenance():
    with fixture_workspace("ham-gen-resume-guards") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        fake = FakePipeline()
        with _overlay(fixture["overlay"]):
            cfg, stage1 = _configs()
            paths = gen.stage2_paths(cfg, NAMESPACE)
            recipes, provenance = gen.validate_generation_inputs(cfg, stage1, NAMESPACE, fixture["split_dir"], paths)

            stray = paths["full_raw"] / f"{recipes['recipe_id'].iloc[0]}.jpg"
            stray.parent.mkdir(parents=True)
            Image.fromarray(np.full((48, 64, 3), 90, dtype=np.uint8)).save(stray, format="JPEG")
            _expect(gen.GenerationContractError, gen.run_generation, recipes, cfg, stage1, paths, "full", provenance, lambda _c, _s: fake)
            stray.unlink()

            gen.run_generation(recipes, cfg, stage1, paths, "full", provenance, lambda _c, _s: fake)
            _expect(gen.GenerationContractError, gen.run_generation, recipes, cfg, stage1, paths, "full", {**provenance, "lora_checkpoint_sha256": "different"}, lambda _c, _s: fake)


def test_output_validation_catches_a_manifest_that_disagrees_with_its_recipes():
    with fixture_workspace("ham-gen-validate") as workspace:
        fixture = _write_fixture(workspace)
        _build(fixture)
        fake = FakePipeline()
        with _overlay(fixture["overlay"]):
            cfg, stage1 = _configs()
            paths = gen.stage2_paths(cfg, NAMESPACE)
            recipes, provenance = gen.validate_generation_inputs(cfg, stage1, NAMESPACE, fixture["split_dir"], paths)
            gen.run_generation(recipes, cfg, stage1, paths, "full", provenance, lambda _c, _s: fake)
            gen.validate_generation_outputs(recipes, cfg, stage1, paths, "full", provenance)
            rows = [json.loads(l) for l in paths["full_manifest"].read_text(encoding="utf-8").splitlines()]
            target = next(r for r in rows if r["status"] == "accepted")
            target["dx"] = "vasc" if target["dx"] != "vasc" else "mel"
            paths["full_manifest"].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            _expect(gen.GenerationContractError, gen.validate_generation_outputs, recipes, cfg, stage1, paths, "full", provenance)


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

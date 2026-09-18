#!/usr/bin/env python3
"""HAM10000 Stage 1 LoRA: config validation, data-role isolation, non-square caching, and a real
train -> checkpoint -> resume round trip on a tiny SDXL-shaped UNet (CPU, no downloads)."""

from __future__ import annotations

import importlib.util
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train import ham10000_train_lora_sdxl as lora  # noqa: E402
from scripts.utils.ham10000_lora_data import LoraRoleError, assert_role_isolation, native_aspect_resize  # noqa: E402
from fixture_workspace import fixture_workspace

SPLITS = ["gen_train", "gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"]


def _prep_module():
    spec = importlib.util.spec_from_file_location("ham_prep_lora", REPO / "scripts" / "data" / "ham10000" / "04_prepare_lora_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg(workspace: Path | None = None, **overrides):
    cfg = lora.load_config()
    patch = {"geometry": {"lora_training_size": [64, 48]}}
    if workspace is not None:
        patch["paths"] = {
            "lora_inputs_dir": str(workspace / "lora_inputs"),
            "lora_cache_dir": str(workspace / "lora_cache"),
            "checkpoints_dir": str(workspace / "checkpoints"),
            "logs_dir": str(workspace / "logs"),
        }
        patch["split"] = {"namespace": "fixture-ns"}
    merged = OmegaConf.merge(cfg, patch, overrides)
    OmegaConf.resolve(merged)
    return merged


def _expect(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


# ---------------------------------------------------------------- config validation

def test_committed_config_validates_and_pins_roles():
    cfg = lora.load_config()
    lora.validate_config(cfg)
    assert cfg.split.train_split == "gen_train" and cfg.split.monitor_split == "gen_val"
    assert list(cfg.geometry.lora_training_size) == list(cfg.geometry.generation_size) == [768, 576]


def test_cli_overrides_reach_the_config():
    cfg = lora.load_config(["training.max_train_steps=1234", "optimizer.learning_rate=5e-5"])
    assert cfg.training.max_train_steps == 1234 and abs(cfg.optimizer.learning_rate - 5e-5) < 1e-12


def test_config_refuses_any_other_split_as_a_training_or_monitor_role():
    for key, split in [("train_split", "final_eval_heldout"), ("train_split", "classifier_train"), ("monitor_split", "asism_tuning_heldout"), ("monitor_split", "classifier_val")]:
        _expect(LoraRoleError, lora.validate_config, _cfg(split={key: split}))


def test_config_refuses_bad_geometry_precision_and_budgets():
    _expect(ValueError, lora.validate_config, _cfg(geometry={"lora_training_size": [100, 75]}))   # not /8
    _expect(ValueError, lora.validate_config, _cfg(geometry={"lora_training_size": [64, 64]}))    # not 4:3
    _expect(ValueError, lora.validate_config, _cfg(training={"precision": "int8"}))
    _expect(ValueError, lora.validate_config, _cfg(training={"max_train_steps": 0}))
    _expect(ValueError, lora.validate_config, _cfg(captions={"template_version": "v1"}))
    _expect(ValueError, lora.validate_config, _cfg(model={"revision": ""}))


def test_training_identity_changes_with_rank_but_not_with_logging():
    base = lora.training_identity_hash(_cfg())
    assert lora.training_identity_hash(_cfg(lora={"rank": 8})) != base
    assert lora.training_identity_hash(_cfg(logging={"tensorboard": False})) == base


# ---------------------------------------------------------------- inputs + role isolation

def test_native_aspect_resize_never_crops_or_pads():
    out = native_aspect_resize(Image.new("RGB", (600, 450), (200, 120, 90)), (768, 576))
    assert out.size == (768, 576) and out.mode == "RGB"
    _expect(ValueError, native_aspect_resize, Image.new("RGB", (600, 600)), (768, 576))


def _fixture_dataset(workspace: Path, leak_final_eval_into_gen_train: bool = False) -> Path:
    rng = np.random.default_rng(0)
    raw = workspace / "raw" / "part1"
    raw.mkdir(parents=True)
    split_dir = workspace / "splits" / "fixture-ns"
    split_dir.mkdir(parents=True)
    labels = ["nv", "mel", "df"]
    for split in SPLITS:
        rows = []
        for i in range(3):
            image_id = f"ISIC_{split}_{i}"
            Image.fromarray(rng.integers(30, 220, (45, 60, 3), dtype=np.uint8)).save(raw / f"{image_id}.jpg")
            rows.append({"image_id": image_id, "lesion_id": f"L_{image_id}", "dx": labels[i], "age": 50, "sex": "male", "localization": "back"})
        pd.DataFrame(rows).to_csv(split_dir / f"{split}.csv", index=False)
    if leak_final_eval_into_gen_train:
        gen = pd.read_csv(split_dir / "gen_train.csv")
        leaked = pd.read_csv(split_dir / "final_eval_heldout.csv").iloc[[0]]
        pd.concat([gen, leaked]).to_csv(split_dir / "gen_train.csv", index=False)
    (split_dir / "split_manifest_v2.json").write_text(json.dumps({"manifest_hash": "fixture-hash"}), encoding="utf-8")
    return split_dir


def _prepare(workspace: Path, cfg, split_dir: Path) -> None:
    prep = _prep_module()
    from scripts.utils.manifest import sha256_file

    for split in ("gen_train", "gen_val"):
        frame = pd.read_csv(split_dir / f"{split}.csv")
        prep.prepare_split(split, frame, workspace / "raw", ["part1"], Path(cfg.paths.lora_inputs_dir) / "fixture-ns", cfg, {
            "split_namespace": "fixture-ns", "split_manifest_hash": "fixture-hash",
            "split_csv_sha256": sha256_file(split_dir / f"{split}.csv"), "forbidden_splits_checked": [],
        })


def test_prepared_inputs_load_with_full_provenance():
    with fixture_workspace("ham-lora-inputs") as workspace:
        cfg = _cfg(workspace)
        split_dir = _fixture_dataset(workspace)
        _prepare(workspace, cfg, split_dir)
        records, provenance = lora.load_role_inputs(cfg, split_dir)
        with Image.open(Path(cfg.paths.lora_inputs_dir) / "fixture-ns" / records["gen_train"][0]["image_relpath"]) as image:
            assert image.size == (64, 48) and image.mode == "RGB"
    assert set(records) == {"gen_train", "gen_val"} and len(records["gen_train"]) == 3
    assert provenance["train_split"] == "gen_train" and provenance["monitor_split"] == "gen_val"
    assert set(provenance["forbidden_splits_checked"]) == {"classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"}
    assert len(records["gen_train"][0]["caption_variants"]) == 4


def test_training_inputs_FAIL_when_a_final_eval_image_is_in_gen_train():
    with fixture_workspace("ham-lora-leak") as workspace:
        cfg = _cfg(workspace)
        split_dir = _fixture_dataset(workspace, leak_final_eval_into_gen_train=True)
        _prepare(workspace, cfg, split_dir)
        _expect(LoraRoleError, lora.load_role_inputs, cfg, split_dir)


def test_role_isolation_is_checked_by_image_id():
    forbidden = {"final_eval_heldout": frozenset({"ISIC_X"})}
    _expect(LoraRoleError, assert_role_isolation, ["ISIC_A", "ISIC_X"], "gen_train", forbidden)
    assert_role_isolation(["ISIC_A"], "gen_train", forbidden)


def test_training_inputs_FAIL_when_captions_changed_after_preparation():
    with fixture_workspace("ham-lora-stale") as workspace:
        cfg = _cfg(workspace)
        split_dir = _fixture_dataset(workspace)
        _prepare(workspace, cfg, split_dir)
        captions = Path(cfg.paths.lora_inputs_dir) / "fixture-ns" / "gen_train_captions.jsonl"
        captions.write_text(captions.read_text(encoding="utf-8").replace("melanoma", "melanocytic nevus"), encoding="utf-8")
        _expect(ValueError, lora.load_role_inputs, cfg, split_dir)


def test_training_inputs_FAIL_when_the_split_csv_changed_after_preparation():
    with fixture_workspace("ham-lora-stale-split") as workspace:
        cfg = _cfg(workspace)
        split_dir = _fixture_dataset(workspace)
        _prepare(workspace, cfg, split_dir)
        frame = pd.read_csv(split_dir / "gen_val.csv")
        frame.iloc[:2].to_csv(split_dir / "gen_val.csv", index=False)
        _expect(ValueError, lora.load_role_inputs, cfg, split_dir)


# ---------------------------------------------------------------- non-square dataset

def _fake_caches(n: int, width: int, height: int):
    records = [{"image_id": f"I{i}", "caption_variants": [f"cap {i} a", f"cap {i} b"]} for i in range(n)]
    latent = {"image_ids": [r["image_id"] for r in records], "moments": torch.zeros(n, 8, height // 8, width // 8, dtype=torch.float16), "scaling_factor": 0.13, "size": [width, height]}
    prompts = {c: {"prompt_embeds": torch.zeros(77, 12, dtype=torch.float16), "pooled_embeds": torch.zeros(8, dtype=torch.float16)} for r in records for c in r["caption_variants"]}
    return latent, prompts, records


def test_dataset_uses_non_square_latents_and_sdxl_time_ids_height_first():
    latent, prompts, records = _fake_caches(3, 768, 576)
    item = lora.HAMCachedSDXLDataset(latent, prompts, records, 768, 576)[0]
    assert tuple(item["latents"].shape) == (4, 72, 96)
    assert item["time_ids"].tolist() == [576.0, 768.0, 0.0, 0.0, 576.0, 768.0]


def test_dataset_refuses_a_cache_of_the_wrong_size_or_order():
    latent, prompts, records = _fake_caches(3, 768, 576)
    _expect(ValueError, lora.HAMCachedSDXLDataset, latent, prompts, records, 768, 768)
    _expect(ValueError, lora.HAMCachedSDXLDataset, latent, prompts, list(reversed(records)), 768, 576)


# ---------------------------------------------------------------- resume safety

def test_resume_refuses_any_data_or_training_identity_difference():
    provenance = {key: f"value-{key}" for key in lora.RESUME_MUST_MATCH}
    lora.check_resume_compatible(dict(provenance), provenance)
    for key in ("train_image_ids_sha256", "split_manifest_hash", "training_identity_sha256", "train_split"):
        _expect(ValueError, lora.check_resume_compatible, {**provenance, key: "different"}, provenance)


def test_auto_resume_finds_the_newest_checkpoint_of_the_latest_run():
    with fixture_workspace("ham-lora-autoresume") as workspace:
        assert lora.find_auto_resume(workspace) == (None, None)
        (workspace / "latest_run.json").write_text(json.dumps({"run_id": "r1"}), encoding="utf-8")
        assert lora.find_auto_resume(workspace) == ("r1", None)
        checkpoint = workspace / "r1" / "checkpoint-4"
        checkpoint.mkdir(parents=True)
        (checkpoint / "metadata.json").write_text("{}", encoding="utf-8")
        (workspace / "r1" / "latest.json").write_text(json.dumps({"checkpoint_dir": str(checkpoint), "step": 4}), encoding="utf-8")
        assert lora.find_auto_resume(workspace) == ("r1", checkpoint)


# ---------------------------------------------------------------- real loop: train, checkpoint, resume

def _tiny_unet(seed: int):
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig

    torch.manual_seed(seed)
    unet = UNet2DConditionModel(
        sample_size=None, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=12, attention_head_dim=2, norm_num_groups=4, addition_embed_type="text_time",
        addition_time_embed_dim=4, projection_class_embeddings_input_dim=6 * 4 + 8,
    )
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=2, lora_alpha=2, init_lora_weights="gaussian", target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    return unet


def _loop_parts(workspace: Path, seed: int, max_steps: int):
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader

    cfg = _cfg(workspace, geometry={"lora_training_size": [96, 72]}, training={"max_train_steps": max_steps, "precision": "fp32", "train_batch_size": 2, "gradient_accumulation_steps": 1},
               checkpointing={"save_every_n_steps": 2, "keep_last_n_full_checkpoints": 1}, validation={"val_every_n_steps": 2})
    run_dir = workspace / "run"
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1, project_config=ProjectConfiguration(project_dir=str(run_dir)))
    latent, prompts, records = _fake_caches(4, 96, 72)
    latent["moments"] = torch.randn_like(latent["moments"].float()).half()
    dataset = lora.HAMCachedSDXLDataset(latent, prompts, records, 96, 72)
    unet = _tiny_unet(seed)
    optimizer = torch.optim.AdamW([p for p in unet.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _s: 1.0)
    train_loader, val_loader = DataLoader(dataset, batch_size=2, shuffle=True), DataLoader(dataset, batch_size=2)
    unet, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(unet, optimizer, train_loader, val_loader, scheduler)
    lora.register_lora_state_hooks(accelerator, unet)
    ema = lora.LoraEMA(accelerator.unwrap_model(unet), 0.99, warmup=True)
    accelerator.register_for_checkpointing(ema)
    provenance = {key: f"p-{key}" for key in lora.RESUME_MUST_MATCH}
    return cfg, accelerator, unet, dataset, train_loader, val_loader, DDPMScheduler(num_train_timesteps=100), optimizer, scheduler, ema, run_dir, provenance


def _lora_params(unet):
    return {name: p.detach().clone() for name, p in unet.named_parameters() if p.requires_grad}


def test_training_loop_checkpoints_and_resumes_the_exact_lora_state():
    with fixture_workspace("ham-lora-loop") as workspace:
        cfg, acc, unet, ds, tl, vl, noise, opt, sched, ema, run_dir, prov = _loop_parts(workspace, seed=0, max_steps=4)
        before = _lora_params(acc.unwrap_model(unet))
        step = lora.train_loop(acc, cfg, unet, ds, tl, vl, noise, opt, sched, ema, torch.float32, run_dir, prov, "adamw")
        trained = _lora_params(acc.unwrap_model(unet))

        assert step == 4
        assert any(not torch.equal(before[k], trained[k]) for k in before), "LoRA weights did not train"
        assert (run_dir / "checkpoint-4").is_dir() and not (run_dir / "checkpoint-2").exists(), "retention keeps only the newest full checkpoint"
        assert (run_dir / "lora_weights" / "step_2").is_dir() and (run_dir / "lora_weights" / "step_4").is_dir()
        latest = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
        assert latest["step"] == 4
        metadata = json.loads((run_dir / "checkpoint-4" / "metadata.json").read_text(encoding="utf-8"))
        lora.check_resume_compatible(metadata, prov)
        assert metadata["resolved_optimizer_name"] == "adamw" and metadata["step"] == 4
        full_size = sum(f.stat().st_size for f in (run_dir / "checkpoint-4").rglob("*") if f.is_file())
        assert not list((run_dir / "checkpoint-4").glob("model*.safetensors")), "frozen base weights must not be in the resumable state"

        cfg2, acc2, unet2, ds2, tl2, vl2, noise2, opt2, sched2, ema2, _, _ = _loop_parts(workspace, seed=123, max_steps=6)
        assert any(not torch.equal(trained[k], v) for k, v in _lora_params(acc2.unwrap_model(unet2)).items()), "fresh model must differ before resume"
        acc2.load_state(str(run_dir / "checkpoint-4"))
        resumed = _lora_params(acc2.unwrap_model(unet2))
        for name, value in trained.items():
            assert torch.allclose(value, resumed[name]), f"resumed LoRA parameter {name} differs"
        assert ema2.num_updates == ema.num_updates == 4
        final_step = lora.train_loop(acc2, cfg2, unet2, ds2, tl2, vl2, noise2, opt2, sched2, ema2, torch.float32, run_dir, prov, "adamw", start_step=4)
        assert final_step == 6 and (run_dir / "checkpoint-6").is_dir()
        assert full_size > 0


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

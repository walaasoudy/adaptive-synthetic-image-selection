#!/usr/bin/env python3
"""HAM10000 Stage 1: SDXL + LoRA fine-tuning on unpadded native-aspect dermoscopy images.

REUSED UNCHANGED from scripts/train/train_lora_sdxl.py (the CheXpert Stage 1):
  build_models      SDXL load, frozen base, UNet-only LoRA adapter (cfg.model / cfg.lora / cfg.training)
  LoraEMA           EMA of the adapter weights with warmup
  build_optimizer   AdamW8bit with a real step probe and the AdamW fallback
  run_validation    seeded, RNG-forked validation loss
  tracker_hparams, _load_captions

HAM10000-SPECIFIC, and why:
  * Non-square training. HAM10000 trains at 768x576 (geometry.lora_training_size), so latents are
    96x72 and SDXL's size conditioning is time_ids = [H, W, 0, 0, H, W]. The CheXpert cache and
    Dataset hard-code a square (resolution//8 x resolution//8, [S, S, 0, 0, S, S]).
  * Data roles. Inputs come only from 04_prepare_lora_inputs.py for split.train_split (gen_train) and
    split.monitor_split (gen_val); both names are asserted, and every caption record's image id is
    checked against the id lists of the other four splits before any model is loaded.
  * Provenance. Checkpoints record the HAM10000 split manifest hash, both split CSV hashes, both
    input manifests' captions/image-id hashes, the training-relevant config hash and the code
    identity. A resume refuses on any difference.
  * Loop + save/load hooks. In the CheXpert script these are inline in main() and cannot be imported
    without editing a CheXpert file; they are reproduced here as functions so they can be tested
    with a tiny UNet on CPU.

Usage (GPU):
    accelerate launch --config_file configs/accelerate_config.yaml scripts/train/ham10000_train_lora_sdxl.py \\
        [--auto-resume | --resume-from DIR --run-id RUN_ID] [override.key=value ...]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.train.train_lora_sdxl import (  # noqa: E402  (CheXpert Stage 1 building blocks, imported unchanged)
    LoraEMA,
    _load_captions,
    build_models,
    build_optimizer,
    run_validation,
    tracker_hparams,
)
from scripts.utils.artifact_contracts import current_code_identity_hash  # noqa: E402
from scripts.utils.caption_builder import select_variant_index  # noqa: E402
from scripts.utils.config import CONFIGS_DIR, _smoke_overlay  # noqa: E402
from scripts.utils.ham10000 import TEMPLATE_VERSION  # noqa: E402
from scripts.utils.ham10000_lora_data import (  # noqa: E402
    LoraRoleError,
    assert_role_isolation,
    forbidden_image_ids,
    validate_roles,
)
from scripts.utils.manifest import build_checkpoint_metadata, hash_dict, make_run_id, read_json, sha256_file, write_json  # noqa: E402
from scripts.utils.seed import set_seed  # noqa: E402

PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
# Config blocks that change what a checkpoint IS. A resume across a change in any of them is refused.
TRAINING_IDENTITY_KEYS = ("model", "lora", "optimizer", "captions", "split", "geometry", "run")


# ------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------

def load_config(overrides: list[str] | None = None):
    config = OmegaConf.load(CONFIGS_DIR / "ham10000_stage1.yaml")
    overlay = _smoke_overlay("ham_stage1")
    if overlay is not None:
        config = OmegaConf.merge(config, overlay)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def validate_config(cfg) -> None:
    """Fail on an unusable or unsafe configuration before any data or model is touched."""
    validate_roles(cfg.split)
    width, height = (int(v) for v in cfg.geometry.lora_training_size)
    if width % 8 or height % 8:
        raise ValueError(f"geometry.lora_training_size {width}x{height} must be multiples of 8 (SDXL VAE)")
    source_w, source_h = (int(v) for v in cfg.geometry.source_aspect)
    if width * source_h != height * source_w:
        raise ValueError(f"geometry.lora_training_size {width}x{height} is not the source aspect {source_w}:{source_h}")
    if str(cfg.captions.template_version) != TEMPLATE_VERSION:
        raise ValueError(f"captions.template_version {cfg.captions.template_version!r} != code {TEMPLATE_VERSION!r}")
    if str(cfg.training.precision).lower() not in PRECISIONS:
        raise ValueError(f"training.precision must be one of {sorted(PRECISIONS)}")
    if not str(cfg.model.revision or "").strip():
        raise ValueError("model.revision must pin an exact base-model revision")
    if bool(cfg.lora.train_text_encoders):
        raise ValueError("lora.train_text_encoders=true is not implemented (shared build_models refuses it)")
    positive = {
        "training.train_batch_size": cfg.training.train_batch_size,
        "training.gradient_accumulation_steps": cfg.training.gradient_accumulation_steps,
        "training.max_train_steps": cfg.training.max_train_steps,
        "checkpointing.save_every_n_steps": cfg.checkpointing.save_every_n_steps,
        "checkpointing.keep_last_n_full_checkpoints": cfg.checkpointing.keep_last_n_full_checkpoints,
        "validation.val_every_n_steps": cfg.validation.val_every_n_steps,
        "lora.rank": cfg.lora.rank,
    }
    bad = {key: value for key, value in positive.items() if int(value) < 1}
    if bad:
        raise ValueError(f"must be >= 1: {bad}")
    if int(cfg.training.min_train_steps) > int(cfg.training.max_train_steps):
        raise ValueError("training.min_train_steps exceeds training.max_train_steps")


def training_identity_hash(cfg) -> str:
    container = OmegaConf.to_container(cfg, resolve=True)
    return hash_dict({key: container.get(key) for key in TRAINING_IDENTITY_KEYS}, length=64)


# ------------------------------------------------------------------------------------------
# Inputs + provenance
# ------------------------------------------------------------------------------------------

def load_role_inputs(cfg, split_dir: Path) -> tuple[dict[str, list[dict]], dict]:
    """Caption records for gen_train / gen_val with every provenance and isolation check applied.

    Returns ({split_name: records}, provenance). Refuses when an input manifest is stale (different
    split manifest, split CSV, size, caption policy or captions file) or when any record's image id
    is absent from its split or present in any other split.
    """
    namespace = str(cfg.split.namespace)
    roles = validate_roles(cfg.split)
    split_manifest_path = Path(split_dir) / "split_manifest_v2.json"
    if not split_manifest_path.is_file():
        raise FileNotFoundError(f"missing split manifest {split_manifest_path}")
    split_manifest = read_json(split_manifest_path)
    forbidden = forbidden_image_ids(split_dir, roles.values())
    inputs_root = Path(cfg.paths.lora_inputs_dir) / namespace
    size = [int(v) for v in cfg.geometry.lora_training_size]

    records_by_split, provenance = {}, {
        "dataset": "ham10000",
        "split_namespace": namespace,
        "split_manifest_hash": split_manifest["manifest_hash"],
        "train_split": roles["train_split"],
        "monitor_split": roles["monitor_split"],
        "forbidden_splits_checked": sorted(forbidden),
    }
    for split_name in roles.values():
        split_csv = Path(split_dir) / f"{split_name}.csv"
        manifest_path = inputs_root / f"{split_name}_lora_inputs_manifest.json"
        captions_path = inputs_root / f"{split_name}_captions.jsonl"
        if not manifest_path.is_file() or not captions_path.is_file():
            raise FileNotFoundError(f"missing LoRA inputs for {split_name} under {inputs_root}; run 04_prepare_lora_inputs.py")
        manifest = read_json(manifest_path)
        expected = {
            "split_namespace": namespace,
            "split_name": split_name,
            "split_manifest_hash": split_manifest["manifest_hash"],
            "split_csv_sha256": sha256_file(split_csv),
            "size": size,
            "caption_template_version": TEMPLATE_VERSION,
            "num_paraphrase_variants": int(cfg.captions.num_paraphrase_variants),
            "age_bucket_width_years": int(cfg.captions.age_bucket_width_years),
            "captions_sha256": sha256_file(captions_path),
        }
        stale = {key: (manifest.get(key), value) for key, value in expected.items() if manifest.get(key) != value}
        if stale:
            raise ValueError(f"stale LoRA inputs for {split_name}: {stale}; rerun 04_prepare_lora_inputs.py")

        records = _load_captions(captions_path)
        ids = [record["image_id"] for record in records]
        split_ids = set(pd_read_ids(split_csv))
        outside = sorted(set(ids) - split_ids)
        if outside:
            raise LoraRoleError(f"{len(outside)} {split_name} caption record(s) are not in {split_name}.csv (e.g. {outside[:3]})")
        assert_role_isolation(ids, split_name, forbidden)
        for record in records:
            if not (inputs_root / record["image_relpath"]).is_file():
                raise FileNotFoundError(f"{split_name}: missing training image {record['image_relpath']}")
        records_by_split[split_name] = records
        prefix = "train" if split_name == roles["train_split"] else "monitor"
        provenance[f"{prefix}_split_csv_sha256"] = expected["split_csv_sha256"]
        provenance[f"{prefix}_captions_sha256"] = expected["captions_sha256"]
        provenance[f"{prefix}_image_ids_sha256"] = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
        provenance[f"{prefix}_num_images"] = len(records)
    return records_by_split, provenance


def pd_read_ids(path: Path) -> list[str]:
    import pandas as pd

    return pd.read_csv(path, usecols=["image_id"])["image_id"].astype(str).tolist()


def full_provenance(cfg, data_provenance: dict) -> dict:
    return {
        **data_provenance,
        "training_identity_sha256": training_identity_hash(cfg),
        "code_identity_sha256": current_code_identity_hash(),
        "base_model_id": str(cfg.model.base_model_id),
        "base_model_revision": str(cfg.model.revision),
        "lora_training_size": [int(v) for v in cfg.geometry.lora_training_size],
    }


# Keys that must match exactly for a resume. code_identity is deliberately NOT one of them: a
# comment edit must not strand a 6-hour run. It is recorded in every checkpoint instead.
RESUME_MUST_MATCH = (
    "dataset", "split_namespace", "split_manifest_hash", "train_split", "monitor_split",
    "train_split_csv_sha256", "monitor_split_csv_sha256", "train_captions_sha256", "monitor_captions_sha256",
    "train_image_ids_sha256", "monitor_image_ids_sha256", "training_identity_sha256",
    "base_model_id", "base_model_revision", "lora_training_size",
)


def check_resume_compatible(checkpoint_metadata: dict, provenance: dict) -> None:
    mismatches = {
        key: (checkpoint_metadata.get(key), provenance.get(key))
        for key in RESUME_MUST_MATCH
        if checkpoint_metadata.get(key) != provenance.get(key)
    }
    if mismatches:
        raise ValueError(f"refusing unsafe resume; checkpoint differs from current data/config: {mismatches}")


def find_auto_resume(checkpoints_root: Path) -> tuple[str | None, Path | None]:
    """(run_id, newest resumable checkpoint) from latest_run.json / latest.json, or (None, None)."""
    latest_run = Path(checkpoints_root) / "latest_run.json"
    if not latest_run.is_file():
        return None, None
    run_id = read_json(latest_run)["run_id"]
    latest = Path(checkpoints_root) / run_id / "latest.json"
    if not latest.is_file():
        return run_id, None
    checkpoint = Path(read_json(latest)["checkpoint_dir"])
    if not (checkpoint / "metadata.json").is_file():
        raise FileNotFoundError(f"latest.json points at {checkpoint}, which has no metadata.json")
    return run_id, checkpoint


# ------------------------------------------------------------------------------------------
# Non-square caches + dataset
# ------------------------------------------------------------------------------------------

def cache_key(cfg, split_name: str, records: list[dict], provenance: dict) -> str:
    prefix = "train" if split_name == str(cfg.split.train_split) else "monitor"
    payload = {
        "captions": provenance[f"{prefix}_captions_sha256"],
        "ids": [record["image_id"] for record in records],
        "model": [str(cfg.model.base_model_id), str(cfg.model.revision)],
        "size": [int(v) for v in cfg.geometry.lora_training_size],
        "precision": str(cfg.training.precision),
    }
    return hash_dict(payload, length=16)


def build_or_load_caches(cfg, models, split_name: str, records: list[dict], provenance: dict, device) -> tuple[Path, Path]:
    """VAE latent moments (H/8 x W/8) and per-caption text embeddings, cached once per content key."""
    namespace = str(cfg.split.namespace)
    width, height = (int(v) for v in cfg.geometry.lora_training_size)
    inputs_root = Path(cfg.paths.lora_inputs_dir) / namespace
    cache_dir = Path(cfg.paths.lora_cache_dir) / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(cfg, split_name, records, provenance)
    latent_path = cache_dir / f"{split_name}_latents_{key}.pt"
    prompt_path = cache_dir / f"{split_name}_prompts_{key}.pt"

    if not latent_path.exists():
        from PIL import Image
        from torchvision.transforms import functional as TF

        vae = models["vae"].to(device, dtype=torch.float32).eval()
        moments = torch.empty((len(records), 8, height // 8, width // 8), dtype=torch.float16)
        with torch.inference_mode():
            for index, record in enumerate(tqdm(records, desc=f"[{split_name}] VAE latent cache")):
                with Image.open(inputs_root / record["image_relpath"]) as image:
                    if image.size != (width, height):
                        raise ValueError(f"{record['image_id']} is {image.size}, expected {(width, height)}")
                    pixels = TF.normalize(TF.to_tensor(image.convert("RGB")), [0.5] * 3, [0.5] * 3)
                params = vae.encode(pixels.unsqueeze(0).to(device, torch.float32)).latent_dist.parameters
                moments[index] = params[0].cpu().half()
        tmp = latent_path.with_suffix(".tmp")
        torch.save({"image_ids": [r["image_id"] for r in records], "moments": moments, "scaling_factor": vae.config.scaling_factor, "size": [width, height]}, tmp)
        tmp.replace(latent_path)
        vae.to("cpu")

    if not prompt_path.exists():
        dtype = PRECISIONS[str(cfg.training.precision).lower()]
        text_one = models["text_encoder_one"].to(device=device, dtype=dtype).eval()
        text_two = models["text_encoder_two"].to(device=device, dtype=dtype).eval()
        cache: dict[str, dict] = {}
        with torch.inference_mode():
            for caption in tqdm(sorted({c for r in records for c in r["caption_variants"]}), desc=f"[{split_name}] text cache"):
                ids_one = models["tokenizer_one"](caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
                ids_two = models["tokenizer_two"](caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
                out_one = text_one(ids_one, output_hidden_states=True, return_dict=True)
                out_two = text_two(ids_two, output_hidden_states=True, return_dict=True)
                cache[caption] = {
                    "prompt_embeds": torch.cat([out_one.hidden_states[-2], out_two.hidden_states[-2]], dim=-1)[0].cpu().half(),
                    "pooled_embeds": out_two.text_embeds[0].cpu().half(),
                }
        tmp = prompt_path.with_suffix(".tmp")
        torch.save(cache, tmp)
        tmp.replace(prompt_path)
        text_one.to("cpu")
        text_two.to("cpu")
    return latent_path, prompt_path


def sdxl_time_ids(width: int, height: int) -> torch.Tensor:
    """SDXL micro-conditioning: (original_h, original_w, crop_top, crop_left, target_h, target_w)."""
    return torch.tensor([height, width, 0, 0, height, width], dtype=torch.float16)


class HAMCachedSDXLDataset(Dataset):
    """Cached latents + text embeddings at a NON-square size. Fresh latent sample per access, and a
    deterministic (image id, epoch) caption-variant choice — the CheXpert selection rule."""

    def __init__(self, latent_cache: dict, prompt_cache: dict, records: list[dict], width: int, height: int):
        if latent_cache.get("image_ids") != [r["image_id"] for r in records]:
            raise ValueError("latent cache image order does not match the caption records; refusing stale cache")
        if list(latent_cache.get("size", [])) != [width, height]:
            raise ValueError(f"latent cache size {latent_cache.get('size')} != {[width, height]}")
        if tuple(latent_cache["moments"].shape[-2:]) != (height // 8, width // 8):
            raise ValueError("latent cache moments do not have H/8 x W/8 spatial shape")
        self.latent_cache, self.prompt_cache, self.records = latent_cache, prompt_cache, records
        self.time_ids = sdxl_time_ids(width, height)
        self.epoch = 0

    @classmethod
    def from_paths(cls, latent_path: Path, prompt_path: Path, records: list[dict], width: int, height: int):
        return cls(torch.load(latent_path, map_location="cpu", weights_only=False), torch.load(prompt_path, map_location="cpu", weights_only=False), records, width, height)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        moments = self.latent_cache["moments"][index]
        mean, logvar = moments[:4].float(), moments[4:].float().clamp(-30, 20)
        latent = (mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)) * self.latent_cache["scaling_factor"]
        caption = record["caption_variants"][select_variant_index(record["image_id"], self.epoch, len(record["caption_variants"]))]
        prompt = self.prompt_cache[caption]
        return {"latents": latent.half(), "prompt_embeds": prompt["prompt_embeds"], "pooled_embeds": prompt["pooled_embeds"], "time_ids": self.time_ids}


# ------------------------------------------------------------------------------------------
# Checkpointing (LoRA-only resumable state) + training loop
# ------------------------------------------------------------------------------------------

def register_lora_state_hooks(accelerator, unet) -> None:
    """Keep only the LoRA adapter in Accelerate's resumable state (same pattern as the CheXpert run)."""
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
    from peft import set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict

    unet_type = type(accelerator.unwrap_model(unet))

    def save_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        layers = None
        for model in models:
            if not isinstance(accelerator.unwrap_model(model), unet_type):
                raise ValueError(f"unexpected model in save hook: {model.__class__.__name__}")
            layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(model)))
            weights.pop()
        if layers is None:
            raise ValueError("save hook received no UNet")
        StableDiffusionXLPipeline.save_lora_weights(str(output_dir), unet_lora_layers=layers, safe_serialization=True)

    def load_hook(models, input_dir):
        target = None
        while models:
            model = models.pop()
            if not isinstance(accelerator.unwrap_model(model), unet_type):
                raise ValueError(f"unexpected model in load hook: {model.__class__.__name__}")
            target = accelerator.unwrap_model(model)
        if target is None:
            raise ValueError("load hook received no UNet")
        state, _ = StableDiffusionXLPipeline.lora_state_dict(str(input_dir))
        unet_state = {key.removeprefix("unet."): value for key, value in state.items() if key.startswith("unet.")}
        if not unet_state:
            raise ValueError(f"no UNet LoRA weights in {input_dir}")
        incompatible = set_peft_model_state_dict(target, convert_unet_state_dict_to_peft(unet_state), adapter_name="default")
        if getattr(incompatible, "unexpected_keys", None):
            raise ValueError(f"unexpected LoRA keys resuming from {input_dir}: {incompatible.unexpected_keys}")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def save_checkpoint(accelerator, unet, ema, cfg, run_dir: Path, step: int, epoch: int, provenance: dict, optimizer_name: str | None) -> Path:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    accelerator.wait_for_everyone()
    checkpoint_dir = Path(run_dir) / f"checkpoint-{step}"
    if not accelerator.is_main_process:
        return checkpoint_dir
    accelerator.save_state(str(checkpoint_dir))
    lora_dir = Path(run_dir) / "lora_weights" / f"step_{step}"
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
    StableDiffusionXLPipeline.save_lora_weights(str(lora_dir), unet_lora_layers=state, safe_serialization=True)
    if ema is not None:
        torch.save(ema.state_dict(), lora_dir / "ema_shadow.pt")
    metadata = build_checkpoint_metadata(
        step, epoch, OmegaConf.to_container(cfg, resolve=True), provenance["split_manifest_hash"], int(cfg.run.seed),
        extra={**provenance, "resolved_optimizer_name": optimizer_name, "ema_applied": False},
    )
    write_json(checkpoint_dir / "metadata.json", metadata)
    write_json(lora_dir / "metadata.json", metadata)
    write_json(Path(run_dir) / "latest.json", {"checkpoint_dir": str(checkpoint_dir), "step": step, "epoch": epoch})
    full = sorted((p for p in Path(run_dir).glob("checkpoint-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[-1]))
    for old in full[: -int(cfg.checkpointing.keep_last_n_full_checkpoints)]:
        shutil.rmtree(old, ignore_errors=True)
    return checkpoint_dir


def training_step(accelerator, unet, batch, noise_scheduler, weight_dtype, optimizer, lr_scheduler, trainable_params, max_grad_norm: float):
    with accelerator.accumulate(unet):
        latents = batch["latents"].to(accelerator.device, weight_dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)
        prediction = unet(
            noisy, timesteps,
            encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, weight_dtype),
            added_cond_kwargs={"text_embeds": batch["pooled_embeds"].to(accelerator.device, weight_dtype), "time_ids": batch["time_ids"].to(accelerator.device, weight_dtype)},
            return_dict=False,
        )[0]
        target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(prediction.float(), target.float())
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return loss


def train_loop(accelerator, cfg, unet, train_dataset, train_loader, val_loader, noise_scheduler, optimizer, lr_scheduler, ema,
               weight_dtype, run_dir: Path, provenance: dict, optimizer_name: str | None, start_step: int = 0, log=None) -> int:
    """Run optimizer steps from start_step to training.max_train_steps with periodic checkpoints and
    validation. Returns the final global step."""
    trainable = [p for p in unet.parameters() if p.requires_grad]
    max_steps = int(cfg.training.max_train_steps)
    global_step = start_step
    epoch = global_step // max(1, len(train_loader))
    progress = tqdm(total=max_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    unet.train()
    while global_step < max_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            loss = training_step(accelerator, unet, batch, noise_scheduler, weight_dtype, optimizer, lr_scheduler, trainable, float(cfg.optimizer.max_grad_norm))
            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                if ema is not None:
                    ema.update(accelerator.unwrap_model(unet))
                if log:
                    log({"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]}, global_step)
                if global_step % int(cfg.checkpointing.save_every_n_steps) == 0:
                    save_checkpoint(accelerator, unet, ema, cfg, run_dir, global_step, epoch, provenance, optimizer_name)
                if global_step % int(cfg.validation.val_every_n_steps) == 0:
                    val_loss = run_validation(accelerator, unet, val_loader, noise_scheduler, weight_dtype, int(cfg.validation.val_seed))
                    if log:
                        log({"val/loss": val_loss}, global_step)
                    unet.train()
            if global_step >= max_steps:
                break
        epoch += 1
    progress.close()
    return global_step


def export_final(accelerator, unet, ema, cfg, run_dir: Path, step: int, provenance: dict) -> Path:
    from diffusers import StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    final_dir = Path(run_dir) / "final"
    if accelerator.is_main_process:
        raw = accelerator.unwrap_model(unet)
        if ema is not None:
            ema.copy_to(raw)
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw))
        StableDiffusionXLPipeline.save_lora_weights(str(final_dir), unet_lora_layers=state, safe_serialization=True)
        write_json(final_dir / "metadata.json", build_checkpoint_metadata(
            step, 0, OmegaConf.to_container(cfg, resolve=True), provenance["split_manifest_hash"], int(cfg.run.seed),
            extra={**provenance, "ema_applied": ema is not None},
        ))
    return final_dir


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--auto-resume", action="store_true", help="Continue the latest run from its newest checkpoint, if any")
    group.add_argument("--resume-from", default=None, help="Explicit checkpoint-N directory to resume from (needs --run-id)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--validate-only", action="store_true", help="Check config, roles and inputs; load no model")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. training.max_train_steps=4000")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.overrides)
    validate_config(cfg)
    splits_cfg = OmegaConf.load(CONFIGS_DIR / "splits_ham10000.yaml")
    OmegaConf.resolve(splits_cfg)
    split_dir = Path(splits_cfg.paths.splits_root) / str(cfg.split.namespace)
    records, data_provenance = load_role_inputs(cfg, split_dir)
    provenance = full_provenance(cfg, data_provenance)
    if args.validate_only:
        print(json.dumps({k: v for k, v in provenance.items()}, indent=2, default=str))
        return 0

    checkpoints_root = Path(cfg.paths.checkpoints_dir)
    resume_from = Path(args.resume_from) if args.resume_from else None
    run_id = args.run_id
    if args.auto_resume:
        run_id, resume_from = find_auto_resume(checkpoints_root)
    if resume_from is not None and not run_id:
        raise SystemExit("--resume-from requires --run-id")
    run_id = run_id or make_run_id(OmegaConf.to_container(cfg, resolve=True))
    run_dir = checkpoints_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoints_root / "latest_run.json", {"run_id": run_id})
    if resume_from is not None:
        check_resume_compatible(read_json(resume_from / "metadata.json"), provenance)

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    set_seed(int(cfg.run.seed))
    precision = str(cfg.training.precision).lower()
    trackers = (["tensorboard"] if cfg.logging.tensorboard else []) + (["wandb"] if cfg.logging.wandb.enabled else [])
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.training.gradient_accumulation_steps),
        mixed_precision="no" if precision == "fp32" else precision,
        log_with=trackers or None,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(Path(cfg.paths.logs_dir) / run_id)),
    )
    if trackers:
        accelerator.init_trackers("ham10000_stage1_lora_sdxl", config=tracker_hparams(cfg))
    weight_dtype = PRECISIONS[precision]
    models = build_models(cfg, weight_dtype)

    width, height = (int(v) for v in cfg.geometry.lora_training_size)
    train_split, monitor_split = str(cfg.split.train_split), str(cfg.split.monitor_split)
    train_paths = build_or_load_caches(cfg, models, train_split, records[train_split], provenance, accelerator.device)
    val_paths = build_or_load_caches(cfg, models, monitor_split, records[monitor_split], provenance, accelerator.device)
    for frozen in ("vae", "text_encoder_one", "text_encoder_two"):
        models.pop(frozen, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dataset = HAMCachedSDXLDataset.from_paths(*train_paths, records[train_split], width, height)
    val_dataset = HAMCachedSDXLDataset.from_paths(*val_paths, records[monitor_split], width, height)
    train_loader = DataLoader(train_dataset, batch_size=int(cfg.training.train_batch_size), shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg.training.train_batch_size), shuffle=False, num_workers=0, pin_memory=True)

    unet = models["unet"]
    optimizer, optimizer_name = build_optimizer(cfg, [p for p in unet.parameters() if p.requires_grad], accelerator.device)
    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(str(cfg.optimizer.lr_scheduler), optimizer=optimizer, num_warmup_steps=int(cfg.optimizer.lr_warmup_steps), num_training_steps=int(cfg.training.max_train_steps))
    unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(unet, optimizer, train_loader, val_loader, lr_scheduler)
    register_lora_state_hooks(accelerator, unet)
    ema = LoraEMA(accelerator.unwrap_model(unet), float(cfg.training.ema.decay), warmup=bool(cfg.training.ema.warmup)) if cfg.training.ema.enabled else None
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    start_step = 0
    if resume_from is not None:
        accelerator.load_state(str(resume_from))
        start_step = int(read_json(resume_from / "metadata.json")["step"])
        print(f"Resumed {run_id} from {resume_from} at step {start_step}", flush=True)

    log = (lambda values, step: accelerator.log(values, step=step)) if trackers else None
    step = train_loop(accelerator, cfg, unet, train_dataset, train_loader, val_loader, models["noise_scheduler"], optimizer, lr_scheduler, ema,
                      weight_dtype, run_dir, provenance, optimizer_name, start_step, log)
    save_checkpoint(accelerator, unet, ema, cfg, run_dir, step, step // max(1, len(train_loader)), provenance, optimizer_name)
    final_dir = export_final(accelerator, unet, ema, cfg, run_dir, step, provenance)
    print(f"Final LoRA (EMA applied: {ema is not None}) -> {final_dir}", flush=True)
    if trackers:
        accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

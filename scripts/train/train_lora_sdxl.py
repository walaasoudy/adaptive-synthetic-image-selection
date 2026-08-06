#!/usr/bin/env python3
"""SDXL + LoRA fine-tuning on CheXpert (docs/stage1_plan.md §3, §8).

Adapted from diffusers' text_to_image_lora_sdxl reference pattern, with:
  - UNet-only LoRA (self+cross attention projections), text encoders frozen
  - bf16 mixed precision, PyTorch SDPA attention, gradient checkpointing
  - one-time VAE latent + text embedding caching (captions are deterministic-from-labels)
  - EMA of the LoRA adapter weights
  - resumable checkpointing via Accelerate save_state/load_state, all paths under PROJECT_ROOT
    (expected to be the persistent volume on RunPod — see configs/stage1_lora_sdxl.yaml)
  - TensorBoard (+ optional Weights & Biases) logging

Run directly, or via scripts/train/launch_resumable.sh which handles the resume-on-restart logic.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml scripts/train/train_lora_sdxl.py \\
        [--resume-from CHECKPOINT_DIR] [--run-id RUN_ID] [override.key=value ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.caption_builder import select_variant_index  # noqa: E402
from scripts.utils.config import ensure_dirs, load_stage1_config  # noqa: E402
from scripts.utils.manifest import build_checkpoint_metadata, make_run_id, read_json, write_json  # noqa: E402
from scripts.utils.seed import set_seed  # noqa: E402


# --------------------------------------------------------------------------------------
# Model / cache construction
# --------------------------------------------------------------------------------------

def build_models(cfg, device_dtype: torch.dtype):
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
    from peft import LoraConfig

    model_id = cfg.model.base_model_id
    cache_dir = cfg.paths.hf_cache_dir
    revision = cfg.model.revision if cfg.model.revision else None

    tokenizer_one = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision, cache_dir=cache_dir)
    tokenizer_two = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", revision=revision, cache_dir=cache_dir)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler", revision=revision, cache_dir=cache_dir)

    text_encoder_one = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", revision=revision, cache_dir=cache_dir, torch_dtype=device_dtype
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2", revision=revision, cache_dir=cache_dir, torch_dtype=device_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", revision=revision, cache_dir=cache_dir,
        torch_dtype=torch.float32 if cfg.model.vae_upcast_fp32 else device_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=revision, cache_dir=cache_dir, torch_dtype=device_dtype
    )

    for model in (vae, text_encoder_one, text_encoder_two, unet):
        model.requires_grad_(False)

    if cfg.lora.train_text_encoders:
        raise NotImplementedError(
            "lora.train_text_encoders=true is a documented fallback (docs/stage1_plan.md §3) "
            "for weak rare-label conditioning, not implemented in v1. Set it back to false, or "
            "implement CLIPText LoRA attachment here before enabling it."
        )

    lora_config = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        init_lora_weights="gaussian",
        target_modules=list(cfg.lora.target_modules),
    )
    unet.add_adapter(lora_config)
    for p in unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if cfg.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
    }


def _load_captions(captions_path: Path) -> list[dict]:
    records = []
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_or_load_cache(cfg, models, split_name: str, device: torch.device):
    """One-time VAE latent + text embedding cache per split (docs/stage1_plan.md §8).

    Latents are cached per image (as encoder-distribution moments, so a fresh sample is still
    drawn each epoch). Text embeddings are cached per unique caption string across all
    paraphrase variants of all images, since many images share the low-cardinality caption
    vocabulary.
    """
    images_dir = Path(cfg.paths.images_dir)
    captions_path = Path(cfg.paths.captions_dir) / f"{split_name}_captions.jsonl"
    cache_dir = Path(cfg.paths.processed_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_cache_path = cache_dir / f"{split_name}_latent_cache.pt"
    prompt_cache_path = cache_dir / f"{split_name}_prompt_cache.pt"

    records = _load_captions(captions_path)
    if not records:
        raise FileNotFoundError(
            f"No caption records found at {captions_path} — run 03_preprocess_images.py and "
            "04_generate_captions.py first."
        )

    resolution = cfg.data.resolution

    if not latent_cache_path.exists():
        from PIL import Image
        from torchvision.transforms import functional as TF

        vae = models["vae"].to(device, dtype=torch.float32).eval()
        moments = torch.empty((len(records), 8, resolution // 8, resolution // 8), dtype=torch.float16)
        with torch.inference_mode():
            for i, rec in enumerate(tqdm(records, desc=f"[{split_name}] VAE latent cache")):
                image_path = images_dir / rec["image_relpath"]
                with Image.open(image_path) as im:
                    pixels = TF.normalize(TF.to_tensor(im.convert("RGB")), [0.5] * 3, [0.5] * 3)
                params = vae.encode(pixels.unsqueeze(0).to(device, torch.float32)).latent_dist.parameters
                moments[i] = params[0].cpu().half()
        torch.save(
            {
                "image_ids": [r["image_id"] for r in records],
                "moments": moments,
                "scaling_factor": vae.config.scaling_factor,
                "target_size": resolution,
            },
            latent_cache_path,
        )
        vae.to("cpu")
        del moments
        torch.cuda.empty_cache()
    else:
        print(f"[{split_name}] using existing latent cache: {latent_cache_path}")

    if not prompt_cache_path.exists():
        text_encoder_one = models["text_encoder_one"].to(device).eval()
        text_encoder_two = models["text_encoder_two"].to(device).eval()
        tokenizer_one, tokenizer_two = models["tokenizer_one"], models["tokenizer_two"]

        unique_captions = sorted({c for r in records for c in r["caption_variants"]})
        prompt_cache: dict[str, dict] = {}
        with torch.inference_mode():
            for caption in tqdm(unique_captions, desc=f"[{split_name}] text embedding cache"):
                ids1 = tokenizer_one(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
                ids2 = tokenizer_two(caption, padding="max_length", truncation=True, max_length=77, return_tensors="pt").input_ids.to(device)
                o1 = text_encoder_one(ids1, output_hidden_states=True, return_dict=True)
                o2 = text_encoder_two(ids2, output_hidden_states=True, return_dict=True)
                prompt_cache[caption] = {
                    "prompt_embeds": torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)[0].cpu().half(),
                    "pooled_embeds": o2.text_embeds[0].cpu().half(),
                }
        torch.save(prompt_cache, prompt_cache_path)
        text_encoder_one.to("cpu")
        text_encoder_two.to("cpu")
        del prompt_cache
        torch.cuda.empty_cache()
    else:
        print(f"[{split_name}] using existing prompt cache: {prompt_cache_path}")

    return latent_cache_path, prompt_cache_path, records


class CachedSDXLDataset(Dataset):
    """Reads precomputed latents + text embeddings; draws a fresh VAE-latent sample and a
    deterministic (per image id + epoch) caption-variant selection each access."""

    def __init__(self, latent_cache_path: Path, prompt_cache_path: Path, records: list[dict], target_size: int):
        self.latent_cache = torch.load(latent_cache_path, map_location="cpu", weights_only=False)
        self.prompt_cache = torch.load(prompt_cache_path, map_location="cpu", weights_only=False)
        self.records = records
        self.target_size = target_size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        moments = self.latent_cache["moments"][index]
        mean, logvar = moments[:4].float(), moments[4:].float().clamp(-30, 20)
        std = torch.exp(0.5 * logvar)
        latent = (mean + std * torch.randn_like(mean)) * self.latent_cache["scaling_factor"]

        variant_idx = select_variant_index(rec["image_id"], self.epoch, len(rec["caption_variants"]))
        caption = rec["caption_variants"][variant_idx]
        prompt = self.prompt_cache[caption]

        time_ids = torch.tensor(
            [self.target_size, self.target_size, 0, 0, self.target_size, self.target_size], dtype=torch.float16
        )
        return {
            "latents": latent.half(),
            "prompt_embeds": prompt["prompt_embeds"],
            "pooled_embeds": prompt["pooled_embeds"],
            "time_ids": time_ids,
        }


# --------------------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------------------

class LoraEMA:
    """EMA over the UNet's LoRA parameters only (docs/stage1_plan.md §8: cheap since LoRA
    weights are small; base/frozen weights are never touched)."""

    def __init__(self, unet, decay: float):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in unet.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, unet) -> None:
        for name, param in unet.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, unet) -> None:
        for name, param in unet.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name].to(param.dtype))

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------

def save_checkpoint(accelerator, unet, ema, cfg, run_dir: Path, step: int, epoch: int, split_manifest_hash: str, seed: int, keep_last_n: int) -> None:
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionXLPipeline

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    full_checkpoint_dir = run_dir / f"checkpoint-{step}"
    accelerator.save_state(str(full_checkpoint_dir))

    lora_dir = run_dir / "lora_weights" / f"step_{step}"
    raw_unet = accelerator.unwrap_model(unet)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw_unet))
    StableDiffusionXLPipeline.save_lora_weights(str(lora_dir), unet_lora_layers=state, safe_serialization=True)

    if ema is not None:
        torch.save(ema.state_dict(), lora_dir / "ema_shadow.pt")

    metadata = build_checkpoint_metadata(
        step=step,
        epoch=epoch,
        config=dict(cfg),
        split_manifest_hash=split_manifest_hash,
        seed=seed,
    )
    write_json(full_checkpoint_dir / "metadata.json", metadata)
    write_json(lora_dir / "metadata.json", metadata)

    write_json(run_dir / "latest.json", {"checkpoint_dir": str(full_checkpoint_dir), "step": step, "epoch": epoch})

    # Retention: keep only the last N full resumable checkpoints; lora_weights/* is kept in full.
    full_checkpoints = sorted(
        [p for p in run_dir.glob("checkpoint-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for old in full_checkpoints[:-keep_last_n]:
        import shutil

        shutil.rmtree(old, ignore_errors=True)

    print(f"Saved checkpoint at step {step} -> {full_checkpoint_dir} (+ lora_weights snapshot)")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-from", type=str, default=None, help="Full checkpoint directory to resume from")
    parser.add_argument("--run-id", type=str, default=None, help="Existing run_id to continue writing into")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. training.train_batch_size=4")
    return parser.parse_args()


def main() -> int:
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    args = parse_args()
    cfg = load_stage1_config(overrides=args.overrides)
    ensure_dirs(cfg)
    set_seed(cfg.run.seed)

    checkpoints_root = Path(cfg.paths.checkpoints_dir)
    run_id = args.run_id or make_run_id(dict(cfg))
    run_dir = checkpoints_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoints_root / "latest_run.json", {"run_id": run_id})

    split_manifest_path = Path(cfg.paths.splits_dir) / "split_manifest.json"
    split_manifest_hash = "unknown"
    if split_manifest_path.exists():
        from scripts.utils.manifest import hash_dict

        split_manifest_hash = hash_dict(read_json(split_manifest_path))

    log_dir = Path(cfg.paths.logs_dir) / run_id
    trackers = []
    if cfg.logging.tensorboard:
        trackers.append("tensorboard")
    if cfg.logging.wandb.enabled:
        trackers.append("wandb")

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.precision,
        log_with=trackers or None,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(log_dir)),
    )
    if trackers:
        accelerator.init_trackers("stage1_lora_sdxl", config=dict(cfg))

    weight_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
    models = build_models(cfg, weight_dtype)

    train_latent_cache, train_prompt_cache, train_records = build_or_load_cache(cfg, models, "gen_train", accelerator.device)
    val_latent_cache, val_prompt_cache, val_records = build_or_load_cache(cfg, models, "gen_val", accelerator.device)

    train_dataset = CachedSDXLDataset(train_latent_cache, train_prompt_cache, train_records, cfg.data.resolution)
    val_dataset = CachedSDXLDataset(val_latent_cache, val_prompt_cache, val_records, cfg.data.resolution)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    unet = models["unet"]
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    optimizer = None
    if cfg.optimizer.name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)
        except Exception as e:
            print(f"bitsandbytes AdamW8bit unavailable ({e}); falling back to torch.optim.AdamW (docs/stage1_plan.md §3/§12).")
    if optimizer is None:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

    from diffusers.optimization import get_scheduler

    lr_scheduler = get_scheduler(
        cfg.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.optimizer.lr_warmup_steps,
        num_training_steps=cfg.training.max_train_steps,
    )

    unet, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_loader, val_loader, lr_scheduler
    )

    ema = LoraEMA(accelerator.unwrap_model(unet), cfg.training.ema.decay) if cfg.training.ema.enabled else None

    global_step = 0
    if args.resume_from:
        accelerator.load_state(args.resume_from)
        resumed_metadata = read_json(Path(args.resume_from) / "metadata.json")
        global_step = resumed_metadata["step"]
        print(f"Resumed from {args.resume_from} at step {global_step}")

    noise_scheduler = models["noise_scheduler"]
    progress = tqdm(total=cfg.training.max_train_steps, initial=global_step, disable=not accelerator.is_local_main_process)
    epoch = global_step // max(1, len(train_loader))

    unet.train()
    while global_step < cfg.training.max_train_steps:
        train_dataset.set_epoch(epoch)
        for batch in train_loader:
            with accelerator.accumulate(unet):
                latents = batch["latents"].to(accelerator.device, weight_dtype)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                added_cond_kwargs = {
                    "text_embeds": batch["pooled_embeds"].to(accelerator.device, weight_dtype),
                    "time_ids": batch["time_ids"].to(accelerator.device, weight_dtype),
                }
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, weight_dtype),
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                target = (
                    noise if noise_scheduler.config.prediction_type == "epsilon"
                    else noise_scheduler.get_velocity(latents, noise, timesteps)
                )
                loss = F.mse_loss(model_pred.float(), target.float())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, cfg.optimizer.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.item():.4f}")
                if ema is not None:
                    ema.update(accelerator.unwrap_model(unet))

                if trackers:
                    accelerator.log({"train/loss": loss.item(), "train/lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % cfg.checkpointing.save_every_n_steps == 0:
                    save_checkpoint(
                        accelerator, unet, ema, cfg, run_dir, global_step, epoch,
                        split_manifest_hash, cfg.run.seed, cfg.checkpointing.keep_last_n_full_checkpoints,
                    )

                if global_step % cfg.validation.val_every_n_steps == 0:
                    val_loss = run_validation(accelerator, unet, val_loader, noise_scheduler, weight_dtype)
                    if trackers:
                        accelerator.log({"val/loss": val_loss}, step=global_step)
                    print(f"[step {global_step}] val_loss={val_loss:.4f}")
                    unet.train()

            if global_step >= cfg.training.max_train_steps:
                break
        epoch += 1

    progress.close()
    save_checkpoint(
        accelerator, unet, ema, cfg, run_dir, global_step, epoch,
        split_manifest_hash, cfg.run.seed, cfg.checkpointing.keep_last_n_full_checkpoints,
    )
    final_dir = run_dir / "final"
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionXLPipeline

    if accelerator.is_main_process:
        raw_unet = accelerator.unwrap_model(unet)
        if ema is not None:
            ema.copy_to(raw_unet)
        state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw_unet))
        StableDiffusionXLPipeline.save_lora_weights(str(final_dir), unet_lora_layers=state, safe_serialization=True)
        write_json(
            final_dir / "metadata.json",
            build_checkpoint_metadata(global_step, epoch, dict(cfg), split_manifest_hash, cfg.run.seed, extra={"ema_applied": ema is not None}),
        )
        print(f"Final LoRA weights (EMA applied: {ema is not None}) -> {final_dir}")

    if trackers:
        accelerator.end_training()

    return 0


@torch.no_grad()
def run_validation(accelerator, unet, val_loader, noise_scheduler, weight_dtype) -> float:
    unet.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        latents = batch["latents"].to(accelerator.device, weight_dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        added_cond_kwargs = {
            "text_embeds": batch["pooled_embeds"].to(accelerator.device, weight_dtype),
            "time_ids": batch["time_ids"].to(accelerator.device, weight_dtype),
        }
        model_pred = unet(
            noisy_latents, timesteps, encoder_hidden_states=batch["prompt_embeds"].to(accelerator.device, weight_dtype),
            added_cond_kwargs=added_cond_kwargs, return_dict=False,
        )[0]
        target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
        loss = F.mse_loss(model_pred.float(), target.float())
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


if __name__ == "__main__":
    raise SystemExit(main())

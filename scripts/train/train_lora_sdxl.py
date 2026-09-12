#!/usr/bin/env python3
"""SDXL + LoRA fine-tuning on CheXpert (docs/stage1_plan.md §3, §8).

Adapted from diffusers' text_to_image_lora_sdxl reference pattern, with:
  - UNet-only LoRA (self+cross attention projections), text encoders frozen
  - bf16 mixed precision, PyTorch SDPA attention, gradient checkpointing
  - one-time VAE latent + text embedding caching (captions are deterministic-from-labels)
  - EMA of the LoRA adapter weights
  - resumable checkpointing via Accelerate save_state/load_state, all paths under PROJECT_ROOT
    (expected to be the persistent volume on RunPod — see configs/stage1_lora_sdxl.yaml).
    save_model_hook/load_model_hook keep only the LoRA adapter in the resumable state, so a
    checkpoint is ~11 MB rather than ~9.5 GB of re-serialized frozen SDXL base weights.
  - TensorBoard (+ optional Weights & Biases) logging

Run directly, or via scripts/train/launch_resumable.sh which handles the resume-on-restart logic.

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml scripts/train/train_lora_sdxl.py \\
        [--resume-from CHECKPOINT_DIR] [--run-id RUN_ID] [override.key=value ...]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
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

def tracker_hparams(cfg) -> dict[str, int | float | str | bool]:
    """Flatten a resolved OmegaConf config into TensorBoard-safe scalar hparams."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    flattened: dict[str, int | float | str | bool] = {}

    def visit(prefix: str, value) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, (list, tuple)):
            flattened[prefix] = json.dumps(value, sort_keys=True)
        elif value is None:
            flattened[prefix] = "null"
        elif isinstance(value, (int, float, str, bool)):
            flattened[prefix] = value
        else:
            flattened[prefix] = str(value)

    visit("", resolved)
    return flattened

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
    seen = set()
    with open(captions_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                image_id = record.get("image_id")
                if not image_id or image_id in seen:
                    raise ValueError(f"Missing or duplicate image_id {image_id!r} in {captions_path}")
                if not record.get("caption_variants"):
                    raise ValueError(f"Caption record {image_id!r} has no variants")
                seen.add(image_id)
                records.append(record)
    return records


def validate_training_inputs(cfg) -> None:
    """Fail before model downloads when captions/preprocessing no longer match the config."""
    namespace = str(cfg.split.namespace)
    metadata_path = Path(cfg.paths.captions_dir) / namespace / "caption_template_version.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing caption metadata: {metadata_path}; run 04_generate_captions.py")
    metadata = read_json(metadata_path)
    expected_caption = {
        "template_version": cfg.captions.template_version,
        "num_paraphrase_variants": cfg.captions.num_paraphrase_variants,
        "age_bucket_width_years": cfg.captions.age_bucket_width_years,
        "uncertain_label_policy": cfg.captions.uncertain_label_policy,
        "no_finding_overrides_positives": cfg.captions.no_finding_overrides_positives,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected_caption.items() if metadata.get(key) != value}
    if metadata.get("split_namespace") != namespace:
        mismatches["split_namespace"] = (metadata.get("split_namespace"), namespace)
    from scripts.utils.splits import read_split_manifest, resolve_splits_dir
    split_manifest = read_split_manifest(namespace)
    if metadata.get("split_manifest_hash") != split_manifest.get("manifest_hash"):
        mismatches["split_manifest_hash"] = (metadata.get("split_manifest_hash"), split_manifest.get("manifest_hash"))
    preprocessing_by_split = metadata.get("preprocessing_manifests", {})
    preprocessing = preprocessing_by_split.get("gen_train", {})
    output_settings = preprocessing.get("output_settings", {})
    if output_settings.get("resolution") != cfg.data.resolution:
        mismatches["preprocessing_resolution"] = (output_settings.get("resolution"), cfg.data.resolution)
    for split_name in ("gen_train", "gen_val"):
        split_path = resolve_splits_dir(namespace) / f"{split_name}.csv"
        recorded = preprocessing_by_split.get(split_name, {}).get("source_splits", {}).get(split_name, {}).get("sha256")
        current = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.is_file() else None
        if recorded != current:
            mismatches[f"{split_name}_sha256"] = (recorded, current)
    if mismatches:
        raise ValueError(f"Caption/preprocessing provenance is stale or incompatible: {mismatches}. Rerun preprocessing and captions.")


def build_or_load_cache(cfg, models, split_name: str, device: torch.device, max_samples: int | None = None):
    """One-time VAE latent + text embedding cache per split (docs/stage1_plan.md §8).

    Latents are cached per image (as encoder-distribution moments, so a fresh sample is still
    drawn each epoch). Text embeddings are cached per unique caption string across all
    paraphrase variants of all images, since many images share the low-cardinality caption
    vocabulary.
    """
    namespace = str(cfg.split.namespace)
    images_dir = Path(cfg.paths.images_dir) / namespace
    captions_path = Path(cfg.paths.captions_dir) / namespace / f"{split_name}_captions.jsonl"
    cache_dir = Path(cfg.paths.processed_dir) / "cache" / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = _load_captions(captions_path)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples-per-split must be positive")
        records = records[:max_samples]
    if not records:
        raise FileNotFoundError(
            f"No caption records found at {captions_path} — run 03_preprocess_images.py and "
            "04_generate_captions.py first."
        )

    resolution = cfg.data.resolution
    record_identity = json.dumps([record["image_id"] for record in records], separators=(",", ":"))
    captions_hash = hashlib.sha256((hashlib.sha256(captions_path.read_bytes()).hexdigest() + record_identity).encode()).hexdigest()[:12]
    model_key = hashlib.sha256(f"{cfg.model.base_model_id}:{cfg.model.revision}:{resolution}".encode()).hexdigest()[:8]
    cache_key = f"{captions_hash}_{model_key}"
    latent_cache_path = cache_dir / f"{split_name}_latent_cache_{cache_key}.pt"
    prompt_cache_path = cache_dir / f"{split_name}_prompt_cache_{cache_key}.pt"

    for record in records:
        image_path = images_dir / record["image_relpath"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Caption record {record['image_id']} references missing image: {image_path}")

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
        text_dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[str(cfg.training.precision).lower()]
        # Some SDXL checkpoints contain mixed stored dtypes (notably text_projection).  Cast the
        # complete frozen encoders explicitly so their hidden states and projection weights agree.
        text_encoder_one = models["text_encoder_one"].to(device=device, dtype=text_dtype).eval()
        text_encoder_two = models["text_encoder_two"].to(device=device, dtype=text_dtype).eval()
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
        expected_ids = [record["image_id"] for record in records]
        if self.latent_cache.get("image_ids") != expected_ids:
            raise ValueError("Latent cache image order does not match the caption records; refusing stale cache")
        if self.latent_cache.get("target_size") != target_size:
            raise ValueError("Latent cache resolution does not match the configured target size")

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
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        if set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA checkpoint parameters do not match the current LoRA adapter")
        self.shadow = {name: value.float() for name, value in state["shadow"].items()}


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------

def _adamw_8bit_step_works(device: torch.device) -> tuple[bool, str]:
    """Actually run one AdamW8bit step on `device` before committing the real run to it.

    Constructing bnb.optim.AdamW8bit succeeds whenever the package imports; the 8-bit CUDA kernel
    is only invoked on .step(). The RunPod pytorch-2.8/cu128 image ships a bitsandbytes build
    without a matching CUDA binary, so construction passed and training then died at the first
    optimizer step -- after the latent cache had been built. Probing a real step here moves that
    failure to before any training and lets the configured "adamw" fallback actually engage
    (docs/stage1_plan.md §3/§12).

    Deterministic by construction: fixed tensors only, so no RNG draw perturbs the seeded run.
    """
    try:
        import bitsandbytes as bnb

        probe = torch.ones(1, device=device, requires_grad=True)
        probe_optimizer = bnb.optim.AdamW8bit([probe], lr=1.0e-4, weight_decay=0.0)
        (probe * probe).sum().backward()
        probe_optimizer.step()
        probe_optimizer.zero_grad(set_to_none=True)
        del probe_optimizer, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True, ""
    except Exception as exc:  # ImportError, CUDA kernel/symbol errors, anything else
        return False, f"{type(exc).__name__}: {exc}"


def build_optimizer(cfg, trainable_params, device: torch.device) -> tuple[torch.optim.Optimizer, str]:
    """Build the configured optimizer, honouring the documented adamw_8bit -> adamw fallback.

    Returns (optimizer, resolved_name) so the checkpoint metadata records which optimizer actually
    ran rather than which one was requested.
    """
    learning_rate, weight_decay = cfg.optimizer.learning_rate, cfg.optimizer.weight_decay
    if cfg.optimizer.name == "adamw_8bit":
        works, reason = _adamw_8bit_step_works(device)
        if works:
            import bitsandbytes as bnb

            print("optimizer: bitsandbytes AdamW8bit (8-bit step verified on this device).", flush=True)
            return bnb.optim.AdamW8bit(trainable_params, lr=learning_rate, weight_decay=weight_decay), "adamw_8bit"
        print(
            f"optimizer: bitsandbytes AdamW8bit failed its step probe ({reason}); "
            "falling back to torch.optim.AdamW (docs/stage1_plan.md §3/§12).",
            flush=True,
        )
    elif cfg.optimizer.name != "adamw":
        raise ValueError(
            f"Unsupported optimizer.name={cfg.optimizer.name!r}; expected \"adamw_8bit\" or \"adamw\" "
            "(configs/stage1_lora_sdxl.yaml)."
        )
    return torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay), "adamw"


def save_checkpoint(accelerator, unet, ema, cfg, run_dir: Path, step: int, epoch: int, split_manifest_hash: str, seed: int, keep_last_n: int, resolved_optimizer_name: str | None = None) -> None:
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionXLPipeline

    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    # Resumable state: LoRA adapter + optimizer/scheduler/RNG/EMA. The registered
    # save_model_hook (see main()) is what keeps the frozen UNet base weights out of this.
    full_checkpoint_dir = run_dir / f"checkpoint-{step}"
    accelerator.save_state(str(full_checkpoint_dir))

    # Separate inference-ready export, kept for every step (keep_all_lora_weight_snapshots) and
    # pinned by Stage 2's checkpoint.lora_weights_dir. Distinct purpose from the resumable state
    # above, so both are written even though both now contain the same adapter.
    lora_dir = run_dir / "lora_weights" / f"step_{step}"
    raw_unet = accelerator.unwrap_model(unet)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(raw_unet))
    StableDiffusionXLPipeline.save_lora_weights(str(lora_dir), unet_lora_layers=state, safe_serialization=True)

    if ema is not None:
        torch.save(ema.state_dict(), lora_dir / "ema_shadow.pt")

    from scripts.utils.manifest import sha256_file
    from scripts.utils.splits import read_split_manifest, resolve_splits_dir
    namespace = str(cfg.split.namespace)
    manifest = read_split_manifest(namespace)
    split_dir = resolve_splits_dir(namespace)
    source_csv = Path(manifest["input_csv_path"])
    metadata = build_checkpoint_metadata(
        step=step,
        epoch=epoch,
        config=dict(cfg),
        split_manifest_hash=split_manifest_hash,
        seed=seed,
        extra={
            "split_namespace": namespace,
            "split_manifest_version": manifest["manifest_version"],
            "source_csv_hash": manifest.get("source_csv_sha256") or (sha256_file(source_csv) if source_csv.is_file() else None),
            "gen_train_csv_hash": sha256_file(split_dir / "gen_train.csv"),
            "gen_val_csv_hash": sha256_file(split_dir / "gen_val.csv"),
            # What actually ran, which is not always config.optimizer.name -- see build_optimizer.
            "resolved_optimizer_name": resolved_optimizer_name,
        },
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
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Diagnostic/smoke-test limit; uses separate caches")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. training.train_batch_size=4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration

    cfg = load_stage1_config(overrides=args.overrides)
    ensure_dirs(cfg)
    validate_training_inputs(cfg)
    set_seed(cfg.run.seed)

    checkpoints_root = Path(cfg.paths.checkpoints_dir)
    run_id = args.run_id or make_run_id(dict(cfg))
    run_dir = checkpoints_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoints_root / "latest_run.json", {"run_id": run_id})

    from scripts.utils.manifest import sha256_file
    from scripts.utils.splits import read_split_manifest, resolve_splits_dir
    namespace = str(cfg.split.namespace)
    split_manifest = read_split_manifest(namespace)
    split_manifest_hash = split_manifest["manifest_hash"]
    split_dir = resolve_splits_dir(namespace)
    source_csv = Path(split_manifest["input_csv_path"])
    stage1_provenance = {
        "split_namespace": namespace,
        "split_manifest_version": split_manifest["manifest_version"],
        "split_manifest_hash": split_manifest_hash,
        "source_csv_hash": split_manifest.get("source_csv_sha256") or (sha256_file(source_csv) if source_csv.is_file() else None),
        "gen_train_csv_hash": sha256_file(split_dir / "gen_train.csv"),
        "gen_val_csv_hash": sha256_file(split_dir / "gen_val.csv"),
    }

    log_dir = Path(cfg.paths.logs_dir) / run_id
    trackers = []
    if cfg.logging.tensorboard:
        trackers.append("tensorboard")
    if cfg.logging.wandb.enabled:
        trackers.append("wandb")

    precision = str(cfg.training.precision).lower()
    accelerate_precision = "no" if precision == "fp32" else precision
    if accelerate_precision not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported training.precision={cfg.training.precision!r}; "
            "expected one of: fp32, fp16, bf16"
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=accelerate_precision,
        log_with=trackers or None,
        project_config=ProjectConfiguration(project_dir=str(run_dir), logging_dir=str(log_dir)),
    )
    if trackers:
        accelerator.init_trackers("stage1_lora_sdxl", config=tracker_hparams(cfg))

    weight_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]
    models = build_models(cfg, weight_dtype)

    train_latent_cache, train_prompt_cache, train_records = build_or_load_cache(cfg, models, "gen_train", accelerator.device, args.max_samples_per_split)
    val_latent_cache, val_prompt_cache, val_records = build_or_load_cache(cfg, models, "gen_val", accelerator.device, args.max_samples_per_split)
    # Training reads only the caches from here on; validation and the final export use the UNet
    # alone. Drop the VAE and both text encoders (~2 GB of host RAM) for the rest of the run.
    for frozen_only in ("vae", "text_encoder_one", "text_encoder_two"):
        models.pop(frozen_only, None)
    gc.collect()
    torch.cuda.empty_cache()

    train_dataset = CachedSDXLDataset(train_latent_cache, train_prompt_cache, train_records, cfg.data.resolution)
    val_dataset = CachedSDXLDataset(val_latent_cache, val_prompt_cache, val_records, cfg.data.resolution)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.train_batch_size, shuffle=False, num_workers=0, pin_memory=True)

    unet = models["unet"]
    trainable_params = [p for p in unet.parameters() if p.requires_grad]

    optimizer, resolved_optimizer_name = build_optimizer(cfg, trainable_params, accelerator.device)

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

    # Accelerate's save_state serializes every prepared model IN FULL. For a LoRA run that means
    # ~9.5 GB of FROZEN SDXL UNet base weights written into every checkpoint alongside ~11 MB of
    # actually-trainable LoRA weights — 99.9% of each checkpoint duplicating weights already pinned
    # immutably by model.revision in the HF cache. At save_every_n_steps=1000 over 30k steps that is
    # ~272 GB written and ~27 GB resident (keep_last_n_full_checkpoints=3), plus the GPU sitting idle
    # through each multi-minute serialization.
    #
    # These hooks are the diffusers text_to_image_lora_sdxl reference pattern this script is adapted
    # from (§docstring); they were the one piece of it that was missing. `weights.pop()` is what
    # stops Accelerate writing the full model, and the LoRA adapter is written instead, so the
    # checkpoint stays fully resumable. Optimizer/scheduler/RNG/EMA state are untouched by this and
    # are still saved by Accelerate as before.
    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        from diffusers import StableDiffusionXLPipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft.utils import get_peft_model_state_dict

        unet_lora_layers = None
        unet_type = type(accelerator.unwrap_model(unet))
        for model in models:
            if not isinstance(accelerator.unwrap_model(model), unet_type):
                raise ValueError(f"save_model_hook got an unexpected model: {model.__class__.__name__}")
            unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
            # Drop the full-model weights Accelerate queued for this model.
            weights.pop()
        if unet_lora_layers is None:
            raise ValueError("save_model_hook received no UNet; refusing to write a checkpoint with no LoRA weights")
        StableDiffusionXLPipeline.save_lora_weights(
            str(output_dir), unet_lora_layers=unet_lora_layers, safe_serialization=True
        )

    def load_model_hook(models, input_dir):
        from diffusers import StableDiffusionXLPipeline
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import set_peft_model_state_dict

        unet_type = type(accelerator.unwrap_model(unet))
        target = None
        while models:
            model = models.pop()
            if not isinstance(accelerator.unwrap_model(model), unet_type):
                raise ValueError(f"load_model_hook got an unexpected model: {model.__class__.__name__}")
            target = model
        if target is None:
            raise ValueError("load_model_hook received no UNet to load LoRA weights into")

        lora_state_dict, _ = StableDiffusionXLPipeline.lora_state_dict(str(input_dir))
        unet_state_dict = {
            key.removeprefix("unet."): value
            for key, value in lora_state_dict.items()
            if key.startswith("unet.")
        }
        if not unet_state_dict:
            raise ValueError(f"No UNet LoRA weights found in {input_dir}; cannot resume")
        incompatible = set_peft_model_state_dict(
            target, convert_unet_state_dict_to_peft(unet_state_dict), adapter_name="default"
        )
        unexpected = getattr(incompatible, "unexpected_keys", None)
        if unexpected:
            raise ValueError(f"Unexpected LoRA keys while resuming from {input_dir}: {unexpected}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    ema = LoraEMA(accelerator.unwrap_model(unet), cfg.training.ema.decay) if cfg.training.ema.enabled else None
    if ema is not None:
        accelerator.register_for_checkpointing(ema)

    global_step = 0
    if args.resume_from:
        resumed_metadata = read_json(Path(args.resume_from) / "metadata.json")
        for key, value in stage1_provenance.items():
            if resumed_metadata.get(key) != value:
                raise ValueError(f"Checkpoint {key} differs from current data ({resumed_metadata.get(key)!r} != {value!r}); refusing unsafe resume")
        accelerator.load_state(args.resume_from)
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
                        resolved_optimizer_name,
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
        resolved_optimizer_name,
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
            build_checkpoint_metadata(global_step, epoch, dict(cfg), split_manifest_hash, cfg.run.seed,
                                      extra={"ema_applied": ema is not None, **stage1_provenance}),
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

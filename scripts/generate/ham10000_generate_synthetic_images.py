#!/usr/bin/env python3
"""HAM10000 Stage 2: build generation recipes, generate synthetic dermoscopy, standardise geometry.

FLOW
  --build-recipes   quotas + prompt context from gen_train ONLY -> recipes.csv + recipes_manifest.json
  --mode pilot      a small per-class batch -> automatic checks -> pilot_approval_manifest.json
  --approve-pilot   explicit human approval (a separate command; never automatic)
  --mode full       refuses without an approved pilot for the SAME inputs, then generates everything,
                    validates every output against the manifest, writes all_candidates.csv (the
                    Stage 4 condition-B input) and generation_completion.json

EACH IMAGE
  1. generated at generation.width x height (unpadded native 4:3) with its own seeded generator;
  2. saved raw (validated at exactly that size);
  3. standardised with the SAME letterbox function real images use, giving an exact content box —
     or REFUSED as `geometry_rejected` if it is off-size or carries generator-drawn padding;
  4. recorded in the jsonl manifest with class, prompt, seed, source role, LoRA identity, config
     identity and provenance.

REUSED UNCHANGED from scripts/generate/02_generate_synthetic_images.py (CheXpert): approve_pilot and
enforce_pilot_gate (they operate on a paths dict), and the same resume semantics (skip only
validated + registered outputs, refuse unregistered images, atomic manifest rewrite).

CONTAMINATION GUARDS
  * recipes: source_split must be gen_train; quotas read gen_train.csv only; every context image id
    is checked against the other five splits;
  * LoRA: metadata.json must record train_split=gen_train, this split namespace and manifest hash,
    and the same base model/revision;
  * outputs: recipe ids are `syn_*` and are checked never to collide with any real image id.

Usage:
    python scripts/generate/ham10000_generate_synthetic_images.py --build-recipes
    python scripts/generate/ham10000_generate_synthetic_images.py --mode pilot checkpoint.lora_weights_dir=<dir>
    python scripts/generate/ham10000_generate_synthetic_images.py --approve-pilot --reviewer "Walaa" checkpoint.lora_weights_dir=<dir>
    python scripts/generate/ham10000_generate_synthetic_images.py --mode full checkpoint.lora_weights_dir=<dir>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.generate.ham10000_recipes import GENERATION_SOURCE_SPLIT, build_contextual_recipes, derive_image_seed, quotas_from_gen_train  # noqa: E402
from scripts.utils.artifact_contracts import config_sha256, current_code_identity_hash  # noqa: E402
from scripts.utils.config import CONFIGS_DIR, _smoke_overlay, load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_geometry import UnknownPaddingError, letterbox_geometry, standardize_generated_image, validate_content_box, write_content_boxes  # noqa: E402
from scripts.utils.ham10000_lora_data import ALL_SPLITS, assert_role_isolation, atomic_save_rgb_jpeg, forbidden_image_ids, validate_rgb_jpeg  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, sha256_directory, sha256_file, write_json  # noqa: E402

SCHEMA_VERSION = 1
MANIFEST_ROW_FIELDS = (
    "image_id", "status", "dx", "class_index", "intended_label_vector", "prompt", "negative_prompt", "seed",
    "source_split", "context_image_id", "raw_image_path", "image_path", "content_box", "mode",
)


class GenerationContractError(SystemExit):
    pass


def _chexpert_stage2():
    spec = importlib.util.spec_from_file_location("chexpert_stage2", Path(__file__).resolve().parent / "02_generate_synthetic_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------
# Config + paths
# ------------------------------------------------------------------------------------------

def load_config(overrides: list[str] | None = None):
    config = OmegaConf.load(CONFIGS_DIR / "ham10000_stage2.yaml")
    overlay = _smoke_overlay("ham_stage2")
    if overlay is not None:
        config = OmegaConf.merge(config, overlay)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config


def validate_config(cfg, stage1_cfg) -> None:
    if str(cfg.recipes.source_split) != GENERATION_SOURCE_SPLIT:
        raise GenerationContractError(f"recipes.source_split must be {GENERATION_SOURCE_SPLIT!r}; got {cfg.recipes.source_split!r}")
    size = [int(cfg.generation.width), int(cfg.generation.height)]
    if size != [int(v) for v in stage1_cfg.geometry.generation_size]:
        raise GenerationContractError(f"generation size {size} != ham10000_stage1 geometry.generation_size {list(stage1_cfg.geometry.generation_size)}")
    for key in ("num_inference_steps", "batch_size"):
        if int(cfg.generation[key]) < 1:
            raise GenerationContractError(f"generation.{key} must be >= 1")
    if not str(cfg.checkpoint.revision or "").strip():
        raise GenerationContractError("checkpoint.revision must pin the base model")


def stage2_paths(cfg, namespace: str) -> dict[str, Path]:
    root = Path(cfg.paths.stage2_root) / namespace
    return {
        "root": root,
        "recipes_path": root / "recipes.csv",
        "recipes_manifest": root / "recipes_manifest.json",
        "pilot_raw": root / "pilot" / "raw",
        "pilot_candidates": root / "pilot" / "candidates",
        "pilot_manifest": root / "pilot" / "generation_manifest.jsonl",
        "pilot_approval_manifest": root / "pilot" / "pilot_approval_manifest.json",
        "full_raw": root / "full" / "raw",
        "full_candidates": root / "full" / "candidates",
        "full_manifest": root / "full" / "generation_manifest.jsonl",
        "all_candidates_csv": root / "all_candidates.csv",
        "generation_completion": root / "generation_completion.json",
    }


def _split_manifest_hash(split_dir: Path) -> str:
    path = Path(split_dir) / "split_manifest_v2.json"
    if not path.is_file():
        raise GenerationContractError(f"missing split manifest {path}")
    return read_json(path)["manifest_hash"]


def real_image_ids(split_dir: Path) -> frozenset[str]:
    ids: set[str] = set()
    for name in ALL_SPLITS:
        ids |= set(pd.read_csv(Path(split_dir) / f"{name}.csv", usecols=["image_id"])["image_id"].astype(str))
    return frozenset(ids)


# ------------------------------------------------------------------------------------------
# Recipes
# ------------------------------------------------------------------------------------------

def build_recipes(cfg, stage1_cfg, namespace: str, split_dir: Path) -> tuple[pd.DataFrame, dict]:
    validate_config(cfg, stage1_cfg)
    gen_train = pd.read_csv(Path(split_dir) / f"{GENERATION_SOURCE_SPLIT}.csv")
    forbidden = forbidden_image_ids(split_dir, [GENERATION_SOURCE_SPLIT])
    assert_role_isolation(gen_train["image_id"], GENERATION_SOURCE_SPLIT, forbidden)
    rc = cfg.recipes
    quotas, counts = quotas_from_gen_train(
        split_dir, base_quota=int(rc.base_quota), rarity_exponent=float(rc.rarity_exponent),
        min_per_class=int(rc.min_per_class), max_per_class=int(rc.max_per_class),
    )
    recipes = build_contextual_recipes(gen_train, quotas, rc, int(cfg.generation.seed), stage1_cfg.captions)
    assert_role_isolation(recipes["context_image_id"], "recipe context", forbidden)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ham10000",
        "split_namespace": namespace,
        "split_manifest_hash": _split_manifest_hash(split_dir),
        "source_split": GENERATION_SOURCE_SPLIT,
        "source_split_csv_sha256": sha256_file(Path(split_dir) / f"{GENERATION_SOURCE_SPLIT}.csv"),
        "forbidden_splits_checked": sorted(forbidden),
        "recipes_config_sha256": config_sha256(cfg.recipes),
        "base_seed": int(cfg.generation.seed),
        "captions_config_sha256": config_sha256(stage1_cfg.captions),
        "gen_train_class_counts": counts,
        "quotas": quotas,
        "num_recipes": int(len(recipes)),
    }
    return recipes, manifest


def write_recipes(recipes: pd.DataFrame, manifest: dict, paths: dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".recipes.", suffix=".tmp", dir=paths["root"])
    os.close(fd)
    try:
        recipes.to_csv(temporary, index=False)
        os.replace(temporary, paths["recipes_path"])
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_json(paths["recipes_manifest"], {**manifest, "recipes_csv_sha256": sha256_file(paths["recipes_path"])})


# ------------------------------------------------------------------------------------------
# Input validation -> provenance
# ------------------------------------------------------------------------------------------

def validate_generation_inputs(cfg, stage1_cfg, namespace: str, split_dir: Path, paths: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    """Every precondition for generation; returns (recipes, provenance). Loads no model."""
    validate_config(cfg, stage1_cfg)
    split_hash = _split_manifest_hash(split_dir)
    if not paths["recipes_manifest"].is_file() or not paths["recipes_path"].is_file():
        raise GenerationContractError(f"missing recipes under {paths['root']}; run --build-recipes")
    manifest = read_json(paths["recipes_manifest"])
    expected = {
        "split_namespace": namespace,
        "split_manifest_hash": split_hash,
        "source_split": GENERATION_SOURCE_SPLIT,
        "source_split_csv_sha256": sha256_file(Path(split_dir) / f"{GENERATION_SOURCE_SPLIT}.csv"),
        "recipes_config_sha256": config_sha256(cfg.recipes),
        "base_seed": int(cfg.generation.seed),
        "captions_config_sha256": config_sha256(stage1_cfg.captions),
        "recipes_csv_sha256": sha256_file(paths["recipes_path"]),
    }
    stale = {k: (manifest.get(k), v) for k, v in expected.items() if manifest.get(k) != v}
    if stale:
        raise GenerationContractError(f"recipes are stale or tampered: {stale}; rerun --build-recipes")

    recipes = pd.read_csv(paths["recipes_path"])
    ids = recipes["recipe_id"].astype(str)
    if ids.duplicated().any():
        raise GenerationContractError("duplicate recipe_id in recipes.csv")
    if not ids.str.startswith("syn_").all():
        raise GenerationContractError("every synthetic recipe_id must start with 'syn_'")
    collisions = sorted(set(ids) & real_image_ids(split_dir))
    if collisions:
        raise GenerationContractError(f"recipe ids collide with real image ids: {collisions[:3]}")
    if not recipes["source_split"].eq(GENERATION_SOURCE_SPLIT).all():
        raise GenerationContractError("a recipe names a source split other than gen_train")
    forbidden = forbidden_image_ids(split_dir, [GENERATION_SOURCE_SPLIT])
    assert_role_isolation(recipes["context_image_id"], "recipe context", forbidden)
    for row in recipes.itertuples():
        if normalize_diagnosis(row.dx) != row.dx or CLASSIFIER_TARGET_LABELS[int(row.class_index)] != row.dx:
            raise GenerationContractError(f"{row.recipe_id}: dx/class_index inconsistent")
        if int(row.seed) != derive_image_seed(int(cfg.generation.seed), row.recipe_id):
            raise GenerationContractError(f"{row.recipe_id}: seed is not the derived seed")

    lora_dir = cfg.checkpoint.lora_weights_dir
    if not lora_dir:
        raise GenerationContractError("checkpoint.lora_weights_dir is not set; pass an explicit HAM10000 LoRA weights directory")
    lora_path = Path(str(lora_dir))
    if not (lora_path / "metadata.json").is_file():
        raise GenerationContractError(f"LoRA weights directory has no metadata.json: {lora_path}")
    metadata = read_json(lora_path / "metadata.json")
    required = {
        "dataset": "ham10000",
        "split_namespace": namespace,
        "split_manifest_hash": split_hash,
        "train_split": GENERATION_SOURCE_SPLIT,
        "base_model_id": str(cfg.checkpoint.base_model_id),
        "base_model_revision": str(cfg.checkpoint.revision),
        "lora_training_size": [int(cfg.generation.width), int(cfg.generation.height)],
    }
    mismatch = {k: (metadata.get(k), v) for k, v in required.items() if metadata.get(k) != v}
    if mismatch:
        raise GenerationContractError(f"LoRA checkpoint does not match this generation: {mismatch}")

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ham10000",
        "split_namespace": namespace,
        "split_manifest_hash": split_hash,
        "source_split": GENERATION_SOURCE_SPLIT,
        "recipes_csv_sha256": expected["recipes_csv_sha256"],
        "recipes_manifest_sha256": sha256_file(paths["recipes_manifest"]),
        "lora_weights_dir": str(lora_path),
        "lora_checkpoint_sha256": sha256_directory(lora_path),
        "lora_metadata_sha256": sha256_file(lora_path / "metadata.json"),
        "lora_train_image_ids_sha256": metadata.get("train_image_ids_sha256"),
        "lora_step": metadata.get("step"),
        "generation_config_sha256": config_sha256(cfg.generation),
        "checkpoint_config_sha256": config_sha256(cfg.checkpoint),
        "geometry_sha256": hash_dict(OmegaConf.to_container(stage1_cfg.geometry, resolve=True), length=64) + f":{int(stage1_cfg.data.resolution)}",
        "code_identity_sha256": current_code_identity_hash(),
    }
    return recipes, provenance


# ------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------

def build_pipeline(cfg, stage1_cfg):
    """SDXL + the pinned HAM10000 LoRA. Imported lazily so everything else runs without a GPU stack."""
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        str(cfg.checkpoint.base_model_id), revision=str(cfg.checkpoint.revision), torch_dtype=dtype, cache_dir=str(stage1_cfg.paths.hf_cache_dir)
    )
    if str(cfg.generation.scheduler) != "DPMSolverMultistep":
        raise GenerationContractError(f"unsupported generation.scheduler {cfg.generation.scheduler!r}")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(str(cfg.checkpoint.lora_weights_dir))
    pipeline = pipeline.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def _read_manifest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.is_file():
        return rows
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in rows:
            raise GenerationContractError(f"missing/duplicate image_id at line {number} of {path}")
        rows[image_id] = row
    return rows


def _rewrite_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def expected_content_box(cfg, stage1_cfg) -> tuple[float, float, float, float]:
    return letterbox_geometry(int(cfg.generation.width), int(cfg.generation.height), int(stage1_cfg.data.resolution))["content_box"]


def _row_is_complete(row: dict, cfg, stage1_cfg) -> bool:
    if row.get("status") == "geometry_rejected":
        return True  # deterministic seed: regenerating would reproduce the same refused image
    raw_ok, _ = validate_rgb_jpeg(Path(row["raw_image_path"]), (int(cfg.generation.width), int(cfg.generation.height)))
    resolution = int(stage1_cfg.data.resolution)
    canvas_ok, _ = validate_rgb_jpeg(Path(row["image_path"]), (resolution, resolution))
    return raw_ok and canvas_ok


def run_generation(recipes: pd.DataFrame, cfg, stage1_cfg, paths: dict[str, Path], mode: str, provenance: dict, pipeline_factory=build_pipeline) -> dict:
    """Generate every recipe not already complete. Returns counters. Resume-safe and idempotent."""
    raw_dir, candidate_dir, manifest_path = paths[f"{mode}_raw"], paths[f"{mode}_candidates"], paths[f"{mode}_manifest"]
    width, height = int(cfg.generation.width), int(cfg.generation.height)
    resolution = int(stage1_cfg.data.resolution)

    existing = _read_manifest(manifest_path)
    recipe_ids = set(recipes["recipe_id"].astype(str))
    unknown = sorted(set(existing) - recipe_ids)
    if unknown:
        raise GenerationContractError(f"manifest has {len(unknown)} ids absent from the recipe set: {unknown[:3]}")
    for image_id, row in existing.items():
        for key, value in provenance.items():
            if row.get(key) != value:
                raise GenerationContractError(f"manifest provenance mismatch for {image_id} at {key}; refusing stale reuse")

    retained, pending = [], []
    counts = {"generated": 0, "geometry_rejected": 0, "skipped_complete": 0, "failed": 0}
    for recipe in recipes.to_dict("records"):
        image_id = str(recipe["recipe_id"])
        row = existing.get(image_id)
        if row is not None and _row_is_complete(row, cfg, stage1_cfg):
            retained.append(row)
            counts["skipped_complete"] += 1
            continue
        raw_path = raw_dir / f"{image_id}.jpg"
        if row is None and validate_rgb_jpeg(raw_path, (width, height))[0]:
            raise GenerationContractError(f"unregistered generated image exists at {raw_path}; refusing to treat it as complete")
        pending.append(recipe)
    if len(retained) != len(existing):
        _rewrite_manifest(manifest_path, retained)
    print(f"[{mode}] {counts['skipped_complete']} complete, {len(pending)} to generate", flush=True)
    if not pending:
        return counts

    import torch

    pipeline = pipeline_factory(cfg, stage1_cfg)
    batch_size = int(cfg.generation.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    negative = str(cfg.generation.negative_prompt) or None
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8", newline="\n", buffering=1) as handle:
        for start in tqdm(range(0, len(pending), batch_size), desc=f"[{mode}] generate", unit="batch"):
            batch = pending[start : start + batch_size]
            try:
                generators = [torch.Generator(device=device).manual_seed(int(r["seed"])) for r in batch]
                result = pipeline(
                    prompt=[r["prompt"] for r in batch],
                    negative_prompt=[negative] * len(batch) if negative else None,
                    num_inference_steps=int(cfg.generation.num_inference_steps),
                    guidance_scale=float(cfg.generation.guidance_scale),
                    height=height, width=width, generator=generators,
                )
            except Exception as exc:  # noqa: BLE001 - one failed batch must not kill a long run
                counts["failed"] += len(batch)
                print(f"[{mode}] FAILED batch starting {batch[0]['recipe_id']}: {type(exc).__name__}: {exc}", flush=True)
                continue
            for recipe, image in zip(batch, result.images):
                image_id = str(recipe["recipe_id"])
                raw_path, canvas_path = raw_dir / f"{image_id}.jpg", candidate_dir / f"{image_id}.jpg"
                row = {
                    "image_id": image_id, "dx": recipe["dx"], "class_index": int(recipe["class_index"]),
                    "intended_label_vector": json.loads(recipe["intended_label_vector"]), "prompt": recipe["prompt"],
                    "negative_prompt": negative, "seed": int(recipe["seed"]), "source_split": recipe["source_split"],
                    "context_image_id": recipe["context_image_id"], "raw_image_path": str(raw_path), "mode": mode,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(), **provenance,
                }
                try:
                    atomic_save_rgb_jpeg(image.convert("RGB"), raw_path, (width, height), int(cfg.generation.jpeg_quality))
                    try:
                        canvas, box_row = standardize_generated_image(
                            image.convert("RGB"), image_id, (width, height), resolution,
                            tuple(stage1_cfg.data.pad_colour), float(stage1_cfg.geometry.generated_min_edge_std),
                        )
                    except UnknownPaddingError as exc:
                        row.update({"status": "geometry_rejected", "rejection_reason": str(exc), "image_path": None, "content_box": None})
                        counts["geometry_rejected"] += 1
                    else:
                        atomic_save_rgb_jpeg(canvas, canvas_path, (resolution, resolution), int(cfg.generation.jpeg_quality))
                        row.update({"status": "accepted", "image_path": str(canvas_path), "content_box": [box_row["x0"], box_row["y0"], box_row["x1"], box_row["y1"]]})
                        counts["generated"] += 1
                except Exception as exc:  # noqa: BLE001
                    counts["failed"] += 1
                    print(f"[{mode}] FAILED {image_id}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def validate_generation_outputs(recipes: pd.DataFrame, cfg, stage1_cfg, paths: dict[str, Path], mode: str, provenance: dict) -> dict:
    """Manifest <-> recipes <-> files consistency. Raises GenerationContractError on any violation."""
    rows = _read_manifest(paths[f"{mode}_manifest"])
    by_id = {str(r["recipe_id"]): r for r in recipes.to_dict("records")}
    problems = []
    if set(rows) != set(by_id):
        problems.append(f"manifest ids != recipe ids (missing {len(set(by_id) - set(rows))}, extra {len(set(rows) - set(by_id))})")
    box = list(expected_content_box(cfg, stage1_cfg))
    resolution = int(stage1_cfg.data.resolution)
    accepted = rejected = 0
    for image_id, row in rows.items():
        recipe = by_id.get(image_id)
        if recipe is None:
            continue
        if any(row.get(k) != v for k, v in provenance.items()):
            problems.append(f"{image_id}: provenance differs")
        if row.get("dx") != recipe["dx"] or int(row.get("seed", -1)) != int(recipe["seed"]) or row.get("prompt") != recipe["prompt"]:
            problems.append(f"{image_id}: dx/seed/prompt differ from recipe")
        if row.get("source_split") != GENERATION_SOURCE_SPLIT:
            problems.append(f"{image_id}: source_split {row.get('source_split')!r}")
        if row.get("status") == "geometry_rejected":
            rejected += 1
            continue
        if row.get("status") != "accepted":
            problems.append(f"{image_id}: unknown status {row.get('status')!r}")
            continue
        accepted += 1
        if not validate_rgb_jpeg(Path(row["raw_image_path"]), (int(cfg.generation.width), int(cfg.generation.height)))[0]:
            problems.append(f"{image_id}: raw image invalid")
        if not validate_rgb_jpeg(Path(row["image_path"]), (resolution, resolution))[0]:
            problems.append(f"{image_id}: standardised image invalid")
        try:
            if list(validate_content_box(row["content_box"])) != box:
                problems.append(f"{image_id}: content box {row['content_box']} != {box}")
        except (TypeError, ValueError) as exc:
            problems.append(f"{image_id}: content box invalid ({exc})")
    if problems:
        raise GenerationContractError(f"generation outputs failed validation ({len(problems)}): {problems[:10]}")
    return {"rows": len(rows), "accepted": accepted, "geometry_rejected": rejected}


def write_candidate_tables(recipes: pd.DataFrame, cfg, stage1_cfg, paths: dict[str, Path], mode: str) -> dict:
    rows = [r for r in _read_manifest(paths[f"{mode}_manifest"]).values() if r.get("status") == "accepted"]
    rows.sort(key=lambda r: r["image_id"])
    write_content_boxes(paths[f"{mode}_candidates"] / "content_boxes.csv", [
        {"image_id": r["image_id"], "source_width": int(cfg.generation.width), "source_height": int(cfg.generation.height),
         "x0": r["content_box"][0], "y0": r["content_box"][1], "x1": r["content_box"][2], "y1": r["content_box"][3]}
        for r in rows
    ])
    table = pd.DataFrame([{"image_id": r["image_id"], "image_path": r["image_path"], "dx": r["dx"], "seed": r["seed"]} for r in rows], columns=["image_id", "image_path", "dx", "seed"])
    return {"candidates": table, "per_class": table["dx"].value_counts().reindex(CLASSIFIER_TARGET_LABELS, fill_value=0).astype(int).to_dict()}


def select_pilot(recipes: pd.DataFrame, per_class: int) -> pd.DataFrame:
    return recipes.groupby("dx", sort=False).head(int(per_class)).reset_index(drop=True)


def run_pilot_checks(recipes: pd.DataFrame, cfg, stage1_cfg, paths: dict[str, Path]) -> dict:
    checks_cfg = cfg.pilot.checks
    rows = _read_manifest(paths["pilot_manifest"])
    total = len(recipes)
    rejected = sum(1 for r in rows.values() if r.get("status") == "geometry_rejected")
    valid = near_uniform = 0
    resolution = int(stage1_cfg.data.resolution)
    for row in rows.values():
        if row.get("status") != "accepted" or not validate_rgb_jpeg(Path(row["image_path"]), (resolution, resolution))[0]:
            continue
        valid += 1
        with Image.open(row["raw_image_path"]) as image:
            if float(np.asarray(image.convert("L"), dtype=np.float32).std()) < float(checks_cfg.blank_std_threshold):
                near_uniform += 1
    # Geometry rejections have their own threshold; valid_fraction is measured over the images that
    # were NOT geometry-rejected, so one refused image is not counted against two limits at once.
    eligible = total - rejected
    results = {
        "total": total, "valid": valid, "geometry_rejected": rejected, "near_uniform": near_uniform,
        "valid_fraction": valid / eligible if eligible else 0.0,
        "geometry_rejected_fraction": rejected / total if total else 1.0,
        "near_uniform_fraction": near_uniform / total if total else 1.0,
    }
    results["passed"] = (
        results["valid_fraction"] >= float(checks_cfg.min_valid_image_fraction)
        and results["near_uniform_fraction"] <= float(checks_cfg.max_near_uniform_fraction)
        and results["geometry_rejected_fraction"] <= float(checks_cfg.max_geometry_rejected_fraction)
    )
    results["thresholds"] = OmegaConf.to_container(checks_cfg, resolve=True)
    return results


# ------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------

def main(argv=None, pipeline_factory=build_pipeline) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-recipes", action="store_true")
    action.add_argument("--mode", choices=["pilot", "full"])
    action.add_argument("--approve-pilot", action="store_true")
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. checkpoint.lora_weights_dir=<dir>")
    args = parser.parse_args(argv)

    cfg = load_config(args.overrides)
    stage1_cfg = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    namespace = str(cfg.split_namespace)
    split_dir = Path(cfg.paths.splits_root) / namespace
    paths = stage2_paths(cfg, namespace)

    if args.build_recipes:
        if paths["recipes_path"].is_file():
            raise GenerationContractError(f"recipes already exist at {paths['recipes_path']}; recipes are immutable once written (use a new split namespace or remove them deliberately)")
        recipes, manifest = build_recipes(cfg, stage1_cfg, namespace, split_dir)
        write_recipes(recipes, manifest, paths)
        print(json.dumps({"recipes": len(recipes), "quotas": manifest["quotas"], "path": str(paths["recipes_path"])}, indent=2))
        return 0

    recipes, provenance = validate_generation_inputs(cfg, stage1_cfg, namespace, split_dir, paths)
    chexpert = _chexpert_stage2()
    if args.approve_pilot:
        if not args.reviewer:
            raise GenerationContractError("--approve-pilot requires --reviewer <name>")
        return chexpert.approve_pilot(cfg, paths, provenance, args.reviewer, args.notes)

    if args.mode == "pilot":
        selected = select_pilot(recipes, int(cfg.pilot.num_images_per_class))
        counts = run_generation(selected, cfg, stage1_cfg, paths, "pilot", provenance, pipeline_factory)
        checks = run_pilot_checks(selected, cfg, stage1_cfg, paths)
        approval = paths["pilot_approval_manifest"]
        previous = read_json(approval) if approval.is_file() else {}
        keep_approval = previous.get("pilot_manifest_sha256") == sha256_file(paths["pilot_manifest"]) and previous.get("split_provenance") == provenance
        write_json(approval, {
            "schema_version": SCHEMA_VERSION, "pilot_completed": counts["failed"] == 0, "counts": counts, "checks": checks,
            "split_provenance": provenance, "pilot_manifest_sha256": sha256_file(paths["pilot_manifest"]),
            "candidates_dir": str(paths["pilot_candidates"]), "raw_dir": str(paths["pilot_raw"]),
            "git_commit_hash": get_git_commit_hash(), "approval": previous.get("approval", {"approved": False}) if keep_approval else {"approved": False},
        })
        print(json.dumps({"counts": counts, "checks": checks}, indent=2), flush=True)
        return 0 if checks["passed"] else 3

    chexpert.enforce_pilot_gate(cfg, paths, provenance)
    counts = run_generation(recipes, cfg, stage1_cfg, paths, "full", provenance, pipeline_factory)
    if counts["failed"]:
        print(json.dumps(counts), flush=True)
        raise GenerationContractError(f"{counts['failed']} image(s) failed; rerun the same command to resume")
    report = validate_generation_outputs(recipes, cfg, stage1_cfg, paths, "full", provenance)
    tables = write_candidate_tables(recipes, cfg, stage1_cfg, paths, "full")
    tables["candidates"].to_csv(paths["all_candidates_csv"], index=False)
    write_json(paths["generation_completion"], {
        **provenance, "status": "complete", "counts": counts, "validation": report, "accepted_per_class": tables["per_class"],
        "generation_manifest_sha256": sha256_file(paths["full_manifest"]),
        "all_candidates_csv_sha256": sha256_file(paths["all_candidates_csv"]),
        "content_boxes_sha256": sha256_file(paths["full_candidates"] / "content_boxes.csv"),
    })
    print(json.dumps({"counts": counts, "validation": report, "accepted_per_class": tables["per_class"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

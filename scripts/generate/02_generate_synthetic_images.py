#!/usr/bin/env python3
"""Stage 2b — generate synthetic CXRs from the recipe table (docs/stages2_to_5_plan.md §3).

Loads base SDXL + the explicitly-pinned Stage 1 LoRA weights and generates one image per recipe.
Captions come from scripts/utils/caption_builder.py VERBATIM — the same module Stage 1 trained
with, because any train/generation phrasing drift degrades conditioning fidelity
(docs/stage1_plan.md §11).

Resumable and idempotent on the same terms as 03_preprocess_images.py: an existing output is
skipped only after being decoded and validated (correct format, mode, and resolution), never on
mere existence; every write is atomic (temp file + os.replace) so an interrupted pod cannot leave a
half-written JPEG.

PILOT GATE (§3, FROZEN): `--mode full` refuses to start unless a pilot_approval_manifest.json
exists, records a completed pilot, records passing automatic checks, and carries an explicit human
approval record. Approval is a separate deliberate command; this script never self-approves.

Usage:
    python scripts/generate/02_generate_synthetic_images.py --mode pilot
    python scripts/generate/02_generate_synthetic_images.py --approve-pilot --reviewer "Walaa"
    python scripts/generate/02_generate_synthetic_images.py --mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from scripts.utils.caption_builder import CaptionConfig, build_caption  # noqa: E402
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.experiment_registry import ExperimentRun  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS  # noqa: E402
from scripts.utils.artifact_contracts import (  # noqa: E402
    ArtifactContractError, config_sha256, current_code_identity_hash, namespace_identity,
    require_manifest_fields, stage2_paths,
)
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, sha256_directory, sha256_file, write_json  # noqa: E402

SCHEMA_VERSION = 2


def load_stage2_config():
    return load_named_config("stage2_generation.yaml", "stage2")


def derive_seed(base_seed: int, recipe_id: str) -> int:
    """Deterministic per-image seed: same recipe id always regenerates the same image, so a resumed
    run reproduces exactly what an uninterrupted run would have produced."""
    digest = hashlib.sha256(f"{base_seed}:{recipe_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def recipe_to_caption_row(recipe: dict) -> dict:
    """Map a recipe row into the dict shape caption_builder expects.

    Support Devices is injected here as a CONTEXT attribute (it drives the device clause) while
    staying out of the primary disease vector — §3.4.
    """
    intended = json.loads(recipe["intended_label_vector"])
    row = {label: intended.get(label, 0) for label in PRIMARY_ENDPOINT_LABELS}
    row["No Finding"] = 1 if bool(recipe["is_no_finding"]) else 0
    row["Support Devices"] = int(recipe["support_devices"])
    row["Age"] = int(recipe["age_bucket_start"])
    row["Sex"] = str(recipe["sex"])
    row["Frontal/Lateral"] = str(recipe["view"])
    row["AP/PA"] = "AP"
    return row


def validate_generated_image(path: Path, resolution: int) -> tuple[bool, str | None]:
    """Fully decode and check an existing output. Existence alone is never sufficient."""
    if not path.is_file():
        return False, "missing"
    try:
        with Image.open(path) as image:
            if image.format != "JPEG":
                return False, f"format_{image.format or 'unknown'}"
            if image.mode != "RGB":
                return False, f"mode_{image.mode}"
            if image.size != (resolution, resolution):
                return False, f"size_{image.size[0]}x{image.size[1]}"
            image.load()
    except Exception as exc:
        return False, f"unreadable_{type(exc).__name__}"
    return True, None


def atomic_save_jpeg(image: Image.Image, destination: Path, quality: int = 95) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        image.save(temp_path, format="JPEG", quality=quality)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def check_upstream_ready(config) -> Path:
    """Validate Stage 2's real upstream gate (an explicit, existing Stage 1 LoRA checkpoint) using
    only config and the filesystem — before importing the GPU stack, so a missing checkpoint fails
    with a clear message rather than an unrelated ImportError."""
    lora_dir = config.checkpoint.lora_weights_dir
    if not lora_dir:
        raise SystemExit(
            "UPSTREAM GATE: checkpoint.lora_weights_dir is not set in configs/stage2_generation.yaml.\n"
            "Stage 2 requires an EXPLICIT Stage 1 LoRA checkpoint (never 'latest') so generation is "
            "reproducible — see docs/stages2_to_5_plan.md §3.\n"
            "Set it to e.g. checkpoints/stage1_lora_sdxl/<run_id>/lora_weights/step_20000"
        )
    lora_path = Path(lora_dir)
    if not lora_path.is_dir():
        raise SystemExit(
            f"UPSTREAM GATE: LoRA weights directory not found: {lora_path}\n"
            "Stage 1 must produce a checkpoint before Stage 2 can generate."
        )
    return lora_path


def validate_generation_inputs(config, namespace: str, paths: dict[str, Path]) -> dict:
    if not paths["recipes_path"].is_file():
        legacy = Path(config.paths.synthetic_root) / "label_recipes.csv"
        if legacy.is_file():
            raise ArtifactContractError(
                f"Refusing stale unnamespaced Stage 2 recipes at {legacy}. Regenerate recipes into {paths['root']} for split run {namespace!r}."
            )
        raise ArtifactContractError(f"Missing namespaced recipe table: {paths['recipes_path']}")
    identity = namespace_identity(namespace)
    expected = {
        "schema_version": 2,
        "split_namespace": namespace,
        "namespace_class": identity["namespace_class"],
        "split_manifest_hash": identity["split_manifest_hash"],
        "recipe_config_sha256": config_sha256(config.recipes),
        "recipes_csv_sha256": sha256_file(paths["recipes_path"]),
        "code_identity_sha256": current_code_identity_hash(),
    }
    recipe_manifest = require_manifest_fields(paths["recipes_manifest"], expected, "Stage 2 recipe")
    lora_path = check_upstream_ready(config)
    lora_hash = sha256_directory(lora_path)
    metadata_path = lora_path / "metadata.json"
    if not metadata_path.is_file():
        raise ArtifactContractError(f"LoRA checkpoint has no metadata.json: {lora_path}")
    metadata = read_json(metadata_path)
    for key, value in (("split_namespace", namespace), ("split_manifest_hash", identity["split_manifest_hash"])):
        if metadata.get(key) != value:
            raise ArtifactContractError(f"LoRA checkpoint {key} mismatch: expected {value!r}, got {metadata.get(key)!r}")
    return {**identity, "recipes_csv_sha256": expected["recipes_csv_sha256"],
            "recipes_manifest_sha256": sha256_file(paths["recipes_manifest"]),
            "lora_checkpoint_sha256": lora_hash, "lora_metadata_sha256": sha256_file(metadata_path),
            "generation_config_sha256": config_sha256(config.generation),
            "stage2_config_sha256": config_sha256(config), "code_identity_sha256": current_code_identity_hash()}


def build_pipeline(config, stage1_cfg):
    """Load SDXL + the pinned LoRA weights. Imported lazily so the pilot gate and manifest logic
    remain testable on a machine with no GPU stack installed."""
    lora_path = check_upstream_ready(config)

    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
    except ImportError as exc:
        raise SystemExit(
            f"UPSTREAM GATE: the generation stack is not installed here ({exc}).\n"
            "Stage 2 generation is GPU work — run it on the RunPod pod after:\n"
            "  pip install -r environment/requirements.txt"
        ) from exc

    # Keep every SDXL component and the injected LoRA adapter in the same dtype.
    # The pinned SDXL checkpoint/LoRA path is float16-compatible; forcing bfloat16 here can
    # leave projection weights in fp16 and fail at inference with Half != BFloat16.
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        config.checkpoint.base_model_id,
        revision=config.checkpoint.revision,
        torch_dtype=dtype,
        cache_dir=str(stage1_cfg.paths.hf_cache_dir),
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.load_lora_weights(str(lora_path))
    if torch.cuda.is_available():
        pipeline = pipeline.to(device="cuda", dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def run_generation(
    recipes: pd.DataFrame,
    config,
    stage1_cfg,
    images_dir: Path,
    manifest_path: Path,
    mode: str,
    artifact_provenance: dict,
) -> dict:
    """Generate images for `recipes`, skipping already-valid outputs. Returns counters."""
    resolution = int(config.generation.resolution)
    caption_config = CaptionConfig(
        age_bucket_width_years=int(stage1_cfg.captions.age_bucket_width_years),
        uncertain_label_policy=str(stage1_cfg.captions.uncertain_label_policy),
        no_finding_overrides_positives=bool(stage1_cfg.captions.no_finding_overrides_positives),
        num_paraphrase_variants=int(stage1_cfg.captions.num_paraphrase_variants),
        template_version=str(stage1_cfg.captions.template_version),
    )

    # Determine what still needs generating BEFORE loading the model, so a fully-complete resume
    # costs nothing and never touches the GPU.
    existing_rows: dict[str, dict] = {}
    if manifest_path.is_file():
        for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id", ""))
            if not image_id or image_id in existing_rows:
                raise SystemExit(f"Generation manifest has a missing/duplicate image_id at line {line_number}: {manifest_path}")
            for key, expected in artifact_provenance.items():
                if row.get(key) != expected:
                    raise SystemExit(f"Generation manifest provenance mismatch for {image_id} at {key}; refusing stale reuse")
            existing_rows[image_id] = row

    pending = []
    retained_rows = []
    counts = {"generated": 0, "skipped_valid": 0, "failed": 0}
    for recipe in recipes.to_dict("records"):
        destination = images_dir / f"{recipe['recipe_id']}.jpg"
        valid, _ = validate_generated_image(destination, resolution)
        existing = existing_rows.get(str(recipe["recipe_id"]))
        if valid and existing is None:
            raise SystemExit(
                f"Unregistered generated image exists at {destination}; refusing to treat a stale/unpinned image as complete"
            )
        if valid:
            counts["skipped_valid"] += 1
            retained_rows.append(existing)
        else:
            pending.append(recipe)

    unknown = sorted(set(existing_rows) - set(recipes["recipe_id"].astype(str)))
    if unknown:
        raise SystemExit(f"Generation manifest contains {len(unknown)} IDs absent from the current recipe set: {unknown[:5]}")

    # Remove rows whose image failed validation before appending replacements. Publication is
    # atomic, so an interruption cannot create duplicate manifest rows on the next resume.
    if manifest_path.is_file() and len(retained_rows) != len(existing_rows):
        fd, temp_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent)
        os.close(fd)
        try:
            with open(temp_name, "w", encoding="utf-8") as handle:
                for row in retained_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temp_name, manifest_path)
        finally:
            Path(temp_name).unlink(missing_ok=True)

    print(
        f"[{mode}] {counts['skipped_valid']} already valid, {len(pending)} to generate",
        flush=True,
    )
    if not pending:
        return counts

    pipeline = build_pipeline(config, stage1_cfg)
    import torch  # available by construction once build_pipeline() returned

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    generation_config_hash = hash_dict(OmegaConf.to_container(config.generation, resolve=True))
    base_seed = int(config.generation.seed)

    with open(manifest_path, "a", encoding="utf-8", buffering=1) as manifest_handle:
        progress = tqdm(pending, desc=f"[{mode}] generate", unit="image")
        for recipe in progress:
            recipe_id = recipe["recipe_id"]
            destination = images_dir / f"{recipe_id}.jpg"
            caption_row = recipe_to_caption_row(recipe)
            caption = build_caption(caption_row, caption_config, variant_index=0)
            seed = derive_seed(base_seed, recipe_id)

            try:
                generator = torch.Generator(
                    device="cuda" if torch.cuda.is_available() else "cpu"
                ).manual_seed(seed)
                result = pipeline(
                    prompt=caption,
                    negative_prompt=str(config.generation.negative_prompt) or None,
                    num_inference_steps=int(config.generation.num_inference_steps),
                    guidance_scale=float(config.generation.guidance_scale),
                    height=resolution,
                    width=resolution,
                    generator=generator,
                )
                image = result.images[0].convert("RGB")
                atomic_save_jpeg(image, destination)

                valid, reason = validate_generated_image(destination, resolution)
                if not valid:
                    counts["failed"] += 1
                    progress.set_postfix(counts, refresh=False)
                    continue

                counts["generated"] += 1
                manifest_handle.write(
                    json.dumps(
                        {
                            "image_id": recipe_id,
                            "schema_version": SCHEMA_VERSION,
                            "intended_label_vector": json.loads(recipe["intended_label_vector"]),
                            "is_no_finding": bool(recipe["is_no_finding"]),
                            "support_devices": int(recipe["support_devices"]),
                            "caption": caption,
                            "seed": seed,
                            "generation_config_hash": generation_config_hash,
                            "lora_weights_dir": str(config.checkpoint.lora_weights_dir),
                            **artifact_provenance,
                            "mode": mode,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as exc:  # noqa: BLE001 - one bad recipe must not kill a long run
                counts["failed"] += 1
                print(f"[{mode}] FAILED {recipe_id}: {type(exc).__name__}: {exc}", flush=True)
            progress.set_postfix(counts, refresh=False)
        progress.close()

    return counts


def run_pilot_checks(images_dir: Path, recipe_ids: list[str], config) -> dict:
    """Automatic artifact checks the pilot must pass before it is eligible for approval."""
    checks_cfg = config.pilot.checks
    resolution = int(config.generation.resolution)

    total = len(recipe_ids)
    valid = 0
    wrong_resolution = 0
    not_rgb = 0
    near_uniform = 0
    missing = 0

    for recipe_id in recipe_ids:
        path = images_dir / f"{recipe_id}.jpg"
        if not path.is_file():
            missing += 1
            continue
        try:
            with Image.open(path) as image:
                if image.mode != "RGB":
                    not_rgb += 1
                if image.size != (resolution, resolution):
                    wrong_resolution += 1
                array = np.asarray(image.convert("L"), dtype=np.float32)
                if array.std() < float(checks_cfg.blank_std_threshold):
                    near_uniform += 1
                valid += 1
        except Exception:
            missing += 1

    valid_fraction = valid / total if total else 0.0
    near_uniform_fraction = near_uniform / total if total else 1.0

    results = {
        "total": total,
        "valid": valid,
        "missing_or_unreadable": missing,
        "wrong_resolution": wrong_resolution,
        "not_rgb": not_rgb,
        "near_uniform": near_uniform,
        "valid_fraction": round(valid_fraction, 4),
        "near_uniform_fraction": round(near_uniform_fraction, 4),
    }
    passed = (
        valid_fraction >= float(checks_cfg.min_valid_image_fraction)
        and near_uniform_fraction <= float(checks_cfg.max_near_uniform_fraction)
        and (wrong_resolution == 0 if bool(checks_cfg.require_correct_resolution) else True)
        and (not_rgb == 0 if bool(checks_cfg.require_rgb) else True)
    )
    results["passed"] = passed
    results["thresholds"] = OmegaConf.to_container(checks_cfg, resolve=True)
    return results


def approve_pilot(config, paths: dict[str, Path], provenance: dict, reviewer: str, notes: str) -> int:
    """Record explicit human approval. Refuses if the pilot did not run or its checks did not pass.

    This is deliberately a separate command a person runs after looking at the images — the
    generation script never approves its own output.
    """
    approval_path = paths["pilot_approval_manifest"]
    if not approval_path.is_file():
        raise SystemExit(
            f"No pilot manifest at {approval_path}. Run --mode pilot first, review the images, "
            "then approve."
        )
    manifest = read_json(approval_path)
    if manifest.get("split_provenance") != provenance:
        raise SystemExit("Pilot provenance no longer matches the current split/recipe/checkpoint/config; rerun the pilot")
    pilot_manifest = paths["pilot_manifest"]
    if not pilot_manifest.is_file() or manifest.get("pilot_manifest_sha256") != sha256_file(pilot_manifest):
        raise SystemExit("Pilot generation manifest is missing or changed; approval refused")
    if not manifest.get("pilot_completed", False):
        raise SystemExit("Pilot did not complete — cannot approve.")
    if not manifest.get("checks", {}).get("passed", False):
        raise SystemExit(
            "Pilot automatic checks did not pass — cannot approve.\n"
            f"{json.dumps(manifest.get('checks', {}), indent=2)}"
        )

    manifest["approval"] = {
        "approved": True,
        "reviewer": reviewer,
        "notes": notes,
        "approved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(approval_path, manifest)
    print(f"Pilot APPROVED by {reviewer} at {manifest['approval']['approved_at_utc']}", flush=True)
    print(f"Manifest: {approval_path}", flush=True)
    print("Full generation is now unblocked.", flush=True)
    return 0


def enforce_pilot_gate(config, paths: dict[str, Path], provenance: dict) -> dict:
    """FROZEN gate (§3): full generation refuses to start without a completed, checked, and
    explicitly approved pilot."""
    approval_path = paths["pilot_approval_manifest"]
    if not approval_path.is_file():
        raise SystemExit(
            "PILOT GATE: full generation refused.\n"
            f"  No pilot approval manifest at {approval_path}\n"
            "  Run:  python scripts/generate/02_generate_synthetic_images.py --mode pilot\n"
            "  Then review the pilot images and approve with --approve-pilot --reviewer <name>."
        )
    manifest = read_json(approval_path)
    if manifest.get("split_provenance") != provenance:
        raise SystemExit("PILOT GATE: approval provenance differs from current production inputs")
    if manifest.get("pilot_manifest_sha256") != sha256_file(paths["pilot_manifest"]):
        raise SystemExit("PILOT GATE: pilot manifest hash mismatch")
    if not manifest.get("pilot_completed", False):
        raise SystemExit("PILOT GATE: full generation refused — pilot did not complete.")
    if not manifest.get("checks", {}).get("passed", False):
        raise SystemExit(
            "PILOT GATE: full generation refused — pilot automatic checks did not pass.\n"
            f"{json.dumps(manifest.get('checks', {}), indent=2)}"
        )
    if not manifest.get("approval", {}).get("approved", False):
        raise SystemExit(
            "PILOT GATE: full generation refused — pilot completed and passed checks, but carries "
            "no explicit approval record.\n"
            "  Approve with: --approve-pilot --reviewer <name>"
        )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pilot", "full"], default=None)
    parser.add_argument("--approve-pilot", action="store_true", help="Record explicit pilot approval")
    parser.add_argument("--reviewer", default=None, help="Required with --approve-pilot")
    parser.add_argument("--notes", default="", help="Optional reviewer notes")
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    config = load_stage2_config()
    stage1_cfg = load_stage1_config()
    namespace = args.namespace or str(config.split_namespace)
    paths = stage2_paths(config, namespace)

    if args.approve_pilot:
        if not args.reviewer:
            raise SystemExit("--approve-pilot requires --reviewer <name>")
        provenance = validate_generation_inputs(config, namespace, paths)
        return approve_pilot(config, paths, provenance, args.reviewer, args.notes)

    if not args.mode:
        raise SystemExit("--mode {pilot,full} is required (or use --approve-pilot)")

    paths = stage2_paths(config, namespace)
    provenance = validate_generation_inputs(config, namespace, paths)
    recipes_path = paths["recipes_path"]
    recipes = pd.read_csv(recipes_path)

    if args.mode == "pilot":
        images_dir = paths["pilot_images"]
        # Deterministic spanning subset: evenly strided so the pilot covers the recipe space
        # rather than sampling one corner of it.
        pilot_n = min(int(config.pilot.num_images), len(recipes))
        stride = max(1, len(recipes) // pilot_n)
        selected = recipes.iloc[::stride].head(pilot_n).reset_index(drop=True)
        manifest_path = paths["pilot_manifest"]
    else:
        enforce_pilot_gate(config, paths, provenance)
        images_dir = paths["images_dir"]
        selected = recipes
        manifest_path = paths["manifest_path"]

    images_dir.mkdir(parents=True, exist_ok=True)

    registry_config = {
        "stage2": OmegaConf.to_container(config, resolve=True),
        "split_provenance": provenance,
        "mode": args.mode,
    }
    with ExperimentRun(
        stage=f"stage2_generation_{args.mode}",
        config=registry_config,
        dataset_version=provenance["split_manifest_hash"],
    ) as run:
        counts = run_generation(
            selected, config, stage1_cfg, images_dir, manifest_path, args.mode, provenance
        )
        run.set_checkpoint_path(images_dir)
        run.set_metrics(counts)

    if args.mode == "pilot":
        checks = run_pilot_checks(images_dir, selected["recipe_id"].tolist(), config)
        approval_path = paths["pilot_approval_manifest"]
        existing_approval = {}
        if approval_path.is_file():
            existing_approval = read_json(approval_path).get("approval", {})
        write_json(
            approval_path,
            {
                "schema_version": SCHEMA_VERSION,
                "pilot_completed": True,
                "pilot_generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_pilot_images": len(selected),
                "counts": counts,
                "checks": checks,
                "images_dir": str(images_dir),
                "split_provenance": provenance,
                "pilot_manifest_sha256": sha256_file(manifest_path),
                "git_commit_hash": get_git_commit_hash(),
                # Approval is NEVER auto-granted here; a prior approval is preserved but a new
                # pilot run invalidates it by resetting to unapproved unless it already existed
                # for this same manifest.
                "approval": existing_approval if existing_approval else {"approved": False},
            },
        )
        print(f"\nPilot checks: {'PASSED' if checks['passed'] else 'FAILED'}", flush=True)
        print(json.dumps(checks, indent=2), flush=True)
        print(f"\nManifest: {approval_path}", flush=True)
        if checks["passed"]:
            print(
                "\nNext: review the pilot images yourself, then run:\n"
                "  python scripts/generate/02_generate_synthetic_images.py "
                "--approve-pilot --reviewer <your name>",
                flush=True,
            )
        return 0 if checks["passed"] else 3

    print(f"\n[full] generated={counts['generated']} skipped_valid={counts['skipped_valid']} "
          f"failed={counts['failed']}", flush=True)
    print(f"Images:   {images_dir}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    write_json(paths["generation_provenance"], provenance)
    if counts["failed"] == 0:
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_ids = [str(row.get("image_id")) for row in manifest_rows]
        expected_ids = selected["recipe_id"].astype(str).tolist()
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise SystemExit("Generation completion refused: manifest IDs are duplicate, missing, or not the exact recipe partition")
        write_json(paths["generation_completion"], {**provenance, "status": "complete",
                   "generation_manifest_sha256": sha256_file(manifest_path),
                   "num_rows": len(manifest_rows), "image_id_set_sha256": hash_dict({"image_ids": sorted(actual_ids)}, length=64)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

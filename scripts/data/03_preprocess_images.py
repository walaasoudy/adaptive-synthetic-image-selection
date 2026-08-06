#!/usr/bin/env python3
"""Preprocess gen_train / gen_val images for SDXL LoRA training (docs/stage1_plan.md §6).

Applies, per image:
  - frontal-only filtering (lateral images are flagged and skipped, not deleted from raw/)
  - grayscale -> 3-channel RGB
  - aspect-preserving resize + centered letterbox padding to a square target resolution
    (never a non-uniform stretch: that would distort the cardiothoracic ratio; never a
    center-crop: that risks losing peripheral pathology)
  - quality filtering (corrupt/unreadable, below minimum source resolution, near-uniform/blank)

classifier_heldout.csv is intentionally NOT read or processed by this script — it must stay
completely untouched by Stage 1 (docs/stage1_plan.md §6).

Every filtering decision is logged to <split>_preprocessing_log.jsonl with image id + reason;
nothing is silently dropped.

Usage:
    python scripts/data/03_preprocess_images.py [--splits gen_train gen_val]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.identifiers import sanitize_image_id  # noqa: E402
from scripts.utils.manifest import write_json  # noqa: E402

BLANK_STD_THRESHOLD = 5.0  # pixel intensity std below this is treated as a likely blank/corrupt export


def resolve_source_path(raw_dir: Path, raw_path: str) -> Path:
    p = str(raw_path).replace("\\", "/")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if p.startswith(prefix):
            p = p[len(prefix):]
    return raw_dir / p


def letterbox_resize(image: Image.Image, target_size: int) -> Image.Image:
    """Aspect-preserving resize so the longer side fits target_size, then center-pad to a
    target_size x target_size square canvas (docs/stage1_plan.md §6: never stretch, never crop)."""
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def process_one_image(
    raw_path: str,
    raw_dir: Path,
    out_dir: Path,
    resolution: int,
    min_source_resolution: int,
    view_filter: str,
    frontal_lateral_value: str,
) -> tuple[str | None, str | None]:
    """Returns (image_id, filter_reason). filter_reason is None on success."""
    image_id = sanitize_image_id(raw_path)

    is_frontal = str(frontal_lateral_value).strip().lower().startswith("front")
    if view_filter == "frontal" and not is_frontal:
        return image_id, "filtered_lateral_view"

    source_path = resolve_source_path(raw_dir, raw_path)
    if not source_path.exists():
        return image_id, "source_file_missing"

    try:
        with Image.open(source_path) as im:
            im = im.convert("L")
            w, h = im.size
            if min(w, h) < min_source_resolution:
                return image_id, "below_min_source_resolution"

            arr = np.asarray(im, dtype=np.float32)
            if arr.std() < BLANK_STD_THRESHOLD:
                return image_id, "near_uniform_blank"

            im_rgb = im.convert("RGB")
            processed = letterbox_resize(im_rgb, resolution)
    except Exception:
        return image_id, "corrupt_or_unreadable"

    out_path = out_dir / f"{image_id}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(out_path, format="JPEG", quality=95)
    return image_id, None


def process_split(
    split_name: str,
    df: pd.DataFrame,
    raw_dir: Path,
    images_dir: Path,
    dataset_cfg,
    data_cfg,
) -> None:
    out_dir = images_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = dataset_cfg.schema

    log_records = []
    n_ok = 0
    for _, row in df.iterrows():
        raw_path = row[schema.path_column]
        fl_value = row.get(schema.frontal_lateral_column, "Frontal")
        image_id, reason = process_one_image(
            raw_path=raw_path,
            raw_dir=raw_dir,
            out_dir=out_dir,
            resolution=data_cfg.resolution,
            min_source_resolution=data_cfg.min_source_resolution,
            view_filter=data_cfg.view_filter,
            frontal_lateral_value=fl_value,
        )
        if reason is None:
            n_ok += 1
        log_records.append({"image_id": image_id, "raw_path": raw_path, "kept": reason is None, "reason": reason})

    log_path = images_dir / f"{split_name}_preprocessing_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in log_records:
            f.write(json.dumps(rec))
            f.write("\n")

    print(f"[{split_name}] kept {n_ok}/{len(df)} images -> {out_dir}")
    print(f"[{split_name}] filtering log -> {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["gen_train", "gen_val"])
    args = parser.parse_args()

    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    ensure_dirs(stage1_cfg)

    raw_dir = Path(stage1_cfg.paths.raw_dir)
    splits_dir = Path(stage1_cfg.paths.splits_dir)
    images_dir = Path(stage1_cfg.paths.images_dir)
    data_cfg = stage1_cfg.data

    for split_name in args.splits:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"Skipping {split_name}: {csv_path} not found (run 02_build_patient_splits.py first)")
            continue
        df = pd.read_csv(csv_path)
        process_split(split_name, df, raw_dir, images_dir, dataset_cfg, data_cfg)

    write_json(
        images_dir / "preprocessing_config_used.json",
        {
            "resolution": data_cfg.resolution,
            "aspect_mode": data_cfg.aspect_mode,
            "view_filter": data_cfg.view_filter,
            "min_source_resolution": data_cfg.min_source_resolution,
            "horizontal_flip": data_cfg.horizontal_flip,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

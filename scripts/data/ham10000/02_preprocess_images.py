#!/usr/bin/env python3
"""Preprocess HAM10000 images to a fixed-size RGB canvas, one split at a time, resumably.

WHAT THIS DOES DIFFERENTLY FROM THE CheXpert PATH, AND WHY
  RGB is preserved.        scripts/data/03_preprocess_images.py converts to "L" because a
                           radiograph is achromatic. Colour is the primary diagnostic cue in
                           dermoscopy, so the same conversion here would delete the signal being
                           classified. Images go in RGB and come out RGB.
  No view filter.          CheXpert drops lateral radiographs; HAM10000 has no view concept.
  Letterbox, never crop.   A centre-crop can cut a lesion that sits off-centre or runs to the frame
                           edge. Padding wastes border pixels instead of destroying content.
  Mid-grey padding.        Black padding looks like the dark circular vignette a real dermatoscope
                           produces, which would make "our padding" and "a real scope edge"
                           indistinguishable to both the generator and the explainability check.

Resumable per image: an already-written, still-valid output is validated and skipped, so an
interrupted run is continued by re-running the identical command.

Usage:
    python scripts/data/ham10000/02_preprocess_images.py --namespace production-thesis-v1
    python scripts/data/ham10000/02_preprocess_images.py --namespace production-thesis-v1 --splits gen_train gen_val
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.manifest import read_json, write_json  # noqa: E402

SPLIT_NAMES = [
    "gen_train",
    "gen_val",
    "classifier_train",
    "classifier_val",
    "asism_tuning_heldout",
    "final_eval_heldout",
]

# A dermoscopy frame with almost no variation is a failed capture, not a lesion. Thresholded on the
# RGB standard deviation rather than a luminance conversion, so a flat but strongly coloured frame
# is still caught.
BLANK_STD_THRESHOLD = 5.0


def letterbox_resize_rgb(image: Image.Image, target_size: int, pad_colour: tuple[int, int, int]) -> Image.Image:
    """Aspect-preserving resize onto a padded square canvas, in RGB throughout.

    Deliberately duplicated rather than imported from scripts/data/03_preprocess_images.py: that
    module is the CheXpert path, its filename starts with a digit (so importing it needs importlib
    gymnastics), and coupling the two datasets' preprocessing through a shared helper is exactly how
    a CheXpert-specific change would later leak into this one.
    """
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), tuple(pad_colour))
    canvas.paste(resized, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
    return canvas


def validate_processed_jpeg(path: Path, resolution: int) -> tuple[bool, str | None]:
    """Fully decode an output JPEG and require RGB mode at the configured size.

    The mode check is load-bearing here: it is what would catch a greyscale file produced by an
    older or mis-wired run, which is the specific failure this whole script exists to prevent.
    """
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


def atomic_save_jpeg(image: Image.Image, destination: Path, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=quality)
        valid, reason = validate_processed_jpeg(temporary, image.width)
        if not valid:
            raise OSError(f"temporary JPEG validation failed: {reason}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_source(raw_dir: Path, image_id: str, image_directories: list[str]) -> Path | None:
    for directory in image_directories:
        candidate = raw_dir / directory / f"{image_id}.jpg"
        if candidate.is_file():
            return candidate
    return None


def expected_manifest(stage1_cfg, dataset_cfg, namespace: str, split_name: str, split_sha: str) -> dict:
    data = stage1_cfg.data
    return {
        "manifest_version": 2,
        "dataset": "ham10000",
        "split_namespace": namespace,
        "split_name": split_name,
        "split_csv_sha256": split_sha,
        "output_settings": {
            "colour_mode": str(data.colour_mode),
            "resolution": int(data.resolution),
            "aspect_mode": str(data.aspect_mode),
            "pad_colour": list(data.pad_colour),
            "jpeg_quality": int(data.jpeg_quality),
            "min_source_resolution": int(data.min_source_resolution),
        },
        "source_image_directories": [str(name) for name in dataset_cfg.source.image_directories],
    }


def process_split(
    split_name: str,
    frame: pd.DataFrame,
    raw_dir: Path,
    out_root: Path,
    image_directories: list[str],
    data_cfg,
    image_id_column: str,
) -> Counter:
    out_dir = out_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(data_cfg.resolution)
    pad_colour = tuple(int(value) for value in data_cfg.pad_colour)
    quality = int(data_cfg.jpeg_quality)
    min_source = int(data_cfg.min_source_resolution)

    log_path = out_root / f"{split_name}_preprocessing_log.jsonl"
    counts = Counter(processed=0, skipped_existing=0, filtered=0, failed=0)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{log_path.name}.", suffix=".tmp", dir=out_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", buffering=1) as log:
            progress = tqdm(
                frame[image_id_column].astype(str),
                total=len(frame),
                desc=f"[{split_name}] preprocess",
                unit="image",
            )
            for image_id in progress:
                destination = out_dir / f"{image_id}.jpg"
                reason: str | None = None
                detail: str | None = None

                valid, validation_reason = validate_processed_jpeg(destination, resolution)
                if valid:
                    action, reason = "skipped_existing", "validated_existing"
                else:
                    source = resolve_source(raw_dir, image_id, image_directories)
                    if source is None:
                        action, reason = "failed", "source_file_missing"
                    else:
                        try:
                            with Image.open(source) as source_image:
                                rgb = source_image.convert("RGB")  # RGB in, RGB out — never "L"
                                width, height = rgb.size
                                if min(width, height) < min_source:
                                    action, reason = "filtered", "below_min_source_resolution"
                                elif float(np.asarray(rgb, dtype=np.float32).std()) < BLANK_STD_THRESHOLD:
                                    action, reason = "filtered", "near_uniform_blank"
                                else:
                                    atomic_save_jpeg(
                                        letterbox_resize_rgb(rgb, resolution, pad_colour),
                                        destination,
                                        quality,
                                    )
                                    action = "processed"
                                    detail = validation_reason and f"replaced_{validation_reason}"
                        except Exception as exc:
                            action, reason, detail = "failed", "corrupt_or_unreadable", f"{type(exc).__name__}: {exc}"

                counts[action] += 1
                log.write(
                    json.dumps(
                        {
                            "image_id": image_id,
                            "kept": action in ("processed", "skipped_existing"),
                            "action": action,
                            "reason": reason,
                            "detail": detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                progress.set_postfix({key: counts[key] for key in ("processed", "skipped_existing", "filtered", "failed")}, refresh=False)
            progress.close()
            log.flush()
            os.fsync(log.fileno())
        os.replace(temporary_name, log_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    print(
        f"[{split_name}] processed={counts['processed']} skipped-existing={counts['skipped_existing']} "
        f"filtered={counts['filtered']} failed={counts['failed']} log={log_path}",
        flush=True,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", required=True, help="Split run id, e.g. production-thesis-v1")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=SPLIT_NAMES)
    args = parser.parse_args()

    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    stage1_cfg = load_named_config("ham10000_stage1.yaml", "ham_stage1")

    if str(stage1_cfg.data.colour_mode).upper() != "RGB":
        raise SystemExit(
            f"colour_mode is {stage1_cfg.data.colour_mode!r}; HAM10000 preprocessing is RGB-only. "
            "Colour is the diagnostic signal in dermoscopy and must not be discarded."
        )

    raw_dir = Path(splits_cfg.paths.raw_dir)
    split_dir = Path(splits_cfg.paths.splits_root) / args.namespace
    if not split_dir.is_dir():
        raise SystemExit(
            f"Missing split run {split_dir}.\n"
            "Run: python scripts/data/ham10000/01_build_splits.py --run-id <id> --freeze"
        )

    out_root = Path(stage1_cfg.paths.images_dir) / args.namespace
    out_root.mkdir(parents=True, exist_ok=True)
    image_directories = [str(name) for name in dataset_cfg.source.image_directories]
    image_id_column = str(dataset_cfg.schema.image_id_column)

    import hashlib

    for split_name in args.splits:
        split_path = split_dir / f"{split_name}.csv"
        if not split_path.is_file():
            raise SystemExit(f"Missing split file: {split_path}")

        split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
        manifest = expected_manifest(stage1_cfg, dataset_cfg, args.namespace, split_name, split_sha)
        manifest_path = out_root / f"{split_name}_preprocessing_manifest.json"

        if manifest_path.is_file():
            existing = read_json(manifest_path)
            if existing != manifest:
                raise SystemExit(
                    "Existing preprocessing outputs were produced from different settings or a "
                    f"different split file.\n  manifest: {manifest_path}\n"
                    f"  existing: {json.dumps(existing, indent=2, sort_keys=True)}\n"
                    f"  requested: {json.dumps(manifest, indent=2, sort_keys=True)}\n"
                    "Choose a new namespace rather than mixing two settings in one directory."
                )
        write_json(manifest_path, manifest)

        process_split(
            split_name,
            pd.read_csv(split_path),
            raw_dir,
            out_root,
            image_directories,
            stage1_cfg.data,
            image_id_column,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.identifiers import sanitize_image_id  # noqa: E402
from scripts.utils.manifest import hash_dict, read_json, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import SPLIT_NAMES, read_split_manifest, resolve_splits_dir  # noqa: E402

BLANK_STD_THRESHOLD = 5.0
JPEG_QUALITY = 95
MANIFEST_VERSION = 2


def resolve_source_path(raw_dir: Path, raw_path: str) -> Path:
    value = str(raw_path).replace("\\", "/")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return raw_dir / value


def validate_processed_jpeg(path: Path, resolution: int) -> tuple[bool, str | None]:
    """Fully decode a JPEG and require its encoded mode and dimensions to match the config."""
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


def letterbox_resize(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(new_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(resized, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
    return canvas


def atomic_save_jpeg(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="JPEG", quality=JPEG_QUALITY)
        valid, reason = validate_processed_jpeg(temporary, image.width)
        if not valid:
            raise OSError(f"temporary JPEG validation failed: {reason}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def expected_manifest(stage1_cfg, split_paths: dict[str, Path], namespace: str | None = None, split_manifest: dict | None = None) -> dict:
    data = stage1_cfg.data
    namespace = namespace or str(stage1_cfg.split.namespace)
    split_manifest = split_manifest or read_split_manifest(namespace)
    settings = {
        "resolution": int(data.resolution), "aspect_mode": str(data.aspect_mode),
        "view_filter": str(data.view_filter), "min_source_resolution": int(data.min_source_resolution),
        "horizontal_flip": bool(data.horizontal_flip), "blank_std_threshold": BLANK_STD_THRESHOLD,
        "jpeg_quality": JPEG_QUALITY,
    }
    return {
        "manifest_version": MANIFEST_VERSION,
        "preprocessing_algorithm_version": 2,
        "output_schema_version": 2,
        "split_namespace": namespace,
        "split_manifest_hash": split_manifest.get("manifest_hash"),
        "preprocessing_config_hash": hash_dict(settings, length=64),
        "output_settings": settings,
        "source_splits": {
            name: {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in sorted(split_paths.items())
        },
    }


def check_manifest(path: Path, expected: dict, adopt_legacy: bool) -> None:
    if not path.exists():
        return
    actual = read_json(path)
    if actual == expected:
        return
    if "manifest_version" not in actual and adopt_legacy:
        legacy_settings = {k: actual.get(k) for k in expected["output_settings"] if k in actual}
        current_settings = {k: expected["output_settings"][k] for k in legacy_settings}
        if legacy_settings != current_settings:
            raise SystemExit(f"Legacy preprocessing settings conflict with the current configuration:\nold={legacy_settings}\nnew={current_settings}")
        print("WARNING: adopting a legacy preprocessing manifest with no split hashes. Existing files will still be decoded and validated before skipping.", flush=True)
        return
    raise SystemExit(
        "Existing preprocessing outputs were created with incompatible or unverifiable settings/splits.\n"
        f"Manifest: {path}\nExisting: {json.dumps(actual, indent=2, sort_keys=True)}\n"
        f"Requested: {json.dumps(expected, indent=2, sort_keys=True)}\n"
        "Use the original configuration/split files. For a legacy manifest only, review it and rerun with --adopt-legacy-manifest."
    )


def process_split(split_name, df, raw_dir, images_dir, dataset_cfg, data_cfg) -> Counter:
    out_dir = images_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = dataset_cfg.schema
    image_ids = [sanitize_image_id(value) for value in df[schema.path_column]]
    duplicates = [item for item, count in Counter(image_ids).items() if count > 1]
    if duplicates:
        raise SystemExit(f"{split_name}: {len(duplicates)} duplicate output image IDs; first examples: {duplicates[:5]}")

    log_path = images_dir / f"{split_name}_preprocessing_log.jsonl"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{log_path.name}.", suffix=".tmp", dir=images_dir)
    counts = Counter(processed=0, skipped_existing=0, filtered=0, failed=0)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", buffering=1) as log:
            progress = tqdm(df.itertuples(index=False, name=None), total=len(df), desc=f"[{split_name}] preprocess", unit="image")
            columns = {name: index for index, name in enumerate(df.columns)}
            for values in progress:
                raw_path = values[columns[schema.path_column]]
                image_id = sanitize_image_id(raw_path)
                destination = out_dir / f"{image_id}.jpg"
                frontal_value = values[columns[schema.frontal_lateral_column]] if schema.frontal_lateral_column in columns else "Frontal"
                reason = None
                detail = None
                action = "processed"

                if str(data_cfg.view_filter) == "frontal" and not str(frontal_value).strip().lower().startswith("front"):
                    action, reason = "filtered", "filtered_lateral_view"
                else:
                    valid, validation_reason = validate_processed_jpeg(destination, int(data_cfg.resolution))
                    if valid:
                        action, reason = "skipped_existing", "validated_existing"
                    else:
                        source = resolve_source_path(raw_dir, raw_path)
                        if not source.is_file():
                            action, reason = "failed", "source_file_missing"
                        else:
                            try:
                                with Image.open(source) as source_image:
                                    gray = source_image.convert("L")
                                    width, height = gray.size
                                    if min(width, height) < int(data_cfg.min_source_resolution):
                                        action, reason = "filtered", "below_min_source_resolution"
                                    elif np.asarray(gray, dtype=np.float32).std() < BLANK_STD_THRESHOLD:
                                        action, reason = "filtered", "near_uniform_blank"
                                    else:
                                        atomic_save_jpeg(letterbox_resize(gray.convert("RGB"), int(data_cfg.resolution)), destination)
                                        detail = validation_reason and f"replaced_{validation_reason}"
                            except Exception as exc:
                                action, reason, detail = "failed", "corrupt_or_unreadable", f"{type(exc).__name__}: {exc}"

                counts[action] += 1
                record = {"image_id": image_id, "raw_path": str(raw_path), "kept": action in ("processed", "skipped_existing"), "action": action, "reason": reason, "detail": detail}
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                progress.set_postfix({key: counts[key] for key in ("processed", "skipped_existing", "filtered", "failed")}, refresh=False)
            progress.close()
            log.flush()
            os.fsync(log.fileno())
        os.replace(temporary_name, log_path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    print(f"[{split_name}] processed={counts['processed']} skipped-existing={counts['skipped_existing']} filtered={counts['filtered']} failed={counts['failed']} log={log_path}", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None, help="Explicit v2 split namespace/run (dev or production run ID)")
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=SPLIT_NAMES)
    parser.add_argument("--adopt-legacy-manifest", action="store_true", help="Explicitly accept an old manifest lacking split hashes after checking its recorded settings")
    args = parser.parse_args()
    cfg, dataset_cfg = load_stage1_config(), load_dataset_config()
    ensure_dirs(cfg)
    namespace = args.namespace or str(cfg.split.namespace)
    split_manifest = read_split_manifest(namespace)
    split_dir = resolve_splits_dir(namespace)
    split_paths = {name: split_dir / f"{name}.csv" for name in args.splits}
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing split files (run 02_build_patient_splits.py first): " + ", ".join(missing))
    images_dir = Path(cfg.paths.images_dir) / namespace
    images_dir.mkdir(parents=True, exist_ok=True)
    for name, path in split_paths.items():
        manifest = expected_manifest(cfg, {name: path}, namespace, split_manifest)
        manifest_path = images_dir / f"{name}_preprocessing_manifest.json"
        split_output_dir = images_dir / name
        if not manifest_path.is_file() and split_output_dir.is_dir() and any(split_output_dir.iterdir()):
            raise SystemExit(
                f"Unpinned preprocessing outputs exist at {split_output_dir} without {manifest_path}; "
                "refusing stale cross-assignment reuse. Preserve them and choose a new versioned namespace."
            )
        check_manifest(manifest_path, manifest, args.adopt_legacy_manifest)
        write_json(manifest_path, manifest)
        process_split(name, pd.read_csv(path), Path(cfg.paths.raw_dir), images_dir, dataset_cfg, cfg.data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

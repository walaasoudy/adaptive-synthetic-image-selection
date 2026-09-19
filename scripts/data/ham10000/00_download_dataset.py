#!/usr/bin/env python3
"""Download HAM10000 from Kaggle, link it into data/ham10000/raw/, and verify it is usable.

Download and verification are ONE script here, unlike the CheXpert path's two. The CheXpert split
existed so a re-verify could be run without touching the download; with a 10k-image dataset the
verify is fast enough that a single idempotent command is less to remember and less to get wrong.

Idempotent: an already-present, already-passing raw/ directory is left alone and the script exits 0.

Usage:
    python scripts/data/ham10000/00_download_dataset.py
    python scripts/data/ham10000/00_download_dataset.py --force       # re-link even if present
    python scripts/data/ham10000/00_download_dataset.py --sample-images 50
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import DIAGNOSIS_CLASSES, validate_metadata  # noqa: E402


def link_or_copy(source: Path, target: Path, force_copy: bool) -> str:
    """Idempotent per-entry: never overwrites an existing target. Prefers a symlink so the dataset
    is not duplicated on the volume; falls back to copying where symlinks are not permitted."""
    if target.exists() or target.is_symlink():
        return "skipped (already present)"
    if not force_copy:
        try:
            target.symlink_to(source, target_is_directory=source.is_dir())
            return "symlinked"
        except (OSError, NotImplementedError) as exc:
            print(f"  symlink failed for {target.name} ({exc}); copying instead.", flush=True)
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return "copied"


def find_data_root(base: Path, metadata_filename: str) -> Path:
    """Locate the directory that directly contains the metadata CSV.

    kagglehub's extraction layout is not guaranteed stable across dataset versions, so the exact
    nesting is discovered rather than assumed.
    """
    direct = base / metadata_filename
    if direct.is_file():
        return base
    for candidate in sorted(base.rglob(metadata_filename)):
        return candidate.parent
    raise FileNotFoundError(f"Could not locate {metadata_filename} anywhere under {base}")


def resolve_image_path(raw_dir: Path, image_id: str, image_directories: list[str]) -> Path | None:
    """HAM10000 ships its images split across two directories; an image may be in either."""
    for directory in image_directories:
        candidate = raw_dir / directory / f"{image_id}.jpg"
        if candidate.is_file():
            return candidate
    return None


def verify(raw_dir: Path, dataset_cfg, sample_images: int) -> list[str]:
    """Structural integrity checks. Returns a list of failures (empty means the dataset is usable)."""
    failures: list[str] = []
    schema = dataset_cfg.schema
    metadata_path = raw_dir / str(dataset_cfg.source.metadata_filename)

    if not metadata_path.is_file():
        return [f"metadata file not found: {metadata_path}"]

    frame = pd.read_csv(metadata_path)
    expected_rows = int(dataset_cfg.source.expected_metadata_rows)
    deviation_pct = abs(len(frame) - expected_rows) / expected_rows * 100.0
    if deviation_pct > 1.0:
        failures.append(
            f"metadata row count {len(frame)} deviates {deviation_pct:.2f}% from the expected "
            f"{expected_rows} (tolerance 1%)"
        )

    try:
        validate_metadata(frame, str(schema.diagnosis_column), str(schema.group_column))
    except ValueError as exc:
        failures.append(str(exc))

    image_id_column = str(schema.image_id_column)
    if image_id_column not in frame.columns:
        failures.append(f"missing {image_id_column!r} column")
        return failures

    duplicates = frame[image_id_column].duplicated().sum()
    if duplicates:
        failures.append(f"{duplicates} duplicate {image_id_column} value(s) in the metadata")

    # Class distribution: reported unconditionally. The imbalance IS the thesis's motivation, so it
    # belongs in the run's output, not only in a paper the reader has to go and find.
    counts = frame[str(schema.diagnosis_column)].value_counts()
    print("\nClass distribution (images per diagnosis):", flush=True)
    for label in DIAGNOSIS_CLASSES:
        count = int(counts.get(label, 0))
        share = count / max(len(frame), 1) * 100.0
        print(f"  {label:6s} {count:6d}  ({share:5.2f}%)", flush=True)
    lesions = frame[str(schema.group_column)].nunique()
    print(
        f"\n{len(frame)} images across {lesions} lesions "
        f"({len(frame) / max(lesions, 1):.2f} images per lesion on average)",
        flush=True,
    )

    image_directories = [str(name) for name in dataset_cfg.source.image_directories]
    missing_directories = [name for name in image_directories if not (raw_dir / name).is_dir()]
    if missing_directories:
        failures.append(f"missing image directories: {missing_directories}")
        return failures

    # Spot-check a random sample rather than opening all 10k: enough to catch a truncated or
    # half-extracted download, fast enough to run on every invocation.
    from PIL import Image

    sample = frame.sample(n=min(sample_images, len(frame)), random_state=0)
    for image_id in sample[image_id_column]:
        path = resolve_image_path(raw_dir, str(image_id), image_directories)
        if path is None:
            failures.append(f"Spot-check: image not found in any image directory: {image_id}")
            continue
        try:
            with Image.open(path) as image:
                width, height = image.size
            if min(width, height) < 64:
                failures.append(f"Spot-check: implausible dimensions {width}x{height} for {image_id}")
        except Exception as exc:
            failures.append(f"Spot-check: cannot open {path}: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download and re-link even if raw/ looks valid")
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlinking into data/ham10000/raw/")
    parser.add_argument("--sample-images", type=int, default=100)
    args = parser.parse_args()

    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    raw_dir = Path(splits_cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata_filename = str(dataset_cfg.source.metadata_filename)

    if not args.force and (raw_dir / metadata_filename).is_file():
        print(f"Found existing {metadata_filename} in {raw_dir}; verifying before re-downloading...", flush=True)
        failures = verify(raw_dir, dataset_cfg, args.sample_images)
        if not failures:
            print("\nOK: already downloaded and verified — nothing to do. Use --force to redo it.", flush=True)
            return 0
        print(f"\nExisting data did not pass verification ({len(failures)} issue(s)); re-downloading.\n", flush=True)

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install kagglehub", flush=True)
        return 1

    slug = str(dataset_cfg.source.kaggle_dataset_slug)
    print(f"Downloading Kaggle dataset '{slug}' via kagglehub...", flush=True)
    download_path = Path(kagglehub.dataset_download(slug))
    print(f"kagglehub cached the dataset at: {download_path}", flush=True)

    data_root = find_data_root(download_path, metadata_filename)
    print(f"Located dataset root: {data_root}", flush=True)

    # Link only what the pipeline reads: the metadata plus the two image directories. The Kaggle
    # mirror also ships flattened 28x28 pixel CSVs (hmnist_*.csv) that this project never uses;
    # linking them would just add confusion about which file is authoritative.
    wanted = [metadata_filename, *[str(name) for name in dataset_cfg.source.image_directories]]
    for name in wanted:
        source = data_root / name
        if not source.exists():
            print(f"  MISSING in download: {name}", flush=True)
            continue
        action = link_or_copy(source, raw_dir / name, force_copy=args.copy)
        print(f"  {action}: {raw_dir / name}", flush=True)

    print("\nVerifying...", flush=True)
    failures = verify(raw_dir, dataset_cfg, args.sample_images)
    if failures:
        print(f"\nFAILED: {len(failures)} check(s) did not pass:", flush=True)
        for failure in failures[:25]:
            print(f"  - {failure}", flush=True)
        if len(failures) > 25:
            print(f"  ... and {len(failures) - 25} more", flush=True)
        return 1

    print("\nOK: all structural integrity checks passed.", flush=True)
    print(f"OK: HAM10000 downloaded, linked into {raw_dir}, and verified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

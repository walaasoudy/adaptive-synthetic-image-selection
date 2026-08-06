#!/usr/bin/env python3
"""Verify the extracted CheXpert-v1.0-small download before anything else runs.

Since the exact Kaggle mirror isn't pinned with a checksum (docs/stage1_plan.md §6), this checks
*structural* integrity instead: expected files, row counts, column schema, label-value domain,
a random image-openability spot-check, and a coarse label-prevalence sanity print for manual
comparison against published CheXpert statistics.

Usage:
    python scripts/data/01_verify_download.py [--sample-images N]

Exits non-zero (and prints every failure, not just the first) if a hard check fails.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_dataset_config, load_stage1_config  # noqa: E402

VALID_LABEL_VALUES = {-1.0, 0.0, 1.0}
ROW_COUNT_TOLERANCE = 0.01  # 1%


def check_files_exist(raw_dir: Path) -> tuple[bool, list[str]]:
    errors = []
    train_csv = raw_dir / "train.csv"
    valid_csv = raw_dir / "valid.csv"
    if not train_csv.exists():
        errors.append(f"Missing {train_csv}")
    if not valid_csv.exists():
        errors.append(f"Missing {valid_csv}")
    return (len(errors) == 0, errors)


def check_row_counts(df_train: pd.DataFrame, df_valid: pd.DataFrame, dataset_cfg) -> tuple[bool, list[str]]:
    errors = []
    expected_train = dataset_cfg.source.expected_train_rows
    expected_valid = dataset_cfg.source.expected_valid_rows

    train_diff = abs(len(df_train) - expected_train) / expected_train
    if train_diff > ROW_COUNT_TOLERANCE:
        errors.append(
            f"train.csv has {len(df_train)} rows, expected ~{expected_train} "
            f"(off by {train_diff:.1%}, tolerance {ROW_COUNT_TOLERANCE:.0%})"
        )

    if len(df_valid) != expected_valid:
        errors.append(f"valid.csv has {len(df_valid)} rows, expected exactly {expected_valid}")

    return (len(errors) == 0, errors)


def check_schema(df: pd.DataFrame, dataset_cfg, csv_name: str) -> tuple[bool, list[str]]:
    errors = []
    schema = dataset_cfg.schema
    required_columns = [
        schema.path_column,
        schema.sex_column,
        schema.age_column,
        schema.frontal_lateral_column,
    ] + list(schema.pathology_columns)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"{csv_name}: missing expected columns: {missing}")

    for col in schema.pathology_columns:
        if col not in df.columns:
            continue
        observed = set(df[col].dropna().unique().tolist())
        invalid = {v for v in observed if v not in VALID_LABEL_VALUES}
        if invalid:
            errors.append(f"{csv_name}: column {col!r} has out-of-domain values: {invalid}")

    return (len(errors) == 0, errors)


def check_image_spotcheck(df: pd.DataFrame, raw_dir: Path, dataset_cfg, n: int, seed: int) -> tuple[bool, list[str]]:
    errors = []
    rng = random.Random(seed)
    path_col = dataset_cfg.schema.path_column
    n = min(n, len(df))
    sample_rows = df.sample(n=n, random_state=seed)

    for _, row in sample_rows.iterrows():
        raw_path = str(row[path_col]).replace("\\", "/")
        # Paths in train.csv are typically like "CheXpert-v1.0-small/train/patient.../view1_frontal.jpg"
        for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
            if raw_path.startswith(prefix):
                raw_path = raw_path[len(prefix):]
        image_path = raw_dir / raw_path
        if not image_path.exists():
            errors.append(f"Spot-check: image not found: {image_path}")
            continue
        try:
            with Image.open(image_path) as im:
                im.verify()
            with Image.open(image_path) as im:
                w, h = im.size
                if w < 50 or h < 50:
                    errors.append(f"Spot-check: implausibly small image {image_path} ({w}x{h})")
        except Exception as e:
            errors.append(f"Spot-check: failed to open {image_path}: {e}")

    return (len(errors) == 0, errors)


def print_label_prevalence(df: pd.DataFrame, dataset_cfg) -> None:
    print("\nLabel prevalence in train.csv (compare against published CheXpert statistics manually --")
    print("this script does not hard-fail on prevalence, since exact published rates aren't asserted here):")
    for col in dataset_cfg.schema.pathology_columns:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        pos = counts.get(1.0, 0)
        neg = counts.get(0.0, 0)
        unc = counts.get(-1.0, 0)
        blank = total - pos - neg - unc
        print(
            f"  {col:<28} positive={pos / total:6.1%}  negative={neg / total:6.1%}  "
            f"uncertain={unc / total:6.1%}  blank={blank / total:6.1%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-images", type=int, default=200, help="Number of images to spot-check")
    args = parser.parse_args()

    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    raw_dir = Path(stage1_cfg.paths.raw_dir)
    seed = stage1_cfg.run.seed

    all_errors: list[str] = []

    ok, errors = check_files_exist(raw_dir)
    all_errors += errors
    if not ok:
        print("FAILED: required files missing, cannot continue further checks.")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    df_train = pd.read_csv(raw_dir / "train.csv")
    df_valid = pd.read_csv(raw_dir / "valid.csv")

    for check_fn, args_ in (
        (check_row_counts, (df_train, df_valid, dataset_cfg)),
        (check_schema, (df_train, dataset_cfg, "train.csv")),
        (check_schema, (df_valid, dataset_cfg, "valid.csv")),
        (check_image_spotcheck, (df_train, raw_dir, dataset_cfg, args.sample_images, seed)),
    ):
        _, errors = check_fn(*args_)
        all_errors += errors

    print_label_prevalence(df_train, dataset_cfg)

    print()
    if all_errors:
        print(f"FAILED: {len(all_errors)} check(s) did not pass:")
        for e in all_errors:
            print(f"  - {e}")
        return 1

    print("OK: all structural integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

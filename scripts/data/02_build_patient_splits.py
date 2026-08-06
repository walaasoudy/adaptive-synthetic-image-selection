#!/usr/bin/env python3
"""Patient-level train/val/heldout split for CheXpert (docs/stage1_plan.md §6 — the single most
consequence-laden decision in Stage 1).

Splits by patient ID, never by image ID, into three disjoint groups:
  - gen_train:          used for LoRA gradient updates.
  - gen_val:             held out from training, used for validation loss / qualitative monitoring.
  - classifier_heldout:  NEVER touched by Stage 1 at all — reserved so Stage 5's eventual
                          real-vs-synthetic comparison isn't contaminated by generator exposure.

The official CheXpert valid.csv (234 curated images) is left untouched in raw/ and is not
consumed by this script at all, for the same reason.

Usage:
    python scripts/data/02_build_patient_splits.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.manifest import hash_dict, write_json  # noqa: E402


def extract_patient_id(path_value: str, patient_id_regex: str) -> str:
    match = re.search(patient_id_regex, str(path_value))
    if not match:
        raise ValueError(f"Could not extract patient id from path: {path_value!r}")
    return match.group(0)


def split_patients(
    patient_ids: list[str],
    gen_train_fraction: float,
    gen_val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    import random

    rng = random.Random(seed)
    shuffled = list(patient_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * gen_train_fraction))
    n_val = int(round(n * gen_val_fraction))

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    heldout_ids = set(shuffled[n_train + n_val :])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(heldout_ids)
    assert val_ids.isdisjoint(heldout_ids)
    assert len(train_ids) + len(val_ids) + len(heldout_ids) == n

    return train_ids, val_ids, heldout_ids


def main() -> int:
    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    ensure_dirs(stage1_cfg)

    raw_dir = Path(stage1_cfg.paths.raw_dir)
    splits_dir = Path(stage1_cfg.paths.splits_dir)
    schema = dataset_cfg.schema
    split_cfg = stage1_cfg.split

    df = pd.read_csv(raw_dir / "train.csv")
    df["patient_id"] = df[schema.path_column].apply(
        lambda p: extract_patient_id(p, schema.patient_id_regex)
    )

    unique_patients = sorted(df["patient_id"].unique().tolist())
    train_ids, val_ids, heldout_ids = split_patients(
        unique_patients,
        gen_train_fraction=split_cfg.gen_train_fraction,
        gen_val_fraction=split_cfg.gen_val_fraction,
        seed=split_cfg.split_seed,
    )

    df_train = df[df["patient_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["patient_id"].isin(val_ids)].reset_index(drop=True)
    df_heldout = df[df["patient_id"].isin(heldout_ids)].reset_index(drop=True)

    # Belt-and-braces: verify no patient leaked across the image-level dataframes too.
    assert set(df_train["patient_id"]).isdisjoint(set(df_val["patient_id"]))
    assert set(df_train["patient_id"]).isdisjoint(set(df_heldout["patient_id"]))
    assert set(df_val["patient_id"]).isdisjoint(set(df_heldout["patient_id"]))

    df_train.to_csv(splits_dir / "gen_train.csv", index=False)
    df_val.to_csv(splits_dir / "gen_val.csv", index=False)
    df_heldout.to_csv(splits_dir / "classifier_heldout.csv", index=False)

    manifest = {
        "split_unit": split_cfg.split_unit,
        "split_seed": split_cfg.split_seed,
        "gen_train_fraction": split_cfg.gen_train_fraction,
        "gen_val_fraction": split_cfg.gen_val_fraction,
        "classifier_heldout_fraction": split_cfg.classifier_heldout_fraction,
        "num_patients_total": len(unique_patients),
        "num_patients_gen_train": len(train_ids),
        "num_patients_gen_val": len(val_ids),
        "num_patients_classifier_heldout": len(heldout_ids),
        "num_images_gen_train": len(df_train),
        "num_images_gen_val": len(df_val),
        "num_images_classifier_heldout": len(df_heldout),
        "official_valid_csv": "reserved untouched at data/chexpert/raw/valid.csv — not read by this script",
        "source_train_csv_hash": hash_dict({"n_rows": len(df), "n_patients": len(unique_patients)}),
    }
    write_json(splits_dir / "split_manifest.json", manifest)

    print(f"Total patients: {len(unique_patients)}")
    print(f"  gen_train:          {len(train_ids):>6} patients, {len(df_train):>7} images")
    print(f"  gen_val:            {len(val_ids):>6} patients, {len(df_val):>7} images")
    print(f"  classifier_heldout: {len(heldout_ids):>6} patients, {len(df_heldout):>7} images")
    print(f"Wrote split CSVs and split_manifest.json to {splits_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

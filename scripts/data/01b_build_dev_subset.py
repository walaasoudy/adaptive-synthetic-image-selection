from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.manifest import (  # noqa: E402
    get_git_commit_hash,
    hash_dict,
    sha256_file,
    write_json,
)

ALGORITHM_VERSION = 1


def compute_prevalence(df: pd.DataFrame, pathology_columns: list[str]) -> pd.Series:
    return (df[pathology_columns] == 1).mean()


def select_dev_subset_patients(
    df: pd.DataFrame,
    patient_column: str,
    pathology_columns: list[str],
    target_size: int,
    seed: int,
) -> set[str]:
    """Rarity-first greedy quota sampling. Returns the set of selected patient IDs.

    Sorts pathology columns ascending by full-dataset image-level prevalence (rarest first),
    and for each, greedily draws whole patients carrying that finding until its proportional
    quota of `target_size` is met (or the eligible pool is exhausted). Because CheXpert
    pathologies co-occur, later quotas are frequently partially satisfied "for free" by earlier
    draws. Any shortfall against `target_size` after processing all columns is filled by
    uniform random patient draws. All randomness comes from one seeded `random.Random` stream.
    """
    prevalence = compute_prevalence(df, pathology_columns)
    rarity_order = prevalence.sort_values(ascending=True).index.tolist()
    quotas = (prevalence * target_size).round().astype(int)

    positive_counts_per_patient = (df[pathology_columns] == 1).groupby(df[patient_column]).sum()
    patient_image_counts = df.groupby(patient_column).size()
    positive_patients_by_col = {
        col: sorted(positive_counts_per_patient.index[positive_counts_per_patient[col] > 0].tolist())
        for col in pathology_columns
    }
    all_patients = sorted(patient_image_counts.index.tolist())

    rng = random.Random(seed)
    selected: set[str] = set()
    running_count = pd.Series(0, index=pathology_columns, dtype="int64")

    for col in rarity_order:
        quota = int(quotas[col])
        if quota <= 0:
            continue
        candidates = [p for p in positive_patients_by_col[col] if p not in selected]
        rng.shuffle(candidates)
        for patient in candidates:
            if running_count[col] >= quota:
                break
            selected.add(patient)
            running_count = running_count.add(positive_counts_per_patient.loc[patient], fill_value=0)

    total_images = int(patient_image_counts[list(selected)].sum()) if selected else 0
    if total_images < target_size:
        remaining = [p for p in all_patients if p not in selected]
        rng.shuffle(remaining)
        for patient in remaining:
            if total_images >= target_size:
                break
            selected.add(patient)
            total_images += int(patient_image_counts[patient])

    return selected


def build_prevalence_report(
    df_full: pd.DataFrame,
    df_subset: pd.DataFrame,
    pathology_columns: list[str],
    tolerance_pct: float,
) -> dict:
    full_pos = compute_prevalence(df_full, pathology_columns)
    subset_pos = compute_prevalence(df_subset, pathology_columns)
    full_unc = (df_full[pathology_columns] == -1).mean()
    subset_unc = (df_subset[pathology_columns] == -1).mean()

    per_column = {}
    num_exceeding = 0
    for col in pathology_columns:
        deviation_pct = abs(float(subset_pos[col]) - float(full_pos[col])) * 100.0
        within = deviation_pct <= tolerance_pct
        if not within:
            num_exceeding += 1
        per_column[col] = {
            "full_dataset_positive_rate": float(full_pos[col]),
            "subset_positive_rate": float(subset_pos[col]),
            "absolute_deviation_pct": deviation_pct,
            "within_tolerance": within,
            "full_dataset_uncertain_rate": float(full_unc[col]),
            "subset_uncertain_rate": float(subset_unc[col]),
        }

    return {
        "tolerance_pct": tolerance_pct,
        "num_columns_exceeding_tolerance": num_exceeding,
        "all_within_tolerance": num_exceeding == 0,
        "columns": per_column,
    }


def main() -> int:
    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    dev_cfg = stage1_cfg.dev_subset

    if not bool(dev_cfg.enabled):
        print("dev_subset.enabled=false — 01b_build_dev_subset.py is a no-op. "
              "split.input_csv should point at the full train.csv.")
        return 0

    raw_dir = Path(stage1_cfg.paths.raw_dir)
    train_csv_path = raw_dir / "train.csv"
    if not train_csv_path.is_file():
        raise SystemExit(f"Missing {train_csv_path} (run 00_download_dataset.py / 01_verify_download.py first).")

    schema = dataset_cfg.schema
    pathology_columns = list(schema.pathology_columns)
    target_size = int(dev_cfg.target_size)
    if target_size <= 0:
        raise SystemExit(f"dev_subset.target_size must be positive, got {target_size}")

    df_full = pd.read_csv(train_csv_path)
    original_columns = list(df_full.columns)

    patient_splits_module = importlib.import_module("scripts.data.02_build_patient_splits")
    extract_patient_id = patient_splits_module.extract_patient_id

    df_full = df_full.copy()
    df_full["__patient_id"] = df_full[schema.path_column].apply(
        lambda p: extract_patient_id(p, schema.patient_id_regex)
    )

    selected_patients = select_dev_subset_patients(
        df_full,
        patient_column="__patient_id",
        pathology_columns=pathology_columns,
        target_size=target_size,
        seed=int(dev_cfg.seed),
    )

    df_subset = df_full[df_full["__patient_id"].isin(selected_patients)][original_columns].reset_index(drop=True)

    output_csv = Path(dev_cfg.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_subset.to_csv(output_csv, index=False)

    report = build_prevalence_report(
        df_full, df_subset, pathology_columns, float(dev_cfg.prevalence_tolerance_pct)
    )
    write_json(Path(dev_cfg.prevalence_report_path), report)

    manifest = {
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm": "rarity_first_greedy_quota_patient_sampling",
        "target_size": target_size,
        "actual_num_images": len(df_subset),
        "actual_num_patients": len(selected_patients),
        "seed": int(dev_cfg.seed),
        "source_train_csv_path": str(train_csv_path),
        "source_train_csv_hash": sha256_file(train_csv_path),
        "source_num_images": len(df_full),
        "source_num_patients": int(df_full["__patient_id"].nunique()),
        "dev_subset_config_hash": hash_dict(
            {
                "target_size": target_size,
                "seed": int(dev_cfg.seed),
                "prevalence_tolerance_pct": float(dev_cfg.prevalence_tolerance_pct),
            }
        ),
        "git_commit_hash": get_git_commit_hash(),
        "all_within_tolerance": report["all_within_tolerance"],
    }
    write_json(Path(dev_cfg.manifest_path), manifest)

    print(f"Source: {len(df_full)} images / {df_full['__patient_id'].nunique()} patients")
    print(f"Dev subset: {len(df_subset)} images / {len(selected_patients)} patients (target {target_size})")
    if report["all_within_tolerance"]:
        print(f"Prevalence check: all {len(pathology_columns)} pathology columns within "
              f"{dev_cfg.prevalence_tolerance_pct}pp tolerance.")
    else:
        print(f"WARNING: {report['num_columns_exceeding_tolerance']} column(s) exceed "
              f"{dev_cfg.prevalence_tolerance_pct}pp prevalence-deviation tolerance — see "
              f"{dev_cfg.prevalence_report_path} (non-fatal).")
    print(f"Wrote {output_csv}, {dev_cfg.manifest_path}, {dev_cfg.prevalence_report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
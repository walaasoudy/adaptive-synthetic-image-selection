"""Shared CheXpert source-schema and path validation used by verification and split building."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from scripts.utils.labels import validate_label_domain
from scripts.utils.manifest import hash_dict


def resolve_chexpert_image(raw_dir: Path, value: object) -> Path:
    relative = str(value).replace("\\", "/")
    for prefix in ("CheXpert-v1.0-small/", "CheXpert-v1.0/"):
        if relative.startswith(prefix):
            relative = relative[len(prefix):]
    return raw_dir / relative


def validate_chexpert_frame(frame: pd.DataFrame, dataset_cfg, raw_dir: Path | None = None,
                            require_all_images: bool = False) -> dict:
    schema = dataset_cfg.schema
    required = [schema.path_column, schema.sex_column, schema.age_column,
                schema.frontal_lateral_column, *list(schema.pathology_columns)]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    validate_label_domain(frame, list(schema.pathology_columns), str(schema.path_column))
    paths = frame[schema.path_column]
    if paths.isna().any() or paths.astype(str).str.strip().eq("").any():
        raise ValueError("blank image path")
    duplicates = paths[paths.duplicated()].astype(str).head().tolist()
    if duplicates:
        raise ValueError(f"duplicate image paths: {duplicates}")
    invalid_patient = [value for value in paths.astype(str) if not re.search(str(schema.patient_id_regex), value)]
    if invalid_patient:
        raise ValueError(f"invalid patient ID/path: {invalid_patient[:5]}")
    if require_all_images:
        if raw_dir is None:
            raise ValueError("raw_dir is required for full image-path validation")
        missing_images = [str(resolve_chexpert_image(raw_dir, value)) for value in paths if not resolve_chexpert_image(raw_dir, value).is_file()]
        if missing_images:
            raise ValueError(f"{len(missing_images)} image paths do not exist; examples={missing_images[:5]}")
    return {"source_schema_hash": hash_dict({"columns": list(frame.columns),
                                              "dtypes": {c: str(t) for c, t in frame.dtypes.items()}}, length=64)}

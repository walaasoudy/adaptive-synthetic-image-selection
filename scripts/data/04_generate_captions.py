from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.caption_builder import CaptionConfig, PATHOLOGY_COLUMNS, build_caption_variants  # noqa: E402
from scripts.utils.config import ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.identifiers import sanitize_image_id  # noqa: E402
from scripts.utils.manifest import read_json, write_json  # noqa: E402
from scripts.utils.splits import read_split_manifest, resolve_splits_dir  # noqa: E402

_preprocess = importlib.import_module("scripts.data.03_preprocess_images")
expected_manifest = _preprocess.expected_manifest
validate_processed_jpeg = _preprocess.validate_processed_jpeg


def build_caption_config(cfg) -> CaptionConfig:
    value = cfg.captions
    return CaptionConfig(value.age_bucket_width_years, value.uncertain_label_policy, value.no_finding_overrides_positives, value.num_paraphrase_variants, value.template_version)


def raw_label_vector(row: pd.Series) -> dict:
    result = {}
    for column in PATHOLOGY_COLUMNS:
        value = row.get(column)
        result[column] = None if value is None or pd.isna(value) else int(value)
    return result


def load_preprocessing_status(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise SystemExit(f"Missing preprocessing log: {path}. Complete preprocessing for this split first.")
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Invalid preprocessing log JSON at {path}:{line_number}: {exc}") from exc
            image_id = record.get("image_id")
            if not image_id or image_id in result:
                raise SystemExit(f"Missing/duplicate image_id in {path}:{line_number}: {image_id!r}")
            result[image_id] = record
    return result


def previous_caption_ids(path: Path) -> tuple[set[str], int]:
    ids, duplicates = set(), 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                image_id = json.loads(line).get("image_id")
                duplicates += image_id in ids
                ids.add(image_id)
    return ids, duplicates


def process_split(name, df, images_dir, captions_dir, schema, caption_cfg, resolution) -> Counter:
    ids = [sanitize_image_id(value) for value in df[schema.path_column]]
    duplicate_source_ids = [key for key, count in Counter(ids).items() if count > 1]
    if duplicate_source_ids:
        raise SystemExit(f"{name}: duplicate image IDs in split CSV: {duplicate_source_ids[:5]}")
    status = load_preprocessing_status(images_dir / f"{name}_preprocessing_log.jsonl")
    split_ids = set(ids)
    unknown_log_ids = set(status) - split_ids
    missing_log_ids = split_ids - set(status)
    if unknown_log_ids or missing_log_ids:
        raise SystemExit(f"{name}: stale/incomplete preprocessing log: missing={len(missing_log_ids)}, not-in-split={len(unknown_log_ids)}")

    destination = captions_dir / f"{name}_captions.jsonl"
    previous_ids, previous_duplicates = previous_caption_ids(destination)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=captions_dir)
    counts = Counter(written=0, filtered_or_failed=0, missing_or_invalid=0)
    written_ids = set()
    try:
        with os.fdopen(fd, "w", encoding="utf-8", buffering=1) as output:
            for (_, row), image_id in tqdm(zip(df.iterrows(), ids), total=len(df), desc=f"[{name}] captions", unit="record"):
                preprocess_record = status[image_id]
                if not preprocess_record.get("kept", False):
                    counts["filtered_or_failed"] += 1
                    continue
                image_path = images_dir / name / f"{image_id}.jpg"
                valid, reason = validate_processed_jpeg(image_path, resolution)
                if not valid:
                    counts["missing_or_invalid"] += 1
                    print(f"[{name}] WARNING: excluding {image_id}: processed image is {reason}", flush=True)
                    continue
                record = {
                    "image_id": image_id, "split": name,
                    "image_relpath": (Path(name) / f"{image_id}.jpg").as_posix(),
                    "patient_id": row.get("patient_id"),
                    "caption_variants": build_caption_variants(row.to_dict(), caption_cfg),
                    "raw_labels": raw_label_vector(row), "sex": row.get(schema.sex_column),
                    "age": row.get(schema.age_column), "frontal_lateral": row.get(schema.frontal_lateral_column),
                    "ap_pa": row.get(schema.ap_pa_column),
                }
                output.write(json.dumps(record, ensure_ascii=False, default=lambda value: None if pd.isna(value) else str(value)) + "\n")
                written_ids.add(image_id)
                counts["written"] += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    stale = previous_ids - written_ids
    missing_previous = written_ids - previous_ids if previous_ids else set()
    print(f"[{name}] written={counts['written']} filtered/failed={counts['filtered_or_failed']} missing/invalid={counts['missing_or_invalid']} prior-duplicates={previous_duplicates} stale-prior={len(stale)} new-vs-prior={len(missing_previous)} -> {destination}", flush=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--splits", nargs="+", default=["gen_train", "gen_val"])
    args = parser.parse_args()
    cfg, dataset_cfg = load_stage1_config(), load_dataset_config()
    ensure_dirs(cfg)
    namespace = args.namespace or str(cfg.split.namespace)
    split_manifest = read_split_manifest(namespace)
    split_paths = {name: resolve_splits_dir(namespace) / f"{name}.csv" for name in args.splits}
    missing = [str(path) for path in split_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing split files: " + ", ".join(missing))
    images_dir = Path(cfg.paths.images_dir) / namespace
    captions_dir = Path(cfg.paths.captions_dir) / namespace
    captions_dir.mkdir(parents=True, exist_ok=True)
    for name, path in split_paths.items():
        expected = expected_manifest(cfg, {name: path}, namespace, split_manifest)
        manifest_path = images_dir / f"{name}_preprocessing_manifest.json"
        if not manifest_path.is_file() or read_json(manifest_path) != expected:
            raise SystemExit(f"Preprocessing provenance for {name} does not match the current namespace/settings. Rerun preprocessing first.")
        process_split(name, pd.read_csv(path), images_dir, captions_dir, dataset_cfg.schema, build_caption_config(cfg), int(cfg.data.resolution))
    write_json(captions_dir / "caption_template_version.json", {
        "template_version": cfg.captions.template_version,
        "num_paraphrase_variants": cfg.captions.num_paraphrase_variants,
        "age_bucket_width_years": cfg.captions.age_bucket_width_years,
        "uncertain_label_policy": cfg.captions.uncertain_label_policy,
        "no_finding_overrides_positives": cfg.captions.no_finding_overrides_positives,
        "preprocessing_manifests": {name: read_json(images_dir / f"{name}_preprocessing_manifest.json") for name in args.splits},
        "split_namespace": namespace,
        "split_manifest_hash": split_manifest.get("manifest_hash"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

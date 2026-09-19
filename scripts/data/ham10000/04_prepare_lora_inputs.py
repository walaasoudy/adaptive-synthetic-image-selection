#!/usr/bin/env python3
"""Prepare HAM10000 Stage 1 LoRA training inputs: unpadded native-aspect images + captions.

WHY THIS IS NOT THE PREPROCESSED CANVAS
  02_preprocess_images.py letterboxes every image onto a padded square for the classifier and ASISM.
  A LoRA trained on those canvases would learn to draw grey bars, and generated images would then
  carry padding of unknown extent. The geometry contract (configs/ham10000_stage1.yaml `geometry`)
  therefore trains the generator on UNPADDED images at the native 4:3 aspect ratio. This script
  writes exactly those, and refuses any source whose aspect ratio is not the contracted one instead
  of cropping or padding it silently.

DATA ROLES
  Only `split.train_split` (gen_train) and `split.monitor_split` (gen_val) are prepared; both names
  are asserted. Before anything is written, each split's image ids are checked against the ids of
  the other four splits — the classifier, ASISM-tuning and final-eval images can never reach the
  generator's inputs even through a mislabelled split file.

Resumable: an existing output is kept only after being decoded and validated (RGB, exact size).
Captions and the manifest are written atomically after every image is in place.

Usage:
    python scripts/data/ham10000/04_prepare_lora_inputs.py --namespace ham-stratified-v1
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import TEMPLATE_VERSION, build_caption_variants, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_lora_data import (  # noqa: E402
    assert_role_isolation,
    atomic_save_rgb_jpeg,
    forbidden_image_ids,
    native_aspect_resize,
    validate_rgb_jpeg,
    validate_roles,
)
from scripts.utils.manifest import sha256_file, write_json  # noqa: E402



def _preprocess_module():
    spec = importlib.util.spec_from_file_location("ham_preprocess", Path(__file__).resolve().parent / "02_preprocess_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare_split(split_name: str, frame: pd.DataFrame, raw_dir: Path, image_directories, out_root: Path, stage1_cfg, provenance: dict) -> dict:
    preprocess = _preprocess_module()
    size = tuple(int(v) for v in stage1_cfg.geometry.lora_training_size)
    captions_cfg = stage1_cfg.captions
    if str(captions_cfg.template_version) != TEMPLATE_VERSION:
        raise SystemExit(f"captions.template_version {captions_cfg.template_version!r} != code TEMPLATE_VERSION {TEMPLATE_VERSION!r}")

    out_dir = out_root / split_name
    records, counts = [], {"written": 0, "skipped_valid": 0}
    for row in tqdm(frame.to_dict("records"), desc=f"[{split_name}] lora inputs", unit="image"):
        image_id = str(row["image_id"])
        destination = out_dir / f"{image_id}.jpg"
        valid, _ = validate_rgb_jpeg(destination, size)
        if valid:
            counts["skipped_valid"] += 1
        else:
            source = preprocess.resolve_source(raw_dir, image_id, list(image_directories))
            if source is None:
                raise SystemExit(f"{split_name}: source image missing for {image_id}; training inputs must be complete")
            with Image.open(source) as image:
                resized = native_aspect_resize(image, size)
            atomic_save_rgb_jpeg(resized, destination, size, int(stage1_cfg.data.jpeg_quality))
            counts["written"] += 1
        records.append(
            {
                "image_id": image_id,
                "image_relpath": f"{split_name}/{image_id}.jpg",
                "dx": normalize_diagnosis(row["dx"]),
                "lesion_id": str(row.get("lesion_id")),
                "caption_variants": build_caption_variants(
                    row, image_id, int(captions_cfg.num_paraphrase_variants), int(captions_cfg.age_bucket_width_years)
                ),
            }
        )

    captions_path = out_root / f"{split_name}_captions.jsonl"
    _atomic_write_text(captions_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    ids = sorted(r["image_id"] for r in records)
    manifest = {
        "dataset": "ham10000",
        "split_name": split_name,
        **provenance,
        "size": list(size),
        "resize": "native_aspect_bicubic_no_crop_no_pad",
        "caption_template_version": TEMPLATE_VERSION,
        "num_paraphrase_variants": int(captions_cfg.num_paraphrase_variants),
        "age_bucket_width_years": int(captions_cfg.age_bucket_width_years),
        "num_images": len(records),
        "image_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "captions_sha256": sha256_file(captions_path),
    }
    write_json(out_root / f"{split_name}_lora_inputs_manifest.json", manifest)
    return {**counts, "images": len(records), "captions": str(captions_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default=None, help="Split run id (default: split.namespace in ham10000_stage1.yaml)")
    args = parser.parse_args()

    stage1_cfg = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    roles = validate_roles(stage1_cfg.split)
    namespace = args.namespace or str(stage1_cfg.split.namespace)

    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    split_manifest_path = split_dir / "split_manifest_v2.json"
    if not split_manifest_path.is_file():
        raise SystemExit(f"missing split manifest {split_manifest_path}; run 01_build_splits.py first")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    forbidden = forbidden_image_ids(split_dir, roles.values())

    out_root = Path(stage1_cfg.paths.lora_inputs_dir) / namespace
    for split_name in roles.values():
        frame = pd.read_csv(split_dir / f"{split_name}.csv")
        assert_role_isolation(frame["image_id"], split_name, forbidden)
        provenance = {
            "split_namespace": namespace,
            "split_manifest_hash": split_manifest["manifest_hash"],
            "split_csv_sha256": sha256_file(split_dir / f"{split_name}.csv"),
            "forbidden_splits_checked": sorted(forbidden),
        }
        result = prepare_split(
            split_name, frame, Path(splits_cfg.paths.raw_dir), dataset_cfg.source.image_directories, out_root, stage1_cfg, provenance
        )
        print(f"[{split_name}] {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate structured label-to-text captions for preprocessed gen_train / gen_val images
(docs/stage1_plan.md §7), using scripts/utils/caption_builder.py — the same module Stage 2 will
import for generation prompts.

Only images that survived 03_preprocess_images.py (i.e. exist on disk under images_dir/<split>/)
get captions; the raw ternary label vector is preserved unmodified alongside each record so a
future revision of the captioning policy can regenerate captions without redoing preprocessing.

Usage:
    python scripts/data/04_generate_captions.py [--splits gen_train gen_val]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.caption_builder import (  # noqa: E402
    CaptionConfig,
    PATHOLOGY_COLUMNS,
    build_caption_variants,
)
from scripts.utils.config import ensure_dirs, load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.identifiers import sanitize_image_id  # noqa: E402
from scripts.utils.manifest import write_json  # noqa: E402


def build_caption_config(stage1_cfg) -> CaptionConfig:
    c = stage1_cfg.captions
    return CaptionConfig(
        age_bucket_width_years=c.age_bucket_width_years,
        uncertain_label_policy=c.uncertain_label_policy,
        no_finding_overrides_positives=c.no_finding_overrides_positives,
        num_paraphrase_variants=c.num_paraphrase_variants,
        template_version=c.template_version,
    )


def raw_label_vector(row: pd.Series) -> dict:
    """Preserve the raw ternary labels verbatim (docs/stage1_plan.md §6/§7) regardless of what
    the caption policy does with them, so captions can be regenerated later without redoing
    image preprocessing."""
    vector = {}
    for col in PATHOLOGY_COLUMNS:
        value = row.get(col)
        if value is None or (isinstance(value, float) and value != value):  # NaN
            vector[col] = None
        else:
            vector[col] = int(value)
    return vector


def process_split(
    split_name: str,
    df: pd.DataFrame,
    images_dir: Path,
    captions_dir: Path,
    schema,
    caption_config: CaptionConfig,
) -> int:
    split_images_dir = images_dir / split_name
    out_path = captions_dir / f"{split_name}_captions.jsonl"

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            image_id = sanitize_image_id(row[schema.path_column])
            image_path = split_images_dir / f"{image_id}.jpg"
            if not image_path.exists():
                continue  # filtered out during preprocessing (lateral view, corrupt, etc.)

            variants = build_caption_variants(row.to_dict(), caption_config)
            record = {
                "image_id": image_id,
                "split": split_name,
                "image_relpath": str(image_path.relative_to(images_dir)),
                "patient_id": row.get("patient_id"),
                "caption_variants": variants,
                "raw_labels": raw_label_vector(row),
                "sex": row.get(schema.sex_column),
                "age": row.get(schema.age_column),
                "frontal_lateral": row.get(schema.frontal_lateral_column),
                "ap_pa": row.get(schema.ap_pa_column),
            }
            f.write(json.dumps(record))
            f.write("\n")
            n_written += 1

    print(f"[{split_name}] wrote {n_written} caption records -> {out_path}")
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["gen_train", "gen_val"])
    args = parser.parse_args()

    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    ensure_dirs(stage1_cfg)

    splits_dir = Path(stage1_cfg.paths.splits_dir)
    images_dir = Path(stage1_cfg.paths.images_dir)
    captions_dir = Path(stage1_cfg.paths.captions_dir)
    schema = dataset_cfg.schema
    caption_config = build_caption_config(stage1_cfg)

    for split_name in args.splits:
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"Skipping {split_name}: {csv_path} not found (run 02_build_patient_splits.py first)")
            continue
        df = pd.read_csv(csv_path)
        if "patient_id" not in df.columns:
            df["patient_id"] = None
        process_split(split_name, df, images_dir, captions_dir, schema, caption_config)

    write_json(
        captions_dir / "caption_template_version.json",
        {
            "template_version": caption_config.template_version,
            "num_paraphrase_variants": caption_config.num_paraphrase_variants,
            "age_bucket_width_years": caption_config.age_bucket_width_years,
            "uncertain_label_policy": caption_config.uncertain_label_policy,
            "no_finding_overrides_positives": caption_config.no_finding_overrides_positives,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

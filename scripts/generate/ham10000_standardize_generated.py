#!/usr/bin/env python3
"""Put generated HAM10000 candidates into the same geometry contract as real images.

For each generated image in --input-dir: verify the contracted generation size and the absence of
generator-drawn padding, letterbox it with the SAME function real preprocessing uses, write the
canvas JPEG to --output-dir, and write <output-dir>/content_boxes.csv. Images that break the
contract are written to <output-dir>/rejected_geometry.csv with the reason and are NOT standardised:
they never enter the candidate pool with a guessed box.

Usage:
    python scripts/generate/ham10000_standardize_generated.py --input-dir <raw_generated> --output-dir <candidates>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000_geometry import UnknownPaddingError, standardize_generated_image, write_content_boxes  # noqa: E402

GENERATED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def standardize_directory(input_dir: Path, output_dir: Path, stage1_cfg) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry, data = stage1_cfg.geometry, stage1_cfg.data
    boxes, rejected = [], []
    for path in sorted(p for p in Path(input_dir).iterdir() if p.suffix.lower() in GENERATED_SUFFIXES):
        with Image.open(path) as image:
            try:
                canvas, row = standardize_generated_image(
                    image,
                    path.stem,
                    tuple(geometry.generation_size),
                    int(data.resolution),
                    tuple(data.pad_colour),
                    float(geometry.generated_min_edge_std),
                )
            except UnknownPaddingError as exc:
                rejected.append({"image_id": path.stem, "reason": str(exc)})
                continue
        canvas.save(output_dir / f"{path.stem}.jpg", format="JPEG", quality=int(data.jpeg_quality))
        boxes.append(row)
    write_content_boxes(output_dir / "content_boxes.csv", boxes)
    pd.DataFrame(rejected, columns=["image_id", "reason"]).to_csv(output_dir / "rejected_geometry.csv", index=False)
    return {"standardized": len(boxes), "rejected": len(rejected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = standardize_directory(args.input_dir, args.output_dir, load_named_config("ham10000_stage1.yaml", "ham_stage1"))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evidence + calibration artifact for the HAM10000 IQA border-artifact threshold. Changes no config.

Measures border uniformity on the preprocessed gen_train canvases in BOTH regions (the whole canvas,
as the inherited rule does, and the persisted content box), and writes:

  <outputs_dir>/<namespace>/iqa_border_calibration.json   the upper-quantile calibration for the
                                                           configured border_calibration.region —
                                                           consumed only if border_calibration.enabled
  <outputs_dir>/<namespace>/iqa_border_diagnostics.json    per-class flag rates for the inherited
                                                           constant and for both calibrated regions,
                                                           plus the padding-coincidence statistics
  <outputs_dir>/<namespace>/border_review/*.png            contact sheets for visual review

gen_train pixels only; asism_tuning_heldout and final_eval_heldout are read as id lists to prove
exclusion.

Usage:
    python scripts/asism/ham10000_calibrate_iqa_border.py --namespace ham-stratified-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_signals import border_region_luminance, border_uniform_fraction, calibrate_border_threshold  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import DIAGNOSIS_CLASSES, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_geometry import load_content_boxes  # noqa: E402
from scripts.utils.manifest import write_json  # noqa: E402

HELD_OUT_SPLITS = ("asism_tuning_heldout", "final_eval_heldout")


def contact_sheet(paths: list[Path], destination: Path, columns: int = 6, tile: int = 160) -> None:
    if not paths:
        return
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), (255, 255, 255))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB").resize((tile, tile)), ((index % columns) * tile, (index // columns) * tile))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def run(namespace: str) -> dict:
    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    iqa = stage3.signals.iqa
    rule = iqa.border_calibration
    source = str(rule.source_split)

    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    frame = pd.read_csv(split_dir / f"{source}.csv")
    diagnosis = {str(r.image_id): normalize_diagnosis(r.dx) for r in frame.itertuples()}
    forbidden: set[str] = set()
    for name in HELD_OUT_SPLITS:
        forbidden |= set(pd.read_csv(split_dir / f"{name}.csv", usecols=["image_id"])["image_id"].astype(str))

    images_root = Path(stage1.paths.images_dir) / namespace
    boxes = load_content_boxes(images_root / f"{source}_content_boxes.csv")
    measured = {"canvas": {}, "content_box": {}}
    edge_mean = {}
    for path in tqdm(sorted((images_root / source).glob("*.jpg")), desc="border uniformity", unit="img"):
        with Image.open(path) as image:
            luminance = np.asarray(image.convert("RGB").convert("L"), dtype=np.float32)
        measured["canvas"][path.stem] = border_uniform_fraction(border_region_luminance(luminance, "canvas"))
        measured["content_box"][path.stem] = border_uniform_fraction(border_region_luminance(luminance, "content_box", boxes[path.stem]))
        height, width = luminance.shape
        band = max(1, int(min(height, width) * 0.06))
        edge_mean[path.stem] = float(np.concatenate([luminance[:band].ravel(), luminance[-band:].ravel(), luminance[:, :band].ravel(), luminance[:, -band:].ravel()]).mean())

    calibrations = {
        region: calibrate_border_threshold(values, region, source, diagnosis.keys(), forbidden, float(rule.quantile), diagnosis)
        for region, values in measured.items()
    }
    inherited = float(iqa.border_uniform_fraction)
    ids = list(measured["canvas"])
    canvas_values = np.asarray([measured["canvas"][i] for i in ids])
    labels = np.asarray([diagnosis[i] for i in ids])
    inherited_flag = canvas_values > inherited
    pad_grey = float(np.mean(stage1.data.pad_colour))
    edge = np.asarray([edge_mean[i] for i in ids])
    content_flag = np.asarray([measured["content_box"][i] for i in ids]) > calibrations["content_box"]["border_uniform_fraction"]

    diagnostics = {
        "namespace": namespace,
        "n_images": len(ids),
        "inherited_rule": {"region": "canvas", "threshold": inherited, "overall_flag_rate": float(inherited_flag.mean()),
                           "per_class": {l: float(inherited_flag[labels == l].mean()) for l in DIAGNOSIS_CLASSES if (labels == l).any()}},
        "calibrated": {region: {k: v for k, v in c.items()} for region, c in calibrations.items()},
        "padding_coincidence": {
            "pad_grey": pad_grey,
            "flagged_median_abs_edge_mean_minus_pad": float(np.median(np.abs(edge[inherited_flag] - pad_grey))) if inherited_flag.any() else None,
            "unflagged_median_abs_edge_mean_minus_pad": float(np.median(np.abs(edge[~inherited_flag] - pad_grey))),
        },
        "overlap_inherited_canvas_vs_calibrated_content_box": int((inherited_flag & content_flag).sum()),
        "held_out_splits_excluded": list(HELD_OUT_SPLITS),
        "split_csv_sha256": hashlib.sha256((split_dir / f"{source}.csv").read_bytes()).hexdigest(),
    }
    out_dir = Path(stage3.paths.outputs_dir) / namespace
    write_json(out_dir / str(rule.artifact_filename), {**calibrations[str(rule.region)], "namespace": namespace, "split_csv_sha256": diagnostics["split_csv_sha256"]})
    write_json(out_dir / "iqa_border_diagnostics.json", diagnostics)

    rng = np.random.default_rng(0)
    review = out_dir / "border_review"
    for label in DIAGNOSIS_CLASSES:
        flagged = [images_root / source / f"{i}.jpg" for i, f, l in zip(ids, inherited_flag, labels) if f and l == label]
        contact_sheet([flagged[j] for j in rng.permutation(len(flagged))[:18]], review / f"inherited_flagged_{label}.png")
    top_content = sorted(ids, key=lambda i: measured["content_box"][i], reverse=True)[:18]
    contact_sheet([images_root / source / f"{i}.jpg" for i in top_content], review / "content_box_most_uniform.png")
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.namespace), indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

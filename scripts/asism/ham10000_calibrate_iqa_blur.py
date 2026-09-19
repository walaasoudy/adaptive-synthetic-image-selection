#!/usr/bin/env python3
"""Calibrate the HAM10000 IQA blur threshold from the preprocessed gen_train images.

Writes <outputs_dir>/<namespace>/iqa_blur_calibration.json. Every HAM10000 IQA run must resolve its
config from this artifact (resolve_iqa_config); there is no constant fallback.

Reads pixels of gen_train ONLY. asism_tuning_heldout and final_eval_heldout are read as id lists so
the calibration can prove none of their images was measured.

Usage:
    python scripts/asism/ham10000_calibrate_iqa_blur.py --namespace ham-stratified-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_signals import calibrate_blur_threshold, measure_sharpness  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import write_json  # noqa: E402

HELD_OUT_SPLITS = ("asism_tuning_heldout", "final_eval_heldout")


def run(namespace: str) -> dict:
    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    rule = stage3.signals.iqa.blur_calibration
    source = str(rule.source_split)

    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    source_frame = pd.read_csv(split_dir / f"{source}.csv")
    forbidden = set()
    for name in HELD_OUT_SPLITS:
        forbidden |= set(pd.read_csv(split_dir / f"{name}.csv", usecols=["image_id"])["image_id"].astype(str))

    images_root = Path(stage1.paths.images_dir) / namespace
    image_dir = images_root / source
    manifest_path = images_root / f"{source}_preprocessing_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing preprocessing manifest {manifest_path}; run 02_preprocess_images.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    diagnosis = {str(r.image_id): normalize_diagnosis(r.dx) for r in source_frame.itertuples()}
    paths = sorted(image_dir.glob("*.jpg"))
    sharpness = {path.stem: measure_sharpness(path) for path in tqdm(paths, desc=f"sharpness[{source}]", unit="img")}

    result = calibrate_blur_threshold(sharpness, source, diagnosis.keys(), forbidden, float(rule.quantile), diagnosis)
    result.update(
        {
            "namespace": namespace,
            "held_out_splits_excluded": list(HELD_OUT_SPLITS),
            "resolution": int(manifest["output_settings"]["resolution"]),
            "preprocessing_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "split_csv_sha256": hashlib.sha256((split_dir / f"{source}.csv").read_bytes()).hexdigest(),
            "images_in_split": len(source_frame),
            "images_measured": len(sharpness),
        }
    )
    out = Path(stage3.paths.outputs_dir) / namespace / str(rule.artifact_filename)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    result["artifact_path"] = str(out)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    result = run(args.namespace)
    print(json.dumps({k: v for k, v in result.items() if k != "per_class"}, indent=2))
    for label, entry in result["per_class"].items():
        print(f"  {label:6s} n={entry['n']:5d} rejected={entry['rejection_rate']:6.2%} p5={entry['p5']:7.1f} median={entry['median']:7.1f} p95={entry['p95']:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

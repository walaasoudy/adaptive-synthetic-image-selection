#!/usr/bin/env python3
"""End-to-end smoke test on REAL HAM10000 images, drawn from gen_train only.

Runs every HAM10000-specific stage that does not need a trained model, in pipeline order, on actual
dataset pixels, and reports the numbers rather than only pass/fail. It exists because the failures
it looks for — a greyscale conversion, attention on our own padding, a content box that never
reaches the measurement, evaluation data leaking into calibration — are invisible to fixtures.

WHAT IT TOUCHES
  Pixels are read ONLY for gen_train images. The other five split files are read for their ids
  (disjointness audit, final-eval exclusion); final_eval_heldout pixels are never opened, and the
  run ends by proving that from the list of images it actually processed.

WHAT IS DIAGNOSTIC ONLY
  * IQA: the blur threshold is CALIBRATED here from the gen_train images being tested (lower
    quantile, configs/ham10000_stage3.yaml) and written as an artifact; the other IQA thresholds are
    still the inherited values. Per-class distributions and rejection rates are reported; the
    inherited blur threshold (40) is shown alongside for comparison only.
  * Grad-CAM uses a RANDOMLY INITIALISED DenseNet121: no model download and no training. Its
    attention statistics are therefore meaningless as attention; they are used to prove the
    plumbing (content box -> peripheral mass -> reference -> typicality) on real image geometry.
  * The metric-suite check uses a synthetic probability matrix.

Prerequisites (outside OneDrive; PROJECT_ROOT decides where data/ lives):
    python scripts/data/ham10000/00_download_dataset.py
    python scripts/data/ham10000/01_build_splits.py --run-id <id>

Usage:
    python scripts/data/ham10000/03_smoke_subset.py --namespace <id>
    python scripts/data/ham10000/03_smoke_subset.py --namespace <id> --per-class-lesions 0   # all of gen_train
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.asism.ham10000_explainability import explainability_rows, gradcam  # noqa: E402
from scripts.asism.ham10000_signals import (  # noqa: E402
    EXACT_CONTENT_BOX_SOURCE,
    MIN_REFERENCE_SIZE,
    ReferenceLeakageError,
    UncalibratedExplainabilityError,
    assert_selection_features_allowed,
    build_reference_distribution,
    calibrate_blur_threshold,
    calibrate_explainability,
    compute_iqa_scores,
    measure_sharpness,
    resolve_iqa_config,
    tie_report,
    load_final_eval_image_ids,
    peripheral_mass,
)
from scripts.generate.ham10000_recipes import quotas_from_gen_train  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    DIAGNOSIS_CLASSES,
    DIAGNOSIS_PHRASE,
    build_caption,
    normalize_diagnosis,
    validate_metadata,
)
from scripts.utils.ham10000_geometry import load_content_boxes  # noqa: E402
from scripts.utils.ham10000_metrics import format_confusion_matrix, full_metric_suite  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
SPLIT_NAMES = ["gen_train", "gen_val", "classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"]
HERE = Path(__file__).resolve().parent


def _load_sibling(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(results: list, name: str, ok: bool, detail: str = "") -> bool:
    results.append({"status": PASS if ok else FAIL, "check": name, "detail": detail})
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def quantiles(values) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    p5, p25, p50, p75, p95 = np.percentile(array, [5, 25, 50, 75, 95])
    return {"n": int(array.size), "p5": p5, "p25": p25, "median": p50, "p75": p75, "p95": p95, "mean": float(array.mean())}


def saturation_inside(array: np.ndarray, box) -> float:
    height, width = array.shape[:2]
    x0, y0, x1, y1 = box
    region = array[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)]
    return float(np.mean(region.max(axis=2) - region.min(axis=2)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True, help="Split run id built by 01_build_splits.py")
    parser.add_argument("--per-class-lesions", type=int, default=40, help="gen_train lesions per class (0 = all)")
    parser.add_argument("--cam-per-class", type=int, default=MIN_REFERENCE_SIZE + 5, help="Images per class through Grad-CAM")
    parser.add_argument("--cam-resolution", type=int, default=512)
    parser.add_argument("--work-dir", type=Path, default=None, help="Scratch output dir (default: system temp)")
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--report-json", type=Path, default=None)
    args = parser.parse_args()

    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    stage1_cfg = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3_cfg = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    preprocess = _load_sibling("ham_preprocess", "02_preprocess_images.py")
    splitter = _load_sibling("ham_splits", "01_build_splits.py")

    raw_dir = Path(splits_cfg.paths.raw_dir)
    split_dir = Path(splits_cfg.paths.splits_root) / args.namespace
    image_directories = [str(name) for name in dataset_cfg.source.image_directories]
    results: list[dict] = []
    report: dict = {"namespace": args.namespace, "raw_dir": str(raw_dir), "split_dir": str(split_dir)}
    started = time.time()

    print("\n" + "=" * 70 + "\nHAM10000 SMOKE TEST — real images, gen_train only, no GPU\n" + "=" * 70, flush=True)

    # 1. metadata <-> image files ---------------------------------------------------------------
    print("\n1. Metadata and image files", flush=True)
    metadata_path = raw_dir / str(dataset_cfg.source.metadata_filename)
    if not metadata_path.is_file():
        print(f"  [{FAIL}] metadata not found at {metadata_path}", flush=True)
        return 1
    metadata = pd.read_csv(metadata_path)
    validate_metadata(metadata, "dx", "lesion_id")
    metadata["dx_norm"] = metadata["dx"].map(normalize_diagnosis)
    metadata_ids = set(metadata["image_id"].astype(str))

    files_by_id: dict[str, list[str]] = {}
    for directory in image_directories:
        for path in (raw_dir / directory).glob("*.jpg"):
            files_by_id.setdefault(path.stem, []).append(directory)
    missing_files = sorted(metadata_ids - set(files_by_id))
    orphan_files = sorted(set(files_by_id) - metadata_ids)
    in_both_dirs = sorted(image_id for image_id, dirs in files_by_id.items() if len(dirs) > 1)
    report["matching"] = {
        "metadata_rows": len(metadata),
        "unique_image_ids": len(metadata_ids),
        "image_files": sum(len(v) for v in files_by_id.values()),
        "metadata_without_file": len(missing_files),
        "files_without_metadata": len(orphan_files),
        "ids_present_in_both_directories": len(in_both_dirs),
        "lesions": int(metadata["lesion_id"].nunique()),
        "class_images": {label: int((metadata["dx_norm"] == label).sum()) for label in DIAGNOSIS_CLASSES},
        "class_lesions": {label: int(metadata.loc[metadata["dx_norm"] == label, "lesion_id"].nunique()) for label in DIAGNOSIS_CLASSES},
    }
    check(results, "every metadata row has exactly one image file", not missing_files and not in_both_dirs,
          f"{len(metadata)} rows, {report['matching']['image_files']} files, missing={len(missing_files)}, duplicated={len(in_both_dirs)}")
    check(results, "no image file lacks a metadata row", not orphan_files, f"orphans={len(orphan_files)}")
    print(f"  lesions={report['matching']['lesions']}", flush=True)
    for label in DIAGNOSIS_CLASSES:
        print(f"    {label:6s} images={report['matching']['class_images'][label]:5d} lesions={report['matching']['class_lesions'][label]:5d}", flush=True)

    # 2. frozen splits --------------------------------------------------------------------------
    print("\n2. Frozen lesion-level splits", flush=True)
    missing_splits = [name for name in SPLIT_NAMES if not (split_dir / f"{name}.csv").is_file()]
    if missing_splits:
        print(f"  [{FAIL}] missing split files {missing_splits} in {split_dir}", flush=True)
        return 1
    frames = {name: pd.read_csv(split_dir / f"{name}.csv") for name in SPLIT_NAMES}
    audit = splitter.audit_splits(frames, "lesion_id", "image_id")
    union = set().union(*(set(frame["image_id"].astype(str)) for frame in frames.values()))
    report["splits"] = {
        "audit_status": audit["status"],
        "images_per_split": audit["images_per_split"],
        "lesions_per_split": audit["lesions_per_split"],
        "class_distribution_per_split": audit["class_distribution_per_split"],
    }
    check(results, "splits are disjoint at lesion AND image level", audit["status"] == PASS, f"failed pairs={audit['failed_pairs']}")
    check(results, "splits exactly cover the metadata", union == metadata_ids, f"{len(union)} ids in splits vs {len(metadata_ids)} in metadata")
    for name in SPLIT_NAMES:
        distribution = audit["class_distribution_per_split"][name]
        print(f"    {name:22s} images={audit['images_per_split'][name]:5d} lesions={audit['lesions_per_split'][name]:5d}  "
              + " ".join(f"{label}:{distribution[label]}" for label in DIAGNOSIS_CLASSES), flush=True)

    final_eval_ids = load_final_eval_image_ids(split_dir)
    final_eval_lesions = set(frames["final_eval_heldout"]["lesion_id"].astype(str))

    # 3. subset: gen_train only -----------------------------------------------------------------
    print("\n3. Subset (gen_train only, lesion-grouped)", flush=True)
    gen_train = frames["gen_train"].copy()
    gen_train["dx_norm"] = gen_train["dx"].map(normalize_diagnosis)
    parts = []
    for label in DIAGNOSIS_CLASSES:
        lesions = sorted(gen_train.loc[gen_train["dx_norm"] == label, "lesion_id"].astype(str).unique())
        if args.per_class_lesions > 0:
            rng = np.random.default_rng(0)
            lesions = sorted(rng.permutation(lesions)[: args.per_class_lesions])
        parts.append(gen_train[gen_train["lesion_id"].astype(str).isin(lesions)])
    subset = pd.concat(parts).drop_duplicates(subset=["image_id"]).sort_values("image_id").reset_index(drop=True)
    subset_ids = set(subset["image_id"].astype(str))
    report["subset"] = {
        "images": len(subset),
        "lesions": int(subset["lesion_id"].nunique()),
        "per_class_images": {label: int((subset["dx_norm"] == label).sum()) for label in DIAGNOSIS_CLASSES},
    }
    check(results, "subset is drawn from gen_train only", subset_ids <= set(gen_train["image_id"].astype(str)), f"{len(subset)} images, {subset['lesion_id'].nunique()} lesions")
    check(results, "subset shares no image or lesion with final_eval_heldout",
          not (subset_ids & final_eval_ids) and not (set(subset["lesion_id"].astype(str)) & final_eval_lesions))
    check(results, "all 7 classes present", all(report["subset"]["per_class_images"][label] > 0 for label in DIAGNOSIS_CLASSES),
          ", ".join(f"{label}:{report['subset']['per_class_images'][label]}" for label in DIAGNOSIS_CLASSES))

    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="ham_smoke_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    out_root = work_dir / "processed"
    processed_ids: set[str] = set()

    try:
        # 4. preprocessing with the real process_split -----------------------------------------
        print("\n4. Preprocessing (real process_split: RGB, letterbox, content boxes)", flush=True)
        resolution = int(stage1_cfg.data.resolution)
        counts = preprocess.process_split("gen_train", subset[["image_id"]], raw_dir, out_root, image_directories, stage1_cfg.data, "image_id")
        out_dir = out_root / "gen_train"
        boxes = load_content_boxes(out_root / "gen_train_content_boxes.csv")
        kept = sorted(path.stem for path in out_dir.glob("*.jpg"))
        processed_ids = set(kept)
        with open(out_root / "gen_train_preprocessing_log.jsonl", encoding="utf-8") as handle:
            logged_ids = {json.loads(line)["image_id"] for line in handle}

        modes, sizes, pixel_inconsistent, source_sizes = Counter(), Counter(), [], Counter()
        source_saturation, output_saturation = [], []
        box_rows = pd.read_csv(out_root / "gen_train_content_boxes.csv")
        size_of = {str(r.image_id): (int(r.source_width), int(r.source_height)) for r in box_rows.itertuples()}
        for image_id in kept:
            with Image.open(out_dir / f"{image_id}.jpg") as image:
                modes[image.mode] += 1
                sizes[image.size] += 1
                array = np.asarray(image.convert("RGB"), dtype=np.int16)
            source_sizes[size_of[image_id]] += 1
            x0, y0, x1, y1 = boxes[image_id]
            r0, r1, c0, c1 = round(y0 * resolution), round(y1 * resolution), round(x0 * resolution), round(x1 * resolution)
            # Pixel cross-check of the persisted box: 2 px inside each padding band must be ~128,
            # and the content's outer rows/cols must not be. JPEG ringing is why it is not exact.
            pad_regions = [array[: max(r0 - 2, 0)], array[min(r1 + 2, resolution):], array[:, : max(c0 - 2, 0)], array[:, min(c1 + 2, resolution):]]
            pad_ok = all(region.size == 0 or float(np.abs(region - 128).mean()) < 4.0 for region in pad_regions)
            if not pad_ok:
                pixel_inconsistent.append(image_id)
            output_saturation.append(saturation_inside(array, boxes[image_id]))
            source = preprocess.resolve_source(raw_dir, image_id, image_directories)
            with Image.open(source) as src:
                source_array = np.asarray(src.convert("RGB"), dtype=np.int16)
            source_saturation.append(float(np.mean(source_array.max(axis=2) - source_array.min(axis=2))))

        box_distribution = Counter(tuple(round(v, 6) for v in box) for box in boxes.values())
        report["preprocessing"] = {
            "counts": dict(counts),
            "output_modes": {k: v for k, v in modes.items()},
            "output_sizes": {f"{w}x{h}": v for (w, h), v in sizes.items()},
            "source_sizes": {f"{w}x{h}": v for (w, h), v in source_sizes.most_common()},
            "content_box_values": {json.dumps(list(box)): n for box, n in box_distribution.most_common()},
            "content_box_rows": len(boxes),
            "pixel_inconsistent_boxes": len(pixel_inconsistent),
            "saturation_source": quantiles(source_saturation),
            "saturation_output_inside_box": quantiles(output_saturation),
        }
        check(results, "every subset image was preprocessed", counts["processed"] + counts["skipped_existing"] == len(subset), json.dumps(dict(counts)))
        check(results, "outputs are RGB, never greyscale", set(modes) == {"RGB"}, f"modes={dict(modes)}")
        check(results, "outputs are the configured square size", set(sizes) == {(resolution, resolution)}, f"{dict(sizes)}")
        check(results, "a content box is persisted for exactly the kept images", set(boxes) == processed_ids == (logged_ids & processed_ids), f"{len(boxes)} boxes / {len(kept)} images")
        check(results, "persisted boxes agree with the output pixels", not pixel_inconsistent, f"inconsistent={len(pixel_inconsistent)} {pixel_inconsistent[:3]}")
        print(f"  source sizes: {report['preprocessing']['source_sizes']}", flush=True)
        print(f"  content_box values: {report['preprocessing']['content_box_values']}", flush=True)

        # 5. colour survival -------------------------------------------------------------------
        print("\n5. Colour survival", flush=True)
        ratio = np.asarray(output_saturation) / np.maximum(np.asarray(source_saturation), 1e-6)
        report["preprocessing"]["saturation_ratio_output_over_source"] = quantiles(ratio)
        check(results, "colour survives (per-image output/source saturation inside the content box)",
              float(np.percentile(ratio, 5)) > 0.8,
              f"ratio p5={np.percentile(ratio, 5):.3f} median={np.median(ratio):.3f}; source median={np.median(source_saturation):.1f}")

        # 6. captions --------------------------------------------------------------------------
        print("\n6. Captions", flush=True)
        records = subset[subset["image_id"].astype(str).isin(processed_ids)].to_dict("records")
        captions = [build_caption({**row, "dx": row["dx_norm"]}) for row in records]
        names_match = all(DIAGNOSIS_PHRASE[row["dx_norm"]] in caption for caption, row in zip(captions, records))
        check(results, "one caption per image, naming the right diagnosis", names_match and len(captions) == len(records), captions[0])
        check(results, "no caption contains the literal token 'unknown'", not any("unknown" in c.lower() for c in captions))

        # 7. IQA — blur threshold calibrated on gen_train, then scored ---------------------------
        print("\n7. IQA on real images (blur threshold calibrated from gen_train only)", flush=True)
        blur_rule = stage3_cfg.signals.iqa.blur_calibration
        held_out_ids = set(frames["asism_tuning_heldout"]["image_id"].astype(str)) | final_eval_ids
        sharpness = {str(row["image_id"]): measure_sharpness(out_dir / f"{row['image_id']}.jpg") for row in records}
        diagnosis_of = {str(row["image_id"]): row["dx_norm"] for row in records}
        blur_calibration = calibrate_blur_threshold(
            sharpness, "gen_train", set(gen_train["image_id"].astype(str)), held_out_ids, float(blur_rule.quantile), diagnosis_of
        )
        blur_calibration["calibrated_on_full_gen_train"] = len(sharpness) == len(gen_train)
        (work_dir / str(blur_rule.artifact_filename)).write_text(json.dumps(blur_calibration, indent=2, default=float), encoding="utf-8")
        resolved_iqa_cfg = resolve_iqa_config(stage3_cfg, blur_calibration)
        values = np.asarray(list(sharpness.values()))
        labels = np.asarray([diagnosis_of[i] for i in sharpness])
        report["blur_calibration"] = {
            **{k: v for k, v in blur_calibration.items() if k != "per_class"},
            "per_class": blur_calibration["per_class"],
            "comparison_inherited_threshold_40": {
                "overall_rejection_rate": float(np.mean(values < 40.0)),
                "per_class": {label: float(np.mean(values[labels == label] < 40.0)) for label in DIAGNOSIS_CLASSES},
            },
        }
        refused = {}
        for case, kwargs in {
            "asism_tuning_source": dict(split="asism_tuning_heldout", ids=sharpness),
            "final_eval_image_measured": dict(split="gen_train", ids={**sharpness, sorted(final_eval_ids)[0]: 50.0}),
        }.items():
            try:
                calibrate_blur_threshold(kwargs["ids"], kwargs["split"], set(gen_train["image_id"].astype(str)) | set(kwargs["ids"]), held_out_ids, float(blur_rule.quantile))
                refused[case] = False
            except ReferenceLeakageError:
                refused[case] = True
        report["blur_calibration"]["leakage_refused"] = refused
        check(results, "blur threshold calibrated on the full gen_train only", blur_calibration["calibrated_on_full_gen_train"] and blur_calibration["source_split"] == "gen_train",
              f"threshold={blur_calibration['laplacian_blur_threshold']:.2f} (q={blur_calibration['quantile']}, n={blur_calibration['n_images']})")
        check(results, "blur calibration refuses asism_tuning as source and any final_eval image", all(refused.values()), json.dumps(refused))
        print("  sharpness quantiles (gen_train): " + ", ".join(f"q{float(q):.2f}={v:.1f}" for q, v in blur_calibration["sharpness_quantiles"].items()), flush=True)
        print(f"  blur rejection: calibrated {blur_calibration['overall_rejection_rate']:.2%} vs inherited-40 {report['blur_calibration']['comparison_inherited_threshold_40']['overall_rejection_rate']:.2%}", flush=True)
        for label in DIAGNOSIS_CLASSES:
            entry = blur_calibration["per_class"][label]
            print(f"    {label:6s} n={entry['n']:5d} calibrated={entry['rejection_rate']:6.2%} inherited-40={report['blur_calibration']['comparison_inherited_threshold_40']['per_class'][label]:6.2%} "
                  f"sharpness p5/med/p95={entry['p5']:.1f}/{entry['median']:.1f}/{entry['p95']:.1f}", flush=True)

        iqa_rows = []
        for row in records:
            # The persisted letterbox box, not a default: the committed border_region is
            # "content_box", and compute_iqa_scores refuses to invent a box rather than
            # silently scoring the padding this preprocessing added.
            scores = compute_iqa_scores(
                out_dir / f"{row['image_id']}.jpg", resolved_iqa_cfg,
                content_box=boxes[str(row["image_id"])],
            )
            iqa_rows.append({"image_id": row["image_id"], "dx": row["dx_norm"], **scores})
        iqa = pd.DataFrame(iqa_rows)
        iqa["iqa_is_clipping"] = (iqa["iqa_clipped_low_fraction"] + iqa["iqa_clipped_high_fraction"]) > 0.20
        thresholds = {key: float(resolved_iqa_cfg.signals.iqa[key]) for key in ("blank_std_threshold", "laplacian_blur_threshold", "low_contrast_std", "border_uniform_fraction")}
        continuous = ["iqa_sharpness", "iqa_contrast_std", "iqa_channel_saturation", "iqa_border_uniform_fraction", "iqa_mean_intensity", "iqa_composite"]
        flags = ["iqa_is_near_uniform", "iqa_is_blurry", "iqa_is_low_contrast", "iqa_has_border_artifact", "iqa_is_clipping"]
        report["iqa"] = {"thresholds": thresholds, "clipping_rule": "clipped_low+clipped_high > 0.20", "per_class": {}, "overall": {}}
        for scope, frame in [("ALL", iqa)] + [(label, iqa[iqa["dx"] == label]) for label in DIAGNOSIS_CLASSES]:
            entry = {
                "n": int(len(frame)),
                "distributions": {column: quantiles(frame[column]) for column in continuous},
                "rejection_rates": {flag: float(frame[flag].mean()) for flag in flags},
                "any_flag_rate": float(frame[flags].any(axis=1).mean()),
                "distinct_composites": int(frame["iqa_composite"].nunique()),
            }
            if scope == "ALL":
                report["iqa"]["overall"] = entry
            else:
                report["iqa"]["per_class"][scope] = entry
        check(results, "IQA reads every processed image", bool(iqa["iqa_valid"].all()), f"{int(iqa['iqa_valid'].sum())}/{len(iqa)}")
        check(results, "composite has >= 20 distinct values (Go/No-Go numerical gate)", iqa["iqa_composite"].nunique() >= 20, f"{iqa['iqa_composite'].nunique()} distinct")
        report["iqa"]["saturation_vs_composite_spearman"] = float(iqa[["iqa_channel_saturation", "iqa_composite"]].rank().corr().iloc[0, 1])
        print(f"  thresholds (blur calibrated, others inherited): {thresholds}", flush=True)
        header = f"  {'class':6s} {'n':>5s} " + " ".join(f"{flag.replace('iqa_is_', '').replace('iqa_has_', ''):>15s}" for flag in flags) + f" {'any':>6s}  sharp p5/med/p95        contrast p5/med/p95"
        print(header, flush=True)
        for scope in ["ALL", *DIAGNOSIS_CLASSES]:
            entry = report["iqa"]["overall"] if scope == "ALL" else report["iqa"]["per_class"][scope]
            s, c = entry["distributions"]["iqa_sharpness"], entry["distributions"]["iqa_contrast_std"]
            print(f"  {scope:6s} {entry['n']:5d} " + " ".join(f"{entry['rejection_rates'][flag]:15.1%}" for flag in flags)
                  + f" {entry['any_flag_rate']:6.1%}  {s['p5']:6.1f}/{s['median']:6.1f}/{s['p95']:7.1f}  {c['p5']:5.1f}/{c['median']:5.1f}/{c['p95']:5.1f}", flush=True)

        # 8. explainability plumbing on real geometry ------------------------------------------
        print("\n8. Explainability (random-init DenseNet121 — PLUMBING CHECK, attention values not meaningful)", flush=True)
        import torch

        from scripts.utils.classifier import build_model
        from scripts.utils.ham10000_classifier import LesionRecordDataset

        index_of = {label: position for position, label in enumerate(CLASSIFIER_TARGET_LABELS)}
        cam_records = []
        for label in DIAGNOSIS_CLASSES:
            chosen = [row for row in records if row["dx_norm"] == label][: args.cam_per_class]
            cam_records += [{"image_id": str(row["image_id"]), "image_path": str(out_dir / f"{row['image_id']}.jpg"), "class_index": index_of[label], "dx": label} for row in chosen]
        dataset = LesionRecordDataset(cam_records, args.cam_resolution)
        position = {record["image_id"]: i for i, record in enumerate(cam_records)}
        model = build_model(len(CLASSIFIER_TARGET_LABELS), 0.0, "random", seed=0).eval()
        torch.manual_seed(0)
        cams: dict[str, np.ndarray] = {}

        def cam_for(record):
            cam = gradcam(model, dataset[position[record["image_id"]]]["image"], record["class_index"])
            cams[record["image_id"]] = cam
            return cam

        t0 = time.time()
        rows = pd.DataFrame(explainability_rows(cam_records, cam_for, boxes))
        rows["dx"] = [record["dx"] for record in cam_records]
        cam_seconds = time.time() - t0

        flow_mismatch = [r.image_id for r in rows.itertuples() if json.loads(r.explainability_content_box) != list(boxes[r.image_id])]
        # Independent flow probe on each image's REAL box: a uniform CAM must give peripheral mass
        # exactly 1 - box area. That value differs per aspect ratio, so a dropped/wrong box shows.
        probe_errors = []
        for r in rows.itertuples():
            x0, y0, x1, y1 = boxes[r.image_id]
            shape = cams[r.image_id].shape
            probe_errors.append(abs(peripheral_mass(np.ones(shape), content_box=boxes[r.image_id]) - (1 - (x1 - x0) * (y1 - y0))))
        generic = [peripheral_mass(cams[r.image_id]) for r in rows.itertuples()]
        report["explainability"] = {
            "model": "densenet121 random init (plumbing only)",
            "images": int(len(rows)),
            "seconds": round(cam_seconds, 1),
            "cam_shapes": dict(Counter(rows["explainability_cam_shape"])),
            "content_box_sources": dict(Counter(rows["explainability_content_box_source"])),
            "row_box_mismatches_vs_table": len(flow_mismatch),
            "uniform_cam_probe_max_abs_error": float(max(probe_errors)),
            "zero_cam_rows": int((~np.isfinite(rows["explainability_peripheral_mass"])).sum()),
            "peripheral_mass_exact_box": quantiles(rows["explainability_peripheral_mass"]),
            "peripheral_mass_generic_band_same_cams": quantiles(generic),
            "mean_abs_difference_exact_vs_generic": float(np.mean(np.abs(rows["explainability_peripheral_mass"] - np.asarray(generic)))),
            "focus_area": quantiles(rows["explainability_focus_area"]),
            "raw_plausibility_uncalibrated": quantiles(rows["explainability_raw_plausibility_uncalibrated"]),
        }
        check(results, "every CAM row used the exact persisted box", set(rows["explainability_content_box_source"]) == {EXACT_CONTENT_BOX_SOURCE}, json.dumps(report["explainability"]["content_box_sources"]))
        check(results, "the box on every row equals the preprocessing table", not flow_mismatch, f"mismatches={len(flow_mismatch)}")
        check(results, "uniform-CAM probe: peripheral mass == 1 - box area on every real box", max(probe_errors) < 1e-9, f"max error={max(probe_errors):.2e}")
        print(f"  {len(rows)} CAMs in {cam_seconds:.0f}s, shapes={report['explainability']['cam_shapes']}, all-zero CAMs={report['explainability']['zero_cam_rows']}", flush=True)
        print(f"  peripheral mass exact box: median={report['explainability']['peripheral_mass_exact_box']['median']:.3f}; "
              f"generic band on the same CAMs: median={report['explainability']['peripheral_mass_generic_band_same_cams']['median']:.3f}", flush=True)

        # 9. calibration against gen_train only ------------------------------------------------
        print("\n9. Calibration (reference split: gen_train; final_eval ids used only for exclusion)", flush=True)
        calibration = {}
        calibrated_rows = []
        for label in DIAGNOSIS_CLASSES:
            all_class_rows = rows[rows["dx"] == label]
            # An all-zero CAM has no mass to locate, so its statistics are NaN. Those rows are
            # counted and excluded here rather than silently shrinking the reference below the
            # minimum (build_reference_distribution would refuse it anyway).
            finite = np.isfinite(all_class_rows["explainability_peripheral_mass"]) & np.isfinite(all_class_rows["explainability_focus_area"])
            class_rows = all_class_rows[finite]
            reference_rows, probe_rows = class_rows.iloc[:MIN_REFERENCE_SIZE], class_rows.iloc[MIN_REFERENCE_SIZE:]
            if len(reference_rows) < MIN_REFERENCE_SIZE or probe_rows.empty:
                calibration[label] = {
                    "skipped": f"{len(class_rows)} finite CAM rows < {MIN_REFERENCE_SIZE} reference + 1 probe",
                    "zero_cam_rows": int((~finite).sum()),
                }
                print(f"  {label:6s} SKIPPED: {calibration[label]['skipped']} (zero CAMs: {calibration[label]['zero_cam_rows']})", flush=True)
                continue
            references = {
                statistic: build_reference_distribution(
                    reference_rows[statistic], reference_rows["image_id"], "gen_train", label, statistic, final_eval_ids,
                    cam_model_id="densenet121-random-init-plumbing", cam_model_trained=False,
                )
                for statistic in ("explainability_peripheral_mass", "explainability_focus_area")
            }
            for probe in probe_rows.to_dict("records"):
                calibrated_rows.append({"image_id": probe["image_id"], "dx": label, **calibrate_explainability(probe, label, references)})
            probe_scores = [row["explainability_calibrated_typicality"] for row in calibrated_rows if row["dx"] == label]
            calibration[label] = {
                "reference_split": "gen_train",
                "reference_scientific": False,
                "ties_peripheral_mass_reference": tie_report(references["explainability_peripheral_mass"].values),
                "zero_cam_rows_excluded": int((~finite).sum()),
                "reference_n": references["explainability_peripheral_mass"].size,
                "reference_peripheral_mass": quantiles(references["explainability_peripheral_mass"].values),
                "probe_n": len(probe_rows),
                "probe_calibrated_typicality": [round(v, 4) for v in probe_scores],
            }
            print(f"  {label:6s} ref n={calibration[label]['reference_n']} peripheral median={calibration[label]['reference_peripheral_mass']['median']:.3f} "
                  f"-> probe typicality {calibration[label]['probe_calibrated_typicality']}", flush=True)
        report["calibration"] = calibration

        leakage_refused = {}
        sample_final = sorted(final_eval_ids)[:1]
        for case, kwargs in {
            "final_eval_named": dict(split_name="final_eval_heldout", ids=list(rows["image_id"][:MIN_REFERENCE_SIZE])),
            "final_eval_id_smuggled_as_gen_train": dict(split_name="gen_train", ids=list(rows["image_id"][: MIN_REFERENCE_SIZE - 1]) + sample_final),
        }.items():
            try:
                build_reference_distribution(
                    np.zeros(MIN_REFERENCE_SIZE), kwargs["ids"], kwargs["split_name"], "nv", "explainability_peripheral_mass", final_eval_ids,
                    cam_model_id="densenet121-random-init-plumbing", cam_model_trained=False,
                )
                leakage_refused[case] = False
            except ReferenceLeakageError:
                leakage_refused[case] = True
        try:
            assert_selection_features_allowed(["iqa_composite", "explainability_raw_plausibility_uncalibrated"])
            leakage_refused["raw_composite_as_selection_feature"] = False
        except UncalibratedExplainabilityError:
            leakage_refused["raw_composite_as_selection_feature"] = True
        report["guards"] = leakage_refused
        check(results, "calibration refuses final_eval by name and by smuggled id; selection refuses the raw composite", all(leakage_refused.values()), json.dumps(leakage_refused))
        check(results, "every calibrated row names gen_train as its reference", bool(calibrated_rows) and {r["explainability_reference_split"] for r in calibrated_rows} == {"gen_train"}, f"{len(calibrated_rows)} calibrated probe rows")

        # 10. Stage 2 quotas from gen_train only -----------------------------------------------
        print("\n10. Stage 2 quotas (gen_train.csv only)", flush=True)
        quotas, gen_counts = quotas_from_gen_train(split_dir)
        report["quotas"] = {"source": "gen_train", "gen_train_class_counts": gen_counts, "quotas": quotas, "total": int(sum(quotas.values()))}
        check(results, "rare classes get larger quotas than nv", quotas["df"] > quotas["nv"] and quotas["vasc"] > quotas["nv"], ", ".join(f"{k}:{v}" for k, v in quotas.items()))

        # 11. metric suite (synthetic probabilities) -------------------------------------------
        print("\n11. Metric suite (SYNTHETIC probabilities over the subset labels)", flush=True)
        rng = np.random.default_rng(0)
        y_true = np.array([index_of[row["dx_norm"]] for row in records])
        logits = rng.normal(size=(len(y_true), len(CLASSIFIER_TARGET_LABELS)))
        logits[np.arange(len(y_true)), y_true] += 2.0
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        suite = full_metric_suite(probabilities, y_true)
        check(results, "metric suite runs (accuracy, balanced accuracy, macro-F1, OvR AUROC, confusion matrix)", all(k in suite for k in ("balanced_accuracy", "macro_f1", "macro_auroc_ovr", "confusion_matrix")),
              f"acc={suite['accuracy']:.3f} bal_acc={suite['balanced_accuracy']:.3f} macro_f1={suite['macro_f1']:.3f} macro_auroc={suite['macro_auroc_ovr']:.3f}")
        print(format_confusion_matrix(np.array(suite["confusion_matrix"])), flush=True)

        # 12. final eval untouched -------------------------------------------------------------
        print("\n12. final_eval_heldout untouched", flush=True)
        touched = processed_ids | set(rows["image_id"]) | {r["image_id"] for r in calibrated_rows}
        report["final_eval"] = {
            "final_eval_images": len(final_eval_ids),
            "images_whose_pixels_were_read": len(touched),
            "overlap_with_final_eval": len(touched & final_eval_ids),
        }
        check(results, "no final_eval_heldout image was preprocessed, scored, CAM'd or used as reference", not (touched & final_eval_ids),
              f"{len(touched)} images touched, overlap={len(touched & final_eval_ids)}")
    finally:
        if args.keep_output:
            print(f"\nProcessed images kept at: {work_dir}", flush=True)
        elif args.work_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)

    failed = [r["check"] for r in results if r["status"] == FAIL]
    report["checks"] = results
    report["seconds"] = round(time.time() - started, 1)
    report["status"] = FAIL if failed else PASS
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nReport written to {args.report_json}", flush=True)

    line = "=" * 70
    print(f"\n{line}\n{len(results) - len(failed)}/{len(results)} checks passed in {report['seconds']:.0f}s", flush=True)
    if failed:
        print(f"STATUS: {FAIL}", flush=True)
        for name in failed:
            print(f"  failed: {name}", flush=True)
    else:
        print(f"STATUS: {PASS} — IQA thresholds and CAM attention values above are diagnostic, not validated", flush=True)
    print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""End-to-end smoke test on a SMALL SUBSET OF THE REAL HAM10000 IMAGES.

This is the step that was missing from the CheXpert run and cost a day: everything below is
exercised on genuine dataset images, in the real order, before any full-cohort or GPU run is
started. It uses real pixels precisely because the failures it is looking for — a greyscale
conversion, a colour penalty, attention landing on padding — are invisible to a synthetic fixture.

Checks, in order:
  1. metadata loads and validates
  2. a lesion-grouped subset is drawn, N images per class, with NO lesion split across classes
  3. preprocessing produces RGB at the configured resolution (asserted by decoding the output)
  4. colour survives: output saturation is compared against the source
  5. captions are built for every image and name the right diagnosis
  6. IQA accepts normal colour images and does not penalise them for being colourful
  7. explainability statistics discriminate centred / diffuse / edge attention
  8. Stage 2 quotas oversample the rare classes
  9. the metric suite runs on a synthetic probability matrix and reports argmax-based numbers

No GPU, no model download, no training: this is the cheap gate that runs before any of that.

Usage:
    python scripts/data/ham10000/03_smoke_subset.py --per-class 5
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.asism.ham10000_signals import (  # noqa: E402
    compute_explainability_statistics,
    compute_iqa_scores,
)
from scripts.generate.ham10000_recipes import class_counts, rarity_quotas  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    DIAGNOSIS_CLASSES,
    build_caption,
    normalize_diagnosis,
    validate_metadata,
)
from scripts.utils.ham10000_metrics import format_confusion_matrix, full_metric_suite  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def check(results: list[tuple[str, str, str]], name: str, ok: bool, detail: str = "") -> bool:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=5, help="Real images to draw per diagnosis")
    parser.add_argument("--keep-output", action="store_true", help="Keep the temporary processed images")
    args = parser.parse_args()

    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    stage1_cfg = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3_cfg = load_named_config("stage3_asism.yaml", "stage3")

    raw_dir = Path(splits_cfg.paths.raw_dir)
    metadata_path = raw_dir / str(dataset_cfg.source.metadata_filename)
    results: list[tuple[str, str, str]] = []

    print("\n" + "=" * 62 + "\nHAM10000 SMOKE TEST — real images, no GPU\n" + "=" * 62, flush=True)

    # 1. metadata ------------------------------------------------------------------------------
    print("\n1. Metadata", flush=True)
    if not metadata_path.is_file():
        print(f"  [{FAIL}] metadata not found at {metadata_path}", flush=True)
        print("\nRun: python scripts/data/ham10000/00_download_dataset.py", flush=True)
        return 1
    frame = pd.read_csv(metadata_path)
    validate_metadata(frame, str(dataset_cfg.schema.diagnosis_column), str(dataset_cfg.schema.group_column))
    check(results, "metadata loads and validates", True, f"{len(frame)} rows, {frame['lesion_id'].nunique()} lesions")

    # 2. lesion-grouped subset ------------------------------------------------------------------
    print("\n2. Subset selection (lesion-grouped)", flush=True)
    frame["dx_norm"] = frame[str(dataset_cfg.schema.diagnosis_column)].map(normalize_diagnosis)
    chosen = []
    for label in DIAGNOSIS_CLASSES:
        lesions = frame.loc[frame["dx_norm"] == label, "lesion_id"].drop_duplicates()
        picked = set(lesions.head(args.per_class))
        chosen.append(frame[frame["lesion_id"].isin(picked)])
    subset = pd.concat(chosen).drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    per_lesion_classes = subset.groupby("lesion_id")["dx_norm"].nunique()
    check(
        results,
        "every lesion carries exactly one diagnosis",
        bool((per_lesion_classes == 1).all()),
        f"{len(subset)} images from {subset['lesion_id'].nunique()} lesions",
    )
    check(
        results,
        "all 7 classes represented in the subset",
        set(subset["dx_norm"]) == set(DIAGNOSIS_CLASSES),
        ", ".join(f"{label}:{int((subset['dx_norm'] == label).sum())}" for label in DIAGNOSIS_CLASSES),
    )

    # 3-4. preprocessing: RGB and colour survival ----------------------------------------------
    print("\n3. Preprocessing (RGB, letterbox)", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ham_preprocess", Path(__file__).resolve().parent / "02_preprocess_images.py"
    )
    preprocess = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocess)

    resolution = int(stage1_cfg.data.resolution)
    pad_colour = tuple(int(v) for v in stage1_cfg.data.pad_colour)
    image_directories = [str(name) for name in dataset_cfg.source.image_directories]
    work_dir = Path(tempfile.mkdtemp(prefix="ham_smoke_"))

    processed, source_saturation, output_saturation = [], [], []
    try:
        for image_id in subset["image_id"].astype(str):
            source = preprocess.resolve_source(raw_dir, image_id, image_directories)
            if source is None:
                continue
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                source_array = np.asarray(rgb, dtype=np.float32)
                source_saturation.append(float(np.mean(source_array.max(axis=2) - source_array.min(axis=2))))
                out_path = work_dir / f"{image_id}.jpg"
                preprocess.atomic_save_jpeg(
                    preprocess.letterbox_resize_rgb(rgb, resolution, pad_colour),
                    out_path,
                    int(stage1_cfg.data.jpeg_quality),
                )
            processed.append(out_path)

        check(results, "every subset image was preprocessed", len(processed) == len(subset), f"{len(processed)}/{len(subset)}")

        modes, sizes = set(), set()
        for path in processed:
            with Image.open(path) as image:
                modes.add(image.mode)
                sizes.add(image.size)
                array = np.asarray(image.convert("RGB"), dtype=np.float32)
                output_saturation.append(float(np.mean(array.max(axis=2) - array.min(axis=2))))

        check(results, "outputs are RGB, never greyscale", modes == {"RGB"}, f"modes={sorted(modes)}")
        check(results, "outputs are the configured square size", sizes == {(resolution, resolution)}, f"{sorted(sizes)}")

        print("\n4. Colour survival", flush=True)
        mean_source, mean_output = float(np.mean(source_saturation)), float(np.mean(output_saturation))
        check(
            results,
            "colour survives preprocessing (output saturation close to source)",
            mean_output > 0.5 * mean_source and mean_output > 1.0,
            f"source={mean_source:.2f} output={mean_output:.2f}",
        )

        # 5. captions ---------------------------------------------------------------------------
        print("\n5. Captions", flush=True)
        captions = [
            build_caption(row) for row in subset.rename(columns={"dx_norm": "dx"}).to_dict("records")
        ]
        from scripts.utils.ham10000 import DIAGNOSIS_PHRASE

        names_match = all(
            DIAGNOSIS_PHRASE[row["dx_norm"]].split()[-1] in caption
            for caption, row in zip(captions, subset.to_dict("records"))
        )
        check(results, "one caption per image, naming the right diagnosis", names_match, captions[0])
        check(results, "no caption contains the literal token 'unknown'", not any("unknown" in c.lower() for c in captions))

        # 6. IQA --------------------------------------------------------------------------------
        print("\n6. IQA on real colour images", flush=True)
        scores = [compute_iqa_scores(path, stage3_cfg) for path in processed]
        valid = [s for s in scores if s.get("iqa_valid")]
        composites = [s["iqa_composite"] for s in valid]
        saturations = [s["iqa_channel_saturation"] for s in valid]

        check(results, "IQA reads every processed image", len(valid) == len(processed), f"{len(valid)}/{len(processed)}")
        check(
            results,
            "colourful images are NOT penalised (no colour term in the composite)",
            float(np.min(composites)) > 0.5,
            f"min={np.min(composites):.3f} mean={np.mean(composites):.3f} (colour would have cost 0.10 each)",
        )
        check(
            results,
            "saturation is reported, not penalised",
            float(np.mean(saturations)) > 1.0 and np.corrcoef(saturations, composites)[0, 1] > -0.5,
            f"mean saturation={np.mean(saturations):.2f}",
        )
        check(
            results,
            "composite has enough distinct values for the Go/No-Go gate",
            len(set(composites)) >= min(20, len(composites)),
            f"{len(set(composites))} distinct / {len(composites)} images",
        )

        # 7. explainability ---------------------------------------------------------------------
        print("\n7. Explainability statistics", flush=True)
        size = 32
        yy, xx = np.mgrid[0:size, 0:size]
        centred = np.exp(-(((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * 3.0**2)))
        diffuse = np.ones((size, size))
        edge = np.zeros((size, size))
        edge[:3, :] = edge[-3:, :] = edge[:, :3] = edge[:, -3:] = 1.0

        centred_stats = compute_explainability_statistics(centred)
        diffuse_stats = compute_explainability_statistics(diffuse)
        edge_stats = compute_explainability_statistics(edge)

        check(
            results,
            "compact central attention scores above diffuse attention",
            centred_stats["explainability_plausibility"] > diffuse_stats["explainability_plausibility"],
            f"centred={centred_stats['explainability_plausibility']:.3f} diffuse={diffuse_stats['explainability_plausibility']:.3f}",
        )
        check(
            results,
            "edge-driven attention is flagged by peripheral mass",
            edge_stats["explainability_peripheral_mass"] > 0.5 > centred_stats["explainability_peripheral_mass"],
            f"edge={edge_stats['explainability_peripheral_mass']:.3f} centred={centred_stats['explainability_peripheral_mass']:.3f}",
        )

        # 8. Stage 2 quotas ---------------------------------------------------------------------
        print("\n8. Stage 2 quotas", flush=True)
        counts = class_counts(frame.rename(columns={"dx_norm": "dx"}))
        quotas = rarity_quotas(counts)
        check(
            results,
            "rare classes get larger synthetic quotas than the dominant class",
            quotas["df"] > quotas["nv"] and quotas["vasc"] > quotas["nv"],
            ", ".join(f"{label}:{quotas[label]}" for label in DIAGNOSIS_CLASSES),
        )

        # 9. metrics ----------------------------------------------------------------------------
        print("\n9. Metric suite", flush=True)
        rng = np.random.default_rng(0)
        y_true = np.array([CLASSIFIER_TARGET_LABELS.index(label) for label in subset["dx_norm"]])
        logits = rng.normal(size=(len(y_true), len(CLASSIFIER_TARGET_LABELS)))
        logits[np.arange(len(y_true)), y_true] += 2.0  # a deliberately decent classifier
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        suite = full_metric_suite(probabilities, y_true)

        check(
            results,
            "metric suite reports balanced accuracy, macro-F1 and a confusion matrix",
            all(key in suite for key in ("balanced_accuracy", "macro_f1", "confusion_matrix")),
            f"balanced_acc={suite['balanced_accuracy']:.3f} macro_f1={suite['macro_f1']:.3f}",
        )
        check(
            results,
            "softmax rows sum to 1",
            bool(np.allclose(probabilities.sum(axis=1), 1.0)),
        )
        print("\n" + format_confusion_matrix(np.array(suite["confusion_matrix"])), flush=True)

    finally:
        if args.keep_output:
            print(f"\nProcessed images kept at: {work_dir}", flush=True)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)

    failed = [name for status, name, _ in results if status == FAIL]
    line = "=" * 62
    print(f"\n{line}", flush=True)
    print(f"{len(results) - len(failed)}/{len(results)} checks passed", flush=True)
    if failed:
        print(f"STATUS: {FAIL}", flush=True)
        for name in failed:
            print(f"  failed: {name}", flush=True)
    else:
        print(f"STATUS: {PASS} — the data layer is safe to run at full scale", flush=True)
    print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the Grad-CAM explainability reference from REAL gen_train images.

This is the step docs/ham10000_gradcam_calibration.md leaves as "the exact future step", and it is
what turns the explainability signal from plumbing into a measurement: without this artifact every
candidate's `explainability_calibrated_typicality` is NaN and the feature is refused at selection.

WHAT IT PRODUCES
    <outputs_dir>/<namespace>/explainability_reference.json
        per class, the gen_train values of each statistic with their image_ids, the cam_model_id
        they came from, the tie report, and the split hashes — everything a later run needs to
        verify the reference it is calibrating against rather than trust a filename.

WHAT IT REFUSES
    * a CAM model that saw synthetic images, or was trained/selected on gen_train itself
      (cam_model_identity — the reference would be the model's own memorised training attention);
    * any reference image that appears in final_eval_heldout, checked by id, not only by name;
    * a class with fewer than `min_reference_size` finite values — reported, never padded;
    * a missing content box for any gen_train image. The generic border band is a different
      measurement, and silently mixing the two inside one reference would make the calibrated
      typicality of a candidate depend on which images happened to have boxes.

Usage:
    python scripts/asism/ham10000_00_build_cam_reference.py --namespace ham-stratified-v1
    python scripts/asism/ham10000_00_build_cam_reference.py --namespace ham-stratified-v1         --config-key auxiliary_classifier_v2

`--config-key` picks the auxiliary classifier the CAMs come from. The default is v1, which reads and
writes exactly where it always did. v2 reads its own checkpoint and writes under
paths.aux_v2_outputs_dir, so the v1 reference is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_explainability import explainability_rows, gradcam  # noqa: E402
from scripts.asism.ham10000_signals import (  # noqa: E402
    REFERENCE_STATISTICS,
    build_reference_distribution,
    cam_model_identity,
    load_final_eval_image_ids,
    reference_artifact_payload,
    tie_report,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_classifier import records_from_split  # noqa: E402
from scripts.utils.ham10000_geometry import load_content_boxes  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402

REFERENCE_SPLIT = "gen_train"
V1_CONFIG_KEY = "auxiliary_classifier"
V2_CONFIG_KEY = "auxiliary_classifier_v2"
CONFIG_KEYS = (V1_CONFIG_KEY, V2_CONFIG_KEY)


def auxiliary_outputs_dir(stage3, config_key: str = V1_CONFIG_KEY) -> Path:
    """The root the reference and the classifier-dependent signals of this CAM model live under.

    v1 keeps paths.outputs_dir. v2 gets its own root, and is refused if that root is the v1 one:
    the two references name different models, and one written over the other would leave the v1
    signals calibrated against a reference that no longer describes them.
    """
    if config_key == V1_CONFIG_KEY:
        return Path(stage3.paths.outputs_dir)
    if config_key == V2_CONFIG_KEY:
        root = Path(stage3.paths.aux_v2_outputs_dir)
        if root.resolve() == Path(stage3.paths.outputs_dir).resolve():
            raise SystemExit(
                "paths.aux_v2_outputs_dir is the v1 outputs_dir; the v2 artifacts would overwrite v1's."
            )
        return root
    raise SystemExit(f"Unknown auxiliary classifier config key {config_key!r}; expected one of {CONFIG_KEYS}.")


def load_cam_model(checkpoint_dir: Path, namespace: str, resolution_hint: int | None = None):
    """The frozen auxiliary classifier, plus the identity its own manifest was minted with.

    The identity is recomputed here rather than read from the manifest: the manifest records what
    the training run believed, and this run is about to build a reference that names a model by the
    hash of the file it actually loaded.
    """
    from scripts.utils.classifier import build_model, require_torch

    torch = require_torch()
    out_dir = Path(checkpoint_dir) / namespace
    model_path = out_dir / "model.pt"
    manifest_path = out_dir / "run_manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: no auxiliary classifier at {out_dir}.\n"
            "Run: python scripts/classify/ham10000_train_auxiliary_classifier.py "
            f"--namespace {namespace}"
        )
    manifest = read_json(manifest_path)
    identity = cam_model_identity(manifest, model_path, reference_split=REFERENCE_SPLIT)

    model_cfg = manifest["model"]
    resolution = int(model_cfg.get("resolution", resolution_hint or 512))
    model = build_model(
        len(CLASSIFIER_TARGET_LABELS),
        float(model_cfg["dropout_p"]),
        str(model_cfg["pretrained_source"]),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, resolution, identity, manifest


def cam_rows(model, records: list[dict], resolution: int, content_boxes: dict, explainability_cfg, device: str):
    """Grad-CAM statistics for every reference image, for its TRUE class.

    True class, not the predicted one: the reference answers "where does this model look at real
    images of class c", and using the prediction would build the reference out of the model's
    correct cases only — an optimistic reference that candidates would then be scored against.
    """
    from scripts.utils.ham10000_classifier import LesionRecordDataset

    dataset = LesionRecordDataset(records, resolution)
    model = model.to(device)
    index_by_id = {str(record["image_id"]): position for position, record in enumerate(records)}

    def cam_for_record(record):
        item = dataset[index_by_id[str(record["image_id"])]]
        return gradcam(model, item["image"].to(device), int(record["class_index"]))

    return explainability_rows(
        records,
        cam_for_record,
        content_boxes,
        border_fraction=float(explainability_cfg.border_fraction),
        mass_fraction=float(explainability_cfg.mass_fraction),
    )


def build(
    namespace: str, device: str | None = None, limit: int | None = None, config_key: str = V1_CONFIG_KEY
) -> dict:
    from scripts.utils.classifier import require_torch

    torch = require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    outputs_root = auxiliary_outputs_dir(stage3, config_key)

    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage1.paths.images_dir) / namespace
    split_path = split_dir / f"{REFERENCE_SPLIT}.csv"
    if not split_path.is_file():
        raise SystemExit(f"UPSTREAM GATE: missing {split_path}; build and freeze the splits first.")

    frame = pd.read_csv(split_path)
    records = records_from_split(frame, images_root / REFERENCE_SPLIT)
    if not records:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed {REFERENCE_SPLIT} images under "
            f"{images_root / REFERENCE_SPLIT}; run scripts/data/ham10000/02_preprocess_images.py"
        )
    if limit is not None:
        records = records[: int(limit)]

    box_path = images_root / f"{REFERENCE_SPLIT}_content_boxes.csv"
    if not box_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: {box_path} is missing. The reference must be measured with each image's "
            "exact letterbox content box; the generic band is a different measurement and cannot "
            "stand in for it."
        )
    content_boxes = load_content_boxes(box_path)

    model, resolution, identity, model_manifest = load_cam_model(
        Path(stage3[config_key].checkpoint_dir), namespace
    )
    print(
        f"CAM model {identity['cam_model_id'][:28]}... trained on "
        f"{identity['trained_on_split']}, scoring {len(records)} {REFERENCE_SPLIT} images on {device}",
        flush=True,
    )

    rows = cam_rows(model, records, resolution, content_boxes, stage3.signals.explainability, device)
    row_frame = pd.DataFrame(rows)
    row_frame["dx"] = [CLASSIFIER_TARGET_LABELS[int(index)] for index in row_frame["class_index"]]

    final_eval_ids = load_final_eval_image_ids(split_dir)
    min_size = int(stage3.signals.explainability.min_reference_size)

    references, ties, undersized = {}, {}, {}
    for diagnosis, group in row_frame.groupby("dx", sort=False):
        ties[diagnosis] = {
            statistic: tie_report(group[statistic]) for statistic in REFERENCE_STATISTICS
        }
        for statistic in REFERENCE_STATISTICS:
            finite = int(np.isfinite(group[statistic].to_numpy(dtype=float)).sum())
            try:
                references[(statistic, diagnosis)] = build_reference_distribution(
                    group[statistic].to_numpy(dtype=float),
                    group["image_id"].astype(str).tolist(),
                    REFERENCE_SPLIT,
                    diagnosis,
                    statistic,
                    final_eval_ids,
                    cam_model_id=identity["cam_model_id"],
                    cam_model_trained=bool(identity["cam_model_trained"]),
                    min_size=min_size,
                )
            except ValueError as exc:
                # A class too thin to calibrate is REPORTED and left absent, never padded or
                # back-filled from another class: candidates of that class then carry NaN
                # typicality, which the selection guard already refuses, instead of a number
                # computed against a different lesion's attention.
                undersized[f"{statistic}/{diagnosis}"] = {"finite_values": finite, "reason": str(exc)}

    if not references:
        raise SystemExit(
            "No class produced a usable reference distribution. Nothing downstream can calibrate; "
            f"details: {json.dumps(undersized, indent=2)}"
        )

    out_dir = outputs_root / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = reference_artifact_payload(
        references,
        {
            "split_namespace": namespace,
            "reference_split": REFERENCE_SPLIT,
            "reference_images_scored": len(records),
            "resolution": resolution,
            "content_box_source_file": str(box_path),
            "cam_model": identity,
            "auxiliary_config_key": config_key,
            "cam_model_run_manifest_git_commit": model_manifest.get("git_commit_hash"),
            "split_csv_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "final_eval_heldout_id_count": len(final_eval_ids),
            "explainability_config": {
                "border_fraction": float(stage3.signals.explainability.border_fraction),
                "mass_fraction": float(stage3.signals.explainability.mass_fraction),
                "min_reference_size": min_size,
            },
            "tie_reports": ties,
            "undersized_classes": undersized,
            "git_commit_hash": get_git_commit_hash(),
        },
    )
    out_path = out_dir / "explainability_reference.json"
    write_json(out_path, payload)

    # The raw per-image rows travel with the artifact: the reference values alone cannot be audited
    # back to the images they came from once the run is over.
    row_frame.to_csv(out_dir / "explainability_reference_rows.csv", index=False)
    return {"path": str(out_path), "payload": payload, "undersized": undersized}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--config-key", choices=CONFIG_KEYS, default=V1_CONFIG_KEY,
        help="Which auxiliary classifier the CAMs come from. Default: v1.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N reference images. Smoke runs only: a truncated reference is "
             "still written and is still used by anything that reads it.",
    )
    args = parser.parse_args()

    result = build(args.namespace, device=args.device, limit=args.limit, config_key=args.config_key)
    payload = result["payload"]
    print(json.dumps({
        "artifact": result["path"],
        "cam_model_id": payload["cam_model_id"],
        "classes_with_a_reference": sorted({key["diagnosis"] for key in payload["references"]}),
        "undersized": result["undersized"],
        "peripheral_mass_tie_fraction": {
            diagnosis: report["explainability_peripheral_mass"]["largest_tie_group_fraction"]
            for diagnosis, report in payload["tie_reports"].items()
        },
    }, indent=2))
    if result["undersized"]:
        print(
            "\nNOTE: the classes above have no reference. Their candidates will carry NaN "
            "explainability typicality and the selection guard will refuse that feature for them.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

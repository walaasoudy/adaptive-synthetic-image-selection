#!/usr/bin/env python3
"""Stage 3a — score every synthetic candidate with the five ASISM signals.

FIVE INDEPENDENT ARTIFACTS, one per signal:
    similarity_scores.parquet   iqa_scores.parquet        uncertainty_scores.parquet
    agreement_scores.parquet    explainability_scores.parquet

Separate files rather than one table, so a signal can be recomputed, audited, or EXCLUDED by the
Go/No-Go gate without disturbing the others, and so an excluded-but-valid signal's evidence survives
for the write-up. Each artifact carries a provenance sidecar naming the candidate manifest, the
models and the calibration artifacts it was produced from — a score whose origin cannot be
reconstructed cannot be defended later.

COST ORDER. IQA needs no model and runs on any machine. Similarity needs the DINOv2 encoder;
uncertainty, agreement and explainability need the frozen auxiliary classifier, and explainability
additionally needs the Grad-CAM reference. Each fails at that upstream gate with the command that
produces what is missing, instead of substituting a default.

Usage:
    python scripts/asism/ham10000_01_compute_signals.py --namespace ham-stratified-v1 --signal all
    python scripts/asism/ham10000_01_compute_signals.py --namespace ham-stratified-v1 --signal iqa
    python scripts/asism/ham10000_01_compute_signals.py --namespace ham-stratified-v1         --config-key auxiliary_classifier_v2 --signal uncertainty --signal agreement --signal explainability

`--config-key auxiliary_classifier_v2` recomputes only the three signals that depend on the auxiliary
classifier, with the v2 model, reading the v2 Grad-CAM reference and writing under
paths.aux_v2_outputs_dir. Similarity and IQA do not depend on that model and are refused there, so
they cannot be recomputed by accident. The default (v1) reads and writes exactly as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_00_build_cam_reference import (  # noqa: E402
    CONFIG_KEYS,
    V1_CONFIG_KEY,
    auxiliary_outputs_dir,
    load_cam_model,
)
from scripts.asism.ham10000_explainability import explainability_rows, gradcam  # noqa: E402
from scripts.asism.ham10000_signals import (  # noqa: E402
    calibrate_explainability,
    compute_agreement_scores,
    compute_iqa_scores,
    compute_similarity_scores,
    compute_uncertainty_scores,
    load_final_eval_image_ids,
    references_from_artifact,
    resolve_iqa_config,
)
# Dataset-agnostic writer: an atomic parquet write plus a provenance sidecar. Nothing in it is
# CheXpert-specific, and duplicating it would leave two artifact formats to keep in step.
from scripts.asism.signals import write_score_artifact  # noqa: E402
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_classifier import (  # noqa: E402
    LesionRecordDataset,
    predict_probability_passes,
)
from scripts.utils.ham10000_geometry import load_content_boxes, validate_content_box  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file  # noqa: E402

SIGNALS = ("iqa", "similarity", "uncertainty", "agreement", "explainability")
REFERENCE_SPLIT = "gen_train"
# The signals computed from the auxiliary classifier. The only ones a non-v1 config key may run.
CLASSIFIER_SIGNALS = ("uncertainty", "agreement", "explainability")


class UpstreamGate(SystemExit):
    """A required input is missing. The message always names the command that produces it."""


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def load_candidates(stage2_root: Path, namespace: str, limit: int | None = None) -> pd.DataFrame:
    path = Path(stage2_root) / namespace / "all_candidates.csv"
    if not path.is_file():
        raise UpstreamGate(
            f"UPSTREAM GATE: no candidate manifest at {path}.\n"
            "Run: python scripts/generate/ham10000_generate_synthetic_images.py "
            f"--namespace {namespace} --mode full"
        )
    frame = pd.read_csv(path)
    frame["dx"] = [normalize_diagnosis(value) for value in frame["dx"]]
    frame["image_id"] = frame["image_id"].astype(str)
    if frame["image_id"].duplicated().any():
        raise UpstreamGate(f"{path} contains duplicate image_ids; the candidate pool is not well-formed")
    missing = [row for row in frame["image_path"] if not Path(row).is_file()]
    if missing:
        raise UpstreamGate(
            f"{len(missing)} candidate image(s) listed in {path} are absent on disk (e.g. {missing[:2]}); "
            "scoring a partial pool would silently change which images selection could choose from"
        )
    return frame.head(int(limit)) if limit else frame


def candidate_records(frame: pd.DataFrame) -> list[dict]:
    index_of = {label: position for position, label in enumerate(CLASSIFIER_TARGET_LABELS)}
    return [
        {
            "image_id": str(row["image_id"]),
            "image_path": str(row["image_path"]),
            "class_index": index_of[row["dx"]],
        }
        for row in frame.to_dict("records")
    ]


def candidate_content_boxes(frame: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    """The exact letterbox box each candidate was standardised onto, from the generation manifest.

    Required for every candidate. The generic band is a different measurement and the reference was
    built with exact boxes, so a fallback here would calibrate one quantity against another.

    Stage 2 writes the boxes to content_boxes.csv beside the candidate images rather than as a column
    of all_candidates.csv; when the column is absent they are read from that table, which holds the
    same exact boxes.
    """
    if "content_box" not in frame.columns:
        return _content_boxes_from_candidate_tables(frame)
    boxes = {}
    for row in frame.to_dict("records"):
        raw = row.get("content_box")
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            raise UpstreamGate(
                f"candidate {row['image_id']} has no content_box in the generation manifest; "
                "re-run scripts/generate/ham10000_standardize_generated.py"
            )
        value = json.loads(raw) if isinstance(raw, str) else raw
        boxes[str(row["image_id"])] = validate_content_box(value)
    return boxes


def _content_boxes_from_candidate_tables(frame: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    tables: dict[Path, dict[str, tuple[float, float, float, float]]] = {}
    boxes = {}
    for row in frame.to_dict("records"):
        image_id = str(row["image_id"])
        table_path = Path(row["image_path"]).parent / "content_boxes.csv"
        if table_path not in tables:
            try:
                tables[table_path] = load_content_boxes(table_path)
            except (FileNotFoundError, ValueError) as exc:
                raise UpstreamGate(
                    f"candidate {image_id} has no content_box in the generation manifest ({exc}); "
                    "re-run scripts/generate/ham10000_standardize_generated.py"
                ) from exc
        if image_id not in tables[table_path]:
            raise UpstreamGate(
                f"candidate {image_id} has no content_box in the generation manifest ({table_path}); "
                "re-run scripts/generate/ham10000_standardize_generated.py"
            )
        boxes[image_id] = tables[table_path][image_id]
    return boxes


def base_provenance(namespace: str, candidates_path: Path, frame: pd.DataFrame) -> dict:
    return {
        "dataset": "ham10000",
        "split_namespace": namespace,
        "candidates_csv": str(candidates_path),
        "candidates_csv_sha256": sha256_file(candidates_path),
        "n_candidates_scored": len(frame),
        "git_commit_hash": get_git_commit_hash(),
    }


# ----------------------------------------------------------------------------------------------
# IQA — no model, no GPU
# ----------------------------------------------------------------------------------------------


def run_iqa(frame, stage3, out_dir: Path, provenance: dict) -> None:
    calibration_dir = Path(stage3.paths.outputs_dir) / provenance["split_namespace"]
    blur_path = calibration_dir / str(stage3.signals.iqa.blur_calibration.artifact_filename)
    if not blur_path.is_file():
        raise UpstreamGate(
            f"UPSTREAM GATE: no blur calibration at {blur_path}.\n"
            "The inherited constant flagged a third of real, clinically accepted images as blurry, "
            "so there is no fallback. Run: python scripts/asism/ham10000_calibrate_iqa_blur.py "
            f"--namespace {provenance['split_namespace']}"
        )
    blur_calibration = read_json(blur_path)

    border_calibration = None
    border_cfg = stage3.signals.iqa.border_calibration
    if bool(border_cfg.enabled):
        border_path = calibration_dir / str(border_cfg.artifact_filename)
        if not border_path.is_file():
            raise UpstreamGate(
                f"UPSTREAM GATE: border calibration is enabled but {border_path} is missing.\n"
                "Run: python scripts/asism/ham10000_calibrate_iqa_border.py "
                f"--namespace {provenance['split_namespace']}"
            )
        border_calibration = read_json(border_path)

    resolved = resolve_iqa_config(stage3, blur_calibration, border_calibration)
    boxes = candidate_content_boxes(frame)

    rows = []
    for row in tqdm(frame.to_dict("records"), desc="iqa", unit="img"):
        image_id = str(row["image_id"])
        scores = compute_iqa_scores(
            Path(row["image_path"]), resolved, content_box=boxes[image_id]
        )
        rows.append({"image_id": image_id, **scores})

    scores_frame = pd.DataFrame(rows)
    write_score_artifact(
        scores_frame,
        out_dir / "iqa_scores.parquet",
        "iqa",
        {
            **provenance,
            "blur_calibration_artifact": str(blur_path),
            "blur_calibration_sha256": sha256_file(blur_path),
            "laplacian_blur_threshold": float(resolved.signals.iqa.laplacian_blur_threshold),
            "border_region": str(resolved.signals.iqa.border_region),
            "border_calibration_enabled": bool(border_cfg.enabled),
        },
    )
    valid = int(scores_frame["iqa_valid"].sum()) if "iqa_valid" in scores_frame else len(scores_frame)
    print(f"iqa: {len(scores_frame)} rows, {valid} valid", flush=True)


# ----------------------------------------------------------------------------------------------
# Similarity — DINOv2 against real gen_train images of the same class
# ----------------------------------------------------------------------------------------------


def load_encoder(similarity_cfg, device: str):
    """The pinned DINOv2 revision, loaded into the explicitly named architecture.

    `pretrained=True` would let timm resolve whatever the default checkpoint currently is; a moved
    tag would change every embedding, and with it every similarity score and selection decision,
    with nothing in the artifacts recording that anything had changed.
    """
    try:
        import timm
    except ImportError as exc:
        raise UpstreamGate(
            f"UPSTREAM GATE: the similarity encoder needs `timm` ({exc}). Install it: pip install timm"
        ) from exc
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors

    weights_path = hf_hub_download(
        repo_id=str(similarity_cfg.weights_repo_id),
        filename=str(similarity_cfg.weights_filename),
        revision=str(similarity_cfg.weights_revision),
    )
    encoder = timm.create_model(str(similarity_cfg.encoder), pretrained=False, num_classes=0)
    incompatible = encoder.load_state_dict(load_safetensors(weights_path), strict=False)
    disallowed_missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
    if incompatible.unexpected_keys or disallowed_missing:
        raise RuntimeError(
            "Pinned DINOv2 checkpoint is incompatible with the configured architecture: "
            f"missing={disallowed_missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    encoder = encoder.eval().to(device)
    data_config = timm.data.resolve_model_data_config(encoder)
    return encoder, timm.data.create_transform(**data_config, is_training=False), weights_path


def embed_images(encoder, transform, paths: list[Path], batch_size: int, device: str) -> np.ndarray:
    import torch
    from PIL import Image

    outputs = []
    for start in tqdm(range(0, len(paths), batch_size), desc="embed", unit="batch"):
        batch = []
        for path in paths[start : start + batch_size]:
            with Image.open(path) as image:
                batch.append(transform(image.convert("RGB")))
        with torch.no_grad():
            outputs.append(encoder(torch.stack(batch).to(device)).cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 0))


def sample_reference_images(split_dir: Path, images_root: Path, per_class: int, seed: int = 42):
    """A per-class sample of real gen_train images, drawn reproducibly.

    Flat per class, not proportional: nv is two thirds of HAM10000 and would otherwise supply most
    of every pool, leaving the rare classes — the ones the synthetic data exists for — compared
    against the thinnest references.
    """
    split_path = Path(split_dir) / f"{REFERENCE_SPLIT}.csv"
    if not split_path.is_file():
        raise UpstreamGate(f"UPSTREAM GATE: missing {split_path}; build and freeze the splits first.")
    frame = pd.read_csv(split_path)
    frame["dx"] = [normalize_diagnosis(value) for value in frame["dx"]]

    rng = np.random.default_rng(seed)
    paths, diagnoses, image_ids = [], [], []
    per_class_counts = {}
    for diagnosis, group in frame.groupby("dx", sort=True):
        available = [
            (str(image_id), Path(images_root) / REFERENCE_SPLIT / f"{image_id}.jpg")
            for image_id in group["image_id"].astype(str)
        ]
        available = [(image_id, path) for image_id, path in available if path.is_file()]
        if not available:
            per_class_counts[diagnosis] = 0
            continue
        take = min(int(per_class), len(available))
        chosen = rng.choice(len(available), size=take, replace=False)
        for index in sorted(chosen):
            image_id, path = available[index]
            image_ids.append(image_id)
            paths.append(path)
            diagnoses.append(diagnosis)
        per_class_counts[diagnosis] = take

    if not paths:
        raise UpstreamGate(
            f"UPSTREAM GATE: no preprocessed {REFERENCE_SPLIT} images under {images_root}; "
            "run scripts/data/ham10000/02_preprocess_images.py"
        )
    return paths, diagnoses, image_ids, per_class_counts


def run_similarity(frame, stage1, stage3, splits_cfg, out_dir: Path, provenance: dict, device: str) -> None:
    similarity_cfg = stage3.signals.similarity
    namespace = provenance["split_namespace"]
    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage1.paths.images_dir) / namespace

    reference_paths, reference_diagnoses, reference_ids, per_class_counts = sample_reference_images(
        split_dir, images_root, int(similarity_cfg.reference_sample_per_class)
    )
    # gen_train is disjoint from final_eval_heldout by construction, but "by construction" is what
    # every leak was before it happened. Check the ids.
    leaked = sorted(set(reference_ids) & load_final_eval_image_ids(split_dir))
    if leaked:
        raise UpstreamGate(
            f"{len(leaked)} similarity reference image(s) are in final_eval_heldout (e.g. {leaked[:3]})"
        )

    encoder, transform, weights_path = load_encoder(similarity_cfg, device)
    print(
        f"similarity: embedding {len(frame)} candidates and {len(reference_paths)} real references "
        f"on {device}",
        flush=True,
    )
    batch_size = int(similarity_cfg.batch_size)
    candidate_embeddings = embed_images(
        encoder, transform, [Path(path) for path in frame["image_path"]], batch_size, device
    )
    reference_embeddings = embed_images(encoder, transform, reference_paths, batch_size, device)

    scores = compute_similarity_scores(
        candidate_embeddings,
        reference_embeddings,
        reference_diagnoses,
        list(frame["dx"]),
        stage3,
    )
    scores.insert(0, "image_id", frame["image_id"].values)
    write_score_artifact(
        scores,
        out_dir / "similarity_scores.parquet",
        "similarity",
        {
            **provenance,
            "encoder": str(similarity_cfg.encoder),
            "weights_repo_id": str(similarity_cfg.weights_repo_id),
            "weights_revision": str(similarity_cfg.weights_revision),
            "weights_file_sha256": sha256_file(Path(weights_path)),
            "reference_split": REFERENCE_SPLIT,
            "reference_images_per_class": per_class_counts,
            "reference_image_ids_sha256": hashlib.sha256(
                "\n".join(sorted(reference_ids)).encode("utf-8")
            ).hexdigest(),
            "k_neighbors": int(similarity_cfg.k_neighbors),
            "near_duplicate_similarity": float(similarity_cfg.near_duplicate_similarity),
        },
    )
    fallback = int((scores["similarity_reference_tier"] != "tier1_same_class").sum())
    print(
        f"similarity: {len(scores)} rows, near-duplicates="
        f"{int(scores['novelty_is_near_duplicate'].sum())}, class-agnostic fallback={fallback}",
        flush=True,
    )


# ----------------------------------------------------------------------------------------------
# Uncertainty and agreement — one set of MC-Dropout passes, two independent artifacts
# ----------------------------------------------------------------------------------------------


def run_uncertainty_and_agreement(
    frame, stage3, out_dir: Path, provenance: dict, which: list[str], device: str,
    config_key: str = V1_CONFIG_KEY,
) -> None:
    """Both read the same stochastic forward passes — the expensive part — but are written and
    gated separately. They answer different questions (how sure is the model / does it see the class
    that was asked for) and the Go/No-Go gate may admit one and exclude the other."""
    namespace = provenance["split_namespace"]
    model, resolution, identity, _ = load_cam_model(
        Path(stage3[config_key].checkpoint_dir), namespace
    )
    records = candidate_records(frame)
    passes = int(stage3.signals.uncertainty.mc_dropout_passes)

    print(f"uncertainty/agreement: {passes} MC-Dropout passes over {len(records)} candidates on {device}", flush=True)
    samples = predict_probability_passes(model, records, resolution, passes, device=device)

    shared = {
        **provenance,
        "cam_model_id": identity["cam_model_id"],
        "auxiliary_trained_on_split": identity["trained_on_split"],
        "resolution": resolution,
        "mc_dropout_passes": passes,
    }

    if "uncertainty" in which:
        scores = compute_uncertainty_scores(samples, stage3)
        scores.insert(0, "image_id", frame["image_id"].values)
        write_score_artifact(
            scores,
            out_dir / "uncertainty_scores.parquet",
            "uncertainty",
            {
                **shared,
                "low_band_max": float(stage3.signals.uncertainty.low_band_max),
                "moderate_band_max": float(stage3.signals.uncertainty.moderate_band_max),
                "band_quantity": "normalised_mutual_information",
            },
        )
        print(
            f"uncertainty: {len(scores)} rows, bands={scores['uncertainty_band'].value_counts().to_dict()}",
            flush=True,
        )

    if "agreement" in which:
        # The MC mean, not a separate deterministic pass: it is the prediction the uncertainty
        # columns describe, so the two artifacts cannot disagree about what the model predicted.
        scores = compute_agreement_scores(samples.mean(axis=0), list(frame["dx"]), stage3)
        scores.insert(0, "image_id", frame["image_id"].values)
        write_score_artifact(
            scores,
            out_dir / "agreement_scores.parquet",
            "agreement",
            {
                **shared,
                "probabilities": "mc_dropout_mean",
                "rival_confidence_threshold": float(stage3.signals.agreement.rival_confidence_threshold),
                "penalty_weight": float(stage3.signals.agreement.penalty_weight),
            },
        )
        print(
            f"agreement: {len(scores)} rows, argmax matches intended dx for "
            f"{float(scores['agreement_is_argmax_match'].mean()):.1%}",
            flush=True,
        )


# ----------------------------------------------------------------------------------------------
# Explainability — raw statistics, then calibrated against the real gen_train reference
# ----------------------------------------------------------------------------------------------


def run_explainability(
    frame, stage1, stage3, splits_cfg, out_dir: Path, provenance: dict, device: str,
    config_key: str = V1_CONFIG_KEY,
) -> None:
    namespace = provenance["split_namespace"]
    reference_path = auxiliary_outputs_dir(stage3, config_key) / namespace / "explainability_reference.json"
    if not reference_path.is_file():
        raise UpstreamGate(
            f"UPSTREAM GATE: no Grad-CAM reference at {reference_path}.\n"
            "Without it every candidate's typicality is NaN and the selection guard refuses the "
            "feature. Run: python scripts/asism/ham10000_00_build_cam_reference.py "
            f"--namespace {namespace} --config-key {config_key}"
        )
    split_dir = Path(splits_cfg.paths.splits_root) / namespace
    references = references_from_artifact(
        read_json(reference_path),
        load_final_eval_image_ids(split_dir),
        min_size=int(stage3.signals.explainability.min_reference_size),
    )

    model, resolution, identity, _ = load_cam_model(
        Path(stage3[config_key].checkpoint_dir), namespace
    )
    reference_model_id = read_json(reference_path)["cam_model_id"]
    if identity["cam_model_id"] != reference_model_id:
        raise UpstreamGate(
            "The reference was built with a different CAM model than the one on disk now "
            f"({reference_model_id} vs {identity['cam_model_id']}). Calibrating one model's "
            "attention against another's is not a comparison; rebuild the reference."
        )

    records = candidate_records(frame)
    boxes = candidate_content_boxes(frame)
    dataset = LesionRecordDataset(records, resolution)
    model = model.to(device)
    position_of = {record["image_id"]: index for index, record in enumerate(records)}

    def cam_for_record(record):
        item = dataset[position_of[record["image_id"]]]
        return gradcam(model, item["image"].to(device), int(record["class_index"]))

    print(f"explainability: Grad-CAM for {len(records)} candidates on {device}", flush=True)
    raw_rows = explainability_rows(
        tqdm(records, desc="explainability", unit="img"),
        cam_for_record,
        boxes,
        border_fraction=float(stage3.signals.explainability.border_fraction),
        mass_fraction=float(stage3.signals.explainability.mass_fraction),
    )

    diagnosis_of = dict(zip(frame["image_id"].astype(str), frame["dx"]))
    rows, uncalibrated = [], {}
    for raw in raw_rows:
        diagnosis = diagnosis_of[raw["image_id"]]
        class_references = references.get(diagnosis)
        if not class_references:
            # The reference for this class was too thin to build. The row is kept with NaN
            # typicality — the selection guard refuses NaN — rather than dropped, so the candidate
            # still appears in the pool with a visible reason.
            uncalibrated[diagnosis] = uncalibrated.get(diagnosis, 0) + 1
            rows.append({**raw, "explainability_calibrated": False,
                         "explainability_calibrated_typicality": float("nan")})
            continue
        rows.append({**raw, **calibrate_explainability(raw, diagnosis, class_references)})

    scores = pd.DataFrame(rows)
    write_score_artifact(
        scores,
        out_dir / "explainability_scores.parquet",
        "explainability",
        {
            **provenance,
            "cam_model_id": identity["cam_model_id"],
            "reference_artifact": str(reference_path),
            "reference_artifact_sha256": sha256_file(reference_path),
            "classes_without_a_reference": uncalibrated,
            "resolution": resolution,
        },
    )
    calibrated = int(scores["explainability_calibrated"].astype(bool).sum())
    print(
        f"explainability: {len(scores)} rows, {calibrated} calibrated"
        + (f", no reference for {uncalibrated}" if uncalibrated else ""),
        flush=True,
    )


# ----------------------------------------------------------------------------------------------


def run(
    namespace: str, signals: list[str], device: str | None = None, limit: int | None = None,
    config_key: str = V1_CONFIG_KEY,
) -> dict:
    from scripts.utils.classifier import require_torch

    if config_key not in CONFIG_KEYS:
        raise SystemExit(f"Unknown auxiliary classifier config key {config_key!r}; expected one of {CONFIG_KEYS}.")
    if config_key != V1_CONFIG_KEY:
        outside = [name for name in signals if name not in CLASSIFIER_SIGNALS]
        if outside:
            raise SystemExit(
                f"{config_key} recomputes only {CLASSIFIER_SIGNALS}; {outside} do not depend on the "
                "auxiliary classifier and are not recomputed with it."
            )

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")

    needs_torch = bool(set(signals) - {"iqa"})
    if needs_torch:
        torch = require_torch()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    candidates_path = Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv"
    frame = load_candidates(Path(stage2.paths.stage2_root), namespace, limit)
    provenance = {**base_provenance(namespace, candidates_path, frame), "auxiliary_config_key": config_key}

    out_dir = auxiliary_outputs_dir(stage3, config_key) / namespace / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "iqa" in signals:
        run_iqa(frame, stage3, out_dir, provenance)
    if "similarity" in signals:
        run_similarity(frame, stage1, stage3, splits_cfg, out_dir, provenance, device)
    shared = [name for name in ("uncertainty", "agreement") if name in signals]
    if shared:
        run_uncertainty_and_agreement(frame, stage3, out_dir, provenance, shared, device, config_key)
    if "explainability" in signals:
        run_explainability(frame, stage1, stage3, splits_cfg, out_dir, provenance, device, config_key)

    return {"out_dir": str(out_dir), "signals": signals, "n_candidates": len(frame), "config_key": config_key}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument(
        "--signal", action="append", choices=["all", *SIGNALS], default=None,
        help="Repeatable. Default: all.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--config-key", choices=CONFIG_KEYS, default=V1_CONFIG_KEY,
        help="Which auxiliary classifier the classifier-dependent signals come from. Default: v1. "
             "Any other key runs only uncertainty, agreement and explainability.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Score only the first N candidates. Smoke runs only: a partial artifact is still "
             "written and is still read by the Go/No-Go gate.",
    )
    args = parser.parse_args()

    requested = args.signal or ["all"]
    signals = list(SIGNALS) if "all" in requested else [name for name in SIGNALS if name in requested]

    result = run(args.namespace, signals, device=args.device, limit=args.limit, config_key=args.config_key)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.signals import (  # noqa: E402
    aggregate_pathology_overlaps,
    compute_agreement_scores,
    compute_distinctiveness_scores,
    compute_iqa_scores,
    compute_similarity_scores,
    compute_uncertainty_scores,
    expected_region_for,
    region_overlap_score,
    write_score_artifact,
)
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.labels import PRIMARY_ENDPOINT_LABELS, normalize_label  # noqa: E402
from scripts.utils.artifact_contracts import (  # noqa: E402
    auxiliary_checkpoint_path, config_sha256, current_code_identity_hash, namespace_identity,
    require_generation_complete, stage3_paths,
)
from scripts.utils.manifest import get_git_commit_hash, read_json, sha256_file  # noqa: E402
from scripts.utils.splits import load_split  # noqa: E402

# "distinctiveness" is produced INSIDE run_similarity (it reuses that step's synthetic
# embeddings, so it costs no extra encoder passes) and is therefore not separately runnable.
ALL_SIGNALS = ["similarity", "iqa", "uncertainty", "explainability", "agreement"]
PRODUCED_SIGNALS = ALL_SIGNALS + ["distinctiveness"]


def load_config():
    return load_named_config("stage3_asism.yaml", "stage3")


def load_synthetic_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: missing Stage 2 generation manifest at {path}\n"
            "Run scripts/generate/02_generate_synthetic_images.py --mode full first."
        )
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"UPSTREAM GATE: generation manifest {path} is empty.")
    return pd.DataFrame(rows)


def validate_generation_manifest_rows(manifest: pd.DataFrame, generation: dict) -> None:
    if manifest["image_id"].astype(str).duplicated().any() or len(manifest) != int(generation.get("num_rows", -1)):
        raise SystemExit("Stage 2 generation manifest has duplicate IDs or disagrees with its completion row count")
    keys = (
        "split_namespace", "namespace_class", "split_manifest_hash", "recipes_csv_sha256",
        "recipes_manifest_sha256", "lora_checkpoint_sha256", "lora_metadata_sha256",
        "generation_config_sha256", "stage2_config_sha256", "code_identity_sha256",
    )
    for key in keys:
        if key not in manifest.columns or not manifest[key].map(lambda value: value == generation.get(key)).all():
            raise SystemExit(f"Stage 2 generation row provenance mismatch at {key}; ASISM refused")


def load_auxiliary_classifier(config, namespace: str, expected_provenance: dict):
    """Load the frozen auxiliary classifier (§2). Fails cleanly if Stage 3's prerequisite is absent."""
    from scripts.utils.classifier import build_model, require_torch

    resume_path = auxiliary_checkpoint_path(config, namespace)
    checkpoint_path = resume_path.with_name(f"{resume_path.stem}.best{resume_path.suffix}")
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"UPSTREAM GATE: auxiliary classifier not found at {checkpoint_path}\n"
            "It is a Stage 3 prerequisite (docs/stages2_to_5_plan.md §2). Train it first:\n"
            "  python scripts/classify/00_train_auxiliary_classifier.py"
        )
    torch = require_torch()
    from scripts.utils.labels import CLASSIFIER_TARGET_LABELS

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_provenance = payload.get("provenance", {})
    for key in ("split_namespace", "split_manifest_hash", "code_identity_sha256"):
        expected = expected_provenance.get(key)
        if checkpoint_provenance.get(key) != expected:
            raise SystemExit(f"Auxiliary best-checkpoint provenance mismatch at {key}: expected {expected!r}, got {checkpoint_provenance.get(key)!r}")
    aux_manifest_path = resume_path.with_suffix(".manifest.json")
    if not aux_manifest_path.is_file():
        raise SystemExit(f"Auxiliary checkpoint manifest missing: {aux_manifest_path}")
    aux_manifest = read_json(aux_manifest_path)
    if aux_manifest.get("best_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise SystemExit("Auxiliary best-checkpoint hash differs from its manifest")
    model_config = payload.get("model_config", {})
    model = build_model(len(CLASSIFIER_TARGET_LABELS), float(model_config["dropout_p"]),
                        str(model_config["pretrained_source"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, int(model_config["resolution"])


def synthetic_records(manifest: pd.DataFrame, images_dir: Path) -> list[dict]:
    from scripts.utils.labels import CLASSIFIER_TARGET_LABELS

    records = []
    for row in manifest.to_dict("records"):
        labels = dict(row["intended_label_vector"])
        labels["No Finding"] = 1 if row.get("is_no_finding") else 0
        labels["Support Devices"] = int(row.get("support_devices", 0))
        records.append(
            {
                "image_id": row["image_id"],
                "image_path": str(images_dir / f"{row['image_id']}.jpg"),
                "labels": {label: labels.get(label) for label in CLASSIFIER_TARGET_LABELS},
                "is_synthetic": True,
            }
        )
    return records


def run_iqa(manifest, images_dir, config, provenance, scores_dir) -> None:
    rows = []
    for image_id in tqdm(manifest["image_id"], desc="iqa", unit="img"):
        scores = compute_iqa_scores(images_dir / f"{image_id}.jpg", config)
        scores["image_id"] = image_id
        rows.append(scores)
    frame = pd.DataFrame(rows)
    columns = ["image_id"] + [c for c in frame.columns if c != "image_id"]
    write_score_artifact(frame[columns], scores_dir / "iqa_scores.parquet", "iqa", provenance)
    print(f"iqa: {len(frame)} rows, {int(frame['iqa_valid'].sum())} valid", flush=True)


def run_uncertainty_and_agreement(
    manifest, images_dir, config, provenance, scores_dir, which: list[str]
) -> None:
    """Both signals come from the same auxiliary-classifier forward passes, so they share the
    (expensive) MC-Dropout inference — but they are written as SEPARATE artifacts and are gated
    independently (§4.0, §4.6). They are never merged into a single score."""
    from scripts.utils.classifier import predict_probabilities

    model, resolution = load_auxiliary_classifier(config, provenance["split_namespace"], provenance)
    records = synthetic_records(manifest, images_dir)
    passes = int(config.signals.uncertainty.mc_dropout_passes)

    mean_all, std_all = predict_probabilities(
        model, records, resolution, mc_dropout_passes=passes
    )

    from scripts.utils.labels import CLASSIFIER_TARGET_LABELS

    primary_idx = [CLASSIFIER_TARGET_LABELS.index(label) for label in PRIMARY_ENDPOINT_LABELS]
    mean_primary = mean_all[:, primary_idx]
    std_primary = std_all[:, primary_idx]

    if "uncertainty" in which:
        frame = compute_uncertainty_scores(std_primary, mean_primary, config)
        frame.insert(0, "image_id", manifest["image_id"].values)
        write_score_artifact(
            frame, scores_dir / "uncertainty_scores.parquet", "uncertainty", provenance
        )
        print(f"uncertainty: {len(frame)} rows, bands={frame['uncertainty_band'].value_counts().to_dict()}", flush=True)

    if "agreement" in which:
        intended = [dict(v) for v in manifest["intended_label_vector"]]
        frame = compute_agreement_scores(mean_primary, intended, config)
        frame.insert(0, "image_id", manifest["image_id"].values)
        write_score_artifact(
            frame, scores_dir / "agreement_scores.parquet", "agreement", provenance
        )
        print(f"agreement: {len(frame)} rows, mean={frame['agreement_score'].mean():.4f}", flush=True)


def run_explainability(manifest, images_dir, config, provenance, scores_dir) -> None:
    """Grad-CAM overlap with the expected region for each recipe's intended pathologies.

    An explainability-PLAUSIBILITY signal, not proof of diagnostic correctness (§4.4).
    """
    from scripts.utils.classifier import CXRRecordDataset, require_torch

    torch = require_torch()
    model, resolution = load_auxiliary_classifier(config, provenance["split_namespace"], provenance)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    from scripts.utils.labels import CLASSIFIER_TARGET_LABELS

    # Empirically derived expected regions (00c_derive_expected_regions.py) take precedence per
    # label over the hand-entered config boxes. Absent, the config boxes still apply -- so this is
    # an upgrade, not a new hard dependency.
    derived_regions, regions_source = {}, "config_hand_entered"
    regions_path = Path(config.paths.asism_dir) / "expected_regions.json"
    if regions_path.is_file():
        derived_payload = read_json(regions_path)
        derived_regions = dict(derived_payload.get("regions", {}))
        regions_source = f"derived:{derived_payload.get('method')}"
        print(
            f"explainability: using {len(derived_regions)} empirically derived region(s); "
            "any remaining label falls back to its config box",
            flush=True,
        )
    else:
        print(
            "explainability: no derived expected_regions.json -- using hand-entered config boxes. "
            "Run 00c_derive_expected_regions.py to replace them with measured ones.",
            flush=True,
        )

    records = synthetic_records(manifest, images_dir)
    dataset = CXRRecordDataset(records, resolution=resolution)

    # Hook the last dense block: standard Grad-CAM target for DenseNet.
    activations, gradients = {}, {}
    target_layer = model.features.denseblock4

    def forward_hook(_module, _input, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    rows = []
    try:
        for index in tqdm(range(len(dataset)), desc="explainability", unit="img"):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            intended = dict(manifest.iloc[index]["intended_label_vector"])
            positives = [l for l in PRIMARY_ENDPOINT_LABELS if int(intended.get(l, 0)) == 1]
            target_labels = positives or ["__max_disease__"]
            per_label_overlap = {}
            cam_maxima = []
            for target_label in target_labels:
                model.zero_grad(set_to_none=True)
                logits = model(image)
                if target_label == "__max_disease__":
                    disease_idx = [CLASSIFIER_TARGET_LABELS.index(l) for l in PRIMARY_ENDPOINT_LABELS]
                    score = logits[0, disease_idx].max()
                    region = expected_region_for({}, config, derived_regions)
                else:
                    score = logits[0, CLASSIFIER_TARGET_LABELS.index(target_label)]
                    single_intent = {label: int(label == target_label) for label in PRIMARY_ENDPOINT_LABELS}
                    region = expected_region_for(single_intent, config, derived_regions)
                score.backward()
                weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze(0).cpu().numpy()
                per_label_overlap[target_label] = region_overlap_score(cam, region)
                cam_maxima.append(float(cam.max()))
            # Frozen aggregation: every intended positive contributes equally; mean overlap is the
            # primary score and the minimum is retained to expose a missed co-pathology.
            overlap, minimum_overlap = aggregate_pathology_overlaps(per_label_overlap)
            rows.append(
                {
                    "image_id": manifest.iloc[index]["image_id"],
                    "explainability_region_overlap": overlap,
                    "explainability_min_pathology_overlap": minimum_overlap,
                    "explainability_target_labels": json.dumps(target_labels),
                    "explainability_per_label_overlap": json.dumps(per_label_overlap, sort_keys=True),
                    "explainability_aggregation": "mean_over_all_intended_positive_pathologies",
                    "explainability_cam_max": max(cam_maxima),
                    "explainability_is_plausibility_signal_only": True,
                }
            )
    finally:
        handle_f.remove()
        handle_b.remove()

    frame = pd.DataFrame(rows)
    write_score_artifact(
        frame, scores_dir / "explainability_scores.parquet", "explainability",
        {**provenance, "expected_regions_source": regions_source,
         "n_derived_regions_used": len(derived_regions)},
    )
    print(f"explainability: {len(frame)} rows, mean overlap={frame['explainability_region_overlap'].mean():.4f}", flush=True)


def run_similarity(manifest, images_dir, config, provenance, scores_dir, namespace) -> None:
    """DINOv2 embeddings for synthetic images and a stratified real-reference sample."""
    from scripts.utils.classifier import require_torch
    from scripts.utils.identifiers import sanitize_image_id

    torch = require_torch()
    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            f"UPSTREAM GATE: the similarity encoder needs `timm` ({exc}).\n"
            "Install it on the pod: pip install timm"
        ) from exc

    stage1_cfg = load_stage1_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Never let timm resolve a mutable default checkpoint. Download the exact reviewed revision,
    # then load it into the explicitly named architecture.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as load_safetensors

    similarity_cfg = config.signals.similarity
    weights_path = hf_hub_download(
        repo_id=str(similarity_cfg.weights_repo_id),
        filename=str(similarity_cfg.weights_filename),
        revision=str(similarity_cfg.weights_revision),
    )
    encoder = timm.create_model(str(similarity_cfg.encoder), pretrained=False, num_classes=0)
    incompatible = encoder.load_state_dict(load_safetensors(weights_path), strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [key for key in incompatible.missing_keys if not key.startswith("head.")]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            "Pinned DINOv2 checkpoint is incompatible with the configured architecture: "
            f"missing={disallowed_missing}, unexpected={unexpected}"
        )
    encoder = encoder.eval().to(device)
    data_config = timm.data.resolve_model_data_config(encoder)
    transform = timm.data.create_transform(**data_config, is_training=False)

    from PIL import Image

    def embed(paths: list[Path]) -> np.ndarray:
        outputs = []
        batch_size = int(config.signals.similarity.batch_size)
        for start in tqdm(range(0, len(paths), batch_size), desc="embed", unit="batch"):
            batch = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    batch.append(transform(image.convert("RGB")))
            with torch.no_grad():
                outputs.append(encoder(torch.stack(batch).to(device)).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    # Real reference pool: a stratified sample of gen_train (the generator's own training data is
    # the correct realism reference; classifier_train is reserved for A-E).
    real_frame = load_split("gen_train", namespace, purpose="schema_validation", caller="asism_similarity")
    per_label = int(config.signals.similarity.reference_sample_per_label)
    real_images_dir = Path(stage1_cfg.paths.images_dir) / namespace / "gen_train"

    selected_rows, seen = [], set()
    for label in PRIMARY_ENDPOINT_LABELS:
        subset = real_frame[real_frame[label].map(normalize_label) == 1].head(per_label)
        for row in subset.to_dict("records"):
            key = row["Path"]
            if key not in seen:
                seen.add(key)
                selected_rows.append(row)
    # Plus negatives, so the class-agnostic fallback tier is not all-pathology.
    for row in real_frame.head(per_label).to_dict("records"):
        if row["Path"] not in seen:
            seen.add(row["Path"])
            selected_rows.append(row)

    reference_paths, reference_vectors = [], []
    for row in selected_rows:
        path = real_images_dir / f"{sanitize_image_id(row['Path'])}.jpg"
        if not path.is_file():
            continue
        reference_paths.append(path)
        reference_vectors.append(
            frozenset(l for l in PRIMARY_ENDPOINT_LABELS if normalize_label(row.get(l)) == 1)
        )

    if not reference_paths:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed gen_train images under {real_images_dir}; "
            "Stage 1 preprocessing must run before the similarity signal."
        )

    synthetic_paths = [images_dir / f"{image_id}.jpg" for image_id in manifest["image_id"]]
    print(f"similarity: embedding {len(synthetic_paths)} synthetic, {len(reference_paths)} real", flush=True)

    # Embed ONCE and reuse. The distinctiveness signal (§4.6) is computed from these same synthetic
    # embeddings, so it adds no encoder passes at all -- only a per-class matrix multiply.
    synthetic_embeddings = embed(synthetic_paths)
    intended_vectors = [dict(v) for v in manifest["intended_label_vector"]]

    frame = compute_similarity_scores(
        synthetic_embeddings,
        embed(reference_paths),
        reference_vectors,
        intended_vectors,
        config,
    )
    frame.insert(0, "image_id", manifest["image_id"].values)
    write_score_artifact(frame, scores_dir / "similarity_scores.parquet", "similarity", provenance)
    print(
        f"similarity: {len(frame)} rows, near-duplicates flagged="
        f"{int(frame['novelty_is_near_duplicate'].sum())}",
        flush=True,
    )

    distinct = compute_distinctiveness_scores(synthetic_embeddings, intended_vectors, config)
    distinct.insert(0, "image_id", manifest["image_id"].values)
    write_score_artifact(
        distinct, scores_dir / "distinctiveness_scores.parquet", "distinctiveness", provenance
    )
    defined = distinct.loc[~distinct["distinctiveness_is_undefined"]]
    print(
        f"distinctiveness: {len(distinct)} rows, "
        f"{int(distinct['distinctiveness_is_undefined'].sum())} undefined (single-image class), "
        f"mean={defined['distinctiveness_score'].mean():.4f} "
        f"images-with-a-duplicate={int((defined['distinctiveness_n_duplicates'] > 0).sum())}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", default="all", choices=ALL_SIGNALS + ["all"])
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    config = load_config()
    namespace = args.namespace or str(config.split_namespace)
    identity = namespace_identity(namespace)

    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")

    stage2_artifacts, generation = require_generation_complete(stage2_cfg, namespace)
    images_dir = stage2_artifacts["images_dir"]
    manifest = load_synthetic_manifest(stage2_artifacts["manifest_path"])
    validate_generation_manifest_rows(manifest, generation)
    scores_dir = stage3_paths(config, namespace)["scores_dir"]
    scores_dir.mkdir(parents=True, exist_ok=True)

    artifact_provenance = {
        **identity,
        "asism_config_sha256": config_sha256(config),
        "generation_manifest_sha256": sha256_file(stage2_artifacts["manifest_path"]),
        "generation_completion_sha256": sha256_file(stage2_artifacts["generation_completion"]),
        "lora_checkpoint_sha256": generation["lora_checkpoint_sha256"],
        "code_identity_sha256": current_code_identity_hash(),
        "git_commit_hash": get_git_commit_hash(),
        "n_synthetic_images": len(manifest),
    }

    which = ALL_SIGNALS if args.signal == "all" else [args.signal]

    if "iqa" in which:
        run_iqa(manifest, images_dir, config, artifact_provenance, scores_dir)
    if {"uncertainty", "agreement"} & set(which):
        run_uncertainty_and_agreement(
            manifest, images_dir, config, artifact_provenance, scores_dir, which
        )
    if "explainability" in which:
        run_explainability(manifest, images_dir, config, artifact_provenance, scores_dir)
    if "similarity" in which:
        run_similarity(manifest, images_dir, config, artifact_provenance, scores_dir, namespace)

    print(f"\nArtifacts -> {scores_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

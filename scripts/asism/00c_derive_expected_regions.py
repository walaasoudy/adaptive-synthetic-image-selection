#!/usr/bin/env python3
"""Derive each pathology's expected Grad-CAM region from REAL positives (docs/stages2_to_5_plan.md §4.4).

WHY THIS EXISTS. The explainability signal scores a synthetic image by how much of its Grad-CAM mass
falls inside the region where that pathology is expected to appear. Until now those regions were
hand-entered constants in configs/stage3_asism.yaml, e.g.

    Cardiomegaly: [0.30, 0.40, 0.75, 0.85]

with no recorded derivation. The first question any reviewer asks is where those numbers came from,
and "we chose them" is not an answer that survives. This script replaces the guess with a
measurement: run the SAME Grad-CAM procedure the signal uses, on REAL images the split says are
positive for a pathology, and take the region the classifier's attention actually concentrates in.

SPLIT DISCIPLINE. Regions are derived from `gen_train` ONLY -- the split the auxiliary classifier
was trained on (§2). Deriving them from classifier_train, classifier_val, asism_tuning_heldout or
final_eval_heldout would leak evaluation data into a component that scores every synthetic image,
so the split is not configurable here; it is fixed in code and recorded in the manifest.

WHAT THIS IS NOT. The derived box is where a classifier trained on real data LOOKS for a pathology,
not where the pathology anatomically is. It inherits any shortcut the classifier learned. It is an
explainability-plausibility reference, exactly as §4.4 already states for the signal itself -- the
improvement is that the reference is now measured and reproducible rather than asserted.

Usage:
    python scripts/asism/00c_derive_expected_regions.py --namespace production-thesis-v1
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.signals import region_from_cam_mass  # noqa: E402
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.labels import CLASSIFIER_TARGET_LABELS, PRIMARY_ENDPOINT_LABELS, normalize_label  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, write_frozen_json  # noqa: E402
from scripts.utils.splits import load_split  # noqa: E402
from scripts.utils.artifact_contracts import asism_score_provenance, stage3_paths  # noqa: E402

# Never configurable: see SPLIT DISCIPLINE in the module docstring.
DERIVATION_SPLIT = "gen_train"


def _signals_module():
    """01_compute_signals.py is not importable by name (leading digit); load it by path so the
    Grad-CAM here is literally the same code the signal uses, never a re-implementation that could
    drift from it."""
    path = Path(__file__).with_name("01_compute_signals.py")
    spec = importlib.util.spec_from_file_location("compute_signals_01", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--max-images-per-label", type=int, default=200,
                        help="Cap on real positives averaged per pathology (cost control).")
    parser.add_argument("--mass-fraction", type=float, default=0.80,
                        help="Fraction of averaged Grad-CAM mass the derived box must enclose.")
    args = parser.parse_args()

    cfg = load_named_config("stage3_asism.yaml", "stage3")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths:
            cfg.paths[key] = str(value)
    cfg.split_namespace = namespace

    output_path = Path(cfg.paths.asism_dir) / "expected_regions.json"
    if output_path.exists():
        raise SystemExit(
            f"REFUSING to overwrite existing derived regions at {output_path}.\n"
            "They are a frozen input to every explainability score. Delete the file deliberately "
            "if you intend to re-derive them, and expect downstream hashes to change."
        )

    signals = _signals_module()
    stage2_cfg = load_named_config("stage2_generation.yaml", "stage2")
    provenance = asism_score_provenance(cfg, stage2_cfg, namespace)
    model, resolution = signals.load_auxiliary_classifier(cfg, namespace, provenance)

    from scripts.utils.classifier import CXRRecordDataset, require_torch
    from scripts.utils.identifiers import sanitize_image_id

    torch = require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    frame = load_split(DERIVATION_SPLIT, namespace, purpose="schema_validation",
                       caller="derive_expected_regions")
    images_dir = Path(load_stage1_config().paths.images_dir) / namespace / DERIVATION_SPLIT

    activations, gradients = {}, {}
    target_layer = model.features.denseblock4
    handle_f = target_layer.register_forward_hook(
        lambda _m, _i, output: activations.__setitem__("value", output.detach()))
    handle_b = target_layer.register_full_backward_hook(
        lambda _m, _gi, grad_out: gradients.__setitem__("value", grad_out[0].detach()))

    derived: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    try:
        for label in PRIMARY_ENDPOINT_LABELS:
            positives = frame[frame[label].map(normalize_label) == 1].head(args.max_images_per_label)
            records = []
            for row in positives.to_dict("records"):
                path = images_dir / f"{sanitize_image_id(row['Path'])}.jpg"
                if path.is_file():
                    records.append({"image_path": str(path), "labels": {}, "is_synthetic": False})
            if not records:
                # No confident real positive with a preprocessed image. Recorded explicitly so the
                # gap is visible, rather than inventing a region for this pathology.
                skipped[label] = "no preprocessed real positives found"
                continue

            dataset = CXRRecordDataset(records, resolution=resolution)
            label_index = CLASSIFIER_TARGET_LABELS.index(label)
            accumulated = None
            contributing = 0
            for index in tqdm(range(len(dataset)), desc=f"cam:{label}", unit="img", leave=False):
                image = dataset[index]["image"].unsqueeze(0).to(device)
                model.zero_grad(set_to_none=True)
                model(image)[0, label_index].backward()
                weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations["value"]).sum(dim=1)).squeeze(0)
                peak = float(cam.max())
                if peak <= 0:
                    continue  # no positive attention on this image; contributes nothing
                accumulated = (cam / peak).cpu().numpy() if accumulated is None \
                    else accumulated + (cam / peak).cpu().numpy()
                contributing += 1

            if accumulated is None:
                skipped[label] = "every real positive produced empty Grad-CAM attention"
                continue

            region = region_from_cam_mass(accumulated / contributing, args.mass_fraction)
            derived[label] = {"region": list(region), "n_real_positives_used": contributing}
            print(f"  {label:30s} n={contributing:4d}  region={[round(v, 3) for v in region]}", flush=True)
    finally:
        handle_f.remove()
        handle_b.remove()

    if not derived:
        raise SystemExit("No pathology yielded a derived region; refusing to write an empty artifact.")

    write_frozen_json(output_path, {
        "schema_version": 1,
        "method": "mean_gradcam_mass_box_from_real_positives_v1",
        "frozen": True,
        "split_namespace": namespace,
        "derivation_split": DERIVATION_SPLIT,
        "mass_fraction": args.mass_fraction,
        "max_images_per_label": args.max_images_per_label,
        "regions": {label: value["region"] for label, value in derived.items()},
        "n_real_positives_used": {label: value["n_real_positives_used"] for label, value in derived.items()},
        "skipped_labels": skipped,
        "interpretation": (
            "Where a classifier trained on REAL data attends for each pathology -- an "
            "explainability-plausibility reference, NOT anatomical ground truth, and it inherits "
            "any shortcut the classifier learned."
        ),
        "git_commit_hash": get_git_commit_hash(),
        **provenance,
    })
    print(f"\nDerived {len(derived)}/{len(PRIMARY_ENDPOINT_LABELS)} regions -> {output_path}")
    if skipped:
        print(f"Skipped (config fallback still applies): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

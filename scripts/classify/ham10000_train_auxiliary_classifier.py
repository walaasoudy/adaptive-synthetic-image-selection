#!/usr/bin/env python3
"""HAM10000 ASISM auxiliary classifier — the real-only model Stage 3 scores WITH.

ROLE (fixed here, not renegotiated downstream)
  Its first purpose is the Grad-CAM reference: the gen_train explainability distribution is built
  from THIS model's attention, and `cam_model_identity` turns it into a single `cam_model_id` that
  every calibrated row must name. Reusing it for the uncertainty and agreement signals is permitted
  only while those signals pin the same id and the same reference sets.

WHY NOT THE CHEXPERT CONVENTION
  scripts/classify/00_train_auxiliary_classifier.py trains on gen_train and keeps classifier_train
  unspent. For HAM10000 gen_train is the explainability REFERENCE split, so a model trained on it
  has memorised the images whose attention it would define. `cam_model_identity` refuses that
  combination by name (docs/ham10000_gradcam_calibration.md), and this entry point refuses it a
  step earlier, in config validation, so the failure names the config key rather than a checkpoint.

WHY THERE IS NO BEST-CHECKPOINT SELECTION
  `train_classifier` runs a fixed optimizer-step budget and returns the final model; Stage 4 does
  the same. `selection_split` here is MEASURED, never selected on: it is scored once after training
  so the manifest records what the frozen model does. Adding selection for this model alone would
  make the reference model trained differently from the models it is a reference for.

Refuses to start when:
  * the train or selection split is gen_train (the reference split) or a held-out split;
  * `class_weighting` is anything but "none" — see the config comment;
  * the architecture or class list is not the one Grad-CAM and the metric suite assume;
  * the produced manifest is not accepted by `cam_model_identity`.

Writes <checkpoint_dir>/<namespace>/: model.pt, selection_metrics.json, run_manifest.json

Usage:
    python scripts/classify/ham10000_train_auxiliary_classifier.py --namespace ham-stratified-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_signals import cam_model_identity  # noqa: E402
from scripts.utils.classifier import TrainingBudget  # noqa: E402  (dataclass only; no loss code)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_classifier import (  # noqa: E402
    predict_probabilities,
    records_from_split,
    train_classifier,
    true_class_indices,
)
from scripts.utils.ham10000_metrics import full_metric_suite  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, write_json  # noqa: E402

# The split whose attention this model defines. Training on it would make the reference the model's
# own memorised training attention, which is the failure cam_model_identity exists to prevent.
CAM_REFERENCE_SPLIT = "gen_train"
# Held out from every development decision; final_eval_heldout additionally belongs to Stage 5 only.
FORBIDDEN_AUXILIARY_SPLITS = frozenset({"gen_train", "asism_tuning_heldout", "final_eval_heldout"})


class AuxiliaryClassifierConfigError(SystemExit):
    pass


def validate_config(auxiliary) -> None:
    for key in ("train_split", "selection_split"):
        split = str(auxiliary[key])
        if split in FORBIDDEN_AUXILIARY_SPLITS:
            raise AuxiliaryClassifierConfigError(
                f"auxiliary_classifier.{key}={split!r} is not usable: {CAM_REFERENCE_SPLIT} is the Grad-CAM "
                f"reference split and the held-out splits are not development data "
                f"(forbidden: {sorted(FORBIDDEN_AUXILIARY_SPLITS)})"
            )
    if str(auxiliary.train_split) == str(auxiliary.selection_split):
        raise AuxiliaryClassifierConfigError(
            f"train_split and selection_split are both {auxiliary.train_split!r}; the measured split must be unseen"
        )
    if str(auxiliary.architecture) != "densenet121":
        raise AuxiliaryClassifierConfigError(
            f"architecture={auxiliary.architecture!r}: ham10000_explainability.gradcam hooks "
            "model.features.denseblock4, which only densenet121 provides"
        )
    if str(auxiliary.class_weighting) != "none":
        raise AuxiliaryClassifierConfigError(
            f"class_weighting={auxiliary.class_weighting!r}: the reference describes where an unweighted model "
            "looks; reweighting would change the attention this stage calls normal"
        )


def build_records(auxiliary, splits_root: Path, images_root: Path, namespace: str):
    """(train records, selection records, provenance). No synthetic path exists in this function."""
    split_dir, images_dir = splits_root / namespace, images_root / namespace
    train_split, selection_split = str(auxiliary.train_split), str(auxiliary.selection_split)

    train = records_from_split(pd.read_csv(split_dir / f"{train_split}.csv"), images_dir / train_split)
    selection = records_from_split(pd.read_csv(split_dir / f"{selection_split}.csv"), images_dir / selection_split)
    if not train or not selection:
        raise AuxiliaryClassifierConfigError(
            f"no preprocessed images found for {train_split} ({len(train)}) or {selection_split} ({len(selection)}); "
            "run scripts/data/ham10000/02_preprocess_images.py first"
        )
    provenance = {
        "real_train_split": train_split,
        "real_train_images": len(train),
        # Constant, not computed: this entry point has no code path that reads a synthetic manifest.
        "synthetic_manifest": None,
        "synthetic_images": 0,
        "selection_split": selection_split,
        "selection_images": len(selection),
        "selection_role": "measured_only_never_selected_on",
        "split_csv_sha256": {
            name: hashlib.sha256((split_dir / f"{name}.csv").read_bytes()).hexdigest()
            for name in (train_split, selection_split)
        },
    }
    return train, selection, provenance


def run(auxiliary, splits_root: Path, images_root: Path, namespace: str, device: str | None = None) -> dict:
    validate_config(auxiliary)

    import torch

    train_records, selection_records, provenance = build_records(
        auxiliary, Path(splits_root), Path(images_root), namespace
    )
    budget = TrainingBudget(
        max_steps=int(auxiliary.max_steps),
        batch_size=int(auxiliary.batch_size),
        learning_rate=float(auxiliary.learning_rate),
        weight_decay=float(auxiliary.weight_decay),
        seed=int(auxiliary.seed),
    )
    resolution = int(auxiliary.resolution)

    model, history = train_classifier(
        train_records, budget, float(auxiliary.dropout_p), str(auxiliary.pretrained_source), resolution,
        class_weights=None, device=device, progress_desc="asism-auxiliary",
    )
    probabilities = predict_probabilities(model, selection_records, resolution, device=device)
    metrics = full_metric_suite(probabilities, true_class_indices(selection_records))

    out_dir = Path(auxiliary.checkpoint_dir) / namespace
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    write_json(out_dir / "selection_metrics.json", {"split": provenance["selection_split"], **metrics})

    manifest = {
        "dataset": "ham10000",
        "role": "asism_auxiliary",
        "loss": "cross_entropy",
        "prediction_activation": "softmax",
        "classes": list(CLASSIFIER_TARGET_LABELS),
        "class_weighting": "none",
        "class_weights": None,
        "split_namespace": namespace,
        "model": {
            "architecture": str(auxiliary.architecture),
            "pretrained_source": str(auxiliary.pretrained_source),
            "dropout_p": float(auxiliary.dropout_p),
            "resolution": resolution,
        },
        "budget": {
            "max_steps": budget.max_steps, "batch_size": budget.batch_size,
            "learning_rate": budget.learning_rate, "weight_decay": budget.weight_decay, "seed": budget.seed,
        },
        "checkpoint_selection": "none_fixed_step_budget",
        "final_loss": history[-1]["loss"] if history else None,
        "data": provenance,
        "final_eval_heldout_read": False,
        "git_commit_hash": get_git_commit_hash(),
    }
    # The identity is minted HERE, through the same guard Stage 3 will apply, so an unusable model
    # fails at the moment it is produced instead of in the stage that tries to calibrate with it.
    manifest.update(cam_model_identity(manifest, model_path, reference_split=CAM_REFERENCE_SPLIT))
    write_json(out_dir / "run_manifest.json", manifest)
    return {"out_dir": str(out_dir), "metrics": metrics, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Override the configured budget (smoke runs only)")
    args = parser.parse_args()

    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    auxiliary = stage3.auxiliary_classifier
    if args.max_steps is not None:
        auxiliary.max_steps = int(args.max_steps)

    result = run(
        auxiliary, Path(splits_cfg.paths.splits_root), Path(stage1.paths.images_dir),
        args.namespace, device=args.device,
    )
    manifest = result["manifest"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "cam_model_id": manifest["cam_model_id"],
        "trained_on": manifest["data"]["real_train_split"],
        "measured_on": manifest["data"]["selection_split"],
        "synthetic_images": manifest["data"]["synthetic_images"],
        "balanced_accuracy": result["metrics"]["balanced_accuracy"],
        "macro_f1": result["metrics"]["macro_f1"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

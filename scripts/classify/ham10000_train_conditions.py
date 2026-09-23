#!/usr/bin/env python3
"""HAM10000 Stage 4: train one condition x seed with softmax + CrossEntropy.

WHICH conditions exist is a protocol decision, not a flag: --protocol names one of the frozen
protocols in scripts/utils/ham10000_conditions.py (v1 = A/B/C, v2 = A/B/C2/D2), and that protocol
chooses both the condition names this accepts and the config file they are read from. The default
is v1, so every existing v1 command keeps working unchanged.

The dedicated HAM10000 entry point. It never imports the CheXpert trainer
(scripts/utils/classifier.train_classifier, BCE) — only the dataset-agnostic model builder and
budget dataclass, through scripts/utils/ham10000_classifier.

Refuses to start when:
  * configs/ham10000_stage4.yaml `loss` is not "cross_entropy";
  * `classes` differs from CLASSIFIER_TARGET_LABELS (column order is part of every artifact);
  * the selection or training split is final_eval_heldout (Stage 5 only);
  * a synthetic manifest row has an unknown diagnosis or a missing image.

Writes <results_dir>/<namespace>/<condition>/seed<seed>/: model.pt, selection_metrics.json
(on classifier_val), run_manifest.json.

Usage:
    python scripts/classify/ham10000_train_conditions.py --condition A --seed 42
    python scripts/classify/ham10000_train_conditions.py --protocol v2 --condition C2 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.classifier import TrainingBudget  # noqa: E402  (dataclass only; no loss code)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.ham10000_conditions import DEFAULT_PROTOCOL, PROTOCOLS, get_protocol  # noqa: E402
from scripts.utils.ham10000_classifier import (  # noqa: E402
    class_weights_from_records,
    predict_probabilities,
    records_from_split,
    train_classifier,
    true_class_indices,
)
from scripts.utils.ham10000_metrics import full_metric_suite  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, write_json  # noqa: E402

FORBIDDEN_STAGE4_SPLITS = frozenset({"final_eval_heldout"})
CLASS_WEIGHT_BASES = frozenset({"real_train_split", "condition_training_set"})


def resolve_class_weights(training_cfg, train_records: list[dict]):
    """(weights or None, record) for CrossEntropyLoss(weight=...).

    With basis real_train_split the counts come from the non-synthetic records only, so conditions
    A, B and C, which share the same real split, receive an identical weight vector.
    """
    if str(training_cfg.class_weighting) == "none":
        return None, {"class_weighting": "none", "class_weight_basis": None, "class_weights": None}
    basis = str(training_cfg.class_weight_basis)
    source = [r for r in train_records if not r.get("synthetic")] if basis == "real_train_split" else train_records
    weights = class_weights_from_records(source)
    return weights, {
        "class_weighting": "inverse_frequency",
        "class_weight_basis": basis,
        "class_weight_source_images": len(source),
        "class_weights": dict(zip(CLASSIFIER_TARGET_LABELS, [float(v) for v in weights])),
    }


class Stage4ConfigError(SystemExit):
    pass


def validate_config(config) -> None:
    if str(config.loss) != "cross_entropy":
        raise Stage4ConfigError(f"HAM10000 Stage 4 requires loss=cross_entropy; got {config.loss!r}")
    if list(config.classes) != list(CLASSIFIER_TARGET_LABELS):
        raise Stage4ConfigError(f"classes {list(config.classes)} != CLASSIFIER_TARGET_LABELS {CLASSIFIER_TARGET_LABELS}")
    for key in ("selection_split", "real_train_split"):
        if str(config[key]) in FORBIDDEN_STAGE4_SPLITS:
            raise Stage4ConfigError(f"{key}={config[key]!r}: final_eval_heldout is Stage 5 only")
    if str(config.training.class_weighting) not in {"none", "inverse_frequency"}:
        raise Stage4ConfigError(f"unknown class_weighting {config.training.class_weighting!r}")
    if str(config.training.get("class_weight_basis", "")) not in CLASS_WEIGHT_BASES:
        raise Stage4ConfigError(f"class_weight_basis must be one of {sorted(CLASS_WEIGHT_BASES)}; got {config.training.get('class_weight_basis')!r}")


def synthetic_records(manifest_path: Path) -> list[dict]:
    frame = pd.read_csv(manifest_path)
    missing_columns = {"image_id", "image_path", "dx"} - set(frame.columns)
    if missing_columns:
        raise Stage4ConfigError(f"{manifest_path}: missing columns {sorted(missing_columns)}")
    index_of = {label: position for position, label in enumerate(CLASSIFIER_TARGET_LABELS)}
    records = []
    for row in frame.to_dict("records"):
        path = Path(row["image_path"])
        if not path.is_file():
            raise Stage4ConfigError(f"{manifest_path}: missing synthetic image {path}")
        records.append({"image_id": str(row["image_id"]), "image_path": str(path), "class_index": index_of[normalize_diagnosis(row["dx"])], "synthetic": True})
    return records


def build_condition_records(config, condition: str) -> tuple[list[dict], list[dict], dict]:
    namespace = str(config.split_namespace)
    split_dir = Path(config.paths.splits_root) / namespace
    images_root = Path(config.paths.images_root) / namespace

    real_split, selection_split = str(config.real_train_split), str(config.selection_split)
    train = records_from_split(pd.read_csv(split_dir / f"{real_split}.csv"), images_root / real_split)
    selection = records_from_split(pd.read_csv(split_dir / f"{selection_split}.csv"), images_root / selection_split)
    if not train or not selection:
        raise Stage4ConfigError(f"no preprocessed images found for {real_split} ({len(train)}) or {selection_split} ({len(selection)})")

    manifest = config.conditions[condition].synthetic_manifest
    synthetic = synthetic_records(Path(manifest)) if manifest else []
    provenance = {
        "real_train_split": real_split,
        "real_train_images": len(train),
        "synthetic_manifest": str(manifest) if manifest else None,
        "synthetic_images": len(synthetic),
        "selection_split": selection_split,
        "selection_images": len(selection),
        "split_csv_sha256": {name: hashlib.sha256((split_dir / f"{name}.csv").read_bytes()).hexdigest() for name in (real_split, selection_split)},
    }
    return train + synthetic, selection, provenance


def run(config, condition: str, seed: int, device: str | None = None, protocol_name: str = DEFAULT_PROTOCOL) -> dict:
    validate_config(config)
    if condition not in config.conditions:
        raise Stage4ConfigError(f"unknown condition {condition!r}; configured: {list(config.conditions)}")

    import torch

    train_records, selection_records, provenance = build_condition_records(config, condition)
    training = config.training
    budget = TrainingBudget(
        max_steps=int(training.max_steps),
        batch_size=int(training.batch_size),
        learning_rate=float(training.learning_rate),
        weight_decay=float(training.weight_decay),
        eval_every_n_steps=int(training.eval_every_n_steps),
        seed=int(seed),
    )
    weights, weighting_record = resolve_class_weights(training, train_records)
    resolution = int(config.model.resolution)

    model, history = train_classifier(
        train_records, budget, float(config.model.dropout_p), str(config.model.pretrained_source), resolution,
        class_weights=weights, device=device, progress_desc=f"{condition}/seed{seed}",
    )
    probabilities = predict_probabilities(model, selection_records, resolution, device=device)
    metrics = full_metric_suite(probabilities, true_class_indices(selection_records))

    out_dir = Path(config.paths.results_dir) / str(config.split_namespace) / condition / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    write_json(out_dir / "selection_metrics.json", {"split": str(config.selection_split), **metrics})
    manifest = {
        "dataset": "ham10000",
        "protocol": protocol_name,
        "condition": condition,
        "seed": int(seed),
        "loss": "cross_entropy",
        "prediction_activation": "softmax",
        "classes": list(CLASSIFIER_TARGET_LABELS),
        **weighting_record,
        "model": dict(config.model),
        "budget": {"max_steps": budget.max_steps, "batch_size": budget.batch_size, "learning_rate": budget.learning_rate, "weight_decay": budget.weight_decay},
        "final_loss": history[-1]["loss"] if history else None,
        "data": provenance,
        "final_eval_heldout_read": False,
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    return {"out_dir": str(out_dir), "metrics": metrics, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL, choices=sorted(PROTOCOLS))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    protocol = get_protocol(args.protocol)
    if args.condition not in protocol.conditions:
        raise Stage4ConfigError(
            f"condition {args.condition!r} is not in protocol {protocol.name}: {list(protocol.conditions)}"
        )
    config = load_named_config(protocol.stage4_config, "ham_stage4")
    if args.seed not in list(config.seeds):
        raise Stage4ConfigError(f"seed {args.seed} is not in the frozen seed list {list(config.seeds)}")
    result = run(config, args.condition, args.seed, args.device, protocol.name)
    print(json.dumps({"out_dir": result["out_dir"], "balanced_accuracy": result["metrics"]["balanced_accuracy"], "macro_f1": result["metrics"]["macro_f1"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

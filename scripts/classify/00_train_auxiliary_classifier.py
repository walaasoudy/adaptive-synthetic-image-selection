#!/usr/bin/env python3
"""Train the auxiliary real-only reference classifier (docs/stages2_to_5_plan.md §2).

This is an ASISM PREREQUISITE, not condition A. It exists so Stage 3's uncertainty (§4.3),
explainability (§4.4), and agreement (§4.5) signals have a real-data classifier to query. Condition
A is a separate experiment trained later, from scratch, on classifier_train under §7's frozen
protocol.

Trained on gen_train, validated on gen_val — never classifier_train/classifier_val (those are
reserved unspent for A/B/C) and never a heldout split. Using the generator's own splits for a scorer
of the generator's output is the point: it keeps the entire classifier development pool free.

Initialization is leakage-safe by construction: CheXpert-pretrained weights are REJECTED in
scripts/utils/classifier.build_model(), because their training patients cannot be shown disjoint
from final_eval_heldout.

Usage:
    python scripts/classify/00_train_auxiliary_classifier.py [--namespace dev] [--max-steps N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.classifier import (  # noqa: E402
    TrainingBudget,
    records_from_split,
    require_torch,
    train_classifier,
)
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.experiment_registry import ExperimentRun  # noqa: E402
from scripts.utils.artifact_contracts import auxiliary_checkpoint_path, current_code_identity_hash, namespace_identity  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import load_split, split_provenance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_named_config("stage3_asism.yaml", "stage3")
    stage1_cfg = load_stage1_config()

    namespace = args.namespace or str(config.split_namespace)
    provenance = namespace_identity(namespace)

    images_dir = Path(stage1_cfg.paths.images_dir) / namespace
    train_frame = load_split("gen_train", namespace, purpose="schema_validation", caller="aux_clf")
    val_frame = load_split("gen_val", namespace, purpose="schema_validation", caller="aux_clf")

    train_records = records_from_split(train_frame, images_dir / "gen_train")
    val_records = records_from_split(val_frame, images_dir / "gen_val")

    if not train_records:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed gen_train images found under {images_dir / 'gen_train'}.\n"
            "Stage 1 preprocessing (scripts/data/03_preprocess_images.py) must run first."
        )
    if not val_records:
        raise SystemExit(
            f"UPSTREAM GATE: no preprocessed gen_val images found under {images_dir / 'gen_val'}."
        )

    require_torch()

    checkpoint_path = auxiliary_checkpoint_path(config, namespace)
    budget = TrainingBudget(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        eval_every_n_steps=max(100, args.max_steps // 10),
    )

    registry_config = {
        "auxiliary_classifier": OmegaConf.to_container(config.auxiliary_classifier, resolve=True),
        "budget": budget.__dict__,
        "split_provenance": provenance,
    }

    with ExperimentRun(
        stage="auxiliary_reference_classifier",
        config=registry_config,
        dataset_version=provenance["split_manifest_hash"],
    ) as run:
        model, accounting, history = train_classifier(
            train_records=train_records,
            val_records=val_records,
            budget=budget,
            dropout_p=float(config.auxiliary_classifier.dropout_p),
            pretrained_source=str(config.auxiliary_classifier.pretrained_source),
            resolution=int(config.auxiliary_classifier.resolution),
            progress_desc="aux-classifier",
            checkpoint_path=checkpoint_path,
            resume_from=checkpoint_path if checkpoint_path.is_file() else None,
            checkpoint_metadata={
                "split_manifest_hash": provenance["split_manifest_hash"],
                "split_namespace": namespace,
                "dataset_id": "auxiliary_gen_train_gen_val",
                "config_hash": hash_dict(registry_config),
                "label_policy": "chexpert_mask_uncertain_v1",
                "condition_or_draw_id": "auxiliary",
                "code_version": get_git_commit_hash(),
                "code_identity_sha256": current_code_identity_hash(),
            },
        )
        best = max(history, key=lambda entry: entry["macro_auroc"]) if history else {}
        accounting.config_hash = hash_dict(registry_config)

        run.set_checkpoint_path(checkpoint_path)
        run.set_metrics({"best_macro_auroc": best.get("macro_auroc"), **accounting.to_dict()})

    write_json(
        checkpoint_path.with_suffix(".manifest.json"),
        {
            "stage": "auxiliary_reference_classifier",
            "role": "ASISM prerequisite scorer — NOT condition A",
            "trained_on": "gen_train",
            "validated_on": "gen_val",
            "never_touched": ["classifier_train", "classifier_val", "asism_tuning_heldout", "final_eval_heldout"],
            "pretrained_source": str(config.auxiliary_classifier.pretrained_source),
            "pretrained_provenance": {
                "note": "ImageNet weights via torchvision; no CheXpert patients involved.",
                "chexpert_pretrained_rejected": True,
            },
            "split_provenance": provenance,
            "accounting": accounting.to_dict(),
            "eval_history": history,
            "config_hash": hash_dict(registry_config),
            "resume_checkpoint": str(checkpoint_path),
            "resume_checkpoint_sha256": sha256_file(checkpoint_path),
            "best_checkpoint": accounting.extra.get("best_checkpoint_path"),
            "best_checkpoint_sha256": sha256_file(accounting.extra["best_checkpoint_path"]),
            "code_identity_sha256": current_code_identity_hash(),
            "git_commit_hash": get_git_commit_hash(),
        },
    )

    print(f"Auxiliary classifier -> {checkpoint_path}", flush=True)
    if history:
        print(f"Best gen_val macro-AUROC: {max(e['macro_auroc'] for e in history):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

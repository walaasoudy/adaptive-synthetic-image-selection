#!/usr/bin/env python3
"""Measure real-only and real+subset macro-AUROC for every frozen utility recipe.

Runs are resumable at (subset, fold, seed) granularity. The same proxy budget is used for the
baseline and augmented run; evaluation is restricted to asism_tuning_heldout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.learned import read_jsonl  # noqa: E402
from scripts.utils.artifact_contracts import stage2_paths, stage3_paths  # noqa: E402
from scripts.utils.classifier import TrainingBudget, evaluate_classifier, records_from_split, records_from_synthetic_manifest, train_classifier  # noqa: E402
from scripts.utils.config import load_named_config, load_stage1_config  # noqa: E402
from scripts.utils.manifest import hash_dict  # noqa: E402
from scripts.utils.splits import load_split  # noqa: E402


def proxy_run_count(n_recipes: int) -> dict:
    """How many proxy trainings --phase run performs. Informational only: there is no GPU-hour cap.
    The search size is fixed by subset_design.total_subsets before any result is seen."""
    return {
        "n_recipes": n_recipes,
        "total_proxy_runs": 1 + n_recipes,
        "formula": "total = 1 (real-only baseline) + n_recipes",
    }


def train_and_score(train_records, eval_records, cfg, seed, checkpoint, tag):
    budget = TrainingBudget(max_steps=int(cfg.tuning.coarse.proxy_max_steps),
                            batch_size=int(cfg.tuning.proxy.batch_size),
                            learning_rate=float(cfg.tuning.proxy.learning_rate), seed=seed,
                            eval_every_n_steps=0)
    model, _, _ = train_classifier(
        train_records=train_records, val_records=[], budget=budget,
        dropout_p=float(cfg.tuning.proxy.dropout_p),
        pretrained_source=str(cfg.tuning.proxy.pretrained_source),
        resolution=int(cfg.tuning.proxy.resolution), progress_desc=tag,
        checkpoint_path=checkpoint, resume_from=checkpoint if checkpoint.is_file() else None,
        checkpoint_metadata={"dataset_id": tag, "condition_or_draw_id": tag,
                             "config_hash": hash_dict({"tag": tag, "seed": seed}, length=64),
                             "label_policy": "chexpert_mask_uncertain_v1"})
    return float(evaluate_classifier(model, eval_records, int(cfg.tuning.proxy.resolution))["macro_auroc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Pilot-only number of recipes")
    parser.add_argument("--phase", choices=["estimate", "run"], default="run")
    args = parser.parse_args()
    cfg = load_named_config("stage3_asism.yaml", "stage3"); stage2 = load_named_config("stage2_generation.yaml", "stage2")
    namespace = args.namespace or str(cfg.split_namespace)
    for key, value in stage3_paths(cfg, namespace).items():
        if key in cfg.paths: cfg.paths[key] = str(value)
    for key, value in stage2_paths(stage2, namespace).items():
        if key in stage2.paths: stage2.paths[key] = str(value)
    if args.phase == "estimate":
        # Uses the CONFIGURED total_subsets, not an actual utility_subsets.jsonl — that file need
        # not exist yet.
        n_recipes = int(cfg.learned_asism.subset_design.total_subsets)
        if args.limit is not None:
            n_recipes = min(n_recipes, args.limit)
        print(json.dumps(proxy_run_count(n_recipes), indent=2), flush=True)
        return 0

    recipes = read_jsonl(Path(cfg.paths.utility_subsets))
    if args.limit is not None: recipes = recipes[:args.limit]
    print(f"{proxy_run_count(len(recipes))['total_proxy_runs']} proxy runs planned", flush=True)
    output = Path(cfg.paths.utility_results); output.parent.mkdir(parents=True, exist_ok=True)
    completed = {(row["subset_id"], int(row["fold"]), int(row["seed"])) for row in read_jsonl(output)} if output.is_file() else set()
    stage1 = load_stage1_config(); image_root = Path(stage1.paths.images_dir) / namespace
    real = records_from_split(load_split("classifier_train", namespace, purpose="schema_validation", caller="learned_asism_utility"), image_root / "classifier_train")
    evaluation = records_from_split(load_split("asism_tuning_heldout", namespace, purpose="schema_validation", caller="learned_asism_utility"), image_root / "asism_tuning_heldout")
    checkpoint_dir = Path(cfg.paths.learned_dir) / "proxy_checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.learned_asism.subset_design.seed)
    baseline_checkpoint = checkpoint_dir / f"real_only_seed{seed}.pt"
    baseline_score = train_and_score(real, evaluation, cfg, seed, baseline_checkpoint, f"learned-real-only-s{seed}")
    with open(output, "a", encoding="utf-8", buffering=1) as handle:
        for index, recipe in enumerate(recipes):
            key = (recipe["subset_id"], 0, seed)
            if key in completed: continue
            synthetic = records_from_synthetic_manifest(Path(stage2.paths.manifest_path), Path(stage2.paths.images_dir), recipe["image_ids"])
            checkpoint = checkpoint_dir / f"{recipe['subset_id']}_seed{seed}.pt"
            augmented = train_and_score(real + synthetic, evaluation, cfg, seed, checkpoint, f"learned-{recipe['subset_id']}-s{seed}")
            row = {"subset_id": recipe["subset_id"], "fold": 0, "seed": seed,
                   "role": recipe.get("role"),
                   "real_only_macro_auroc": baseline_score, "augmented_macro_auroc": augmented,
                   "utility_delta": augmented - baseline_score}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[{index + 1}/{len(recipes)}] {recipe['subset_id']}: utility={row['utility_delta']:+.6f}", flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 3c — build controlled candidate subsets, then MEASURE what each one does downstream.

The ranking network needs a target, and the only honest target is measured: train a proxy classifier
on the real training split plus one subset of synthetic candidates, and see what happens to balanced
accuracy on asism_tuning_heldout relative to a real-only baseline. Nothing here estimates utility
from the signals themselves — that would be circular, since the signals are the model's inputs.

THREE PHASES, deliberately separate commands:

    --phase feasibility   CPU, seconds. Can the configured design be drawn from the real pool without
                          replacement? Writes a report. Nothing is built.
    --phase build         CPU, seconds. Writes utility_subsets.jsonl. Refuses unless the newest
                          feasibility report matches this config and shows zero failures.
    --phase measure       GPU, and the expensive one: one proxy training run per subset, plus one
                          real-only baseline per seed. Resumable — an already-measured subset is
                          skipped, so an interrupted run continues rather than restarting.

WHY THE PHASES ARE SEPARATE. The feasibility numbers must be seen BEFORE any subset exists, so that
a design which cannot be drawn is revised rather than quietly truncated; and the build must be
frozen before any utility is measured, so that no subset can be added or dropped after someone has
seen which ones scored well.

WHAT IS NEVER READ. final_eval_heldout. Utility is measured on asism_tuning_heldout, which exists
precisely so that the selector can be tuned against real held-out performance without touching the
split Stage 5 reports on.

Usage:
    python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase feasibility
    python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase build
    python scripts/asism/ham10000_03_build_utility_subsets.py --namespace ham-stratified-v1 --phase measure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_ranking import (  # noqa: E402
    DEFAULT_UTILITY_METRIC,
    PoolGate,
    active_feature_columns,
    contributing_signals,
    load_candidate_pool,
    pool_feasibility_report,
    utility_columns,
    write_jsonl,
)
# Dataset-agnostic: these operate on an image_id frame and know nothing about labels.
from scripts.asism.learned import (  # noqa: E402
    build_role_conditioned_subsets,
    read_jsonl,
    split_image_pool,
    verify_built_subsets,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, hash_dict, read_json, write_json  # noqa: E402

TUNING_SPLIT = "asism_tuning_heldout"
FORBIDDEN_SPLIT = "final_eval_heldout"


def design_hash(design_cfg, feature_columns: list[str], metric: str) -> str:
    """Identifies the design a feasibility report was computed for.

    --phase build compares this against the report on disk, so a config edited after a report was
    written cannot be built from that report's blessing.
    """
    from omegaconf import OmegaConf

    return hash_dict(
        {
            "design": OmegaConf.to_container(design_cfg, resolve=True),
            "feature_columns": list(feature_columns),
            "utility_metric": metric,
        }
    )


def resolve_features(namespace: str, stage3, stage2_root: Path):
    frame, surviving, pool_report = load_candidate_pool(namespace, stage3, stage2_root)
    configured = [str(column) for column in stage3.learned_asism.feature_columns]
    columns = active_feature_columns(configured, surviving)
    return frame, surviving, columns, pool_report


# ==============================================================================================
# Phases
# ==============================================================================================


def run_feasibility(namespace: str, stage3, stage2_root: Path, out_dir: Path) -> dict:
    frame, surviving, columns, pool_report = resolve_features(namespace, stage3, stage2_root)
    design = stage3.learned_asism.subset_design
    metric = str(stage3.learned_asism.utility_metric)

    report = pool_feasibility_report(
        frame,
        columns,
        [int(size) for size in design.subset_sizes],
        int(design.quantile_bins),
        float(design.val_pool_fraction),
        int(design.total_subsets),
        {key: int(value) for key, value in dict(design.feasibility_thresholds).items()},
    )
    report.update(
        {
            "namespace": namespace,
            "surviving_signals": surviving,
            "contributing_signals": contributing_signals(columns),
            "feature_columns": columns,
            "utility_metric": metric,
            "design_hash": design_hash(design, columns, metric),
            "pool": pool_report,
            "git_commit_hash": get_git_commit_hash(),
        }
    )
    path = out_dir / "utility_subset_feasibility.json"
    write_json(path, report)

    print(f"Feasibility -> {path}", flush=True)
    print(f"  candidates {report['n_candidates']}  train pool {report['train_pool_size']}  "
          f"val pool {report['val_pool_size']}", flush=True)
    for failure in report["failures"]:
        print(f"  FAIL: {failure}", flush=True)
    if report["feasible"]:
        print("  design is drawable from this pool", flush=True)
    return report


def run_build(namespace: str, stage3, stage2_root: Path, out_dir: Path) -> dict:
    frame, surviving, columns, _ = resolve_features(namespace, stage3, stage2_root)
    design = stage3.learned_asism.subset_design
    metric = str(stage3.learned_asism.utility_metric)
    expected_hash = design_hash(design, columns, metric)

    feasibility_path = out_dir / "utility_subset_feasibility.json"
    if not feasibility_path.is_file():
        raise PoolGate(
            f"no feasibility report at {feasibility_path}. Run --phase feasibility first: a design "
            "that cannot be drawn without replacement must be revised before it is built, not "
            "truncated afterwards."
        )
    feasibility = read_json(feasibility_path)
    if feasibility.get("design_hash") != expected_hash:
        raise PoolGate(
            "the feasibility report was computed for a different design than the one configured now "
            f"({feasibility.get('design_hash')} vs {expected_hash}). Re-run --phase feasibility."
        )
    if not feasibility.get("feasible"):
        raise PoolGate(
            "the most recent feasibility report lists failures:\n  "
            + "\n  ".join(feasibility.get("failures", []))
            + "\nRevise the design. No size or ratio is adjusted automatically to make it pass."
        )

    val_fraction = float(design.val_pool_fraction)
    total = int(design.total_subsets)
    val_total = round(total * val_fraction)
    train_total = total - val_total

    train_frame, val_frame = split_image_pool(frame, val_fraction, int(design.seed))
    records = build_role_conditioned_subsets(
        train_frame,
        val_frame,
        columns,
        train_total,
        val_total,
        [int(size) for size in design.subset_sizes],
        float(design.random_fraction),
        float(design.single_signal_fraction),
        float(design.mixed_fraction),
        int(design.seed),
        quantile_bins=int(design.quantile_bins),
    )
    verify_built_subsets(records, train_total, val_total)

    exposures = pd.Series([image_id for record in records for image_id in record["image_ids"]]).value_counts()
    minimum_exposures = int(design.minimum_image_exposures)
    under_exposed = int((exposures < minimum_exposures).sum()) + int(len(frame) - len(exposures))

    subsets_path = out_dir / "utility_subsets.jsonl"
    write_jsonl(subsets_path, records)
    manifest = {
        "namespace": namespace,
        "n_subsets": len(records),
        "train_subsets": train_total,
        "val_subsets": val_total,
        "feature_columns": columns,
        "surviving_signals": surviving,
        "contributing_signals": contributing_signals(columns),
        "utility_metric": metric,
        "design_hash": expected_hash,
        "feasibility_report_hash": feasibility.get("design_hash"),
        # Reported, not enforced: an image measured in too few subsets gets a noisy marginal target,
        # which the training stage handles by requiring a minimum exposure per image there. Failing
        # the build over it would discard a design that is otherwise sound.
        "images_below_minimum_exposure": under_exposed,
        "minimum_image_exposures": minimum_exposures,
        "git_commit_hash": get_git_commit_hash(),
    }
    write_json(out_dir / "utility_subsets_manifest.json", manifest)
    print(f"Built {len(records)} subsets ({train_total} train / {val_total} val) -> {subsets_path}", flush=True)
    return manifest


# ----------------------------------------------------------------------------------------------
# Measurement — the GPU phase
# ----------------------------------------------------------------------------------------------


def _split_records(splits_dir: Path, images_root: Path, split: str):
    from scripts.utils.ham10000_classifier import records_from_split

    path = Path(splits_dir) / f"{split}.csv"
    if not path.is_file():
        raise PoolGate(f"UPSTREAM GATE: missing split {path}; build and freeze the splits first.")
    return records_from_split(pd.read_csv(path), Path(images_root) / split)


def _candidate_records(frame: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, dict]:
    from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis

    index_of = {label: position for position, label in enumerate(CLASSIFIER_TARGET_LABELS)}
    paths = dict(zip(manifest["image_id"].astype(str), manifest["image_path"].astype(str)))
    records = {}
    for image_id, diagnosis in zip(frame["image_id"].astype(str), frame["dx"]):
        records[image_id] = {
            "image_id": image_id,
            "image_path": paths[image_id],
            "class_index": index_of[normalize_diagnosis(diagnosis)],
        }
    return records


def _measure(train_records, tuning_records, proxy_cfg, seed: int, device: str | None) -> dict:
    from scripts.utils.classifier import TrainingBudget
    from scripts.utils.ham10000_classifier import predict_probabilities, train_classifier, true_class_indices
    from scripts.utils.ham10000_metrics import full_metric_suite

    budget = TrainingBudget(
        max_steps=int(proxy_cfg.max_steps),
        batch_size=int(proxy_cfg.batch_size),
        learning_rate=float(proxy_cfg.learning_rate),
        weight_decay=float(proxy_cfg.weight_decay),
        seed=int(seed),
        eval_every_n_steps=10**9,  # no intermediate evaluation: the budget is fixed and final
    )
    model, _ = train_classifier(
        train_records,
        budget,
        float(proxy_cfg.dropout_p),
        str(proxy_cfg.pretrained_source),
        int(proxy_cfg.resolution),
        device=device,
        progress_desc="proxy",
    )
    probabilities = predict_probabilities(model, tuning_records, int(proxy_cfg.resolution), device=device)
    return full_metric_suite(probabilities, true_class_indices(tuning_records))


def run_measure(namespace: str, stage1, stage2, stage3, splits_cfg, out_dir: Path,
                device: str | None = None, limit: int | None = None) -> dict:
    subsets_path = out_dir / "utility_subsets.jsonl"
    if not subsets_path.is_file():
        raise PoolGate(f"no subsets at {subsets_path}; run --phase build first.")
    subsets = read_jsonl(subsets_path)

    learned = stage3.learned_asism
    metric = str(learned.utility_metric)
    real_column, augmented_column = utility_columns(metric)
    proxy_cfg = learned.proxy
    seed = int(proxy_cfg.seed)

    splits_dir = Path(splits_cfg.paths.splits_root) / namespace
    images_root = Path(stage1.paths.images_dir) / namespace
    if FORBIDDEN_SPLIT in {str(learned.real_train_split), TUNING_SPLIT}:
        raise PoolGate("the protected split cannot be used to measure utility")

    real_records = _split_records(splits_dir, images_root, str(learned.real_train_split))
    tuning_records = _split_records(splits_dir, images_root, TUNING_SPLIT)

    frame, _, _ = load_candidate_pool(namespace, stage3, Path(stage2.paths.stage2_root), verbose=False)
    manifest = pd.read_csv(Path(stage2.paths.stage2_root) / namespace / "all_candidates.csv")
    candidate_records = _candidate_records(frame, manifest)

    results_path = out_dir / "utility_results.jsonl"
    done = {row["subset_id"]: row for row in read_jsonl(results_path)} if results_path.is_file() else {}

    # ONE real-only baseline per seed, trained once and reused for every subset. It is the same model
    # each time by construction (same records, same budget, same seed), so retraining it per subset
    # would multiply the GPU cost by two for an identical number.
    baseline_path = out_dir / "utility_baseline.json"
    if baseline_path.is_file():
        baseline = read_json(baseline_path)
    else:
        print(f"Baseline: real-only proxy on {len(real_records)} images, seed {seed}", flush=True)
        metrics = _measure(real_records, tuning_records, proxy_cfg, seed, device)
        baseline = {
            "split": str(learned.real_train_split),
            "evaluated_on": TUNING_SPLIT,
            "seed": seed,
            "metrics": metrics,
            "git_commit_hash": get_git_commit_hash(),
        }
        write_json(baseline_path, baseline)
    baseline_value = float(baseline["metrics"][metric])

    pending = [row for row in subsets if row["subset_id"] not in done]
    if limit:
        pending = pending[: int(limit)]
    print(
        f"Measuring {len(pending)} subset(s); {len(done)} already measured. "
        f"Baseline {metric} = {baseline_value:.4f}",
        flush=True,
    )

    for position, record in enumerate(pending, start=1):
        missing = [image_id for image_id in record["image_ids"] if image_id not in candidate_records]
        if missing:
            raise PoolGate(
                f"subset {record['subset_id']} references {len(missing)} candidate(s) absent from the "
                f"current pool (e.g. {missing[:3]}); the subsets were built against a different pool"
            )
        synthetic = [candidate_records[image_id] for image_id in record["image_ids"]]
        metrics = _measure(real_records + synthetic, tuning_records, proxy_cfg, seed, device)
        row = {
            "subset_id": record["subset_id"],
            "role": record["role"],
            "size": record["size"],
            "seed": seed,
            real_column: baseline_value,
            augmented_column: float(metrics[metric]),
            "utility_metric": metric,
            "augmented_metrics": metrics,
        }
        done[record["subset_id"]] = row
        # Appended after every subset: the expensive phase must survive an interrupted pod.
        write_jsonl(results_path, [done[key] for key in sorted(done)])
        print(
            f"  [{position}/{len(pending)}] {record['subset_id']} size={record['size']} "
            f"{metric}={metrics[metric]:.4f} delta={metrics[metric] - baseline_value:+.4f}",
            flush=True,
        )

    return {"results_path": str(results_path), "measured": len(done), "planned": len(subsets)}


# ==============================================================================================


def run(namespace: str, phase: str, device: str | None = None, limit: int | None = None) -> dict:
    stage1 = load_named_config("ham10000_stage1.yaml", "ham_stage1")
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")

    out_dir = Path(stage3.paths.outputs_dir) / namespace / "learned"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage2_root = Path(stage2.paths.stage2_root)

    if phase == "feasibility":
        return run_feasibility(namespace, stage3, stage2_root, out_dir)
    if phase == "build":
        return run_build(namespace, stage3, stage2_root, out_dir)
    if phase == "measure":
        return run_measure(namespace, stage1, stage2, stage3, splits_cfg, out_dir, device, limit)
    raise ValueError(f"unknown phase {phase!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--phase", required=True, choices=["feasibility", "build", "measure"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="measure only N more subsets this run")
    args = parser.parse_args()

    result = run(args.namespace, args.phase, device=args.device, limit=args.limit)
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, dict)}, indent=2))
    return 0 if result.get("feasible", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

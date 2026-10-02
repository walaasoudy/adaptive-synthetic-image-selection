#!/usr/bin/env python3
"""ASISM v2 final comparison: the frozen Stage 5 analysis plus the predeclared seed-robustness
analysis, on either of two prediction sources.

  --source classifier_val   the Stage 4 monitoring split. Each Stage 4 run already wrote its
                            selection_predictions.parquet; they are gathered here. Permitted by the
                            contract (§2: Stage 4 monitoring) and labelled MONITORING everywhere: it
                            is not the final evaluation and decides nothing.
  --source final            a Stage 5 run directory written by ham10000_stage5_evaluate.py after
                            the test-set policy is signed. Only its prediction files are read.

The analysis itself is not chosen here: compare() is ham10000_compare_conditions.compare (lesion
bootstrap, 2,000 resamples, seed 42, Holm on the confirmatory family, BH on the exploratory one),
and robustness() is ham10000_seed_robustness (protocol §5).

Usage:
    python -m scripts.followup.ham10000_asism_v2_compare --protocol asism_v2 --source classifier_val
    python -m scripts.followup.ham10000_asism_v2_compare --protocol asism_v2 --source final --run-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.classify.ham10000_aggregate_conditions import (
    check_fairness,
    check_protected_split,
    check_training_data_differs,
    discover_runs,
)
from scripts.eval import ham10000_compare_conditions as frozen
from scripts.eval.ham10000_seed_robustness import robustness
from scripts.utils.config import load_named_config
from scripts.utils.ham10000_conditions import get_protocol
from scripts.utils.manifest import get_git_commit_hash, sha256_file, write_json

N_RESAMPLES, SEED, ALPHA = 2000, 42, 0.05


def gather_classifier_val(protocol, namespace: str) -> Path:
    stage4 = load_named_config(protocol.stage4_config, "ham_stage4")
    seeds = [int(s) for s in stage4.seeds]
    runs, missing = discover_runs(Path(stage4.paths.results_dir), namespace, seeds, conditions=protocol.conditions)
    if missing:
        raise SystemExit("the grid is incomplete; missing " + ", ".join(f"{c}/seed{s}" for c, s in missing))
    check_protected_split(runs)
    check_fairness(runs)
    check_training_data_differs(runs, protocol)
    run_dir = Path(stage4.paths.results_dir) / namespace / "_classifier_val_comparison"
    predictions = run_dir / "predictions"
    if predictions.is_dir():
        shutil.rmtree(predictions)          # a pure copy of the per-run files, rebuilt every time
    predictions.mkdir(parents=True)
    hashes = {}
    for (condition, seed), entry in sorted(runs.items()):
        source = Path(entry["run_dir"]) / "selection_predictions.parquet"
        if not source.is_file():
            raise SystemExit(f"{source} is missing; this run predates the predictions file")
        target = predictions / f"{condition}_seed{seed}.parquet"
        shutil.copyfile(source, target)
        hashes[target.name] = sha256_file(target)
    write_json(run_dir / "stage5_manifest.json", {
        "final_eval_run_id": None, "namespace": namespace, "split": "classifier_val",
        "purpose": "MONITORING: Stage 4 monitoring split, not the final evaluation", "predictions_sha256": hashes})
    return run_dir


def run(protocol_name: str, source: str, namespace: str, run_dir: Path | None = None) -> dict:
    protocol = get_protocol(protocol_name)
    if source == "classifier_val":
        run_dir = gather_classifier_val(protocol, namespace)
    elif run_dir is None:
        raise SystemExit("--source final needs --run-dir (a Stage 5 run directory)")
    by_condition, reference = frozen.load_predictions(Path(run_dir), protocol.conditions)
    result = frozen.compare(by_condition, reference, N_RESAMPLES, SEED, ALPHA, protocol)
    pairs = list(protocol.confirmatory) + list(protocol.exploratory)
    result["seed_robustness"] = robustness(by_condition, reference, pairs, N_RESAMPLES, SEED, ALPHA)
    result.update(source=source, split="classifier_val" if source == "classifier_val" else "final_eval_heldout",
                  label=("MONITORING (classifier_val): not the final evaluation" if source == "classifier_val"
                         else "FINAL EVALUATION"),
                  interpretation=protocol.interpretation, git_commit_hash=get_git_commit_hash())
    out = Path(run_dir) / f"asism_v2_comparison_{source}.json"
    write_json(out, result)
    print(f"{result['label']} -> {out}")
    for name, entry in result["confirmatory"].items():
        print(f"  {name}: {entry['observed_difference']:+.4f} [{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}] "
              f"p={entry['p_value']:.4f}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["asism_v2", "asism_v2_none"])
    parser.add_argument("--source", required=True, choices=["classifier_val", "final"])
    parser.add_argument("--namespace", default="ham-stratified-v1")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.protocol, args.source, args.namespace, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

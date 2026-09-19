#!/usr/bin/env python3
"""Stage 5 analysis — compare A/B/C on the protected split, with intervals, and answer the thesis.

Reads only the prediction tables ham10000_stage5_evaluate.py wrote. No torch, no GPU, and the
protected split is never re-opened: every number here comes from predictions already made, so the
analysis and its figures can be iterated as often as needed without a second look at held-out data.

THE COMPARISON THAT IS THE THESIS
    A   real only            B   real + ALL synthetic            C   real + SELECTED synthetic
C versus A shows that synthetic data helps. C versus B is the one that tests the contribution: with
the same generator and the same candidates, does SELECTING them beat using all of them? That is
therefore the confirmatory comparison, fixed here before any result is read, and it is the only one
in the family that controls family-wise error. Everything else is exploratory, corrected with FDR,
and labelled as exploratory wherever it appears.

RESAMPLING IS OVER LESIONS, NOT IMAGES. HAM10000 contains several images of the same lesion.
Resampling images would treat them as independent observations, and every interval would come out
narrower than the evidence supports. A lesion's images move in or out of a resample together.

SEEDS ARE PART OF THE ESTIMATE, NOT AVERAGED AWAY FIRST. A condition's metric inside a resample is
the mean over its seeds computed on THAT resample, so the interval carries both the sampling
variation of the split and the run-to-run variation of training. Averaging the three seeds' metrics
first and bootstrapping the average would hide the second.

PRIMARY METRIC: balanced accuracy. On a seven-class problem that is two-thirds nv, plain accuracy is
maximised by ignoring the rare classes — the failure the whole intervention targets.

Usage:
    python scripts/eval/ham10000_compare_conditions.py --run-dir outputs/ham10000/stage5/<ns>/<run-id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS  # noqa: E402
from scripts.utils.ham10000_metrics import (  # noqa: E402
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    one_vs_rest_auroc,
)
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402
from scripts.utils.metrics import (  # noqa: E402
    benjamini_hochberg,
    holm_bonferroni,
    paired_bootstrap_difference,
    patient_level_bootstrap,
)

CONDITIONS = ("A", "B", "C")

# Fixed before any result is read. C vs B is the only comparison that tests the contribution rather
# than the generator, so it is the only one in the confirmatory family.
CONFIRMATORY_COMPARISONS = (("C", "B"),)
EXPLORATORY_COMPARISONS = (("C", "A"), ("B", "A"))

PRIMARY_METRIC = "balanced_accuracy"


def metric_functions() -> dict:
    """Each takes (probabilities, truth) over a set of rows and returns one number."""

    def _balanced(probabilities, truth):
        return balanced_accuracy(confusion_matrix(truth, probabilities.argmax(axis=1), len(CLASSIFIER_TARGET_LABELS)))

    def _macro_f1(probabilities, truth):
        return macro_f1(confusion_matrix(truth, probabilities.argmax(axis=1), len(CLASSIFIER_TARGET_LABELS)))

    def _macro_auroc(probabilities, truth):
        values = one_vs_rest_auroc(probabilities, truth, len(CLASSIFIER_TARGET_LABELS))
        usable = values[np.isfinite(values)]
        return float(usable.mean()) if usable.size else float("nan")

    def _accuracy(probabilities, truth):
        return float((probabilities.argmax(axis=1) == truth).mean()) if len(truth) else float("nan")

    return {
        "balanced_accuracy": _balanced,
        "macro_f1": _macro_f1,
        "macro_auroc_ovr": _macro_auroc,
        "accuracy": _accuracy,
    }


def per_class_recall_function(class_index: int):
    def _recall(probabilities, truth):
        mask = truth == class_index
        if not mask.any():
            return float("nan")
        return float((probabilities[mask].argmax(axis=1) == class_index).mean())

    return _recall


def load_predictions(run_dir: Path) -> tuple[dict[str, list[pd.DataFrame]], pd.DataFrame]:
    predictions_dir = Path(run_dir) / "predictions"
    if not predictions_dir.is_dir():
        raise SystemExit(
            f"no predictions at {predictions_dir}.\n"
            "Run: python scripts/eval/ham10000_stage5_evaluate.py --namespace <ns> "
            "--final-eval-run-id <id>"
        )
    by_condition: dict[str, list[pd.DataFrame]] = {}
    reference = None
    for path in sorted(predictions_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        condition = str(frame["condition"].iloc[0])
        if reference is None:
            reference = frame[["image_id", "lesion_id", "true_class_index"]].copy()
        elif not frame["image_id"].astype(str).tolist() == reference["image_id"].astype(str).tolist():
            raise SystemExit(
                f"{path.name} covers a different image set than the other prediction files; the "
                "conditions were not evaluated on the same protected split"
            )
        by_condition.setdefault(condition, []).append(frame)

    missing = [condition for condition in CONDITIONS if condition not in by_condition]
    if missing:
        raise SystemExit(
            f"predictions are missing for condition(s) {missing}. The comparison is between all "
            "three conditions or it is not the comparison the thesis makes."
        )
    return by_condition, reference


def _probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"prob_{label}" for label in CLASSIFIER_TARGET_LABELS]].to_numpy(dtype=np.float64)


def condition_metric(frames: list[pd.DataFrame], truth: np.ndarray, metric_fn):
    """A condition's metric on a set of rows: the mean over its seeds, computed on those rows.

    Not the metric of the seed-averaged probabilities — averaging probabilities would build an
    ensemble, which is a different (and better-performing) model than any condition actually
    trained, and would quietly make every condition stronger than what Stage 4 produced.
    """
    matrices = [_probability_matrix(frame) for frame in frames]

    def _evaluate(indices: np.ndarray) -> float:
        values = [metric_fn(matrix[indices], truth[indices]) for matrix in matrices]
        usable = [value for value in values if value is not None and not np.isnan(value)]
        return float(np.mean(usable)) if usable else float("nan")

    return _evaluate


def compare(by_condition, reference, n_resamples: int, seed: int, alpha: float) -> dict:
    truth = reference["true_class_index"].to_numpy(dtype=int)
    lesions = reference["lesion_id"].astype(str).to_numpy()
    metrics = metric_functions()

    per_condition = {}
    for condition in CONDITIONS:
        frames = by_condition[condition]
        entry = {"n_seeds": len(frames), "seeds": sorted(int(frame["seed"].iloc[0]) for frame in frames)}
        for name, metric_fn in metrics.items():
            evaluate = condition_metric(frames, truth, metric_fn)
            entry[name] = patient_level_bootstrap(lesions, evaluate, n_resamples, seed, alpha)
        entry["per_class_recall"] = {}
        for index, label in enumerate(CLASSIFIER_TARGET_LABELS):
            evaluate = condition_metric(frames, truth, per_class_recall_function(index))
            entry["per_class_recall"][label] = patient_level_bootstrap(lesions, evaluate, n_resamples, seed, alpha)
        per_condition[condition] = entry

    def _difference(left: str, right: str, metric_fn) -> dict:
        return paired_bootstrap_difference(
            lesions,
            condition_metric(by_condition[left], truth, metric_fn),
            condition_metric(by_condition[right], truth, metric_fn),
            n_resamples,
            seed,
            alpha,
        )

    confirmatory, exploratory = {}, {}
    for left, right in CONFIRMATORY_COMPARISONS:
        confirmatory[f"{left}_vs_{right}:{PRIMARY_METRIC}"] = _difference(left, right, metrics[PRIMARY_METRIC])
    for left, right in EXPLORATORY_COMPARISONS:
        for name, metric_fn in metrics.items():
            exploratory[f"{left}_vs_{right}:{name}"] = _difference(left, right, metric_fn)
    # Every non-primary metric of the confirmatory pair is exploratory too: only the pre-registered
    # metric on the pre-registered pair carries the confirmatory claim.
    for left, right in CONFIRMATORY_COMPARISONS:
        for name, metric_fn in metrics.items():
            if name == PRIMARY_METRIC:
                continue
            exploratory[f"{left}_vs_{right}:{name}"] = _difference(left, right, metric_fn)
        for index, label in enumerate(CLASSIFIER_TARGET_LABELS):
            exploratory[f"{left}_vs_{right}:recall_{label}"] = _difference(
                left, right, per_class_recall_function(index)
            )

    confirmatory_corrected = holm_bonferroni(
        {name: entry["p_value"] for name, entry in confirmatory.items()}, alpha
    )
    exploratory_corrected = benjamini_hochberg(
        {name: entry["p_value"] for name, entry in exploratory.items()}, alpha
    )
    for name, entry in confirmatory.items():
        entry.update(confirmatory_corrected.get(name, {}))
        entry["family"] = "confirmatory"
    for name, entry in exploratory.items():
        entry.update(exploratory_corrected.get(name, {}))
        entry["family"] = "exploratory"

    return {
        "per_condition": per_condition,
        "confirmatory": confirmatory,
        "exploratory": exploratory,
        "primary_metric": PRIMARY_METRIC,
        "confirmatory_family": [f"{left}_vs_{right}:{PRIMARY_METRIC}" for left, right in CONFIRMATORY_COMPARISONS],
        "resampling_unit": "lesion_id",
        "n_lesions": int(len(set(lesions))),
        "n_images": int(len(truth)),
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
    }


def run(run_dir: Path, n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "stage5_manifest.json"
    stage5 = read_json(manifest_path) if manifest_path.is_file() else {}

    by_condition, reference = load_predictions(run_dir)
    result = compare(by_condition, reference, n_resamples, seed, alpha)
    result.update(
        {
            "stage": "ham10000_stage5_compare",
            "run_dir": str(run_dir),
            "final_eval_run_id": stage5.get("final_eval_run_id"),
            "namespace": stage5.get("namespace"),
            "split": stage5.get("split"),
            "split_csv_sha256": stage5.get("split_csv_sha256"),
            "selection_threshold_policy": stage5.get("selection_threshold_policy"),
            "interpretation": (
                "C vs B is the confirmatory test of the contribution: same generator, same "
                "candidates, selection the only difference. C vs A shows only that synthetic data "
                "helps. Every p-value is reported with its effect size and interval; significance "
                "alone is not a result."
            ),
            "git_commit_hash": get_git_commit_hash(),
        }
    )
    write_json(run_dir / "stage5_comparison.json", result)

    rows = []
    for condition in CONDITIONS:
        entry = result["per_condition"][condition]
        row = {"condition": condition, "n_seeds": entry["n_seeds"]}
        for name in metric_functions():
            row[name] = entry[name]["point_estimate"]
            row[f"{name}_ci_lower"] = entry[name]["ci_lower"]
            row[f"{name}_ci_upper"] = entry[name]["ci_upper"]
        for label in CLASSIFIER_TARGET_LABELS:
            row[f"recall_{label}"] = entry["per_class_recall"][label]["point_estimate"]
        rows.append(row)
    table_path = run_dir / "stage5_condition_table.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    result["table_path"] = str(table_path)

    print(f"Stage 5 comparison — {result.get('namespace')} / {result.get('final_eval_run_id')}", flush=True)
    print(f"  {result['n_images']} images, {result['n_lesions']} lesions, "
          f"{n_resamples} lesion-level bootstrap resamples", flush=True)
    print("=" * 78, flush=True)
    for condition in CONDITIONS:
        entry = result["per_condition"][condition][PRIMARY_METRIC]
        print(
            f"  {condition}  {PRIMARY_METRIC} = {entry['point_estimate']:.4f} "
            f"[{entry['ci_lower']:.4f}, {entry['ci_upper']:.4f}]",
            flush=True,
        )
    print("=" * 78, flush=True)
    print("  CONFIRMATORY (family-wise corrected):", flush=True)
    for name, entry in result["confirmatory"].items():
        print(
            f"    {name}: {entry['observed_difference']:+.4f} "
            f"[{entry['ci_lower']:+.4f}, {entry['ci_upper']:+.4f}]  "
            f"p={entry['p_value']:.4f} adjusted={entry.get('adjusted_p_value', float('nan')):.4f}",
            flush=True,
        )
    print(f"  exploratory: {len(result['exploratory'])} comparisons, FDR-corrected, "
          "labelled exploratory wherever reported", flush=True)
    print(f"\n  -> {run_dir / 'stage5_comparison.json'}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    result = run(Path(args.run_dir), args.n_resamples, args.seed, args.alpha)
    print(json.dumps({"table_path": result["table_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

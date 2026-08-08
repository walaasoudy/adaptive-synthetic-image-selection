#!/usr/bin/env python3
"""Stage 5 analysis — condition comparison, statistics, and result tables (§8).

Reads only the prediction parquet files written by stage5_evaluate.py. No torch, no GPU: the whole
analysis layer runs locally, which is what keeps interpretation and figure iteration off the pod.

Implements the frozen statistical policy (§8.2):
  - patient-level paired bootstrap for effect sizes and 95% CIs;
  - Holm-Bonferroni for the CONFIRMATORY family (primary endpoint / primary comparison);
  - Benjamini-Hochberg FDR for EXPLORATORY analyses, labelled as such everywhere;
  - effect sizes reported alongside every p-value.

Condition D is summarised ACROSS its five draws (mean and spread), never as a single draw.

Usage:
    python scripts/eval/compare_conditions.py --run-dir outputs/stage5/<run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.labels import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    PRIMARY_ENDPOINT_LABELS,
    build_label_arrays,
)
from scripts.utils.manifest import read_json, sha256_file, write_json  # noqa: E402
from scripts.utils.metrics import (  # noqa: E402
    auroc,
    benjamini_hochberg,
    full_metric_suite,
    holm_bonferroni,
    paired_bootstrap_difference,
    patient_level_bootstrap,
)

TAG_PATTERN = re.compile(r"^(?P<condition>[A-E])(?:_draw(?P<draw>\d+))?_seed(?P<seed>\d+)$")

# The CONFIRMATORY family, fixed before results are seen (§8.1/§8.2). Everything else is
# exploratory and corrected with FDR.
CONFIRMATORY_COMPARISONS = [("C", "D")]
SECONDARY_COMPARISONS = [("A", "B"), ("A", "C"), ("A", "E"), ("B", "C")]


def parse_tag(tag: str) -> dict | None:
    match = TAG_PATTERN.match(tag)
    if not match:
        return None
    return {
        "condition": match.group("condition"),
        "draw": int(match.group("draw")) if match.group("draw") is not None else None,
        "seed": int(match.group("seed")),
    }


def load_predictions(run_dir: Path) -> tuple[dict[str, list[pd.DataFrame]], pd.DataFrame]:
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.is_dir():
        raise SystemExit(f"No predictions directory at {predictions_dir}")

    by_condition: dict[str, list[pd.DataFrame]] = defaultdict(list)
    reference = None
    for path in sorted(predictions_dir.glob("*.parquet")):
        info = parse_tag(path.stem)
        if info is None:
            continue
        sidecar_path = path.with_suffix(".complete.json")
        if not sidecar_path.is_file():
            raise SystemExit(f"Prediction completion sidecar missing: {sidecar_path}")
        sidecar = read_json(sidecar_path)
        if sidecar.get("prediction_sha256") != sha256_file(path):
            raise SystemExit(f"Prediction hash mismatch: {path}")
        frame = pd.read_parquet(path)
        frame.attrs.update(info)
        by_condition[info["condition"]].append(frame)
        reference = frame if reference is None else reference

    if reference is None:
        raise SystemExit(f"No prediction files found in {predictions_dir}")
    return by_condition, reference


def build_truth(run_dir: Path, reference: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth arrays for the evaluation rows, aligned to the prediction row order."""
    manifest = read_json(run_dir / "final_eval_run_manifest.json")
    namespace = manifest["namespace"]

    # Non-outcome access: we already hold the outcome-bearing predictions; this reloads the labels
    # for metric computation inside the same declared run.
    from scripts.utils.splits import load_split

    frame = load_split(
        "final_eval_heldout",
        namespace,
        purpose="load_labels_for_evaluation",
        final_eval_run_id=manifest["final_eval_run_id"],
        final_eval_context_path=run_dir / "final_eval_registration.json",
        caller="compare_conditions",
    )
    from scripts.utils.identifiers import sanitize_image_id

    frame["image_id"] = frame["Path"].apply(sanitize_image_id)
    aligned = reference[["image_id"]].merge(frame, on="image_id", how="left")

    for label in CLASSIFIER_TARGET_LABELS:
        if label not in aligned.columns:
            aligned[label] = np.nan
    raw, targets, masks = build_label_arrays(aligned[CLASSIFIER_TARGET_LABELS], CLASSIFIER_TARGET_LABELS)
    return raw, targets, masks


def macro_auroc_for(probabilities: np.ndarray, targets: np.ndarray, masks: np.ndarray,
                    rows: np.ndarray) -> float:
    """Masked macro-AUROC over the 11 primary labels for a subset of rows."""
    values = []
    for label in PRIMARY_ENDPOINT_LABELS:
        index = CLASSIFIER_TARGET_LABELS.index(label)
        mask = masks[rows, index].astype(bool)
        if mask.sum() == 0:
            continue
        y_true = targets[rows][mask, index].astype(int)
        y_score = probabilities[rows][mask, index]
        if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
            continue
        value = auroc(y_score, y_true)
        if not np.isnan(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    by_condition, reference = load_predictions(run_dir)
    _, targets, masks = build_truth(run_dir, reference)
    patient_ids = reference["patient_id"].to_numpy()

    label_columns = CLASSIFIER_TARGET_LABELS

    # Per-condition mean probabilities across seeds (and, for D, across draws AND seeds).
    condition_probabilities: dict[str, np.ndarray] = {}
    condition_spread: dict[str, dict] = {}
    for condition, frames in by_condition.items():
        stacked = np.stack([frame[label_columns].to_numpy() for frame in frames], axis=0)
        condition_probabilities[condition] = stacked.mean(axis=0)

        per_run_macro = []
        for frame in frames:
            value = macro_auroc_for(
                frame[label_columns].to_numpy(), targets, masks, np.arange(len(frame))
            )
            per_run_macro.append(
                {"draw": frame.attrs.get("draw"), "seed": frame.attrs.get("seed"),
                 "macro_auroc": value}
            )
        values = [entry["macro_auroc"] for entry in per_run_macro if not np.isnan(entry["macro_auroc"])]
        condition_spread[condition] = {
            "n_runs": len(frames),
            "per_run": per_run_macro,
            "mean_macro_auroc": float(np.mean(values)) if values else float("nan"),
            "std_macro_auroc": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            "min_macro_auroc": float(np.min(values)) if values else float("nan"),
            "max_macro_auroc": float(np.max(values)) if values else float("nan"),
        }

    # Full metric suite per condition.
    per_condition_metrics = {}
    for condition, probabilities in condition_probabilities.items():
        thresholds_path = next(
            (run_dir / "predictions").glob(f"{condition}*seed*.thresholds.json"), None
        )
        thresholds = read_json(thresholds_path)["thresholds"] if thresholds_path else {}
        suite = full_metric_suite(probabilities, targets, masks, thresholds, label_columns)

        bootstrap = patient_level_bootstrap(
            patient_ids,
            lambda rows, p=probabilities: macro_auroc_for(p, targets, masks, rows),
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )
        suite["macro_auroc_bootstrap"] = bootstrap
        suite["across_run_spread"] = condition_spread[condition]
        per_condition_metrics[condition] = suite

    # Paired comparisons.
    def compare(left: str, right: str) -> dict | None:
        if left not in condition_probabilities or right not in condition_probabilities:
            return None
        return paired_bootstrap_difference(
            patient_ids,
            lambda rows: macro_auroc_for(condition_probabilities[left], targets, masks, rows),
            lambda rows: macro_auroc_for(condition_probabilities[right], targets, masks, rows),
            n_resamples=args.bootstrap_resamples,
            seed=args.seed,
        )

    confirmatory, exploratory = {}, {}
    for left, right in CONFIRMATORY_COMPARISONS:
        result = compare(left, right)
        if result:
            confirmatory[f"{left}_vs_{right}"] = result
    for left, right in SECONDARY_COMPARISONS:
        result = compare(left, right)
        if result:
            exploratory[f"{left}_vs_{right}"] = result

    # Per-label exploratory comparisons for the primary comparison pair.
    per_label_exploratory = {}
    if "C" in condition_probabilities and "D" in condition_probabilities:
        for label in PRIMARY_ENDPOINT_LABELS:
            index = label_columns.index(label)

            def label_auroc(rows, probabilities, idx=index):
                mask = masks[rows, idx].astype(bool)
                if mask.sum() == 0:
                    return float("nan")
                y_true = targets[rows][mask, idx].astype(int)
                if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
                    return float("nan")
                return auroc(probabilities[rows][mask, idx], y_true)

            result = paired_bootstrap_difference(
                patient_ids,
                lambda rows: label_auroc(rows, condition_probabilities["C"]),
                lambda rows: label_auroc(rows, condition_probabilities["D"]),
                n_resamples=max(200, args.bootstrap_resamples // 5),
                seed=args.seed,
            )
            per_label_exploratory[f"C_vs_D::{label}"] = result

    confirmatory_corrected = holm_bonferroni(
        {name: result["p_value"] for name, result in confirmatory.items()}
    )
    exploratory_corrected = benjamini_hochberg(
        {
            **{name: result["p_value"] for name, result in exploratory.items()},
            **{name: result["p_value"] for name, result in per_label_exploratory.items()},
        }
    )

    report = {
        "run_dir": str(run_dir),
        "primary_endpoint": f"macro-AUROC over the {len(PRIMARY_ENDPOINT_LABELS)} primary disease labels",
        "primary_comparison": "C vs D",
        "n_eval_rows": len(reference),
        "n_eval_patients": int(pd.Series(patient_ids).nunique()),
        "per_condition": per_condition_metrics,
        "confirmatory": {
            name: {**result, "correction": confirmatory_corrected.get(name, {})}
            for name, result in confirmatory.items()
        },
        "exploratory": {
            name: {**result, "correction": exploratory_corrected.get(name, {})}
            for name, result in {**exploratory, **per_label_exploratory}.items()
        },
        "statistical_policy": {
            "resampling_unit": "patient",
            "confirmatory_correction": "holm_bonferroni",
            "exploratory_correction": "benjamini_hochberg_fdr",
            "effect_sizes_reported": True,
            "policy_frozen_before_results": True,
        },
    }
    write_json(run_dir / "stage5_comparison_report.json", report)

    # Human-readable tables.
    rows = []
    for condition in sorted(per_condition_metrics):
        suite = per_condition_metrics[condition]
        bootstrap = suite["macro_auroc_bootstrap"]
        spread = suite["across_run_spread"]
        rows.append(
            {
                "condition": condition,
                "macro_auroc": round(suite["macro_auroc_primary"], 4),
                "ci_lower": round(bootstrap["ci_lower"], 4),
                "ci_upper": round(bootstrap["ci_upper"], 4),
                "across_run_mean": round(spread["mean_macro_auroc"], 4),
                "across_run_std": (
                    round(spread["std_macro_auroc"], 4)
                    if not np.isnan(spread["std_macro_auroc"]) else None
                ),
                "n_runs": spread["n_runs"],
                "macro_auprc": round(suite["macro_auprc_primary"], 4),
                "micro_auroc": round(suite["micro_auroc_primary"], 4),
                "mean_brier": round(suite["mean_brier_primary"], 4),
                "mean_ece": round(suite["mean_ece_primary"], 4),
            }
        )
    condition_table = pd.DataFrame(rows)
    condition_table.to_csv(run_dir / "table_conditions.csv", index=False)

    comparison_rows = []
    for family, source, corrected in (
        ("confirmatory", confirmatory, confirmatory_corrected),
        ("exploratory", {**exploratory, **per_label_exploratory}, exploratory_corrected),
    ):
        for name, result in source.items():
            correction = corrected.get(name, {})
            comparison_rows.append(
                {
                    "comparison": name,
                    "family": family,
                    "effect_size": round(result["effect_size"], 4),
                    "ci_lower": round(result["ci_lower"], 4),
                    "ci_upper": round(result["ci_upper"], 4),
                    "p_value": round(result["p_value"], 5),
                    "adjusted_p": (
                        round(correction.get("adjusted_p_value"), 5)
                        if correction.get("adjusted_p_value") is not None
                        and not np.isnan(correction.get("adjusted_p_value", np.nan))
                        else None
                    ),
                    "significant": correction.get("rejected"),
                    "correction": correction.get("method"),
                }
            )
    comparison_table = pd.DataFrame(comparison_rows)
    comparison_table.to_csv(run_dir / "table_comparisons.csv", index=False)

    print(
        f"\nCondition summary (primary endpoint: macro-AUROC over {len(PRIMARY_ENDPOINT_LABELS)} disease labels)",
        flush=True,
    )
    print(condition_table.to_string(index=False), flush=True)
    print("\nComparisons", flush=True)
    print(comparison_table.to_string(index=False), flush=True)
    print(f"\nReport -> {run_dir / 'stage5_comparison_report.json'}", flush=True)
    print(f"Tables -> {run_dir / 'table_conditions.csv'}, {run_dir / 'table_comparisons.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

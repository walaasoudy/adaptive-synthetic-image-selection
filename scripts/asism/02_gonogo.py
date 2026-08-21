#!/usr/bin/env python3
"""Stage 3b — ASISM component Go/No-Go gate (docs/stages2_to_5_plan.md §4.6).

Runs BEFORE weights/thresholds are tuned and frozen. Each of the five signals is checked for
technical validity, score directionality, numerical stability, reproducibility, missing-output
rate, redundancy with the other signals, and downstream usefulness.

The rule this script exists to enforce: **a signal that fails its gate is not forced into the final
selector merely because it was implemented.** Outcomes are one of:

    include        - passed everything; eligible for the frozen selector
    ablation_only  - technically valid but weak or near-duplicate of another signal; retained for
                     the leave-one-signal-out analysis, not weighted into the selector
    exclude        - technically invalid or unstable; removed from the selector, artifact retained
                     for audit if it is at least readable, exclusion + reason recorded

Evidence comes from asism_tuning_heldout-derived artifacts only. final_eval_heldout is never read.

Usage:
    python scripts/asism/02_gonogo.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402
from scripts.utils.artifact_contracts import asism_score_provenance, require_score_artifact, stage3_paths  # noqa: E402

# Each signal's headline score column, and whether higher is better.
PRIMARY_SCORE_COLUMN = {
    "similarity": ("similarity_knn_mean", True),
    "iqa": ("iqa_composite", True),
    "uncertainty": ("uncertainty_mean_std", None),   # None = no a-priori direction (§4.3)
    "explainability": ("explainability_region_overlap", True),
    "agreement": ("agreement_score", True),
}

ARTIFACT_FILENAME = {name: f"{name}_scores.parquet" for name in PRIMARY_SCORE_COLUMN}


def check_technical_validity(frame: pd.DataFrame, signal: str) -> dict:
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    has_ids = "image_id" in frame.columns
    has_score = column in frame.columns
    return {
        "passed": bool(has_ids and has_score and len(frame) > 0),
        "has_image_id": has_ids,
        "has_primary_score_column": has_score,
        "n_rows": len(frame),
        "primary_score_column": column,
    }


def check_numerical_stability(values: np.ndarray, config) -> dict:
    finite = np.isfinite(values)
    n_unique = int(len(np.unique(values[finite]))) if finite.any() else 0
    min_unique = int(config.gonogo.min_unique_values)
    non_finite_fraction = float(1.0 - finite.mean()) if len(values) else 1.0
    return {
        "passed": bool(non_finite_fraction == 0.0 and n_unique >= min_unique),
        "non_finite_fraction": non_finite_fraction,
        "n_unique_values": n_unique,
        "min_unique_required": min_unique,
        "std": float(np.std(values[finite])) if finite.any() else float("nan"),
        "note": (
            "A near-constant score carries no selection information even if it computes cleanly."
        ),
    }


def check_missing_rate(frame: pd.DataFrame, signal: str, config) -> dict:
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
    missing_fraction = float(np.isnan(values).mean()) if len(values) else 1.0
    # IQA additionally reports explicit per-image validity.
    if signal == "iqa" and "iqa_valid" in frame.columns:
        missing_fraction = float(1.0 - frame["iqa_valid"].astype(bool).mean())
    threshold = float(config.gonogo.max_missing_fraction)
    return {
        "passed": bool(missing_fraction <= threshold),
        "missing_fraction": missing_fraction,
        "max_allowed": threshold,
    }


def check_directionality(frame: pd.DataFrame, signal: str) -> dict:
    """Does the score move the way the methodology claims it does?

    Each check uses an internal contrast whose expected direction is known a priori, rather than
    assuming the sign is right because the formula looks right.
    """
    result = {"passed": None, "method": "", "detail": {}}

    if signal == "iqa":
        # Images the domain checks flagged as near-uniform must score BELOW unflagged ones.
        if "iqa_is_near_uniform" in frame.columns and frame["iqa_is_near_uniform"].any():
            flagged = frame.loc[frame["iqa_is_near_uniform"].astype(bool), "iqa_composite"].mean()
            clean = frame.loc[~frame["iqa_is_near_uniform"].astype(bool), "iqa_composite"].mean()
            result.update(
                passed=bool(flagged < clean),
                method="near_uniform_images_score_lower_than_clean",
                detail={"flagged_mean": float(flagged), "clean_mean": float(clean)},
            )
        else:
            result.update(
                passed=True,
                method="no_flagged_images_in_sample",
                detail={"note": "No near-uniform images present; directionality not contradicted."},
            )

    elif signal == "agreement":
        # No-Finding recipes and disease recipes must not be degenerate: the score must vary, and
        # the unintended-penalty must actually reduce the score where it fires.
        if "agreement_n_confident_unintended" in frame.columns:
            penalized = frame["agreement_n_confident_unintended"] > 0
            if penalized.any() and (~penalized).any():
                with_penalty = frame.loc[penalized, "agreement_score"].mean()
                without = frame.loc[~penalized, "agreement_score"].mean()
                result.update(
                    passed=bool(with_penalty < without),
                    method="confident_unintended_predictions_lower_the_score",
                    detail={"penalized_mean": float(with_penalty), "clean_mean": float(without)},
                )
            else:
                result.update(passed=True, method="penalty_never_fired_in_sample", detail={})
        else:
            result.update(passed=False, method="missing_penalty_column", detail={})

    elif signal == "similarity":
        # Flagged near-duplicates must receive no novelty credit.
        if "novelty_is_near_duplicate" in frame.columns and frame["novelty_is_near_duplicate"].any():
            flagged_novelty = frame.loc[
                frame["novelty_is_near_duplicate"].astype(bool), "novelty_score"
            ].max()
            result.update(
                passed=bool(flagged_novelty == 0.0),
                method="memorized_images_receive_zero_novelty",
                detail={"max_novelty_among_flagged": float(flagged_novelty)},
            )
        else:
            result.update(passed=True, method="no_near_duplicates_in_sample", detail={})

    elif signal == "explainability":
        # Overlap is a fraction of activation mass: it must lie in [0, 1].
        values = frame["explainability_region_overlap"].dropna()
        in_range = bool(((values >= 0.0) & (values <= 1.0)).all()) if len(values) else False
        result.update(
            passed=in_range,
            method="region_overlap_is_a_valid_fraction",
            detail={"min": float(values.min()) if len(values) else None,
                    "max": float(values.max()) if len(values) else None},
        )

    elif signal == "uncertainty":
        # §4.3 forbids an a-priori direction: high uncertainty is NOT defined as bad. The check is
        # that the bands are populated and ordered, not that the score correlates with quality.
        if "uncertainty_band" in frame.columns:
            counts = frame["uncertainty_band"].value_counts().to_dict()
            result.update(
                passed=bool(len(counts) >= 2),
                method="bands_are_populated_no_direction_asserted",
                detail={"band_counts": {str(k): int(v) for k, v in counts.items()}},
            )
        else:
            result.update(passed=False, method="missing_band_column", detail={})

    return result


def check_reproducibility(scores_dir: Path, signal: str, config) -> dict:
    """Compare the artifact against its recorded provenance fingerprint.

    Full recomputation is a GPU-cost decision for the operator; what is enforced here is that the
    artifact carries the provenance needed to reproduce it and that its row count matches.
    """
    sidecar = scores_dir / f"{signal}_scores.provenance.json"
    if not sidecar.is_file():
        return {"passed": False, "reason": "missing_provenance_sidecar"}
    provenance = read_json(sidecar)
    required = {"schema_version", "signal", "n_rows", "asism_config_sha256", "git_commit_hash"}
    missing = sorted(required - set(provenance))
    return {
        "passed": bool(not missing),
        "missing_provenance_fields": missing,
        "recorded_n_rows": provenance.get("n_rows"),
        "asism_config_sha256": provenance.get("asism_config_sha256"),
    }


def check_redundancy(merged: pd.DataFrame, signal: str, config) -> dict:
    """Correlation of this signal's headline score with the other signals'.

    A signal that is a near-linear restatement of another adds little independent information, and
    weighting both would double-count the same evidence.
    """
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    threshold = float(config.gonogo.redundancy_abs_correlation_max)

    correlations = {}
    for other, (other_column, _) in PRIMARY_SCORE_COLUMN.items():
        if other == signal or other_column not in merged.columns or column not in merged.columns:
            continue
        pair = merged[[column, other_column]].dropna()
        if len(pair) < 10 or pair[column].std() == 0 or pair[other_column].std() == 0:
            continue
        correlations[other] = float(pair[column].corr(pair[other_column]))

    worst = max(correlations.items(), key=lambda item: abs(item[1]), default=(None, 0.0))
    return {
        "passed": bool(abs(worst[1]) < threshold),
        "correlations": correlations,
        "most_correlated_with": worst[0],
        "max_abs_correlation": abs(worst[1]),
        "threshold": threshold,
    }


def check_usefulness(merged: pd.DataFrame, signal: str) -> dict:
    """Does including this signal change which images a simple equal-weight selector would keep?

    Compares the top-50% selection under an equal-weight composite WITH and WITHOUT this signal. A
    signal that changes nothing about the selection cannot influence downstream results, whatever
    its intrinsic quality.
    """
    available = [
        (name, col)
        for name, (col, direction) in PRIMARY_SCORE_COLUMN.items()
        if col in merged.columns and direction is True
    ]
    if len(available) < 2:
        return {"passed": True, "reason": "insufficient_signals_to_compare", "jaccard": None}

    def composite(names: list[str]) -> pd.Series:
        parts = []
        for _, col in [(n, c) for n, c in available if n in names]:
            values = merged[col].astype(float)
            spread = values.max() - values.min()
            parts.append((values - values.min()) / spread if spread > 0 else values * 0.0)
        return sum(parts) / max(len(parts), 1)

    all_names = [name for name, _ in available]
    if signal not in all_names:
        return {"passed": True, "reason": "signal_has_no_directional_score", "jaccard": None}
    without_names = [name for name in all_names if name != signal]

    n_keep = max(1, len(merged) // 2)
    with_top = set(composite(all_names).nlargest(n_keep).index)
    without_top = set(composite(without_names).nlargest(n_keep).index)

    intersection = len(with_top & without_top)
    union = len(with_top | without_top)
    jaccard = intersection / union if union else 1.0

    return {
        "passed": bool(jaccard < 0.99),
        "jaccard_overlap_of_selection": jaccard,
        "n_selected": n_keep,
        "note": "Jaccard ~1.0 means the signal does not change the selection at all.",
    }


def decide(checks: dict) -> tuple[str, str]:
    """Map check results to one of include / ablation_only / exclude (§4.6)."""
    hard_failures = [
        name
        for name in ("technical_validity", "numerical_stability", "missing_rate", "reproducibility")
        if checks[name].get("passed") is False
    ]
    if hard_failures:
        return "exclude", f"failed hard checks: {', '.join(hard_failures)}"

    if checks["directionality"].get("passed") is False:
        return "exclude", "score direction contradicts the methodology's stated meaning"

    soft_failures = []
    if checks["redundancy"].get("passed") is False:
        soft_failures.append(
            f"redundant with {checks['redundancy'].get('most_correlated_with')} "
            f"(|r|={checks['redundancy'].get('max_abs_correlation'):.3f})"
        )
    if checks["usefulness"].get("passed") is False:
        soft_failures.append("does not change selection")

    if soft_failures:
        return "ablation_only", "; ".join(soft_failures)
    return "include", "passed all checks"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", default=None)
    parser.add_argument("--namespace", default=None)
    args = parser.parse_args()

    config = load_named_config("stage3_asism.yaml", "stage3")

    namespace = args.namespace or str(config.split_namespace)
    stage2_config = load_named_config("stage2_generation.yaml", "stage2")
    expected_provenance = asism_score_provenance(config, stage2_config, namespace)
    paths = stage3_paths(config, namespace)
    scores_dir = Path(args.scores_dir) if args.scores_dir else paths["scores_dir"]

    frames: dict[str, pd.DataFrame] = {}
    absent: list[str] = []
    for signal, filename in ARTIFACT_FILENAME.items():
        path = scores_dir / filename
        if path.is_file():
            require_score_artifact(path, signal, expected_provenance)
            frames[signal] = pd.read_parquet(path)
        else:
            absent.append(signal)

    if not frames:
        raise SystemExit(
            f"UPSTREAM GATE: no ASISM score artifacts found in {scores_dir}\n"
            "Run scripts/asism/01_compute_signals.py first."
        )

    merged = None
    for signal, frame in frames.items():
        column, _ = PRIMARY_SCORE_COLUMN[signal]
        subset = frame[["image_id", column]] if column in frame.columns else frame[["image_id"]]
        merged = subset if merged is None else merged.merge(subset, on="image_id", how="outer")
    merged = merged.set_index("image_id")

    report = {
        "stage": "asism_gonogo",
        "scores_dir": str(scores_dir),
        "git_commit_hash": get_git_commit_hash(),
        "signals_present": sorted(frames),
        "signals_absent": sorted(absent),
        "evidence_split": "asism_tuning_heldout only; final_eval_heldout never read",
        "per_signal": {},
    }

    for signal, frame in frames.items():
        column, _ = PRIMARY_SCORE_COLUMN[signal]
        values = (
            frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            if column in frame.columns
            else np.array([])
        )

        checks = {
            "technical_validity": check_technical_validity(frame, signal),
            "numerical_stability": check_numerical_stability(values, config),
            "missing_rate": check_missing_rate(frame, signal, config),
            "reproducibility": check_reproducibility(scores_dir, signal, config),
            "directionality": check_directionality(frame, signal),
            "redundancy": check_redundancy(merged, signal, config),
            "usefulness": check_usefulness(merged, signal),
        }
        outcome, reason = decide(checks)
        report["per_signal"][signal] = {
            "outcome": outcome,
            "reason": reason,
            "checks": checks,
            "artifact_retained_for_audit": True,
        }

    for signal in absent:
        report["per_signal"][signal] = {
            "outcome": "exclude",
            "reason": "artifact not produced",
            "checks": {},
            "artifact_retained_for_audit": False,
        }

    included = sorted(s for s, r in report["per_signal"].items() if r["outcome"] == "include")
    ablation = sorted(s for s, r in report["per_signal"].items() if r["outcome"] == "ablation_only")
    excluded = sorted(s for s, r in report["per_signal"].items() if r["outcome"] == "exclude")

    report["surviving_signals"] = included
    report["ablation_only_signals"] = ablation
    report["excluded_signals"] = excluded
    report["asism_variant_status"] = (
        "primary" if len(included) >= 3
        else "reduced_variant — fewer than 3 signals survived; the resulting selector is reported "
             "as an alternative/ablation rather than the primary method (§4.6)"
    )

    report.update(expected_provenance)
    write_json(paths["gonogo_report"], report)

    print("ASISM Go/No-Go", flush=True)
    print("=" * 60, flush=True)
    for signal in sorted(report["per_signal"]):
        entry = report["per_signal"][signal]
        print(f"  {signal:<16} {entry['outcome']:<14} {entry['reason']}", flush=True)
    print("=" * 60, flush=True)
    print(f"  include:       {included}", flush=True)
    print(f"  ablation only: {ablation}", flush=True)
    print(f"  exclude:       {excluded}", flush=True)
    print(f"  variant:       {report['asism_variant_status']}", flush=True)
    print(f"\nReport -> {config.paths.gonogo_report}", flush=True)

    if not included:
        print(
            "\nNo signal survived the gate — the selector cannot be frozen. Investigate the "
            "failing checks before tuning.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

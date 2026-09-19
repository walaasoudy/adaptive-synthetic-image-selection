#!/usr/bin/env python3
"""The HAM10000 side of the learned selector: which features it may see, and what "utility" means.

Everything here is the part of the learned-ASISM machinery that cannot be shared with CheXpert. The
dataset-agnostic pieces — controlled subset construction, the image-disjoint train/val pools, the
Deep Sets utility model, the Banzhaf/leave-one-out target estimators, target normalisation — are
imported from `scripts/asism/learned.py` and `scripts/asism/models.py` unchanged. What differs is:

  * WHICH COLUMNS EXIST. The HAM10000 signals produce different columns from the CheXpert ones, and
    two of the CheXpert feature names do not exist here at all.
  * WHAT UTILITY IS. CheXpert measures macro AUROC over 11 independent binary labels. HAM10000 is a
    seven-way decision with a two-thirds-nv prior, where plain accuracy is maximised by a model that
    never predicts a rare class — which is precisely the failure the synthetic data exists to fix.
    Utility here is BALANCED ACCURACY (recall averaged over the seven classes), so a subset is
    credited for what it does to the rare classes rather than for leaving nv alone.
  * WHAT A STRATUM IS. A CheXpert stratum is a co-occurring label set; a HAM10000 stratum is the one
    diagnosis the recipe asked for.

THE SAFETY GATE IS NOT A SIGNAL. Technical validity (`iqa_valid`) and memorisation
(`novelty_is_near_duplicate`) are read straight from their artifacts, whatever the Go/No-Go gate
decided about the iqa and similarity signals. A memorised near-copy of a real patient's lesion must
not become selectable because the similarity signal happened to be redundant with another.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis
from scripts.utils.manifest import read_json

# Which signal owns which feature column. Go/No-Go decides signals; this map turns that decision
# into the set of columns the learned models are allowed to see, so an excluded or ablation-only
# signal cannot reach a model through a column whose name nobody connected back to it.
FEATURE_COLUMNS_BY_SIGNAL: dict[str, frozenset[str]] = {
    "similarity": frozenset({"similarity_knn_mean", "similarity_top1", "similarity_topk_spread"}),
    "iqa": frozenset({"iqa_composite", "iqa_sharpness", "iqa_contrast_std"}),
    # The epistemic term, not the per-class std the CheXpert map uses: under a softmax the std is
    # dominated by whichever class happens to be large, which on this dataset is nv.
    "uncertainty": frozenset({"uncertainty_mutual_information", "uncertainty_predictive_entropy"}),
    # ONE column, and only the calibrated one. `explainability_region_overlap` does not exist in the
    # HAM10000 artifacts, and the raw statistics are not selection-eligible: a model fed raw
    # peripheral mass would be selecting on where a lesion happens to sit in the frame.
    "explainability": frozenset({"explainability_calibrated_typicality"}),
    "agreement": frozenset({"agreement_score", "agreement_margin"}),
}

UTILITY_METRICS = ("balanced_accuracy", "macro_f1")
DEFAULT_UTILITY_METRIC = "balanced_accuracy"


class PoolGate(SystemExit):
    """The candidate pool cannot be assembled. The message names the command that fixes it."""


# ----------------------------------------------------------------------------------------------
# Features admitted by the Go/No-Go gate
# ----------------------------------------------------------------------------------------------


def active_feature_columns(configured_columns, surviving_signals) -> list[str]:
    """The configured feature columns that Go/No-Go admitted, in configured order.

    A configured column owned by no signal is a hard error rather than a silent drop: a typo, or a
    genuinely new score column added to the config without registering an owner here, would
    otherwise shrink the model's input space with no trace in any manifest. The model would train on
    fewer features than the config claims and nothing downstream could detect it.
    """
    surviving = set(map(str, surviving_signals))
    unknown = surviving - set(FEATURE_COLUMNS_BY_SIGNAL)
    if unknown:
        raise ValueError(f"unknown Go/No-Go signal(s) for HAM10000: {sorted(unknown)}")

    owned = {column for columns in FEATURE_COLUMNS_BY_SIGNAL.values() for column in columns}
    unmapped = [column for column in configured_columns if column not in owned]
    if unmapped:
        raise ValueError(
            f"feature_columns contains column(s) with no registered signal owner: {unmapped}. "
            "Register each one in FEATURE_COLUMNS_BY_SIGNAL (so Go/No-Go governs it) or remove it "
            "from the config — it cannot enter a learned model without an owner."
        )

    admitted = {column for signal in surviving for column in FEATURE_COLUMNS_BY_SIGNAL[signal]}
    active = [column for column in configured_columns if column in admitted]
    if not active:
        raise ValueError(
            "Go/No-Go admitted no feature columns; the ranking network has nothing to learn from. "
            "Investigate the failing checks in gonogo_report.json before retrying."
        )
    return active


def contributing_signals(active_columns) -> list[str]:
    """Which signals actually feed a feature set.

    Recorded in the training manifest so the reduced-variant judgement can be made for the LEARNED
    selector too. A "multi-signal" ranking network fed by one surviving signal still runs, but it is
    no longer multi-signal, and the manifest has to say so rather than let the name imply otherwise.
    """
    active = set(active_columns)
    return sorted(signal for signal, columns in FEATURE_COLUMNS_BY_SIGNAL.items() if columns & active)


# ----------------------------------------------------------------------------------------------
# The candidate pool
# ----------------------------------------------------------------------------------------------


def unsafe_candidates(image_ids, iqa: pd.DataFrame | None, similarity: pd.DataFrame | None) -> dict[str, str]:
    """image_id -> reason, for every candidate that fails a technical safety check.

    Fails CLOSED: a candidate with no row in the artifact, or a missing flag, is unsafe. An image
    the gate cannot vouch for is not the same as an image the gate approved, and the difference
    matters most exactly when something upstream went wrong.
    """
    reasons: dict[str, str] = {}
    ids = [str(image_id) for image_id in image_ids]
    if iqa is not None:
        valid = dict(zip(iqa["image_id"].astype(str), iqa["iqa_valid"]))
        for image_id in ids:
            value = valid.get(image_id)
            if value is None or pd.isna(value) or not bool(value):
                reasons[image_id] = "invalid_iqa_safety_gate"
    if similarity is not None:
        duplicate = dict(zip(similarity["image_id"].astype(str), similarity["novelty_is_near_duplicate"]))
        for image_id in ids:
            if image_id in reasons:
                continue
            value = duplicate.get(image_id)
            if value is None or pd.isna(value) or bool(value):
                reasons[image_id] = "near_duplicate_safety_gate"
    return reasons


def _safety_frame(scores_dir: Path, signal: str, column: str) -> pd.DataFrame:
    path = Path(scores_dir) / f"{signal}_scores.parquet"
    if not path.is_file():
        raise PoolGate(
            f"SAFETY GATE: {path} is required even if {signal} did not survive Go/No-Go — the "
            f"{column} check runs on every candidate pool, independently of the gate's verdict.\n"
            f"Run: python scripts/asism/ham10000_01_compute_signals.py --namespace <ns> --signal {signal}"
        )
    frame = pd.read_parquet(path)
    if column not in frame.columns:
        raise PoolGate(f"SAFETY GATE: {path} has no {column!r} column; refusing an unfiltered pool.")
    return frame[["image_id", column]]


def load_candidate_pool(
    namespace: str,
    stage3,
    stage2_root: Path,
    *,
    reject_invalid_iqa: bool | None = None,
    reject_near_duplicates: bool | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str], dict]:
    """The merged, safety-filtered pool the learned selector trains and scores on.

    Returns (frame, surviving_signals, report). The frame carries `image_id`, `dx`, `__stratum` and
    every column of every SURVIVING signal — nothing from a signal the gate excluded or marked
    ablation-only, because those must be absent from the learned model as well as from the weighted
    one, and the merge is where that is enforced.

    The two safety switches default to `learned_asism.safety` in the config rather than to hardcoded
    `True`. They used to be plain Python defaults that no caller overrode, which made the config
    keys decorative: turning one off for a sensitivity run changed nothing and said nothing. An
    explicit argument still wins, so a test can exercise one switch without editing the config.
    """
    safety = stage3.learned_asism.safety
    if reject_invalid_iqa is None:
        reject_invalid_iqa = bool(safety.reject_invalid_iqa)
    if reject_near_duplicates is None:
        reject_near_duplicates = bool(safety.reject_near_duplicates)

    namespace_dir = Path(stage3.paths.outputs_dir) / namespace
    scores_dir = namespace_dir / "signals"
    gonogo_path = namespace_dir / "gonogo_report.json"
    if not gonogo_path.is_file():
        raise PoolGate(
            f"UPSTREAM GATE: no Go/No-Go report at {gonogo_path}.\n"
            "The learned selector may only see signals the gate admitted. Run: "
            f"python scripts/asism/ham10000_02_gonogo.py --namespace {namespace}"
        )
    gonogo = read_json(gonogo_path)
    surviving = [str(name) for name in gonogo["surviving_signals"]]
    if not surviving:
        raise PoolGate(
            "No signal survived the Go/No-Go gate; the ranking network cannot train. The selector "
            "cannot be frozen on evidence the gate refused."
        )

    merged = None
    for signal in surviving:
        path = scores_dir / f"{signal}_scores.parquet"
        if not path.is_file():
            raise PoolGate(f"{path} is missing although Go/No-Go recorded {signal} as surviving.")
        frame = pd.read_parquet(path)
        merged = frame if merged is None else merged.merge(frame, on="image_id", how="inner")
    if merged is None or merged.empty:
        raise PoolGate("no candidates remain after merging the admitted signals")

    reasons = unsafe_candidates(
        merged["image_id"],
        _safety_frame(scores_dir, "iqa", "iqa_valid") if reject_invalid_iqa else None,
        _safety_frame(scores_dir, "similarity", "novelty_is_near_duplicate") if reject_near_duplicates else None,
    )
    counts = {reason: sum(1 for value in reasons.values() if value == reason) for reason in sorted(set(reasons.values()))}
    before = len(merged)
    merged = merged[~merged["image_id"].astype(str).isin(reasons)].reset_index(drop=True)
    if merged.empty:
        raise PoolGate("SAFETY GATE: every candidate failed the technical safety checks.")

    manifest_path = Path(stage2_root) / namespace / "all_candidates.csv"
    if not manifest_path.is_file():
        raise PoolGate(f"UPSTREAM GATE: no candidate manifest at {manifest_path}.")
    manifest = pd.read_csv(manifest_path)
    diagnosis_of = dict(
        zip(manifest["image_id"].astype(str), (normalize_diagnosis(value) for value in manifest["dx"]))
    )
    unknown = sorted(set(merged["image_id"].astype(str)) - set(diagnosis_of))
    if unknown:
        raise PoolGate(
            f"{len(unknown)} scored candidate(s) are absent from the manifest (e.g. {unknown[:3]})"
        )
    merged["dx"] = [diagnosis_of[str(image_id)] for image_id in merged["image_id"]]
    # One diagnosis per candidate, so the stratum IS the class. Named `__stratum` because that is
    # what the shared subset builder looks for when it keeps a subset's class mix close to the pool's.
    merged["__stratum"] = merged["dx"]

    report = {
        "surviving_signals": surviving,
        # Recorded so a pool built with a check switched off cannot be mistaken for one built with
        # it on. A removal count of zero means "nothing failed"; it must not also mean "nothing was
        # checked", and these two flags are what tells the two apart.
        "safety_checks_applied": {
            "reject_invalid_iqa": bool(reject_invalid_iqa),
            "reject_near_duplicates": bool(reject_near_duplicates),
        },
        "candidates_before_safety_gate": int(before),
        "candidates_after_safety_gate": int(len(merged)),
        "safety_gate_removals": counts,
        "per_class_counts": merged["dx"].value_counts().sort_index().to_dict(),
    }
    if verbose:
        print(f"Safety gate: removed {len(reasons)}/{before} candidates {counts}", flush=True)
    return merged, surviving, report


# ----------------------------------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------------------------------


def utility_columns(metric: str) -> tuple[str, str]:
    if metric not in UTILITY_METRICS:
        raise ValueError(f"unknown utility metric {metric!r}; expected one of {list(UTILITY_METRICS)}")
    return f"real_only_{metric}", f"augmented_{metric}"


def validate_utility_results(subsets: list[dict], results: list[dict], metric: str = DEFAULT_UTILITY_METRIC) -> pd.DataFrame:
    """Pair every built subset with its measured downstream utility, or refuse.

    Incomplete results are a hard error rather than a smaller training set: the subsets that fail to
    measure are not a random sample of the design — a subset drawn from an extreme quantile band is
    exactly the one most likely to produce a degenerate proxy run — so dropping them would quietly
    bias the utility model toward the easy middle of the pool.
    """
    real_column, augmented_column = utility_columns(metric)
    expected = {row["subset_id"] for row in subsets}
    result_ids = [row.get("subset_id") for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("duplicate subset_id in the utility results")
    missing = sorted(expected - set(result_ids))
    if missing:
        raise ValueError(
            f"utility results are incomplete: {len(missing)} subset(s) unmeasured (e.g. {missing[:3]}). "
            "Measure them or rebuild the design; a partial set biases the utility model."
        )
    frame = pd.DataFrame(results)
    required = {"subset_id", real_column, augmented_column, "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"utility results need columns {sorted(required)}; got {sorted(frame.columns)}")
    frame["utility_delta"] = frame[augmented_column] - frame[real_column]
    frame["utility_metric"] = metric
    return frame


# ----------------------------------------------------------------------------------------------
# Feasibility — computed against the real pool, before any subset is built
# ----------------------------------------------------------------------------------------------


def pool_feasibility_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    subset_sizes: list[int],
    quantile_bins: int,
    val_fraction: float,
    total_subsets: int,
    thresholds: dict,
) -> dict:
    """Can the configured subset design actually be built from THIS pool, without replacement?

    A pure function of the frame it is given, so it can be checked against a fixture and then run
    against the real pool. It never invents a candidate: the answer is a property of the pool, and
    the design is what gets revised when the answer is no — never the threshold, and never after
    seeing which way it failed.
    """
    n_val = round(len(frame) * float(val_fraction))
    n_train = len(frame) - n_val
    largest = max(int(size) for size in subset_sizes)

    per_class = frame["dx"].value_counts().sort_index().to_dict() if "dx" in frame.columns else {}
    failures: list[str] = []

    min_per_class = int(thresholds.get("min_candidates_per_class", 0))
    for diagnosis in sorted(set(CLASSIFIER_TARGET_LABELS) | set(per_class)):
        count = int(per_class.get(diagnosis, 0))
        if count < min_per_class:
            failures.append(
                f"class {diagnosis} has {count} candidates, below min_candidates_per_class={min_per_class}"
            )

    # A single-signal subset draws entirely from ONE quantile band of ONE feature, within one role
    # pool. That band is the tightest constraint in the whole design, so it is what is checked.
    min_per_band = int(thresholds.get("min_candidates_per_quantile_band", 0))
    band_report: dict[str, dict] = {}
    for column in feature_columns:
        if column not in frame.columns:
            failures.append(f"feature column {column!r} is absent from the pool")
            continue
        bands = pd.qcut(frame[column].rank(method="first"), quantile_bins, labels=False)
        counts = {int(band): int((bands == band).sum()) for band in range(quantile_bins)}
        # Each band is split between the two role pools in the same proportion as the pool itself.
        smallest_val_band = min(int(round(count * float(val_fraction))) for count in counts.values())
        band_report[column] = {"per_band": counts, "smallest_val_side_band": smallest_val_band}
        if min(counts.values()) < min_per_band:
            failures.append(
                f"feature {column!r} has a quantile band with {min(counts.values())} candidates, "
                f"below min_candidates_per_quantile_band={min_per_band}"
            )
        if smallest_val_band < largest:
            failures.append(
                f"feature {column!r}: the smallest val-side quantile band holds {smallest_val_band} "
                f"candidates, fewer than the largest subset size ({largest}); a single-signal "
                "val-role subset of that size cannot be drawn without replacement"
            )

    val_total = round(total_subsets * float(val_fraction))
    min_val_subsets = int(thresholds.get("min_val_subsets", 0))
    if val_total < min_val_subsets:
        failures.append(
            f"the design yields {val_total} val-role subsets, below min_val_subsets={min_val_subsets}; "
            "there would be no image-disjoint validation to early-stop on"
        )

    return {
        "n_candidates": int(len(frame)),
        "train_pool_size": int(n_train),
        "val_pool_size": int(n_val),
        "per_class_counts": {str(key): int(value) for key, value in per_class.items()},
        "quantile_bands": band_report,
        "planned_train_subsets": int(total_subsets - val_total),
        "planned_val_subsets": int(val_total),
        "largest_subset_size": largest,
        "thresholds": dict(thresholds),
        "failures": failures,
        "feasible": not failures,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


__all__ = [
    "DEFAULT_UTILITY_METRIC",
    "FEATURE_COLUMNS_BY_SIGNAL",
    "PoolGate",
    "UTILITY_METRICS",
    "active_feature_columns",
    "contributing_signals",
    "load_candidate_pool",
    "pool_feasibility_report",
    "unsafe_candidates",
    "utility_columns",
    "validate_utility_results",
    "write_jsonl",
]

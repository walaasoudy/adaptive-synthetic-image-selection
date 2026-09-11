"""Pure data-design utilities for learned ASISM."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.utils.labels import patient_level_support
from scripts.utils.manifest import hash_dict
from scripts.utils.splits import load_split


# A feature is eligible only when the Go/No-Go decision admits the signal that
# produced it.  Keeping this mapping here makes the learned selector obey the
# same signal-governance contract as the weighted selector.
FEATURE_COLUMNS_BY_SIGNAL = {
    "similarity": {"similarity_knn_mean", "similarity_top1", "similarity_topk_spread"},
    "iqa": {"iqa_composite", "iqa_sharpness", "iqa_contrast_std"},
    "uncertainty": {"uncertainty_mean_std"},
    "explainability": {"explainability_region_overlap"},
    "agreement": {"agreement_score"},
    "distinctiveness": {"distinctiveness_score"},
}


def active_feature_columns(configured_columns: list[str], surviving_signals: list[str]) -> list[str]:
    """Return configured learned-ASISM features admitted by Go/No-Go.

    ``ablation_only`` and ``exclude`` signals are deliberately absent from the
    merged score frame, so they must also be absent from every learned model.
    The returned ordered list is frozen in the learned-training manifest and
    reused by all downstream learned-ASISM stages.
    """
    surviving = set(surviving_signals)
    unknown = surviving - set(FEATURE_COLUMNS_BY_SIGNAL)
    if unknown:
        raise ValueError(f"Unknown Go/No-Go signal(s): {sorted(unknown)}")

    # A configured column that maps to NO known signal must be a hard error, never a silent drop.
    # Silently dropping it would let a typo, or a genuinely new score column added to
    # learned_asism.feature_columns without registering it above, shrink the learned model's input
    # space with no trace in any manifest — the model would train on fewer features than the config
    # says it uses, and nothing downstream could detect it.
    owned = {column for columns in FEATURE_COLUMNS_BY_SIGNAL.values() for column in columns}
    unmapped = [column for column in configured_columns if column not in owned]
    if unmapped:
        raise ValueError(
            f"learned_asism.feature_columns contains column(s) not registered in "
            f"FEATURE_COLUMNS_BY_SIGNAL: {unmapped}. Add each one to its producing signal there "
            "(so Go/No-Go governs it) or remove it from the config — it cannot be admitted to a "
            "learned model without a signal owner."
        )

    active = [
        column for column in configured_columns
        if any(column in columns and signal in surviving for signal, columns in FEATURE_COLUMNS_BY_SIGNAL.items())
    ]
    if not active:
        raise ValueError("Go/No-Go admitted no learned-ASISM feature columns; learned ASISM cannot train.")
    return active


def contributing_signals(active_columns: list[str]) -> list[str]:
    """Which Go/No-Go signals actually feed a given active feature set.

    Recorded in the learned-training manifest so the §4.6 `reduced_variant` judgement (fewer than
    three admitted signals -> the selector is reported as an ablation/alternative, not the primary
    method) can be made for the LEARNED selector too, not just the weighted one. A
    'Multi-Signal Utility Ranking Network' fed by one surviving signal is still runnable, but it is
    no longer multi-signal, and the manifest must say so rather than let the name imply otherwise.
    (The network is single-objective by design in every case — see MultiSignalUtilityRankingNetwork's
    docstring; what this count governs is how many SIGNALS feed it, not how many objectives.)
    """
    return sorted(
        signal for signal, columns in FEATURE_COLUMNS_BY_SIGNAL.items()
        if columns & set(active_columns)
    )


def safe_feature_frame(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing learned-ASISM feature columns: {missing}")
    values = frame[columns].replace([np.inf, -np.inf], np.nan)
    medians = values.median().fillna(0.0)
    values = values.fillna(medians)
    means = values.mean()
    stds = values.std(ddof=0).replace(0, 1.0).fillna(1.0)
    normalized = (values - means) / stds
    stats = {"columns": columns, "median": medians.to_dict(), "mean": means.to_dict(), "std": stds.to_dict()}
    return normalized, stats


def apply_feature_frame(frame: pd.DataFrame, columns: list[str], stats: dict) -> pd.DataFrame:
    """Normalize using FROZEN stats from a prior safe_feature_frame call — never refit at inference
    time. Refitting on whatever candidate pool happens to be present at selection time silently
    drifts the ranker's inputs away from what it was trained on."""
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing learned-ASISM feature columns: {missing}")
    medians = pd.Series({column: stats["median"][column] for column in columns})
    means = pd.Series({column: stats["mean"][column] for column in columns})
    stds = pd.Series({column: stats["std"][column] for column in columns})
    values = frame[columns].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(medians)
    return (values - means) / stds


def _sample(pool: pd.DataFrame, size: int, rng: np.random.Generator, allow_partial: bool = False) -> list[str]:
    """NEVER samples with replacement.

    By default (allow_partial=False) this is a hard invariant: if the pool has fewer than `size`
    unique images, it raises rather than silently returning a smaller subset. A subset quietly
    smaller than its nominal size is exactly what this refuses to produce — that must surface as an
    explicit failure, either at --phase feasibility (before any subset exists) or, if the gate was
    bypassed or is stale, right here at build time.

    allow_partial=True exists ONLY for _sample_balanced's internal per-stratum draws, where drawing
    less than a stratum's exact quota is expected (a stratum can be small) and handled by that
    function's own backfill + final invariant check — never for a caller that needs an exact count.
    """
    if size <= 0:
        return []
    if len(pool) < size:
        if not allow_partial:
            raise ValueError(
                f"Insufficient pool: requested {size} unique images without replacement, pool has "
                f"only {len(pool)}. This must be caught by --phase feasibility before --phase build."
            )
        size = len(pool)
    positions = rng.choice(len(pool), size=size, replace=False)
    return pool.iloc[positions]["image_id"].astype(str).tolist()


def _sample_balanced(pool: pd.DataFrame, reference: pd.DataFrame, size: int,
                     rng: np.random.Generator, allow_partial: bool = False) -> list[str]:
    """Approximately preserve the reference label-recipe distribution when available.

    Per-stratum draws may legitimately undershoot (a stratum can be small); the shortfall is topped
    up from the rest of `pool`. If `pool` as a whole still can't cover `size` unique images after
    that: with allow_partial=False (the default, used wherever no further backfill exists above
    this call — the "random" and "single_signal" designs) this raises, a genuine invariant
    violation. allow_partial=True is used ONLY by build_controlled_subsets's "mixed" design, whose
    per-column contributions are intentionally allowed to fall short because THAT caller backfills
    the shortfall itself from the whole frame afterward (and raises there if even that isn't enough).
    """
    if "__stratum" not in reference or "__stratum" not in pool:
        return _sample(pool, size, rng, allow_partial=allow_partial)
    proportions = reference["__stratum"].value_counts(normalize=True)
    selected: list[str] = []
    for stratum, proportion in proportions.items():
        count = int(round(size * float(proportion)))
        selected.extend(_sample(pool.loc[pool["__stratum"] == stratum], count, rng, allow_partial=True))
    selected = list(dict.fromkeys(selected))
    if len(selected) < size:
        remaining = pool.loc[~pool.image_id.astype(str).isin(selected)]
        selected.extend(_sample(remaining, size - len(selected), rng, allow_partial=allow_partial))
    return selected[:size]


def build_controlled_subsets(frame: pd.DataFrame, feature_columns: list[str], total: int,
                             sizes: list[int], random_fraction: float, single_fraction: float,
                             mixed_fraction: float, seed: int, quantile_bins: int = 3) -> list[dict]:
    """Create random controls plus randomized single-signal and mixed compositions.

    Sampling remains random inside each pool. The controlled compositions prevent every large
    random subset from collapsing toward the same mean feature vector.
    """
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    if total < 3 or not sizes:
        raise ValueError("At least three subsets and one positive subset size are required")
    fraction_total = random_fraction + single_fraction + mixed_fraction
    if abs(fraction_total - 1.0) > 1e-6:
        raise ValueError(
            "random_fraction + single_signal_fraction + mixed_fraction must sum to 1.0, "
            f"got {fraction_total}"
        )
    if quantile_bins < 2:
        raise ValueError(f"quantile_bins must be >= 2, got {quantile_bins}")
    rng = np.random.default_rng(seed)
    n_random = round(total * random_fraction)
    n_single = round(total * single_fraction)
    n_mixed = total - n_random - n_single
    designs = ["random"] * n_random + ["single_signal"] * n_single + ["mixed"] * n_mixed
    rng.shuffle(designs)
    quantiles = {
        column: pd.qcut(frame[column].rank(method="first"), quantile_bins, labels=False)
        for column in feature_columns
    }
    records = []
    for index, design in enumerate(designs):
        size = min(int(rng.choice(sizes)), len(frame))
        if design == "random":
            ids = _sample_balanced(frame, frame, size, rng)
            recipe = {"type": "matched_random"}
        elif design == "single_signal":
            column = str(rng.choice(feature_columns))
            band = int(rng.integers(0, quantile_bins))
            ids = _sample_balanced(frame.loc[quantiles[column] == band], frame, size, rng)
            recipe = {"type": design, "feature": column, "quantile_band": band}
        else:
            chosen = list(rng.choice(feature_columns, size=min(3, len(feature_columns)), replace=False))
            weights = rng.dirichlet(np.ones(len(chosen)))
            parts: list[str] = []
            allocations = np.floor(weights * size).astype(int)
            allocations[0] += size - int(allocations.sum())
            bands = []
            for column, count in zip(chosen, allocations):
                band = int(rng.integers(0, quantile_bins))
                bands.append(band)
                parts.extend(_sample_balanced(frame.loc[quantiles[column] == band], frame, int(count), rng,
                                              allow_partial=True))
            ids = list(dict.fromkeys(parts))
            if len(ids) < size:
                remaining = frame.loc[~frame["image_id"].astype(str).isin(ids)]
                ids.extend(_sample(remaining, size - len(ids), rng))  # strict: raises if truly insufficient
            recipe = {"type": design, "features": chosen, "quantile_bands": bands, "mixture": weights.tolist()}
        records.append({"subset_id": f"utility_{index:04d}", "image_ids": ids[:size],
                        "size": len(ids[:size]), "seed": seed + index, "design": recipe})
    return records


def split_image_pool(
    frame: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen, seeded image-level split into a train-image-pool and a val-image-pool.

    This must run BEFORE any subset is built. Every downstream subset is built from exactly one
    side (see build_role_conditioned_subsets), so no image can ever appear in both a train-role and
    a val-role subset — disjointness is guaranteed by construction, not by post-hoc filtering.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if frame["image_id"].duplicated().any():
        raise ValueError("image_id must be unique")
    ids = frame["image_id"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(ids)
    n_val = round(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val].tolist())
    is_val = frame["image_id"].astype(str).isin(val_ids)
    return frame.loc[~is_val].copy(), frame.loc[is_val].copy()


def build_role_conditioned_subsets(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    train_total: int,
    val_total: int,
    sizes: list[int],
    random_fraction: float,
    single_fraction: float,
    mixed_fraction: float,
    seed: int,
    quantile_bins: int = 3,
) -> list[dict]:
    """Build train-role and val-role subset populations from two DISJOINT image pools.

    Each call to build_controlled_subsets only ever sees one pool's frame, so it is structurally
    impossible for a subset to mix images from both sides.
    """
    train_records = build_controlled_subsets(
        train_frame, feature_columns, train_total, sizes, random_fraction, single_fraction,
        mixed_fraction, seed, quantile_bins=quantile_bins,
    )
    val_records = build_controlled_subsets(
        val_frame, feature_columns, val_total, sizes, random_fraction, single_fraction,
        mixed_fraction, seed + train_total, quantile_bins=quantile_bins,
    )
    for index, record in enumerate(train_records):
        record["role"] = "train"
        record["subset_id"] = f"train_{record['subset_id']}"
    for index, record in enumerate(val_records):
        record["role"] = "val"
        record["subset_id"] = f"val_{record['subset_id']}"
    return train_records + val_records


def pool_feasibility_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    intended_by_id: dict[str, dict],
    labels: list[str],
    subset_sizes: list[int],
    quantile_bins: int,
    val_fraction: float,
    seed: int,
    total_subsets: int,
    feasibility_thresholds: dict,
) -> dict:
    """Phase-1b feasibility numbers for the image-disjoint subset design.

    Pure function of whatever candidate frame is passed in: run it against a fixture to check the
    calculation is correct, or against a real merged score artifact (once Stage 2 generation + Stage
    3 signal scoring have produced one) to get the real numbers. It never invents a candidate pool.

    `val_fraction` is a PROVISIONAL DEFAULT (configs/stage3_asism.yaml's val_pool_fraction) until a
    real production candidate pool has been run through this report and reviewed — this function
    tags that explicitly rather than implying the ratio is final.
    """
    train_frame, val_frame = split_image_pool(frame, val_fraction, seed)

    # Defensive re-check: split_image_pool guarantees disjointness by construction, but a failure
    # criterion must VERIFY that, not assume it.
    train_ids = set(train_frame["image_id"].astype(str))
    val_ids = set(val_frame["image_id"].astype(str))
    overlap_ids = train_ids & val_ids

    def per_label_counts(pool: pd.DataFrame) -> dict[str, int]:
        return {
            label: sum(
                1
                for image_id in pool["image_id"]
                if int(intended_by_id.get(str(image_id), {}).get(label, 0)) == 1
            )
            for label in labels
        }

    def band_series(pool: pd.DataFrame) -> dict[str, "pd.Series"]:
        if pool.empty or len(pool) < quantile_bins:
            return {}
        return {
            column: pd.qcut(pool[column].rank(method="first"), quantile_bins, labels=False)
            for column in feature_columns
        }

    def per_band_counts(bands: dict) -> dict[str, dict]:
        if not bands:
            return {column: {} for column in feature_columns}
        return {
            column: {int(band): int(count) for band, count in series.value_counts().sort_index().items()}
            for column, series in bands.items()
        }

    def size_achievability(pool_n: int) -> dict:
        # No replacement is ever used (see _sample): a pool smaller than a requested size simply
        # produces a smaller-than-requested subset. That is what "achievable_without_replacement"
        # reports here — there is no separate "replacement rate" to report because replacement never
        # happens; a False here IS the shortfall.
        result = {}
        for size in subset_sizes:
            result[str(size)] = {
                "requested_size": size,
                "pool_size": pool_n,
                "achievable_without_replacement": bool(pool_n >= size),
                "achievable_size_if_smaller": min(size, pool_n),
            }
        return result

    train_bands = band_series(train_frame)
    val_bands = band_series(val_frame)
    train_total = round(int(total_subsets) * (1 - val_fraction))
    val_total = int(total_subsets) - train_total

    report = {
        "total_candidates": len(frame),
        "train_pool_size": len(train_frame),
        "val_pool_size": len(val_frame),
        "val_fraction_requested": val_fraction,
        "val_fraction_actual": (len(val_frame) / len(frame)) if len(frame) else 0.0,
        "val_pool_fraction_status": "provisional_default",
        "planned_train_subsets": train_total,
        "planned_val_subsets": val_total,
        "train_val_overlap_count": len(overlap_ids),
        "candidates_per_label": {"train": per_label_counts(train_frame), "val": per_label_counts(val_frame)},
        "candidates_per_quantile_band": {
            "train": per_band_counts(train_bands),
            "val": per_band_counts(val_bands),
        },
        "subset_size_achievability": {
            "train": size_achievability(len(train_frame)),
            "val": size_achievability(len(val_frame)),
        },
        "quantile_band_size_achievability": {
            "train": {
                column: {
                    str(band): size_achievability(int(count))
                    for band, count in per_band_counts(train_bands).get(column, {}).items()
                }
                for column in feature_columns
            },
            "val": {
                column: {
                    str(band): size_achievability(int(count))
                    for band, count in per_band_counts(val_bands).get(column, {}).items()
                }
                for column in feature_columns
            },
        },
        "feasibility_thresholds": dict(feasibility_thresholds),
    }
    report["failures"] = evaluate_subset_design_feasibility(report, feasibility_thresholds)
    report["passed"] = len(report["failures"]) == 0
    return report


def evaluate_subset_design_feasibility(report: dict, thresholds: dict) -> list[str]:
    """Pure evaluation of a pool_feasibility_report's numbers against config-declared thresholds.

    Every threshold here comes from `feasibility_thresholds` in configs/stage3_asism.yaml — nothing
    is hardcoded except the structural rules that have no numeric threshold to configure (no
    sampling with replacement, ever; train/val pools must be exactly disjoint, not "mostly").
    """
    failures: list[str] = []

    if report["train_val_overlap_count"] > 0:
        failures.append(
            f"train/val pool overlap: {report['train_val_overlap_count']} image(s) appear in both "
            "pools (structural rule, not configurable)"
        )

    min_per_class = int(thresholds["min_candidates_per_class"])
    for role in ("train", "val"):
        for label, count in report["candidates_per_label"][role].items():
            if count < min_per_class:
                failures.append(
                    f"insufficient candidates per class: {role} pool has {count} candidates for "
                    f"{label!r}, below feasibility_thresholds.min_candidates_per_class={min_per_class}"
                )

    min_per_band = int(thresholds["min_candidates_per_quantile_band"])
    for role in ("train", "val"):
        for column, bands in report["candidates_per_quantile_band"][role].items():
            for band, count in bands.items():
                if count < min_per_band:
                    failures.append(
                        f"insufficient candidates per quantile band: {role} pool, feature={column!r}, "
                        f"band={band}, count={count}, below "
                        f"feasibility_thresholds.min_candidates_per_quantile_band={min_per_band}"
                    )

    for role in ("train", "val"):
        for size, entry in report["subset_size_achievability"][role].items():
            if not entry["achievable_without_replacement"]:
                failures.append(
                    f"actual subset size below required: {role} pool cannot build a "
                    f"{entry['requested_size']}-image subset without replacement "
                    f"(pool has {entry['pool_size']}; would only reach "
                    f"{entry['achievable_size_if_smaller']})"
                )
        for column, bands in report["quantile_band_size_achievability"][role].items():
            for band, sizes in bands.items():
                for size, entry in sizes.items():
                    if not entry["achievable_without_replacement"]:
                        failures.append(
                            f"actual subset size below required: {role} pool, feature={column!r}, "
                            f"band={band} cannot build a {entry['requested_size']}-image "
                            f"single_signal/mixed subset without replacement (band has "
                            f"{entry['pool_size']}; would only reach {entry['achievable_size_if_smaller']})"
                        )

    min_val_subsets = int(thresholds["min_val_subsets"])
    if report["planned_val_subsets"] < min_val_subsets:
        failures.append(
            f"val subset count below required: planned {report['planned_val_subsets']} val-role "
            f"subsets, below feasibility_thresholds.min_val_subsets={min_val_subsets}"
        )

    return failures


def verify_built_subsets(records: list[dict], train_total: int, val_total: int) -> None:
    """Post-build defense-in-depth: re-verify the WRITTEN records against every invariant the
    construction was supposed to already guarantee. Construction bugs, stale assumptions, or future
    refactors should never silently ship a broken utility_subsets.jsonl — this is the last check
    before it is written to disk, and raises SystemExit naming exactly which invariant failed."""
    for record in records:
        ids = record["image_ids"]
        if len(set(ids)) != len(ids):
            raise SystemExit(
                f"POST-BUILD INVARIANT VIOLATION: subset {record['subset_id']!r} contains a "
                "duplicate image_id"
            )
        if len(ids) != record["size"]:
            raise SystemExit(
                f"POST-BUILD INVARIANT VIOLATION: subset {record['subset_id']!r} has "
                f"{len(ids)} images but declares size={record['size']}"
            )

    train_ids = {image_id for r in records if r["role"] == "train" for image_id in r["image_ids"]}
    val_ids = {image_id for r in records if r["role"] == "val" for image_id in r["image_ids"]}
    overlap = train_ids & val_ids
    if overlap:
        raise SystemExit(
            f"POST-BUILD INVARIANT VIOLATION: {len(overlap)} image(s) appear in both a train-role "
            "and a val-role subset (e.g. " + ", ".join(sorted(overlap)[:5]) + ")"
        )

    n_train = sum(1 for r in records if r["role"] == "train")
    n_val = sum(1 for r in records if r["role"] == "val")
    if n_train != train_total or n_val != val_total:
        raise SystemExit(
            f"POST-BUILD INVARIANT VIOLATION: built {n_train} train-role and {n_val} val-role "
            f"subsets, planned {train_total}/{val_total}"
        )


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_utility_results(subsets: list[dict], results: list[dict]) -> pd.DataFrame:
    expected = {row["subset_id"] for row in subsets}
    result_ids = [row.get("subset_id") for row in results]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Duplicate subset_id in utility results")
    missing = expected - set(result_ids)
    if missing:
        raise ValueError(f"Utility results are incomplete; missing {len(missing)} subsets")
    frame = pd.DataFrame(results)
    required = {"subset_id", "real_only_macro_auroc", "augmented_macro_auroc", "fold", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Utility results need columns {sorted(required)}")
    frame["utility_delta"] = frame["augmented_macro_auroc"] - frame["real_only_macro_auroc"]
    return frame


def spearman_with_reason(a, b) -> tuple[float | None, str | None]:
    """Spearman rho with an explicit machine-readable reason when it is undefined.

    scipy returns NaN when either input is (numerically) constant or fewer than two points are
    present — the correlation is undefined, not zero, and NaN is not valid JSON. This returns
    ``(rho, None)`` on success and ``(None, reason)`` otherwise so callers never write a bare NaN
    into a frozen manifest.
    """
    import math as _math

    import scipy.stats as _stats

    a = np.asarray(list(a), dtype=np.float64)
    b = np.asarray(list(b), dtype=np.float64)
    if a.size < 2 or b.size < 2 or a.size != b.size:
        return None, "fewer_than_two_paired_points"
    correlation = _stats.spearmanr(a, b)
    statistic = getattr(correlation, "statistic", None)
    value = float(statistic) if statistic is not None else float(correlation[0])
    if _math.isnan(value):
        return None, "constant_predictions_or_targets"
    return value, None


def normalize_ranking_targets(
    targets: dict[str, float], method: str = "standardize", winsorize_quantile: float = 0.0
) -> dict[str, float]:
    """Transform the distilled per-image ranking targets before the Smooth-L1 + pairwise loss.

    On ~96 training images a single noisy proxy-subset measurement can produce an outlier marginal
    value that dominates the regression term. ``"standardize"`` (zero mean, unit std) and ``"rank"``
    (map to evenly-spaced ranks in [0, 1]) both bound that influence; ``"none"`` keeps the raw
    values. ``winsorize_quantile`` in (0, 0.5) additionally clips each tail before the transform.
    Every method is order-preserving, so the pairwise term is unaffected. Motivated by the
    robustness rationale of the Banzhaf value (Wang & Jia, AISTATS 2023) and recent
    marginal-contribution estimators (2D-OOB, NeurIPS 2024; Chi et al., ICML 2026).
    """
    if not targets:
        return {}
    keys = list(targets)
    values = np.asarray([float(targets[key]) for key in keys], dtype=np.float64)
    if winsorize_quantile:
        if not 0.0 < winsorize_quantile < 0.5:
            raise ValueError(f"winsorize_quantile must be in (0, 0.5), got {winsorize_quantile}")
        low, high = np.quantile(values, [winsorize_quantile, 1.0 - winsorize_quantile])
        values = np.clip(values, low, high)
    if method == "none":
        pass
    elif method == "standardize":
        std = float(values.std())
        values = (values - values.mean()) / (std if std > 1e-12 else 1.0)
    elif method == "rank":
        import scipy.stats as _stats
        ranks = _stats.rankdata(values, method="average")
        values = (ranks - 1.0) / max(len(ranks) - 1, 1)
    else:
        raise ValueError(f"unknown target normalization method {method!r}")
    return dict(zip(keys, values.tolist()))


def banzhaf_msr_targets(
    subsets: list[dict],
    utility_by_id: dict[str, float],
    minimum_in_subsets: int = 1,
    minimum_out_subsets: int = 1,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """Per-image Data-Banzhaf value via the Maximum-Sample-Reuse (MSR) estimator.

    Uses ONLY the subsets already measured for SetUtilityNetwork (utility_subsets.jsonl +
    utility_results.jsonl) — no extra proxy-model training, no additional GPU cost. For image i:

        banzhaf(i) = mean( U(S) for S in subsets if i in S )
                   - mean( U(S) for S in subsets if i not in S )

    This is the one-pass MSR estimator of the Banzhaf semivalue (Wang & Jia, "Data Banzhaf: A
    Robust Data Valuation Framework for Machine Learning", AISTATS 2023): every measured subset
    contributes to exactly one of the two averages for every image, so all ``len(subsets)``
    measurements are reused for all images simultaneously.

    An image gets a value ONLY if it appears in at least ``minimum_in_subsets`` measured subsets AND
    is absent from at least ``minimum_out_subsets`` of them — otherwise one mean is undefined and the
    image is simply absent from the returned targets (never 0.0, never a one-sided value). This
    mirrors the exposure filter that ``05_train_learned_asism.py`` already applies to the
    leave-one-out targets. ``counts`` reports ``{"in": ..., "out": ...}`` for every image seen.
    """
    if not subsets:
        return {}, {}
    total_utility = 0.0
    n_subsets = 0
    in_sum: dict[str, float] = {}
    in_count: dict[str, int] = {}
    for subset in subsets:
        subset_id = subset["subset_id"]
        if subset_id not in utility_by_id:
            raise ValueError(f"banzhaf_msr_targets: no measured utility for subset {subset_id!r}")
        utility = float(utility_by_id[subset_id])
        total_utility += utility
        n_subsets += 1
        for image_id in set(map(str, subset["image_ids"])):
            in_sum[image_id] = in_sum.get(image_id, 0.0) + utility
            in_count[image_id] = in_count.get(image_id, 0) + 1

    targets: dict[str, float] = {}
    counts: dict[str, dict[str, int]] = {}
    for image_id, appearances in in_count.items():
        out_appearances = n_subsets - appearances
        counts[image_id] = {"in": appearances, "out": out_appearances}
        if appearances < minimum_in_subsets or out_appearances < minimum_out_subsets:
            continue
        mean_in = in_sum[image_id] / appearances
        mean_out = (total_utility - in_sum[image_id]) / out_appearances
        targets[image_id] = mean_in - mean_out
    return targets, counts


def loo_vs_banzhaf_diagnostic(
    leave_one_out_targets: dict[str, float], banzhaf_targets: dict[str, float]
) -> dict:
    """Rank agreement between the leave-one-out marginal targets (the current supervision for the
    ranking network) and the Banzhaf MSR targets, over the images both methods scored.

    Recorded in the learned-training manifest so the choice to switch the ranking network onto
    Banzhaf supervision is made from evidence, not by default — the plan keeps Banzhaf optional
    "until the diagnostic is reviewed".
    """
    shared = sorted(set(leave_one_out_targets) & set(banzhaf_targets))
    rho, reason = spearman_with_reason(
        [leave_one_out_targets[image_id] for image_id in shared],
        [banzhaf_targets[image_id] for image_id in shared],
    )
    return {
        "n_images_compared": len(shared),
        "n_leave_one_out_targets": len(leave_one_out_targets),
        "n_banzhaf_targets": len(banzhaf_targets),
        "spearman": rho,
        "spearman_undefined_reason": reason,
    }


def freematch_style_percentile_per_class(
    ranking_scores: dict[str, float],
    intended_by_id: dict[str, dict],
    image_ids: list[str],
    primary_labels: list[str],
    real_prevalence_by_label: dict[str, dict],
    base_percentile: float = 50.0,
    min_percentile: float = 10.0,
) -> dict[str, float | None]:
    """Training-free class-adaptive percentile threshold, adapted from FreeMatch (Wang et al.,
    ICLR 2023) self-adaptive thresholding.

    FreeMatch lowers the confidence threshold for classes the model has learned less well so their
    pseudo-labels are not all filtered out. ASISM applies the same idea as a ONE-TIME data-curation
    decision (never a per-training-step recompute): a class's admission percentile is scaled DOWN
    (more lenient) the rarer that class is in the REAL patient population
    (``real_class_support_context`` prevalence), so a single global cut does not starve rare classes
    of synthetic samples. The rarest class present maps to ``min_percentile`` and the most common to
    ``base_percentile`` — leniency is RANGE-normalised across the observed per-class prevalences
    (SST self-adaptive style: Zhao et al., Information Processing & Management, 2025), not merely
    divided by the maximum.

        ratio      = (prevalence(class) - min_prevalence) / (max_prevalence - min_prevalence)   in [0, 1]
        percentile = min_percentile + (base_percentile - min_percentile) * ratio
        threshold  = that percentile of the class's own candidate ranking scores

    A class with unknown or zero real prevalence falls back to ``base_percentile`` — missing
    information never buys a more lenient cut. A class with zero candidates returns ``None``. When
    every class shares the same prevalence, all get ``base_percentile``. No neural network, no
    gradient step; it is a competing threshold policy, not a replacement for the learned network.
    Pair with ``enforce_per_class_selection_floor`` so a rare class is never selected down to zero.
    """
    if not 0.0 <= min_percentile <= base_percentile <= 100.0:
        raise ValueError(
            f"require 0 <= min_percentile <= base_percentile <= 100, got {min_percentile}, {base_percentile}"
        )
    prevalences = {
        label: (real_prevalence_by_label.get(label) or {}).get("real_prevalence")
        for label in primary_labels
    }
    positive_prevalences = [value for value in prevalences.values() if value is not None and value > 0]
    max_prevalence = max(positive_prevalences) if positive_prevalences else None
    min_prevalence = min(positive_prevalences) if positive_prevalences else None
    prevalence_span = (max_prevalence - min_prevalence) if positive_prevalences else 0.0

    result: dict[str, float | None] = {}
    for label in primary_labels:
        class_scores = [
            ranking_scores[image_id]
            for image_id in image_ids
            if int(intended_by_id.get(image_id, {}).get(label, 0)) == 1
        ]
        if not class_scores:
            result[label] = None
            continue
        prevalence = prevalences[label]
        if max_prevalence is None or prevalence is None or prevalence <= 0 or prevalence_span <= 0:
            percentile = base_percentile
        else:
            ratio = min(1.0, max(0.0, (float(prevalence) - min_prevalence) / prevalence_span))
            percentile = min_percentile + (base_percentile - min_percentile) * ratio
        result[label] = float(np.percentile(class_scores, percentile))
    return result


def enforce_per_class_selection_floor(
    thresholds: dict[str, float | None],
    ranking_scores: dict[str, float],
    intended_by_id: dict[str, dict],
    image_ids: list[str],
    primary_labels: list[str],
    min_selected_per_label: int,
) -> dict[str, float | None]:
    """Lower a class's threshold just enough to admit ``min_selected_per_label`` of its own
    candidates whenever the class-adaptive percentile would otherwise select fewer.

    The one-time-curation analogue of SST's class-fairness term (Zhao et al., Information Processing
    & Management, 2025): a class-adaptive threshold must not starve a rare class. A class with no
    threshold set is left untouched; a class with fewer than ``min_selected_per_label`` candidates
    in total has its threshold dropped to admit every candidate it has. Thresholds already
    admitting enough are unchanged.
    """
    adjusted = dict(thresholds)
    for label in primary_labels:
        threshold = adjusted.get(label)
        if threshold is None:
            continue
        class_scores = sorted(
            (ranking_scores[image_id] for image_id in image_ids
             if int(intended_by_id.get(image_id, {}).get(label, 0)) == 1),
            reverse=True,
        )
        if not class_scores:
            continue
        selected = sum(1 for score in class_scores if score >= threshold)
        if selected >= min_selected_per_label:
            continue
        floor_index = min(min_selected_per_label, len(class_scores)) - 1
        adjusted[label] = float(class_scores[floor_index])
    return adjusted


def scientific_status_for_namespace(namespace: str) -> str:
    """Only the 'production' namespace is production evidence. Every other namespace (dev,
    dev-smoke-v1, ...) is development-only and must never be reported as real-world prevalence
    without this label attached."""
    return "production" if namespace == "production" else "non-production / development-only"


def real_class_support_context(
    namespace: str, labels: list[str], frame: pd.DataFrame | None = None
) -> dict:
    """Real patient-level prevalence/support per label from classifier_train (chexpert_mask_uncertain_v1).

    This is the REAL prevalence in the actual patient population of whatever `namespace` was loaded
    — NOT the fraction of synthetic candidate images that happen to carry a label, and NOT
    necessarily production evidence: check `scientific_status` before treating the numbers as
    anything more than development-only. `frame` may be pre-loaded (e.g. in tests) to avoid
    re-reading the split file.
    """
    if frame is None:
        frame = load_split(
            "classifier_train", namespace, purpose="schema_validation", caller="learned_asism_threshold"
        )
    support = patient_level_support(frame, labels)
    labels_context: dict[str, dict] = {}
    for _, row in support.iterrows():
        positive = float(row["positive_patients"])
        negative = float(row["negative_patients"])
        denominator = positive + negative
        labels_context[row["label"]] = {
            "positive_patients": positive,
            "negative_patients": negative,
            "real_prevalence": positive / denominator if denominator > 0 else None,
        }
    return {
        "namespace": namespace,
        "scientific_status": scientific_status_for_namespace(namespace),
        "labels": labels_context,
    }


def class_aware_context_vector(scores: list[float], real_prevalence: dict, budget_context: dict) -> list[float]:
    """Fixed-length (10-dim) context feature vector: candidate-pool score distribution stats + the
    REAL patient-level prevalence for this class (from real_class_support_context, not a fraction of
    synthetic candidates) + budget context. Order is part of the schema — do not reorder without
    also updating threshold_contexts.jsonl's "context_features" consumers."""
    array = np.asarray(scores, dtype=np.float64)
    if array.size:
        q25, q50, q75 = np.quantile(array, [0.25, 0.5, 0.75])
        mean, std = float(array.mean()), float(array.std())
    else:
        q25 = q50 = q75 = mean = std = 0.0
    real_prevalence_value = real_prevalence.get("real_prevalence")
    return [
        mean, std, float(q25), float(q50), float(q75),
        float(array.size), float(np.log1p(array.size)),
        float(real_prevalence_value) if real_prevalence_value is not None else 0.0,
        float(real_prevalence.get("positive_patients") or 0.0),
        float(budget_context.get("target_synthetic_to_real_ratio", 1.0)),
    ]


def bootstrap_class_contexts(
    pool: pd.DataFrame,
    class_id: int,
    label: str,
    intended_by_id: dict,
    score_column: str,
    context_source: str,
    n_replicates: int,
    min_fraction: float,
    max_fraction: float,
    ranking_checkpoint_hash: str,
    critic_checkpoint_hash: str,
    real_prevalence_context: dict,
    budget_context: dict,
    seed: int,
) -> list[dict]:
    """Resampled contexts for one class, from ONE image pool only (train-role or val-role — never
    both; `context_source` records which). These are honestly NOT independent clinical samples: they
    are repeated subsamples of the same underlying candidate pool, using the same ranking/critic
    checkpoints. `independent_clinical_sample: False` is stamped on every record so nothing
    downstream can mistake resampling diversity for genuine independent evidence.

    No sampling with replacement: each context is a distinct, duplicate-free subsample (via
    _sample's strict default), never a padded/duplicated set.
    """
    if not 0.0 < min_fraction <= max_fraction <= 1.0:
        raise ValueError(f"require 0 < min_fraction <= max_fraction <= 1.0, got {min_fraction}, {max_fraction}")
    class_ids = {
        str(image_id) for image_id in pool["image_id"]
        if int(intended_by_id.get(str(image_id), {}).get(label, 0)) == 1
    }
    class_pool = pool.loc[pool["image_id"].astype(str).isin(class_ids)]
    if class_pool.empty:
        return []
    rng = np.random.default_rng(seed)
    contexts = []
    for index in range(n_replicates):
        fraction = float(rng.uniform(min_fraction, max_fraction))
        size = max(1, round(len(class_pool) * fraction))
        image_ids = _sample(class_pool, size, rng)  # strict: no replacement, no silent undersizing
        scores = class_pool.loc[class_pool["image_id"].astype(str).isin(image_ids), score_column].tolist()
        context_features = class_aware_context_vector(scores, real_prevalence_context, budget_context)
        contexts.append({
            "context_id": f"ctx_{label}_{context_source}_{index:04d}",
            "class_id": class_id,
            "label": label,
            "context_source": context_source,
            "independent_clinical_sample": False,
            "candidate_pool_hash": hash_dict({"image_ids": sorted(image_ids)}, length=32),
            "ranking_checkpoint_hash": ranking_checkpoint_hash,
            "critic_checkpoint_hash": critic_checkpoint_hash,
            "fold": None,
            "seed": seed + index,
            "budget_context": dict(budget_context),
            "real_prevalence_context": dict(real_prevalence_context),
            "context_features": context_features,
            "image_ids": image_ids,
        })
    return contexts


def hard_threshold_grid_search(
    critic_utility_fn,
    ranking_scores: dict[str, float],
    image_ids: list[str],
    t_grid,
    budget_weight: float,
    max_selected: int,
    min_selected: int = 0,
) -> list[dict]:
    """Grid search over HARD thresholds only (boolean selection — critic_utility_fn must never be
    called with soft/continuous weights here). Empty subsets are rejected outright, never scored.
    `critic_predicted_utility` is exactly what the name says: the frozen critic's prediction, NOT a
    measured/ground-truth value — callers must never treat it as such.
    """
    evaluations = []
    for threshold in t_grid:
        selected = [image_id for image_id in image_ids if ranking_scores.get(image_id, 0.0) >= threshold]
        if not selected:
            continue  # reject empty subsets outright — never a valid candidate at any threshold
        if len(selected) > max_selected:
            selected = sorted(selected, key=lambda i: -ranking_scores.get(i, 0.0))[:max_selected]
        critic_predicted_utility = float(critic_utility_fn(selected))
        size_cost = len(selected) / max(len(image_ids), 1)
        objective = critic_predicted_utility - float(budget_weight) * size_cost
        evaluations.append({
            "candidate_threshold": float(threshold),
            "selected_count": len(selected),
            "critic_predicted_utility": critic_predicted_utility,
            "below_min_selected": len(selected) < min_selected,
            "_objective": objective,
        })
    if not evaluations:
        raise ValueError(
            "hard_threshold_grid_search: every threshold in t_grid produced an empty subset — "
            "the grid or the candidate pool is misconfigured for this context."
        )
    evaluations.sort(key=lambda entry: -entry["_objective"])
    for rank, entry in enumerate(evaluations, start=1):
        entry["rank_within_context"] = rank
        del entry["_objective"]
    return evaluations


def resolve_verified_only_targets(contexts: list[dict], measurements: list[dict]) -> list[dict]:
    """OFFICIAL training path. A context contributes a target ONLY if it has at least one
    proxy-MEASURED candidate — never a critic guess, never blended, never weighted-down-but-included.
    A context with zero measurements is simply absent from this function's output; there is no
    fallback inside this function to anything critic-derived. `measurements` must come from
    threshold_proxy_measurements.jsonl (07b's output), never from threshold_candidate_evaluations.jsonl
    (07's critic-only output) — mixing the two files here would defeat the whole point of the split.

    `proxy_best_among_verified_candidates` is deliberately NOT called "measured_best_threshold": only
    a subset of the grid (however it was chosen — see diversify_verification_candidates) was ever
    measured, so this is the best of what was checked, not a claim about the true grid optimum.
    """
    by_context: dict[str, list[dict]] = {}
    for row in measurements:
        by_context.setdefault(row["context_id"], []).append(row)

    targets = []
    for context in contexts:
        rows = by_context.get(context["context_id"], [])
        if not rows:
            continue
        best = max(rows, key=lambda row: row["measured_utility"])
        targets.append({
            "context_id": context["context_id"],
            "class_id": context["class_id"],
            "label": context["label"],
            "context_source": context["context_source"],
            "target_threshold": best["candidate_threshold"],
            "proxy_best_among_verified_candidates": best["candidate_threshold"],
            "n_verified_candidates_in_context": len(rows),
            "target_source": "proxy_verified",
            "supervision_weight": 1.0,
        })
    return targets


def resolve_critic_assisted_exploratory_targets(contexts: list[dict], evaluations: list[dict]) -> list[dict]:
    """EXPLORATORY path only — a completely separate model/training run from
    resolve_verified_only_targets. Uses the critic's own top grid pick for every context regardless
    of verification status. Its output must never be merged into, averaged with, or used as a
    fallback for the official verified-only targets; callers that train the OFFICIAL
    AdaptiveThresholdNetwork must not call this function at all.
    """
    by_context: dict[str, list[dict]] = {}
    for row in evaluations:
        by_context.setdefault(row["context_id"], []).append(row)

    targets = []
    for context in contexts:
        rows = by_context.get(context["context_id"], [])
        if not rows:
            continue
        critic_best = min(rows, key=lambda row: row["rank_within_context"])
        targets.append({
            "context_id": context["context_id"],
            "class_id": context["class_id"],
            "label": context["label"],
            "context_source": context["context_source"],
            "target_threshold": critic_best["candidate_threshold"],
            "critic_grid_threshold": critic_best["candidate_threshold"],
            "target_source": "critic_only",
            "supervision_weight": 1.0,  # full weight WITHIN this exploratory-only path only
        })
    return targets


def diversify_verification_candidates(
    evaluations: list[dict], top_k: int, quantile_fractions: list[float], candidate_pool_size: int,
) -> list[dict]:
    """Selects which grid evaluations get proxy-verified for one context: critic top-K PLUS
    quantile-spaced selection-count targets PLUS the two boundary (smallest/largest selected_count)
    candidates — never critic top-K alone. Verifying only where the critic already thinks is good
    would make the critic/proxy correlation check circular. Each returned entry carries
    `selection_reasons` (a threshold can qualify for more than one reason)."""
    if not evaluations:
        return []
    by_rank = sorted(evaluations, key=lambda entry: entry["rank_within_context"])
    chosen: dict[float, set] = {}

    def mark(entry: dict, reason: str) -> None:
        chosen.setdefault(entry["candidate_threshold"], set()).add(reason)

    for entry in by_rank[:top_k]:
        mark(entry, "critic_top_k")

    for fraction in quantile_fractions:
        target_count = max(1, round(candidate_pool_size * fraction))
        closest = min(evaluations, key=lambda entry: abs(entry["selected_count"] - target_count))
        mark(closest, "quantile_spaced")

    by_size = sorted(evaluations, key=lambda entry: entry["selected_count"])
    mark(by_size[0], "boundary")
    mark(by_size[-1], "boundary")

    # A literal 50%-of-pool selection target is included explicitly so verification always covers
    # the OLD baseline's own operating point, not just the new critic's preferences.
    baseline_target_count = max(1, round(candidate_pool_size * 0.5))
    baseline_closest = min(evaluations, key=lambda entry: abs(entry["selected_count"] - baseline_target_count))
    mark(baseline_closest, "baseline_target_ratio")

    by_threshold = {entry["candidate_threshold"]: entry for entry in evaluations}
    return [
        {**by_threshold[threshold], "selection_reasons": sorted(reasons)}
        for threshold, reasons in sorted(chosen.items())
    ]


def compute_verified_context_counts(
    verified_train_targets: list[dict], verified_held_out_targets: list[dict], labels: list[str]
) -> tuple[dict[str, int], dict[str, int]]:
    """Per-class counts of proxy_verified contexts, split by train/held-out source. The single
    source of truth both the eligibility gate and the governance decision are built from."""
    train_counts = {label: 0 for label in labels}
    held_out_counts = {label: 0 for label in labels}
    for target in verified_train_targets:
        train_counts[target["label"]] += 1
    for target in verified_held_out_targets:
        held_out_counts[target["label"]] += 1
    return train_counts, held_out_counts


def eligible_classes_for_official_training(
    train_counts: dict[str, int], held_out_counts: dict[str, int], min_verified_contexts_per_class: int
) -> set[str]:
    """A class is eligible for the OFFICIAL network only if it independently clears the minimum on
    BOTH sides — enough proxy-verified TRAIN contexts to learn from, and enough proxy-verified,
    image-disjoint HELD-OUT contexts to actually evaluate generalization on. Neither side alone is
    sufficient: training data without held-out evidence can't be trusted, and held-out evidence
    without training data has nothing to evaluate."""
    return {
        label for label in train_counts
        if train_counts[label] >= min_verified_contexts_per_class
        and held_out_counts.get(label, 0) >= min_verified_contexts_per_class
    }


def filter_targets_to_eligible_classes(targets: list[dict], eligible_classes: set[str]) -> list[dict]:
    """Restrict official fitting/evaluation to classes that cleared both evidence gates."""
    return [target for target in targets if target["label"] in eligible_classes]


def choose_full_policy(
    results: list[dict], expected_policies: list[str], expected_seeds: list[int],
    tie_noise_band: float, simplicity_order: list[str],
) -> dict:
    """Select a winner from complete ASISM-tuning full-policy measurements.

    A policy needs exactly one finite measurement for every frozen seed. Policies within the
    pre-registered noise band are resolved by the frozen simplicity order.
    """
    by_policy: dict[str, dict[int, float]] = {policy: {} for policy in expected_policies}
    for row in results:
        policy, seed = str(row["policy"]), int(row["seed"])
        if policy not in by_policy or seed not in expected_seeds:
            continue
        utility = float(row["measured_utility"])
        if not np.isfinite(utility):
            raise ValueError(f"non-finite measured utility for policy={policy}, seed={seed}")
        if seed in by_policy[policy]:
            raise ValueError(f"duplicate full-policy result for policy={policy}, seed={seed}")
        by_policy[policy][seed] = utility
    missing = {
        policy: sorted(set(expected_seeds) - set(values))
        for policy, values in by_policy.items() if set(values) != set(expected_seeds)
    }
    if missing:
        raise ValueError(f"incomplete full-policy verification results: {missing}")
    means = {policy: float(np.mean(list(values.values()))) for policy, values in by_policy.items()}
    best_mean = max(means.values())
    tied = {policy for policy, value in means.items() if best_mean - value <= tie_noise_band}
    ordered = [policy for policy in simplicity_order if policy in tied]
    winner = ordered[0] if ordered else sorted(tied)[0]
    return {
        "winning_policy": winner,
        "mean_measured_utility_by_policy": means,
        "best_mean_measured_utility": best_mean,
        "tie_noise_band": float(tie_noise_band),
        "tied_policies": sorted(tied),
        "tie_break_order": list(simplicity_order),
    }


def determine_per_class_official_method(
    verified_train_targets: list[dict],
    verified_held_out_targets: list[dict],
    labels: list[str],
    min_verified_contexts_per_class: int,
    acceptance_passed: bool,
) -> dict[str, str]:
    """Three-way per-class governance — no other outcome is possible:

    - ZERO verified TRAIN contexts for a class -> the OLD, honestly-named
      fixed_target_ratio_threshold_distillation_baseline_v1, REGARDLESS of how many held-out
      measurements exist. Held-out evidence must never be used to decide or construct a threshold —
      it exists only to evaluate generalization of something already built from train data. A class
      with train_count==0 has nothing to build a hard_proxy_best_among_verified threshold from
      either (see aggregate_hard_proxy_best_threshold, which only ever reads train targets).
    - Some verified TRAIN contexts, but below the pre-registered minimum on either side (train or
      image-disjoint held-out) -> hard_proxy_best_among_verified (the median measured-best threshold
      across that class's verified TRAIN contexts; no network trained/trusted on too little
      verified evidence).
    - Enough verified contexts on BOTH sides AND the frozen acceptance criteria passed ->
      adaptive_threshold_network.

    `verified_train_targets`/`verified_held_out_targets` must both come from
    resolve_verified_only_targets — never mix in critic-only exploratory targets here.
    """
    train_counts, held_out_counts = compute_verified_context_counts(verified_train_targets, verified_held_out_targets, labels)
    result = {}
    for label in labels:
        train_count, held_out_count = train_counts[label], held_out_counts[label]
        if train_count == 0:
            result[label] = "fixed_target_ratio_threshold_distillation_baseline_v1"
        elif train_count < min_verified_contexts_per_class or held_out_count < min_verified_contexts_per_class:
            result[label] = "hard_proxy_best_among_verified"
        else:
            result[label] = "adaptive_threshold_network" if acceptance_passed else "hard_proxy_best_among_verified"
    return result


def aggregate_hard_proxy_best_threshold(verified_train_targets: list[dict], label: str) -> float | None:
    """Final per-class threshold for hard_proxy_best_among_verified: the MEDIAN of
    proxy_best_among_verified_candidates (target_threshold) across that class's verified TRAIN-pool
    contexts ONLY. Held-out contexts are never read here — they evaluate generalization, they do not
    get a vote in what the operative threshold is. Median (not mean) is used for robustness against
    any single noisy proxy measurement. Returns None if the class has no verified train targets."""
    values = [target["target_threshold"] for target in verified_train_targets if target["label"] == label]
    if not values:
        return None
    return float(np.median(values))


def held_out_generalization_metrics(
    predicted_threshold: float, target_threshold: float, scores_for_context: dict[str, float]
) -> dict:
    """Absolute threshold error and selected-set Jaccard between a network's predicted threshold and
    the context's real target, evaluated on a held-out (val-pool-sourced) context the network never
    trained on. Utility regret (which needs the critic) is computed by the caller and merged in."""
    absolute_error = abs(predicted_threshold - target_threshold)
    predicted_selected = {image_id for image_id, score in scores_for_context.items() if score >= predicted_threshold}
    target_selected = {image_id for image_id, score in scores_for_context.items() if score >= target_threshold}
    union = predicted_selected | target_selected
    jaccard = (len(predicted_selected & target_selected) / len(union)) if union else 1.0
    return {"absolute_threshold_error": float(absolute_error), "selected_set_jaccard": float(jaccard)}


CRITERIA_DEFINITIONS = {
    "critic_proxy_median_spearman_min": {
        "unit": "spearman_rho (dimensionless, [-1, 1])",
        "definition": (
            "Median, across classes, of the WITHIN-class Spearman correlation between "
            "critic_predicted_utility and measured_utility, computed only on that class's "
            "proxy_verified candidates. The primary correlation criterion — pooled_spearman is "
            "diagnostic only and never gates acceptance, because pooling across classes with "
            "different utility scales can hide a class that individually correlates poorly."
        ),
    },
    "median_critic_predicted_utility_regret_max": {
        "unit": "critic_predicted_utility units (same scale as the SetUtilityNetwork's output)",
        "definition": (
            "Median, across held-out (bootstrap_val_pool) contexts, of (grid-best "
            "critic_predicted_utility - critic_predicted_utility at the network's own predicted "
            "threshold for that context)."
        ),
    },
    "threshold_stability_std_max": {
        "unit": "threshold probability std (the threshold itself is in [0, 1])",
        "definition": (
            "Median, across classes, of the standard deviation of the network's predicted "
            "threshold across that class's held-out contexts."
        ),
    },
}


def freeze_acceptance_criteria(criteria: dict, calibration_artifact_hash: str | None = None) -> dict:
    """Freezes acceptance-criteria numbers with a timestamp + config hash BEFORE any held-out
    evaluation is run. enforce_acceptance_criteria must be called against exactly this frozen
    payload — never against criteria re-read from a config that could have changed after results
    were seen. Tagged provisional_pre_registered_defaults: these numbers are a starting point set
    before any production evidence exists, not numbers validated by prior production runs — they
    must not be treated as settled science, and must NEVER be adjusted after seeing held-out or
    production results (that would be exactly the p-hacking this freezing mechanism exists to
    prevent)."""
    return {
        "criteria_status": "provisional_pre_registered_defaults",
        "criteria": dict(criteria),
        "criteria_definitions": {key: CRITERIA_DEFINITIONS[key] for key in criteria if key in CRITERIA_DEFINITIONS},
        "criteria_frozen_at": datetime.now(timezone.utc).isoformat(),
        "criteria_config_hash": hash_dict(dict(criteria), length=32),
        "calibration_artifact_hash": calibration_artifact_hash,
    }


def enforce_acceptance_criteria(evidence: dict, frozen: dict) -> dict:
    """Pure pass/fail evaluation of measured evidence against a FROZEN criteria payload (from
    freeze_acceptance_criteria). Returns a dict with a per-criterion verdict and an overall "passed"
    — never raises, so the caller can record a failing result rather than crash. Gates ONLY on
    critic_proxy_median_spearman_min (per-class median) — never on a pooled correlation."""
    criteria = frozen["criteria"]
    results = {}
    checks = {
        "critic_proxy_median_spearman_min": ("median_spearman", lambda value, bound: value is not None and value >= bound),
        "median_critic_predicted_utility_regret_max": (
            "median_critic_predicted_utility_regret", lambda value, bound: value is not None and value <= bound
        ),
        "threshold_stability_std_max": ("threshold_stability_std", lambda value, bound: value is not None and value <= bound),
    }
    for criterion_key, (evidence_key, comparator) in checks.items():
        if criterion_key not in criteria:
            continue
        bound = criteria[criterion_key]
        measured = evidence.get(evidence_key)
        results[criterion_key] = {
            "bound": bound,
            "measured": measured,
            "passed": bool(comparator(measured, bound)),
        }
    return {
        "criteria_status": frozen["criteria_status"],
        "criteria_frozen_at": frozen["criteria_frozen_at"],
        "criteria_config_hash": frozen["criteria_config_hash"],
        "per_criterion": results,
        "passed": all(entry["passed"] for entry in results.values()) if results else False,
    }


def critic_proxy_correlation_per_class(evaluations: list[dict], measurements: list[dict], contexts: list[dict]) -> dict:
    """Spearman correlation between critic_predicted_utility and measured_utility, computed WITHIN
    each class separately (never pooled across classes for the primary number — pooling is reported
    only as an explicit diagnostic). Requires >= 2 verified (context, threshold) pairs for a class to
    report a correlation at all; otherwise that class is simply absent from per_class_spearman."""
    import math as _math

    import scipy.stats as _stats

    label_by_context = {context["context_id"]: context["label"] for context in contexts}
    critic_by_key = {(row["context_id"], row["candidate_threshold"]): row["critic_predicted_utility"] for row in evaluations}

    by_label: dict[str, list[tuple[float, float]]] = {}
    for row in measurements:
        label = label_by_context.get(row["context_id"])
        critic_value = critic_by_key.get((row["context_id"], row["candidate_threshold"]))
        if label is None or critic_value is None:
            continue
        by_label.setdefault(label, []).append((critic_value, row["measured_utility"]))

    per_class_spearman: dict[str, float | None] = {}
    n_verified_by_class: dict[str, int] = {}
    for label, pairs in by_label.items():
        n_verified_by_class[label] = len(pairs)
        if len(pairs) < 2:
            per_class_spearman[label] = None
            continue
        predicted, measured = zip(*pairs)
        correlation = _stats.spearmanr(predicted, measured)
        statistic = getattr(correlation, "statistic", None)
        value = float(statistic) if statistic is not None else float(correlation[0])
        per_class_spearman[label] = None if _math.isnan(value) else value

    available = [value for value in per_class_spearman.values() if value is not None]
    macro_spearman = float(np.mean(available)) if available else None
    median_spearman = float(np.median(available)) if available else None

    all_pairs = [pair for pairs in by_label.values() for pair in pairs]
    pooled_spearman_diagnostic_only = None
    if len(all_pairs) >= 2:
        predicted, measured = zip(*all_pairs)
        correlation = _stats.spearmanr(predicted, measured)
        statistic = getattr(correlation, "statistic", None)
        value = float(statistic) if statistic is not None else float(correlation[0])
        pooled_spearman_diagnostic_only = None if _math.isnan(value) else value

    return {
        "per_class_spearman": per_class_spearman,
        "n_verified_by_class": n_verified_by_class,
        "macro_spearman": macro_spearman,
        "median_spearman": median_spearman,
        "pooled_spearman_diagnostic_only": pooled_spearman_diagnostic_only,
    }


def build_policy_selected_manifest(
    per_class_thresholds: dict[str, float | None],
    ranking_scores: dict[str, float],
    intended_by_id: dict[str, dict],
    image_ids: list[str],
    primary_labels: list[str],
) -> list[str]:
    """Combine per-class thresholds into ONE final selected set for a full-policy verification run.
    An image is selected if its ranking score clears the MINIMUM applicable threshold among the
    diseases it was intended for — the same min(applicable_thresholds) multi-label rule
    06_learn_thresholds_select.py already uses, kept identical here so a full-policy comparison
    across (baseline, hard_proxy_best, network, ...) is apples-to-apples on selection LOGIC, varying
    only the per-class thresholds themselves. An image with no applicable class threshold (e.g. a
    No-Finding recipe, or every applicable class currently unset) is never selected."""
    selected = []
    for image_id in image_ids:
        intended = intended_by_id.get(image_id, {})
        applicable = [
            per_class_thresholds[label] for label in primary_labels
            if int(intended.get(label, 0)) == 1 and per_class_thresholds.get(label) is not None
        ]
        if not applicable:
            continue
        threshold = min(applicable)
        if ranking_scores.get(image_id, 0.0) >= threshold:
            selected.append(image_id)
    return sorted(selected)

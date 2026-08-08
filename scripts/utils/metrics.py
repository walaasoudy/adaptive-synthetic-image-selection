"""Masked metrics and patient-level inference for Stage 5 (docs/stages2_to_5_plan.md §5, §6, §8).

Every metric here is MASKED: uncertain (-1) and blank labels are excluded rather than mapped to 0
or 1 (§6), and every result carries its effective N so a metric computed over 40 usable patients is
never silently compared against one computed over 400.

Resampling is PATIENT-level (§8). Bootstrapping images would let a patient with many studies
dominate a confidence interval purely by study count, which understates the true uncertainty.

Pure NumPy — no torch, no GPU. This module is fully testable and runnable locally, which is what
makes the whole Stage 5 analysis layer developable off the pod.
"""

from __future__ import annotations

import numpy as np

from scripts.utils.labels import (
    CLASSIFIER_TARGET_LABELS,
    PRIMARY_ENDPOINT_LABELS,
)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, with correct mid-rank tie handling.

    Returns NaN when either class is absent — an undefined AUROC, not a silent 0.5, so it can be
    excluded from a macro-average rather than dragging it toward chance.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_positive = int((labels == 1).sum())
    n_negative = int((labels == 0).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]

    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        mid_rank = (index + end) / 2.0 + 1.0
        ranks[order[index : end + 1]] = mid_rank
        index = end + 1

    positive_rank_sum = ranks[labels == 1].sum()
    u_statistic = positive_rank_sum - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step interpolation, the standard AP definition)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (labels == 1).sum() == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    predicted_positives = np.arange(1, len(sorted_labels) + 1)
    precision = true_positives / predicted_positives
    total_positives = (labels == 1).sum()
    return float((precision * (sorted_labels == 1)).sum() / total_positives)


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (scores > lower) & (scores <= upper) if lower > 0 else (scores >= lower) & (scores <= upper)
        count = int(in_bin.sum())
        if count == 0:
            continue
        error += (count / len(scores)) * abs(labels[in_bin].mean() - scores[in_bin].mean())
    return float(error)


def sensitivity_specificity_f1(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    predicted = (np.asarray(scores) >= threshold).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    true_positive = int(((predicted == 1) & (labels == 1)).sum())
    false_positive = int(((predicted == 1) & (labels == 0)).sum())
    true_negative = int(((predicted == 0) & (labels == 0)).sum())
    false_negative = int(((predicted == 0) & (labels == 1)).sum())

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else float("nan")
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else float("nan")
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else float("nan")
    if np.isnan(precision) or np.isnan(sensitivity) or (precision + sensitivity) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
    return float(sensitivity), float(specificity), float(f1)


def macro_auroc_masked(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    labels: list[str] | None = None,
    primary_labels: list[str] | None = None,
    min_positive: int = 1,
    min_negative: int = 1,
) -> dict:
    """Per-label and macro AUROC over the PRIMARY endpoint labels, masked.

    Labels failing the minimum-support rule are excluded from the macro-average by the
    pre-specified rule (§5.3) and reported with their support counts rather than an unstable value.
    """
    labels = labels or CLASSIFIER_TARGET_LABELS
    primary_labels = primary_labels or PRIMARY_ENDPOINT_LABELS
    index_of = {label: position for position, label in enumerate(labels)}

    per_label = {}
    usable = []
    for label in primary_labels:
        if label not in index_of:
            continue
        column = index_of[label]
        mask = masks[:, column].astype(bool)
        y_true = targets[mask, column].astype(np.int64)
        y_score = probabilities[mask, column]

        n_positive = int((y_true == 1).sum())
        n_negative = int((y_true == 0).sum())
        eligible = n_positive >= min_positive and n_negative >= min_negative
        value = auroc(y_score, y_true) if eligible else float("nan")

        per_label[label] = {
            "auroc": value,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "effective_n": int(mask.sum()),
            "eligible": bool(eligible and not np.isnan(value)),
        }
        if per_label[label]["eligible"]:
            usable.append(value)

    return {
        "macro_auroc": float(np.mean(usable)) if usable else float("nan"),
        "n_labels_in_macro": len(usable),
        "n_labels_excluded": len(primary_labels) - len(usable),
        "per_label": per_label,
    }


def full_metric_suite(
    probabilities: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    thresholds: dict[str, float] | None = None,
    labels: list[str] | None = None,
    min_positive: int = 1,
    min_negative: int = 1,
) -> dict:
    """Every Stage 5 metric (§8) for one condition, per label, with effective N attached."""
    labels = labels or CLASSIFIER_TARGET_LABELS
    thresholds = thresholds or {}
    index_of = {label: position for position, label in enumerate(labels)}

    per_label = {}
    for label in labels:
        column = index_of[label]
        mask = masks[:, column].astype(bool)
        y_true = targets[mask, column].astype(np.int64)
        y_score = probabilities[mask, column]

        n_positive = int((y_true == 1).sum())
        n_negative = int((y_true == 0).sum())
        eligible = n_positive >= min_positive and n_negative >= min_negative
        threshold = float(thresholds.get(label, 0.5))
        sensitivity, specificity, f1 = (
            sensitivity_specificity_f1(y_score, y_true, threshold)
            if eligible
            else (float("nan"), float("nan"), float("nan"))
        )

        per_label[label] = {
            "auroc": auroc(y_score, y_true) if eligible else float("nan"),
            "auprc": average_precision(y_score, y_true) if eligible else float("nan"),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "brier": brier_score(y_score, y_true) if eligible else float("nan"),
            "ece": expected_calibration_error(y_score, y_true) if eligible else float("nan"),
            "threshold": threshold,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "effective_n": int(mask.sum()),
            "eligible": bool(eligible),
        }

    def macro_over(metric: str, subset: list[str]) -> float:
        values = [
            per_label[label][metric]
            for label in subset
            if label in per_label and per_label[label]["eligible"]
            and not np.isnan(per_label[label][metric])
        ]
        return float(np.mean(values)) if values else float("nan")

    primary = [label for label in PRIMARY_ENDPOINT_LABELS if label in per_label]
    secondary = [label for label in labels if label not in PRIMARY_ENDPOINT_LABELS]

    # Micro-AUROC pools all confidently-labelled (row, primary-label) pairs into one ranking.
    micro_scores, micro_targets = [], []
    for label in primary:
        column = index_of[label]
        mask = masks[:, column].astype(bool)
        micro_scores.append(probabilities[mask, column])
        micro_targets.append(targets[mask, column].astype(np.int64))

    return {
        "macro_auroc_primary": macro_over("auroc", primary),
        "macro_auprc_primary": macro_over("auprc", primary),
        "micro_auroc_primary": (
            auroc(np.concatenate(micro_scores), np.concatenate(micro_targets))
            if micro_scores else float("nan")
        ),
        "macro_f1_primary": macro_over("f1", primary),
        "mean_brier_primary": macro_over("brier", primary),
        "mean_ece_primary": macro_over("ece", primary),
        "primary_labels": primary,
        "secondary_labels": secondary,
        "per_label": per_label,
    }


def patient_level_bootstrap(
    patient_ids: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Patient-level bootstrap CI (§8).

    Resamples PATIENTS with replacement and re-evaluates on all rows belonging to the drawn
    patients, so a patient's whole set of studies moves in or out together — the correct unit,
    since studies from one patient are not independent observations.
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    rows_by_patient = {patient: np.where(patient_ids == patient)[0] for patient in unique_patients}

    point_estimate = metric_fn(np.arange(len(patient_ids)))

    samples = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([rows_by_patient[patient] for patient in drawn])
        value = metric_fn(indices)
        if value is not None and not np.isnan(value):
            samples.append(value)

    if not samples:
        return {
            "point_estimate": point_estimate,
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_resamples_valid": 0,
            "n_patients": len(unique_patients),
        }

    samples = np.array(samples)
    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(np.percentile(samples, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(samples, 100 * (1 - alpha / 2))),
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_std": float(samples.std(ddof=1)) if len(samples) > 1 else float("nan"),
        "n_resamples_valid": len(samples),
        "n_patients": len(unique_patients),
    }


def paired_bootstrap_difference(
    patient_ids: np.ndarray,
    metric_fn_a,
    metric_fn_b,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired patient-level bootstrap of (A - B), with an effect size and a two-sided p-value.

    Paired: both conditions are evaluated on the SAME resampled patients each iteration, which
    removes patient-composition variance from the comparison and is why this is more sensitive than
    comparing two independent CIs.

    The p-value is the standard bootstrap proportion of resamples whose difference crosses zero.
    It is always reported WITH the effect size and CI (§8.2) — significance alone is not a result.
    """
    rng = np.random.default_rng(seed)
    patient_ids = np.asarray(patient_ids)
    unique_patients = np.unique(patient_ids)
    rows_by_patient = {patient: np.where(patient_ids == patient)[0] for patient in unique_patients}

    all_rows = np.arange(len(patient_ids))
    observed = metric_fn_a(all_rows) - metric_fn_b(all_rows)

    differences = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        indices = np.concatenate([rows_by_patient[patient] for patient in drawn])
        value_a, value_b = metric_fn_a(indices), metric_fn_b(indices)
        if not (np.isnan(value_a) or np.isnan(value_b)):
            differences.append(value_a - value_b)

    if not differences:
        return {
            "observed_difference": float(observed),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "n_resamples_valid": 0,
        }

    differences = np.array(differences)
    proportion_crossing = float(np.mean(differences <= 0)) if observed > 0 else float(np.mean(differences >= 0))
    p_value = float(min(1.0, 2 * proportion_crossing))

    return {
        "observed_difference": float(observed),
        "effect_size": float(observed),
        "ci_lower": float(np.percentile(differences, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(differences, 100 * (1 - alpha / 2))),
        "bootstrap_mean_difference": float(differences.mean()),
        "p_value": p_value,
        "n_resamples_valid": len(differences),
        "n_patients": len(unique_patients),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction for the CONFIRMATORY family (§8.2).

    Uniformly more powerful than plain Bonferroni at the same family-wise error rate, which matters
    when the confirmatory family is small and each test is expensive to obtain.
    """
    valid = {name: value for name, value in p_values.items() if not np.isnan(value)}
    ordered = sorted(valid.items(), key=lambda item: item[1])
    n_tests = len(ordered)

    results: dict[str, dict] = {}
    previous_adjusted = 0.0
    still_rejecting = True

    for rank, (name, p_value) in enumerate(ordered):
        adjusted = min(1.0, (n_tests - rank) * p_value)
        adjusted = max(adjusted, previous_adjusted)  # enforce monotonicity
        previous_adjusted = adjusted
        if adjusted > alpha:
            still_rejecting = False
        results[name] = {
            "p_value": p_value,
            "adjusted_p_value": adjusted,
            "rejected": bool(still_rejecting and adjusted <= alpha),
            "rank": rank + 1,
            "n_tests": n_tests,
            "method": "holm_bonferroni",
            "family": "confirmatory",
            "alpha": alpha,
        }

    for name, value in p_values.items():
        if name not in results:
            results[name] = {
                "p_value": value,
                "adjusted_p_value": float("nan"),
                "rejected": False,
                "method": "holm_bonferroni",
                "family": "confirmatory",
                "note": "undefined p-value (insufficient support)",
            }
    return results


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Benjamini-Hochberg FDR control for EXPLORATORY analyses (§8.2).

    Exploratory results corrected this way must be labelled exploratory wherever reported — FDR
    controls the expected false-discovery proportion, not the family-wise error rate, and is not
    interchangeable with the confirmatory correction.
    """
    valid = {name: value for name, value in p_values.items() if not np.isnan(value)}
    ordered = sorted(valid.items(), key=lambda item: item[1])
    n_tests = len(ordered)

    adjusted_values = []
    for rank, (_, p_value) in enumerate(ordered, start=1):
        adjusted_values.append(min(1.0, p_value * n_tests / rank))

    # Enforce monotonicity from the largest p-value downward.
    for index in range(len(adjusted_values) - 2, -1, -1):
        adjusted_values[index] = min(adjusted_values[index], adjusted_values[index + 1])

    results: dict[str, dict] = {}
    for (name, p_value), adjusted in zip(ordered, adjusted_values):
        results[name] = {
            "p_value": p_value,
            "adjusted_p_value": adjusted,
            "rejected": bool(adjusted <= alpha),
            "n_tests": n_tests,
            "method": "benjamini_hochberg_fdr",
            "family": "exploratory",
            "alpha": alpha,
        }

    for name, value in p_values.items():
        if name not in results:
            results[name] = {
                "p_value": value,
                "adjusted_p_value": float("nan"),
                "rejected": False,
                "method": "benjamini_hochberg_fdr",
                "family": "exploratory",
                "note": "undefined p-value (insufficient support)",
            }
    return results

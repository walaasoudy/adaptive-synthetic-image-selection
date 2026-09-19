#!/usr/bin/env python3
"""Adaptive Threshold Learning: how high a candidate must score to be kept, decided per class.

One global cut cannot serve this dataset. nv has thousands of real images and its synthetic
candidates compete against a rich real distribution; df and vasc have a few dozen, and a cut tuned
on the pooled score distribution — two thirds of which is nv — would admit almost nothing for them.
That is the failure mode the synthetic data exists to fix, so the threshold has to know the class.

TWO POLICIES, BOTH CLASS-AWARE, BOTH DECIDED ON TUNING EVIDENCE ONLY.

  percentile  Training-free. A class's admission percentile is scaled DOWN — more lenient — the
              rarer that class is in the REAL patient population, range-normalised across the
              observed prevalences. Adapted from FreeMatch's self-adaptive thresholding (Wang et
              al., ICLR 2023) and SST's class-fairness term (Zhao et al., IP&M 2025), applied as a
              ONE-TIME curation decision rather than a per-step recompute. This is the baseline the
              learned policy has to beat, not a fallback for when the network fails.

  network     The proposed method. A frozen one-dimensional search finds each class's threshold
              under an explicit utility-versus-budget objective, and `AdaptiveThresholdNetwork` is
              then distilled from those targets given a class CONTEXT vector. The point of the
              distillation is that the network learns the RULE relating a class's situation to its
              cut, rather than memorising seven numbers: a class whose candidate pool shifts gets a
              threshold consistent with how the other classes were treated.

ORDER IS PART OF THE METHOD. The quality floor is absolute and comes FIRST: a candidate below it is
rejected whatever its class, so rarity never buys admission for a bad image. Only then does the
class-aware threshold apply, and only then is a rare class's cut lowered to meet its floor. Running
the floor after the quota would let a starved class pull in its own worst candidates.

Nothing here reads a split. The prevalence that makes a class "rare" comes from the real training
split, the scores come from the ranking network, and `final_eval_heldout` is not involved at any
point.
"""

from __future__ import annotations

import numpy as np

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

CONTEXT_FEATURES = (
    "real_prevalence",
    "log1p_n_candidates",
    "score_mean",
    "score_std",
    "score_q25",
    "score_q50",
    "score_q75",
    "target_fraction",
)


def normalise_scores(scores: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map ranking scores onto [0, 1] by min-max over the whole pool.

    The threshold network's head is a sigmoid, so its output lives in [0, 1] and the scores it is
    compared against must live there too. Min-max is strictly monotone, so no candidate's position
    relative to any other changes — only the units do. The transform is recorded so a later run can
    state what a stored threshold meant.
    """
    values = np.asarray(scores, dtype=np.float64)
    if not values.size:
        return values, {"method": "min_max", "min": 0.0, "max": 1.0, "degenerate": True}
    low, high = float(values.min()), float(values.max())
    span = high - low
    if span <= 0:
        # Every candidate scored identically: no ordering exists to threshold on, and pretending
        # otherwise would make the cut an artifact of floating-point noise.
        return np.full_like(values, 0.5), {"method": "min_max", "min": low, "max": high, "degenerate": True}
    return (values - low) / span, {"method": "min_max", "min": low, "max": high, "degenerate": False}


def quality_floor(scores: np.ndarray, percentile: float) -> float:
    """The absolute floor, computed on the WHOLE pool before any class is considered.

    Pooled deliberately: a per-class floor would define "bad" relative to each class's own
    candidates, so the worst class's mediocre images would clear a floor its own mediocrity set.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"quality_floor_percentile must be in [0, 100], got {percentile}")
    values = np.asarray(scores, dtype=np.float64)
    if not values.size:
        return float("-inf")
    return float(np.percentile(values, percentile))


def target_counts(
    real_counts: dict[str, int],
    ratio: float,
    minimum: int,
    maximum: int,
    labels: list[str] | None = None,
) -> dict[str, int]:
    """How many synthetic images each class should ideally receive.

    Proportional to the class's REAL training count, so the target expresses "this much synthetic
    data per real image" rather than a flat quota — a flat quota would hand df as many synthetic
    images as nv and change the class balance far more than the augmentation is meant to.
    """
    if ratio < 0:
        raise ValueError(f"target_synthetic_to_real_ratio must be >= 0, got {ratio}")
    if minimum > maximum:
        raise ValueError(f"min_accepted_per_class ({minimum}) exceeds max_accepted_per_class ({maximum})")
    labels = list(labels or CLASSIFIER_TARGET_LABELS)
    return {
        label: int(np.clip(round(int(real_counts.get(label, 0)) * float(ratio)), minimum, maximum))
        for label in labels
    }


def rarity_scaled_percentiles(
    real_counts: dict[str, int],
    labels: list[str],
    base_percentile: float,
    min_percentile: float,
) -> dict[str, float]:
    """Each class's admission percentile, lower (more lenient) the rarer the class really is.

    Range-normalised across the observed prevalences rather than divided by the maximum: with nv at
    two thirds of the data, dividing by the maximum would leave every other class bunched at almost
    the same leniency, which is the opposite of what a class-adaptive rule is for.

    A class with no real images falls back to `base_percentile` — missing information never buys a
    more lenient cut.
    """
    if not 0.0 <= min_percentile <= base_percentile <= 100.0:
        raise ValueError(
            f"require 0 <= min_percentile <= base_percentile <= 100, got {min_percentile}, {base_percentile}"
        )
    total = sum(int(real_counts.get(label, 0)) for label in labels)
    prevalence = {
        label: (int(real_counts.get(label, 0)) / total if total else 0.0) for label in labels
    }
    positive = [value for value in prevalence.values() if value > 0]
    if not positive or max(positive) - min(positive) <= 0:
        return {label: float(base_percentile) for label in labels}
    low, high = min(positive), max(positive)

    result = {}
    for label in labels:
        value = prevalence[label]
        if value <= 0:
            result[label] = float(base_percentile)
            continue
        ratio = float(np.clip((value - low) / (high - low), 0.0, 1.0))
        result[label] = float(min_percentile + (base_percentile - min_percentile) * ratio)
    return result


def search_threshold(scores: np.ndarray, target_count: int, budget_weight: float, grid: int = 101) -> float:
    """The frozen one-dimensional search the network is distilled FROM.

    Maximises `mean(score of the kept candidates) - budget_weight * |kept - target| / target`: keep
    good candidates, but do not miss the class's budget to do it. The grid, the objective and the
    tie rule are fixed in advance — on a tie the HIGHER threshold wins, so the more selective of two
    equally-scoring cuts is taken rather than whichever the search happened to reach first.
    """
    values = np.asarray(scores, dtype=np.float64)
    if not values.size:
        return 1.0
    best = None
    for threshold in np.linspace(0.0, 1.0, int(grid)):
        kept = values[values >= threshold]
        utility = float(kept.mean()) if kept.size else -1.0
        budget_error = abs(len(kept) - target_count) / max(target_count, 1)
        objective = utility - float(budget_weight) * budget_error
        candidate = (objective, threshold)
        if best is None or candidate > best:
            best = candidate
    return float(best[1])


def class_context(
    scores: np.ndarray, real_prevalence: float, target_count: int
) -> np.ndarray:
    """What the threshold network is allowed to know about a class.

    Deliberately only the SHAPE of the class's own score distribution, how rare the class really is,
    and how many images it is budgeted. No class identity beyond the embedding, and nothing about
    which images those scores belong to — the network decides a cut from a class's situation, not
    from the candidates themselves, which the ranking network has already judged.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size:
        q25, q50, q75 = (float(value) for value in np.quantile(values, [0.25, 0.5, 0.75]))
        mean, std = float(values.mean()), float(values.std())
    else:
        q25 = q50 = q75 = mean = std = 0.0
    return np.asarray(
        [
            float(real_prevalence),
            float(np.log1p(len(values))),
            mean,
            std,
            q25,
            q50,
            q75,
            float(target_count) / max(len(values), 1),
        ],
        dtype=np.float32,
    )


def enforce_class_floor(
    thresholds: dict[str, float],
    scores_by_class: dict[str, np.ndarray],
    minimum_per_class: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Lower a class's threshold just enough to admit `minimum_per_class` of its own candidates.

    Applied AFTER the quality floor has already removed the pool's worst images, so what this can
    admit is a class's best remaining candidates, never its worst overall. A class holding fewer
    candidates than the floor has its threshold dropped to admit all of them, and the shortfall is
    reported rather than hidden: a class that cannot reach its floor is a finding about the
    generator, not something for the selector to paper over.
    """
    adjusted, lowered = dict(thresholds), {}
    for label, threshold in thresholds.items():
        values = np.sort(np.asarray(scores_by_class.get(label, []), dtype=np.float64))[::-1]
        if not values.size:
            continue
        kept = int((values >= threshold).sum())
        if kept >= minimum_per_class:
            continue
        index = min(int(minimum_per_class), len(values)) - 1
        adjusted[label] = float(values[index])
        lowered[label] = kept
    return adjusted, lowered


__all__ = [
    "CONTEXT_FEATURES",
    "class_context",
    "enforce_class_floor",
    "normalise_scores",
    "quality_floor",
    "rarity_scaled_percentiles",
    "search_threshold",
    "target_counts",
]

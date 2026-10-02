#!/usr/bin/env python3
"""Seed-aware robustness analysis of a condition comparison, predeclared in
docs/ham10000_asism_v2_final_protocol.md §5 before any ASISM v2 result.

The frozen Stage 5 interval (ham10000_compare_conditions.py) resamples lesions only. Each
condition's metric there is the mean over its seeds on the resampled rows, so training-seed
variation enters the point estimate but is never resampled. At this recipe the seed SD of balanced
accuracy (0.03-0.07) is far larger than the lesion sampling error, so that interval understates the
uncertainty about a difference between training procedures. Two analyses that do carry it are
reported NEXT to the frozen one, never instead of it:

  welch       per-seed balanced accuracy on the full split, Welch two-sided test of left - right
              and its 95% interval. The unit is the training run.
  seed_lesion two-level bootstrap: each resample draws lesions with replacement (shared by both
              conditions, so the comparison stays paired over lesions) and, independently per
              condition, seeds with replacement. Percentile 95% interval of the difference.

Pure numpy/scipy on prediction tables; reads no split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS

N_CLASSES = len(CLASSIFIER_TARGET_LABELS)


def _predicted(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"prob_{c}" for c in CLASSIFIER_TARGET_LABELS]].to_numpy(dtype=np.float64).argmax(axis=1)


def balanced_accuracy_weighted(truth: np.ndarray, predicted: np.ndarray, weights: np.ndarray) -> float:
    """Balanced accuracy where row i counts weights[i] times (a bootstrap multiplicity), over the
    classes present. Equal to ham10000_metrics.balanced_accuracy on the expanded rows."""
    support = np.bincount(truth, weights=weights, minlength=N_CLASSES)
    hits = np.bincount(truth, weights=weights * (truth == predicted), minlength=N_CLASSES)
    present = support > 0
    return float((hits[present] / support[present]).mean())


def per_seed_balanced_accuracy(frames: list[pd.DataFrame], truth: np.ndarray) -> np.ndarray:
    ones = np.ones(len(truth))
    return np.array([balanced_accuracy_weighted(truth, _predicted(f), ones) for f in frames])


def welch(left: np.ndarray, right: np.ndarray, alpha: float = 0.05) -> dict:
    left, right = np.asarray(left, float), np.asarray(right, float)
    vl, vr = left.var(ddof=1) / len(left), right.var(ddof=1) / len(right)
    se = float(np.sqrt(vl + vr))
    df = float((vl + vr) ** 2 / (vl ** 2 / (len(left) - 1) + vr ** 2 / (len(right) - 1))) if se > 0 else float("inf")
    diff = float(left.mean() - right.mean())
    t = stats.t.ppf(1 - alpha / 2, df)
    p = float(2 * stats.t.sf(abs(diff / se), df)) if se > 0 else float("nan")
    return {"difference": diff, "se": se, "df": df, "ci95": [diff - t * se, diff + t * se], "p_value": p,
            "n_left": int(len(left)), "n_right": int(len(right)),
            "mean_left": float(left.mean()), "mean_right": float(right.mean()),
            "sd_left": float(left.std(ddof=1)), "sd_right": float(right.std(ddof=1))}


def seed_lesion_bootstrap(left: list[pd.DataFrame], right: list[pd.DataFrame], truth: np.ndarray,
                          lesions: np.ndarray, n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(lesions.astype(str), return_inverse=True)
    pl = [_predicted(f) for f in left]
    pr = [_predicted(f) for f in right]
    diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        drawn = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = drawn[inverse].astype(float)
        sl = rng.integers(0, len(pl), len(pl))
        sr = rng.integers(0, len(pr), len(pr))
        ml = np.mean([balanced_accuracy_weighted(truth, pl[i], weights) for i in sl])
        mr = np.mean([balanced_accuracy_weighted(truth, pr[i], weights) for i in sr])
        diffs[b] = ml - mr
    ones = np.ones(len(truth))
    observed = float(np.mean([balanced_accuracy_weighted(truth, p, ones) for p in pl])
                     - np.mean([balanced_accuracy_weighted(truth, p, ones) for p in pr]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))
    return {"observed_difference": observed, "ci95": [float(lo), float(hi)], "bootstrap_p_value": p,
            "n_resamples": int(n_resamples), "seed": int(seed), "resampling": "lesions (paired) x seeds (per condition)"}


def robustness(by_condition: dict[str, list[pd.DataFrame]], reference: pd.DataFrame, pairs,
               n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05) -> dict:
    truth = reference["true_class_index"].to_numpy(dtype=int)
    lesions = reference["lesion_id"].astype(str).to_numpy()
    per_seed = {c: per_seed_balanced_accuracy(frames, truth) for c, frames in by_condition.items()}
    out = {"metric": "balanced_accuracy",
           "per_seed": {c: {"seeds": [int(f["seed"].iloc[0]) for f in by_condition[c]],
                            "values": [float(v) for v in per_seed[c]],
                            "mean": float(per_seed[c].mean()),
                            "sd": float(per_seed[c].std(ddof=1)) if len(per_seed[c]) > 1 else float("nan")}
                        for c in by_condition},
           "comparisons": {}}
    for left, right in pairs:
        out["comparisons"][f"{left}_vs_{right}"] = {
            "welch": welch(per_seed[left], per_seed[right], alpha),
            "seed_lesion_bootstrap": seed_lesion_bootstrap(by_condition[left], by_condition[right], truth,
                                                           lesions, n_resamples, seed, alpha),
        }
    return out

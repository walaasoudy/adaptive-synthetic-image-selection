"""The pre-registered checks around the V2 ranker (contract §8 and §9). CPU only.

    reliability_gate   G1, before any fit: are the 5-seed subset means repeatable enough to learn from?
    acceptance         after the fit, on test subsets read once: does the ranker predict the utility
                       of subsets it never saw, and does it do so better than size and class alone?
    selection_stability  how much of the selected count is the bootstrap: the selection repeated at
                       each pre-registered fit seed.

Thresholds come from scripts/asism_v2/prereg.py. Nothing here chooses or relaxes one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .contracts import validate_measurements, write_new_json
from .pipeline import FittedRanking, _matrix, fit_ranker
from .stopping import progressive_select

SIZE_AND_CLASS_ONLY = (False, False, False, False)
SIMILARITY_ONLY = (True, False, False, False)
COMPOSITE_COLUMNS = ("similarity_knn_mean", "explainability_calibrated_typicality")
ACCEPTANCE_NAME = "ranker_acceptance.json"


def _anova(groups: list[np.ndarray]) -> dict:
    """One-way ANOVA with k repeats per subset -> ICC(1) and the reliability of a k-run mean."""
    k = len(groups[0])
    if len(groups) < 2 or k < 2 or any(len(g) != k for g in groups):
        raise ValueError("Reliability needs >= 2 subsets with the same number (>= 2) of repeats")
    means = np.array([g.mean() for g in groups])
    msb = k * means.var(ddof=1)
    msw = float(np.mean([g.var(ddof=1) for g in groups]))
    between = max((msb - msw) / k, 0.0)
    icc = between / (between + msw) if between + msw > 0 else 0.0
    return {"subsets": len(groups), "repeats": k, "between_variance": between, "within_variance": msw,
            "icc_single_run": icc,
            "reliability_of_mean": between / (between + msw / k) if between + msw > 0 else 0.0}


def reliability_gate(values: dict[str, list[float]], sizes: dict[str, int], threshold: float) -> dict:
    """G1 on the train + validation subsets. `values` is validate_measurements' output.

    The gated number is the reliability of the subset means over all those subsets, as approved.
    Sizes differ between subsets, so that number includes the effect of size. The same quantity
    within each size (which images, at a fixed count) is reported next to it and does not gate.
    """
    groups = {sid: np.asarray(v, dtype=float) for sid, v in values.items()}
    overall = _anova(list(groups.values()))
    within_size = {}
    for size in sorted(set(sizes[sid] for sid in groups)):
        members = [groups[sid] for sid in groups if sizes[sid] == size]
        within_size[str(size)] = _anova(members) if len(members) >= 2 else None
    return {"gate": "G1", "threshold": float(threshold), **overall,
            "passed": bool(overall["reliability_of_mean"] >= threshold),
            "within_size_not_gating": within_size}


def predict_subsets(fitted: FittedRanking, frame: pd.DataFrame, subsets: dict[str, dict],
                    subset_ids: list[str]) -> np.ndarray:
    tensors = _matrix(subsets, subset_ids, frame, fitted.normalizer, fitted.classes)
    fitted.model.eval()
    with torch.no_grad():
        return fitted.model(*tensors).numpy().astype(float)


def _centred_ranks(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    out = np.empty(len(values))
    for s in np.unique(strata):
        rows = np.flatnonzero(strata == s)
        ranks = pd.Series(values[rows]).rank().to_numpy()
        out[rows] = ranks - ranks.mean()
    return out


def stratified_spearman(predicted, measured, strata, permutations: int, seed: int) -> dict:
    """Spearman within strata (ranks centred within each size), with a one-sided permutation p-value
    obtained by permuting the measured values within each size."""
    predicted, measured, strata = map(np.asarray, (predicted, measured, strata))
    a = _centred_ranks(predicted.astype(float), strata)

    def rho(values: np.ndarray) -> float:
        b = _centred_ranks(values, strata)
        denominator = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return float((a * b).sum() / denominator) if denominator > 0 else 0.0

    observed = rho(measured.astype(float))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    at_least = 0
    for _ in range(permutations):
        shuffled = measured.astype(float).copy()
        for rows in groups:
            shuffled[rows] = shuffled[rng.permutation(rows)]
        at_least += rho(shuffled) >= observed
    return {"spearman_within_size": observed, "one_sided_p": (at_least + 1) / (permutations + 1),
            "permutations": int(permutations), "n": int(len(measured))}


def _equal_weight_composite(frame: pd.DataFrame) -> pd.Series:
    parts = []
    for column in COMPOSITE_COLUMNS:
        low = frame.groupby("dx")[column].transform("min")
        spread = frame.groupby("dx")[column].transform("max") - low
        parts.append(((frame[column] - low) / spread.where(spread > 0, 1.0)).fillna(0.0))
    return pd.Series((sum(parts) / len(parts)).to_numpy(), index=frame["image_id"].astype(str))


def acceptance(frame: pd.DataFrame, subsets: dict[str, dict], fit_measurements: list[dict],
               test_measurements: list[dict], seeds: list[int], protocol: dict, prereg: dict,
               fit_kwargs: dict, out_path: Path) -> dict:
    """Fit on train + validation, then read the test subsets ONCE and write the verdict.

    `out_path` is created exclusively: a second acceptance run in the same place is refused, so the
    test subsets cannot be read, adjusted to and read again.
    """
    out_path = Path(out_path)
    if out_path.exists():
        raise ValueError(f"{out_path} exists: the test subsets were already read once")
    rule = prereg["acceptance"]
    test_ids = sorted(sid for sid, s in subsets.items() if s["role"] == "test")
    members = {sid: list(map(str, subsets[sid]["image_ids"])) for sid in test_ids}
    values = validate_measurements(test_measurements, members, seeds, protocol)
    measured = np.array([np.mean(values[sid]) for sid in test_ids])
    sizes = np.array([len(members[sid]) for sid in test_ids])

    no_bootstrap = {**fit_kwargs, "bootstrap": 0}
    models = {"ranker": fit_ranker(frame, subsets, fit_measurements, seeds, protocol, **no_bootstrap),
              "size_and_class_only": fit_ranker(frame, subsets, fit_measurements, seeds, protocol,
                                                signal_mask=SIZE_AND_CLASS_ONLY, **no_bootstrap),
              "similarity_only": fit_ranker(frame, subsets, fit_measurements, seeds, protocol,
                                            signal_mask=SIMILARITY_ONLY, **no_bootstrap)}
    if no_bootstrap.get("architecture", "linear") != "linear":
        # Contract §9, amendment of 2026-10-04: the linear score the network replaced, reported
        # next to it. It does not gate and is never used to select.
        models["linear_additive"] = fit_ranker(frame, subsets, fit_measurements, seeds, protocol,
                                               **{**no_bootstrap, "architecture": "linear"})
    report = {}
    for name, fitted in models.items():
        predicted = predict_subsets(fitted, frame, subsets, test_ids)
        report[name] = {"test_mse": float(np.mean((predicted - measured) ** 2)),
                        **stratified_spearman(predicted, measured, sizes, int(rule["permutations"]),
                                              int(rule["permutation_seed"]))}
    composite = _equal_weight_composite(frame)
    composite_means = np.array([composite.loc[members[sid]].mean() for sid in test_ids])
    report["equal_weight_composite"] = stratified_spearman(
        composite_means, measured, sizes, int(rule["permutations"]), int(rule["permutation_seed"]))

    ranker, null = report["ranker"], report["size_and_class_only"]
    correlation = (ranker["spearman_within_size"] >= float(rule["min_within_size_spearman"])
                   and ranker["one_sided_p"] <= float(rule["max_one_sided_p"]))
    beats_null = ranker["test_mse"] < null["test_mse"]
    result = {"gate": "ranker_acceptance", "prereg_sha256": prereg["prereg_sha256"],
              "test_subsets": len(test_ids), "criteria": dict(rule),
              "a_correlation_passed": bool(correlation), "b_beats_size_and_class_only": bool(beats_null),
              "accepted": bool(correlation and beats_null), "models": report,
              "if_not_accepted": "the learned ranking is not used; this failure is the reported result"}
    write_new_json(out_path, result)
    return result


def selection_stability(frame: pd.DataFrame, subsets: dict[str, dict], fit_measurements: list[dict],
                        seeds: list[int], protocol: dict, candidates: pd.DataFrame, prereg: dict,
                        fit_kwargs: dict) -> dict:
    """The whole fit-and-select repeated at each pre-registered fit seed. Reported, never chosen
    from: the selection used downstream is the one at fit.seed."""
    runs = {}
    for fit_seed in prereg["stopping"]["stability_fit_seeds"]:
        fitted = fit_ranker(frame, subsets, fit_measurements, seeds, protocol,
                            **{**fit_kwargs, "seed": int(fit_seed)})
        selected = progressive_select(fitted, candidates)
        runs[int(fit_seed)] = {"counts": selected["counts"], "total": int(sum(selected["counts"].values())),
                               "ids": set(selected["selected"]["image_id"])}
    reference = runs[int(prereg["fit"]["seed"])]["ids"] if int(prereg["fit"]["seed"]) in runs else None
    totals = [run["total"] for run in runs.values()]
    overlap = {}
    for fit_seed, run in runs.items():
        union = run["ids"] | reference if reference is not None else set()
        overlap[fit_seed] = (len(run["ids"] & reference) / len(union)) if union else 1.0
    return {"fit_seeds": list(runs), "per_seed_counts": {s: run["counts"] for s, run in runs.items()},
            "total_min": int(min(totals)), "total_max": int(max(totals)),
            "jaccard_with_the_reported_selection": overlap,
            "reported_selection_fit_seed": int(prereg["fit"]["seed"])}

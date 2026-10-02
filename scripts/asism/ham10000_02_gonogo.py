#!/usr/bin/env python3
"""Stage 3b — the ASISM component Go/No-Go gate for HAM10000.

Runs AFTER the five signals are computed and BEFORE any weight, threshold or ranking model is fitted.
Each signal is checked for technical validity, selection eligibility, numerical stability, missing
rate, provenance, score directionality, redundancy with the other signals, and whether it changes
the selection at all. Outcomes:

    include        - passed everything; eligible for the ranking network and the frozen selector
    ablation_only  - technically valid but weak or a near-restatement of another signal; kept for the
                     leave-one-signal-out analysis, not weighted into the selector
    exclude        - invalid, unstable, or contradicting its own stated meaning; removed from the
                     selector, artifact retained for audit, exclusion and reason recorded

WHAT IS DIFFERENT FROM THE CHEXPERT GATE (scripts/asism/02_gonogo.py), AND WHY
  * The score columns differ because the signals differ: `uncertainty_mutual_information` (the
    epistemic term of the softmax decomposition) replaces `uncertainty_mean_std`, and
    `explainability_calibrated_typicality` replaces `explainability_region_overlap`. The CheXpert
    columns do not exist in the HAM10000 artifacts, so running that gate here would report every
    signal as technically invalid.
  * Redundancy and usefulness are measured WITHIN CLASS. HAM10000's pool is class-conditioned and
    two-thirds nv; pooled across classes, any two signals that merely separate nv from mel look
    correlated, and a pooled top-50% selection is dominated by whichever class the signals happen to
    score highest. Both would be measuring class structure rather than the signals.
  * Rank (Spearman) rather than linear correlation: a typicality p-value, an entropy and a cosine
    mean are not on a common scale, and what "redundant" means here is "orders the candidates the
    same way", which is exactly what a rank correlation measures.
  * An explainability artifact must additionally pass the Stage 3 selection guard
    (`selection_explainability_column`): raw or diagnostic-only attention statistics are not
    admissible selection features however well they score on the other checks.

EVIDENCE DISCIPLINE. The gate reads the synthetic candidate pool and the signal artifacts only.
final_eval_heldout is never opened, and any artifact whose provenance names it is refused outright
rather than merely reported.

Usage:
    python scripts/asism/ham10000_02_gonogo.py --namespace ham-stratified-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.asism.ham10000_signals import (  # noqa: E402
    FORBIDDEN_REFERENCE_SPLITS,
    UncalibratedExplainabilityError,
    assert_selection_features_allowed,
    selection_explainability_column,
)
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import get_git_commit_hash, read_json, write_json  # noqa: E402

# Each signal's headline score column, and whether higher is better.
#   True  -> higher is better; the column may enter an equal-weight composite
#   None  -> no a-priori direction; the signal is reported and banded, never scored as good or bad
PRIMARY_SCORE_COLUMN = {
    "similarity": ("similarity_knn_mean", True),
    "iqa": ("iqa_composite", True),
    "uncertainty": ("uncertainty_mutual_information", None),
    "explainability": ("explainability_calibrated_typicality", True),
    "agreement": ("agreement_score", True),
}

SIGNALS = tuple(PRIMARY_SCORE_COLUMN)
ARTIFACT_FILENAME = {name: f"{name}_scores.parquet" for name in SIGNALS}

OUTCOME_INCLUDE = "include"
OUTCOME_ABLATION = "ablation_only"
OUTCOME_EXCLUDE = "exclude"

HARD_CHECKS = (
    "technical_validity",
    "selection_eligibility",
    "numerical_stability",
    "missing_rate",
    "provenance",
)


class UpstreamGate(SystemExit):
    """A required input is missing or inconsistent. The message names what produces it."""


# ==============================================================================================
# Per-signal checks
# ==============================================================================================


def check_technical_validity(frame: pd.DataFrame, signal: str) -> dict:
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    has_ids = "image_id" in frame.columns
    has_score = column in frame.columns
    return {
        "passed": bool(has_ids and has_score and len(frame) > 0),
        "has_image_id": has_ids,
        "has_primary_score_column": has_score,
        "n_rows": int(len(frame)),
        "primary_score_column": column,
    }


def check_selection_eligibility(frame: pd.DataFrame, signal: str) -> dict:
    """May this artifact's headline column be used as a selection feature at all?

    For four signals this is the cheap name check. For explainability it is the substantive one:
    Stage 3 admits exactly one attention feature — the typicality calibrated against the real
    gen_train reference — and refuses the raw peripheral mass, the focus-area statistic and anything
    calibrated against a split that is not permitted. A signal that cannot pass this guard cannot be
    included however well it scores on stability or usefulness, so it is a HARD check.
    """
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    try:
        assert_selection_features_allowed([column])
        if signal == "explainability":
            selection_explainability_column(frame)
    except UncalibratedExplainabilityError as exc:
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}", "column": column}
    except Exception as exc:  # ReferenceLeakageError and anything else the guard raises
        return {"passed": False, "reason": f"{type(exc).__name__}: {exc}", "column": column}
    return {"passed": True, "column": column, "guard": "assert_selection_features_allowed"}


def check_numerical_stability(values: np.ndarray, config) -> dict:
    """A score that is constant, or that carries infinities, cannot order candidates.

    NaN is NOT counted as instability here — it is the missing-rate check's business, and for
    explainability a NaN is a documented, expected state (no reference for that class). An infinity
    is different: it is an arithmetic failure, and one of them makes every ranking that touches it
    meaningless.
    """
    values = np.asarray(values, dtype=np.float64)
    defined = values[~np.isnan(values)]
    infinite_fraction = float(np.isinf(defined).mean()) if len(defined) else 0.0
    finite = defined[np.isfinite(defined)]
    n_unique = int(len(np.unique(finite)))
    min_unique = int(config.gonogo.min_unique_values)
    return {
        "passed": bool(infinite_fraction == 0.0 and n_unique >= min_unique),
        "infinite_fraction": infinite_fraction,
        "n_unique_values": n_unique,
        "min_unique_required": min_unique,
        "std": float(np.std(finite)) if len(finite) else float("nan"),
        "note": "A near-constant score carries no selection information even if it computes cleanly.",
    }


def check_missing_rate(frame: pd.DataFrame, signal: str, config) -> dict:
    """How much of the candidate pool this signal cannot score.

    Explainability's NaNs are legitimate — a class whose gen_train reference was too thin to build is
    recorded, not hidden — but legitimate is not the same as usable: those candidates cannot be
    ranked by this signal, and a signal that is missing for a whole class would silently select
    within the classes it does cover. The rate is judged on the same threshold as the others.
    """
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    values = frame[column].to_numpy(dtype=np.float64, na_value=np.nan) if column in frame.columns else np.array([])
    missing_fraction = float(np.isnan(values).mean()) if len(values) else 1.0
    if signal == "iqa" and "iqa_valid" in frame.columns:
        # IQA reports its own per-image validity; a row that failed to load is missing even if the
        # composite column happens to hold a number.
        missing_fraction = max(missing_fraction, float(1.0 - frame["iqa_valid"].astype(bool).mean()))
    threshold = float(config.gonogo.max_missing_fraction)
    return {
        "passed": bool(missing_fraction <= threshold),
        "missing_fraction": missing_fraction,
        "max_allowed": threshold,
    }


def check_provenance(scores_dir: Path, signal: str) -> dict:
    """The artifact carries what is needed to reproduce it, and names no forbidden split.

    Recomputation is a GPU-cost decision for the operator, so what is enforced here is that the
    sidecar exists, identifies the candidate pool and the code that produced it, and does not
    reference final_eval_heldout anywhere in its values — a signal computed against the protected
    split would invalidate Stage 5 whatever its scores look like.
    """
    sidecar = scores_dir / f"{signal}_scores.provenance.json"
    if not sidecar.is_file():
        return {"passed": False, "reason": "missing_provenance_sidecar"}
    provenance = read_json(sidecar)
    required = {"schema_version", "signal", "n_rows", "candidates_csv_sha256", "git_commit_hash"}
    missing = sorted(required - set(provenance))
    forbidden = sorted(_forbidden_split_mentions(provenance))
    return {
        "passed": bool(not missing and not forbidden),
        "missing_provenance_fields": missing,
        "forbidden_split_mentions": forbidden,
        "recorded_n_rows": provenance.get("n_rows"),
        "candidates_csv_sha256": provenance.get("candidates_csv_sha256"),
    }


def _forbidden_split_mentions(node, path: str = "") -> list[str]:
    """Every provenance field whose value names a split this stage may never read."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_forbidden_split_mentions(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found.extend(_forbidden_split_mentions(value, f"{path}[{index}]"))
    elif isinstance(node, str):
        if any(split in node for split in FORBIDDEN_REFERENCE_SPLITS):
            found.append(path or node)
    return found


def check_directionality(frame: pd.DataFrame, signal: str) -> dict:
    """Does the score move the way the methodology claims it does?

    Each signal is checked against an internal contrast whose expected direction is known a priori,
    rather than assuming the sign is right because the formula looks right. Where the contrast is
    absent from the sample the check does not fail — it reports that the claim was not contradicted,
    which is a different statement from "verified" and is recorded as such.
    """
    result: dict = {"passed": None, "method": "", "detail": {}}

    column, _ = PRIMARY_SCORE_COLUMN[signal]
    if column not in frame.columns:
        # technical_validity has already failed; there is nothing to check the direction of, and
        # saying so is more useful in the report than a second copy of the same failure.
        return {"passed": None, "method": "primary_score_column_absent", "detail": {"column": column}}

    if signal == "iqa":
        # Images the domain checks flagged as near-uniform must score BELOW unflagged ones.
        if "iqa_is_near_uniform" in frame.columns and frame["iqa_is_near_uniform"].astype(bool).any():
            flagged_mask = frame["iqa_is_near_uniform"].astype(bool)
            flagged = frame.loc[flagged_mask, "iqa_composite"].mean()
            clean = frame.loc[~flagged_mask, "iqa_composite"].mean()
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
        # The single-label contrast the CheXpert version could not make: under mutually exclusive
        # classes an image either reads back as the class the recipe asked for or it does not, and
        # the score must be higher when it does.
        if {"agreement_is_argmax_match", "agreement_score"} <= set(frame.columns):
            match = frame["agreement_is_argmax_match"].astype(bool)
            if match.any() and (~match).any():
                result.update(
                    passed=bool(frame.loc[match, "agreement_score"].mean() > frame.loc[~match, "agreement_score"].mean()),
                    method="candidates_read_back_as_the_intended_class_score_higher",
                    detail={
                        "match_mean": float(frame.loc[match, "agreement_score"].mean()),
                        "mismatch_mean": float(frame.loc[~match, "agreement_score"].mean()),
                    },
                )
            else:
                result.update(
                    passed=True,
                    method="no_argmax_contrast_in_sample",
                    detail={"all_match": bool(match.all()), "note": "Direction not contradicted."},
                )
        else:
            result.update(passed=False, method="missing_argmax_match_column", detail={})

    elif signal == "similarity":
        # A flagged near-duplicate must receive no novelty credit: memorisation is not fidelity, and
        # the whole point of keeping the two apart is that a copied image cannot win on being real.
        if "novelty_is_near_duplicate" in frame.columns and frame["novelty_is_near_duplicate"].astype(bool).any():
            flagged_novelty = frame.loc[frame["novelty_is_near_duplicate"].astype(bool), "novelty_score"].max()
            result.update(
                passed=bool(float(flagged_novelty) == 0.0),
                method="memorised_images_receive_zero_novelty",
                detail={"max_novelty_among_flagged": float(flagged_novelty)},
            )
        else:
            result.update(
                passed=True,
                method="no_near_duplicates_in_sample",
                detail={"note": "Nothing flagged; directionality not contradicted."},
            )

    elif signal == "explainability":
        # Typicality is a two-sided conformal p-value, so it must lie in [0, 1]; and a row that WAS
        # calibrated with an exact content box must carry a number. A NaN there would mean the
        # calibrated column is silently undefined for candidates the gate believes it covers.
        values = frame["explainability_calibrated_typicality"]
        defined = values.dropna()
        in_range = bool(((defined >= 0.0) & (defined <= 1.0)).all()) if len(defined) else False
        eligible = (
            frame["explainability_calibrated"].astype(bool) & frame["explainability_content_box_exact"].astype(bool)
            if {"explainability_calibrated", "explainability_content_box_exact"} <= set(frame.columns)
            else pd.Series(False, index=frame.index)
        )
        undefined_where_eligible = int(values[eligible].isna().sum())
        result.update(
            passed=bool(in_range and undefined_where_eligible == 0),
            method="typicality_is_a_valid_p_value_and_is_defined_wherever_it_was_calibrated",
            detail={
                "min": float(defined.min()) if len(defined) else None,
                "max": float(defined.max()) if len(defined) else None,
                "undefined_where_calibrated_with_an_exact_box": undefined_where_eligible,
            },
        )

    elif signal == "uncertainty":
        # No a-priori direction is asserted: high epistemic uncertainty is NOT defined as bad, and
        # nothing in the pipeline penalises a band. What IS checkable is the decomposition itself —
        # mutual information is predictive minus expected entropy, and all three are normalised by
        # log(n_classes), so each must lie in [0, 1] and the identity must hold. A violation means
        # the passes were not a valid probability sample, which is an arithmetic fault, not a taste.
        columns = {"uncertainty_predictive_entropy", "uncertainty_expected_entropy", "uncertainty_mutual_information"}
        if not columns <= set(frame.columns):
            result.update(passed=False, method="missing_decomposition_columns", detail={})
        elif "uncertainty_band" not in frame.columns:
            result.update(passed=False, method="missing_band_column", detail={})
        else:
            predictive = frame["uncertainty_predictive_entropy"].astype(float)
            expected = frame["uncertainty_expected_entropy"].astype(float)
            mutual = frame["uncertainty_mutual_information"].astype(float)
            residual = float((mutual - np.clip(predictive - expected, 0.0, None)).abs().max())
            normalised = bool(
                ((predictive >= 0.0) & (predictive <= 1.0)).all() and ((expected >= 0.0) & (expected <= 1.0)).all()
            )
            counts = frame["uncertainty_band"].value_counts().to_dict()
            result.update(
                passed=bool(residual < 1e-9 and normalised),
                method="epistemic_term_matches_the_decomposition_no_direction_asserted",
                detail={
                    "max_identity_residual": residual,
                    "entropies_normalised_to_unit_interval": normalised,
                    "band_counts": {str(key): int(value) for key, value in counts.items()},
                    "note": "Bands are reported, never penalised; only the arithmetic is gated.",
                },
            )

    return result


# ==============================================================================================
# Cross-signal checks — measured within class
# ==============================================================================================


def _within_class_groups(merged: pd.DataFrame, diagnoses: pd.Series):
    aligned = diagnoses.reindex(merged.index)
    for diagnosis, index in aligned.groupby(aligned, sort=True).groups.items():
        yield str(diagnosis), merged.loc[index]


def check_redundancy(merged: pd.DataFrame, diagnoses: pd.Series, signal: str, config) -> dict:
    """Rank correlation with each other signal, computed within class and pooled by class size.

    A signal that is a near-restatement of another adds little independent information, and weighting
    both would count the same evidence twice. Correlating across the whole pool would instead measure
    class structure: nv candidates differ systematically from mel ones on almost every signal, so two
    unrelated signals both separating nv from mel would read as redundant.

    The decision aggregates |rho| per class, not rho: a pair that is +0.98 in some classes and -0.98
    in others is redundant in every one of them, and a signed average would cancel it towards zero.
    The signed average is kept beside it for diagnosis only.
    """
    column, _ = PRIMARY_SCORE_COLUMN[signal]
    threshold = float(config.gonogo.redundancy_abs_correlation_max)
    min_pairs = int(config.gonogo.min_pairs_for_correlation)

    correlations: dict[str, float] = {}
    signed_correlations: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}
    for other, (other_column, _) in PRIMARY_SCORE_COLUMN.items():
        if other == signal or other_column not in merged.columns or column not in merged.columns:
            continue
        abs_sum, signed_sum, weight_total = 0.0, 0.0, 0
        for diagnosis, group in _within_class_groups(merged, diagnoses):
            pair = group[[column, other_column]].dropna()
            if len(pair) < min_pairs or pair[column].nunique() < 2 or pair[other_column].nunique() < 2:
                continue
            correlation = float(pair[column].corr(pair[other_column], method="spearman"))
            if not np.isfinite(correlation):
                continue
            per_class.setdefault(other, {})[diagnosis] = correlation
            abs_sum += abs(correlation) * len(pair)
            signed_sum += correlation * len(pair)
            weight_total += len(pair)
        if weight_total:
            correlations[other] = abs_sum / weight_total
            signed_correlations[other] = signed_sum / weight_total

    worst = max(correlations.items(), key=lambda item: item[1], default=(None, 0.0))
    return {
        # No comparable class survived the minimum-pairs floor: absence of evidence is not evidence
        # of redundancy, so the check does not fail on it.
        "passed": bool(worst[1] < threshold),
        "method": "class_size_weighted_within_class_abs_spearman",
        "correlations": correlations,
        "signed_correlations_diagnostic_only": signed_correlations,
        "per_class": per_class,
        "most_correlated_with": worst[0],
        "max_abs_correlation": float(abs(worst[1])),
        "threshold": threshold,
    }


def check_usefulness(merged: pd.DataFrame, diagnoses: pd.Series, signal: str, config) -> dict:
    """Does this signal change which candidates an equal-weight selector would keep?

    The comparison selector is class-conditioned — a fixed fraction of each class's candidates —
    because that is the shape the real selector has: synthetic data exists to fill the rare classes,
    and a pooled top-fraction would simply keep whichever class scores highest overall.

    A signal that changes nothing about the selection cannot influence any downstream result,
    whatever its intrinsic quality. That is a reason to keep it for the ablation, not to weight it.
    """
    available = [
        (name, column)
        for name, (column, direction) in PRIMARY_SCORE_COLUMN.items()
        if column in merged.columns and direction is True
    ]
    if signal not in [name for name, _ in available]:
        return {
            "passed": True,
            "reason": "signal_has_no_directional_score",
            "note": "Nothing is asserted about where this signal's score should be high.",
            "jaccard_overlap_of_selection": None,
        }
    if len(available) < 2:
        return {
            "passed": True,
            "reason": "insufficient_signals_to_compare",
            "jaccard_overlap_of_selection": None,
        }

    fraction = float(config.gonogo.selection_fraction)

    def selected(names: list[str]) -> set:
        columns = [column for name, column in available if name in names]
        keep: set = set()
        for _, group in _within_class_groups(merged, diagnoses):
            parts = []
            for column in columns:
                # Min-max WITHIN CLASS: the signals are on different scales, and normalising across
                # the pool would let a class's overall level, not a candidate's standing among its
                # own class, decide the composite.
                values = group[column].astype(float)
                spread = values.max() - values.min()
                parts.append((values - values.min()) / spread if spread > 0 else values * 0.0)
            composite = sum(parts) / max(len(parts), 1)
            composite = composite.fillna(composite.min() if composite.notna().any() else 0.0)
            keep |= set(composite.nlargest(max(1, int(round(len(group) * fraction)))).index)
        return keep

    all_names = [name for name, _ in available]
    with_signal = selected(all_names)
    without_signal = selected([name for name in all_names if name != signal])

    union = len(with_signal | without_signal)
    jaccard = len(with_signal & without_signal) / union if union else 1.0
    maximum = float(config.gonogo.usefulness_jaccard_max)
    return {
        "passed": bool(jaccard < maximum),
        "jaccard_overlap_of_selection": jaccard,
        "max_allowed": maximum,
        "selection_fraction_per_class": fraction,
        "n_selected": len(with_signal),
        "note": "Jaccard at the ceiling means the signal does not change the selection at all.",
    }


# ==============================================================================================
# Decision
# ==============================================================================================


def decide(checks: dict) -> tuple[str, str]:
    """Map check results onto include / ablation_only / exclude.

    Hard failures are ones that make the score unusable or inadmissible. Directionality is also hard
    but reported separately, because "the signal contradicts its own stated meaning" is a different
    finding from "the artifact is malformed" and belongs in the write-up as such. Redundancy and
    weak usefulness are soft: the signal is real, it just should not be weighted twice.
    """
    hard_failures = [name for name in HARD_CHECKS if checks.get(name, {}).get("passed") is False]
    if hard_failures:
        return OUTCOME_EXCLUDE, f"failed hard checks: {', '.join(hard_failures)}"

    if checks.get("directionality", {}).get("passed") is False:
        return OUTCOME_EXCLUDE, (
            "score direction contradicts the methodology's stated meaning "
            f"({checks['directionality'].get('method')})"
        )

    soft_failures = []
    redundancy = checks.get("redundancy", {})
    if redundancy.get("passed") is False:
        soft_failures.append(
            f"redundant with {redundancy.get('most_correlated_with')} "
            f"(|rho|={redundancy.get('max_abs_correlation', float('nan')):.3f})"
        )
    if checks.get("usefulness", {}).get("passed") is False:
        soft_failures.append("does not change the selection")

    if soft_failures:
        return OUTCOME_ABLATION, "; ".join(soft_failures)
    return OUTCOME_INCLUDE, "passed all checks"


# ==============================================================================================
# Driver
# ==============================================================================================


def load_candidate_diagnoses(stage2_root: Path, namespace: str) -> pd.Series:
    """image_id -> dx from the generation manifest, the authoritative record of what was asked for.

    Taken from the manifest rather than from any signal artifact: the agreement artifact carries an
    intended diagnosis too, but then the class structure used to judge the signals would come from
    one of the signals being judged.
    """
    path = Path(stage2_root) / namespace / "all_candidates.csv"
    if not path.is_file():
        raise UpstreamGate(
            f"UPSTREAM GATE: no candidate manifest at {path}.\n"
            "Run: python scripts/generate/ham10000_generate_synthetic_images.py "
            f"--namespace {namespace} --mode full"
        )
    frame = pd.read_csv(path)
    return pd.Series(
        [normalize_diagnosis(value) for value in frame["dx"]],
        index=frame["image_id"].astype(str),
        name="dx",
    )


def load_artifacts(scores_dir: Path) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames, absent = {}, []
    for signal, filename in ARTIFACT_FILENAME.items():
        path = scores_dir / filename
        if path.is_file():
            frames[signal] = pd.read_parquet(path)
        else:
            absent.append(signal)
    if not frames:
        raise UpstreamGate(
            f"UPSTREAM GATE: no ASISM score artifacts in {scores_dir}.\n"
            "Run: python scripts/asism/ham10000_01_compute_signals.py --namespace <ns> --signal all"
        )
    return frames, absent


def assert_one_candidate_pool(scores_dir: Path, frames: dict[str, pd.DataFrame]) -> str:
    """Every artifact must have been computed from the SAME candidate manifest.

    Redundancy and usefulness compare signals against each other; comparing two signals scored on
    different pools — a full run against a --limit smoke run, say — produces numbers that look
    ordinary and mean nothing. The manifest hash recorded by each sidecar settles it.
    """
    hashes: dict[str, str] = {}
    for signal in frames:
        sidecar = scores_dir / f"{signal}_scores.provenance.json"
        if sidecar.is_file():
            recorded = read_json(sidecar).get("candidates_csv_sha256")
            if recorded:
                hashes[signal] = str(recorded)
    distinct = sorted(set(hashes.values()))
    if len(distinct) > 1:
        raise UpstreamGate(
            "The signal artifacts were computed from different candidate manifests "
            f"({json.dumps(hashes, indent=2)}). Cross-signal comparison is meaningless across pools; "
            "recompute the stale signals against the current manifest."
        )
    return distinct[0] if distinct else ""


def merge_primary_scores(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for signal, frame in frames.items():
        column, _ = PRIMARY_SCORE_COLUMN[signal]
        subset = frame[["image_id", column]] if column in frame.columns else frame[["image_id"]]
        merged = subset if merged is None else merged.merge(subset, on="image_id", how="outer")
    return merged.set_index(merged["image_id"].astype(str)).drop(columns=["image_id"])


def evaluate(frames: dict[str, pd.DataFrame], diagnoses: pd.Series, scores_dir: Path, config) -> dict:
    merged = merge_primary_scores(frames)
    unknown = sorted(set(merged.index) - set(diagnoses.index))
    if unknown:
        raise UpstreamGate(
            f"{len(unknown)} scored image_id(s) are absent from the candidate manifest "
            f"(e.g. {unknown[:3]}); the artifacts do not describe this pool"
        )

    per_signal = {}
    for signal, frame in frames.items():
        column, _ = PRIMARY_SCORE_COLUMN[signal]
        values = (
            frame[column].to_numpy(dtype=np.float64, na_value=np.nan)
            if column in frame.columns
            else np.array([])
        )
        checks = {
            "technical_validity": check_technical_validity(frame, signal),
            "selection_eligibility": check_selection_eligibility(frame, signal),
            "numerical_stability": check_numerical_stability(values, config),
            "missing_rate": check_missing_rate(frame, signal, config),
            "provenance": check_provenance(scores_dir, signal),
            "directionality": check_directionality(frame, signal),
            "redundancy": check_redundancy(merged, diagnoses, signal, config),
            "usefulness": check_usefulness(merged, diagnoses, signal, config),
        }
        outcome, reason = decide(checks)
        per_signal[signal] = {
            "outcome": outcome,
            "reason": reason,
            "checks": checks,
            "artifact_retained_for_audit": True,
        }
    return per_signal


def build_report(per_signal: dict, absent: list[str], scores_dir: Path, namespace: str, pool_hash: str) -> dict:
    for signal in absent:
        per_signal[signal] = {
            "outcome": OUTCOME_EXCLUDE,
            "reason": "artifact not produced",
            "checks": {},
            "artifact_retained_for_audit": False,
        }

    included = sorted(name for name, entry in per_signal.items() if entry["outcome"] == OUTCOME_INCLUDE)
    ablation = sorted(name for name, entry in per_signal.items() if entry["outcome"] == OUTCOME_ABLATION)
    excluded = sorted(name for name, entry in per_signal.items() if entry["outcome"] == OUTCOME_EXCLUDE)

    return {
        "stage": "ham10000_asism_gonogo",
        "dataset": "ham10000",
        "split_namespace": namespace,
        "scores_dir": str(scores_dir),
        "candidates_csv_sha256": pool_hash,
        "git_commit_hash": get_git_commit_hash(),
        "evidence": "synthetic candidate pool and signal artifacts only; final_eval_heldout never read",
        "per_signal": per_signal,
        "surviving_signals": included,
        "ablation_only_signals": ablation,
        "excluded_signals": excluded,
        "asism_variant_status": (
            "primary"
            if len(included) >= 3
            else "reduced_variant — fewer than 3 signals survived the gate; the resulting selector is "
                 "reported as an alternative/ablation rather than as the primary method"
        ),
    }


def run(namespace: str, scores_dir: Path | None = None) -> dict:
    stage2 = load_named_config("ham10000_stage2.yaml", "ham_stage2")
    stage3 = load_named_config("ham10000_stage3.yaml", "ham_stage3")

    namespace_dir = Path(stage3.paths.outputs_dir) / namespace
    scores_dir = Path(scores_dir) if scores_dir else namespace_dir / "signals"

    frames, absent = load_artifacts(scores_dir)
    pool_hash = assert_one_candidate_pool(scores_dir, frames)
    diagnoses = load_candidate_diagnoses(Path(stage2.paths.stage2_root), namespace)

    per_signal = evaluate(frames, diagnoses, scores_dir, stage3)
    report = build_report(per_signal, absent, scores_dir, namespace, pool_hash)

    report_path = namespace_dir / "gonogo_report.json"
    write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--scores-dir", default=None)
    args = parser.parse_args()

    report = run(args.namespace, Path(args.scores_dir) if args.scores_dir else None)

    print("HAM10000 ASISM Go/No-Go", flush=True)
    print("=" * 72, flush=True)
    for signal in sorted(report["per_signal"]):
        entry = report["per_signal"][signal]
        print(f"  {signal:<16} {entry['outcome']:<14} {entry['reason']}", flush=True)
    print("=" * 72, flush=True)
    print(f"  include:       {report['surviving_signals']}", flush=True)
    print(f"  ablation only: {report['ablation_only_signals']}", flush=True)
    print(f"  exclude:       {report['excluded_signals']}", flush=True)
    print(f"  variant:       {report['asism_variant_status']}", flush=True)
    print(f"\nReport -> {report['report_path']}", flush=True)

    if not report["surviving_signals"]:
        print(
            "\nNo signal survived the gate — the selector cannot be frozen. Investigate the failing "
            "checks before fitting any ranking model or threshold.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

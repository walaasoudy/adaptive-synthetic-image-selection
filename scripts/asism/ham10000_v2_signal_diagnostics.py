#!/usr/bin/env python3
"""The v1-vs-v2 diagnostic of the three classifier-dependent signals, and the pre-registered verdict.

Applies docs/ham10000_v2_signal_criteria.md as written: Q1-Q5 on the uncertainty, agreement and
explainability artifacts of both auxiliary classifiers, then the decision rule (v2 sufficient, or
build v3) on the v2 numbers. v1 is computed the same way and reported beside it as the comparison;
the decision is about v2 only.

READ-ONLY and CPU-only. It reads a directory of already-downloaded artifacts:

    <input>/v1/{uncertainty,agreement,explainability}_scores.{parquet,provenance.json}
    <input>/v2/...                                               (same six files)
    <input>/stage2/all_candidates.csv

and writes diagnostics.json and diagnostics.md to --output-dir. Nothing under outputs/ is opened,
no configuration is read, and no threshold here may be changed after the result is seen: the
constants below are the document's, and tests/test_ham10000_v2_signal_diagnostics.py pins them.

The run stops (SystemExit) rather than reporting on inputs it cannot vouch for: a cam_model_id other
than the expected one, a parquet whose hash differs from its provenance, a candidate count or
image_id set that differs between v1, v2 and the Stage 2 pool, an intended diagnosis that disagrees
with the pool's dx, or a mutual-information column that does not reproduce its own uncertainty
bands (the unit check of Q3, Amendment 1).

Usage (after approval only):
    python scripts/asism/ham10000_v2_signal_diagnostics.py --input-dir <downloaded artifacts>
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.ham10000 import CLASSIFIER_TARGET_LABELS, normalize_diagnosis  # noqa: E402
from scripts.utils.manifest import read_json, sha256_file, write_json  # noqa: E402

CLASSES: list[str] = list(CLASSIFIER_TARGET_LABELS)
SIGNALS = ("uncertainty", "agreement", "explainability")
VERSIONS = ("v1", "v2")
EXPECTED_CANDIDATES = 3168

EXPECTED_MODEL_IDS = {
    "v1": "ham10000-classifier:a0bf59b1f448f4528f6e0f1c4b72cc114e29bacbc288f7d5975dca396c012bf0",
    "v2": "ham10000-classifier:db8f5001086c4507825f644134f91d4a62f7c51eb48945f8df769cf19f8f82e6",
}

# Real recall on classifier_val as (correct, support), recorded before the criteria were written.
# v1: snapshot_cmp/stage2_snapshot_cmp/evaluation/classifier_val_recall.json (balanced acc. 0.478).
# v2: the diagonal and row sums of the v2 acceptance confusion matrix (balanced acc. 0.600).
REAL_RECALL_COUNTS = {
    "v1": {"nv": (898, 955), "mel": (51, 149), "bkl": (74, 149), "bcc": (50, 68),
           "akiec": (15, 45), "vasc": (10, 20), "df": (0, 15)},
    "v2": {"nv": (886, 955), "mel": (69, 149), "bkl": (85, 149), "bcc": (49, 68),
           "akiec": (21, 45), "vasc": (13, 20), "df": (6, 15)},
}

# docs/ham10000_v2_signal_criteria.md, fixed 2026-09-27 before the analysis ran.
CRITERIA = {
    "q1_chance": 1.0 / 7.0,
    "q1_max_classes_at_or_below_chance": 1,
    "q1_min_spearman_with_real_recall": 0.5,
    "q2_max_mel_to_nv_share": 0.30,
    "q2_max_nv_share_of_mismatches": 0.50,
    "q3_degenerate_p90_below": 0.01,
    "q3_degenerate_iqr_below": 0.005,
    "q4_min_iqr": 0.15,
    "q4_max_tie_share": 0.20,
    "q4_min_discriminative_classes": 4,
    "q5_min_abs_spearman": 0.7,
    "q5_min_redundant_classes": 4,
}
# The uncertainty bands the artifacts were written with (configs/ham10000_stage3.yaml); only
# re-applied to check the unit, never changed.
UNCERTAINTY_LOW_BAND_MAX = 0.05
UNCERTAINTY_MODERATE_BAND_MAX = 0.20

Q5_COLUMNS = {
    "agreement": "agreement_score",
    "uncertainty": "uncertainty_mutual_information",
    "explainability": "explainability_calibrated_typicality",
}
WILSON_Z = 1.959963984540054


# ==============================================================================================
# Small statistics
# ==============================================================================================


def wilson_interval(successes: int, total: int, z: float = WILSON_Z) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; (nan, nan) for an empty denominator."""
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def spearman(x, y) -> float:
    """Rank correlation (average ranks for ties), NaN pairs dropped; NaN if either side is constant
    or fewer than 3 pairs remain."""
    frame = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    if len(frame) < 3:
        return float("nan")
    ranks = frame.rank(method="average")
    if ranks["x"].nunique() < 2 or ranks["y"].nunique() < 2:
        return float("nan")
    return float(np.corrcoef(ranks["x"], ranks["y"])[0, 1])


def proportion(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {"k": int(successes), "n": int(total),
            "rate": float(successes / total) if total else float("nan"),
            "ci95": [low, high]}


def iqr(values) -> float:
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(0.75) - values.quantile(0.25))


# ==============================================================================================
# Loading, with the refusals
# ==============================================================================================


def load_version(directory: Path, version: str, expected_n: int = EXPECTED_CANDIDATES) -> pd.DataFrame:
    """The three artifacts of one classifier, joined on image_id, after the provenance checks."""
    frames = []
    for signal in SIGNALS:
        parquet = directory / f"{signal}_scores.parquet"
        provenance = read_json(directory / f"{signal}_scores.provenance.json")
        if provenance.get("cam_model_id") != EXPECTED_MODEL_IDS[version]:
            raise SystemExit(
                f"{version} {signal}: cam_model_id {provenance.get('cam_model_id')} is not the "
                f"expected {EXPECTED_MODEL_IDS[version]}. Refusing to report on another model."
            )
        recorded = provenance.get("parquet_sha256")
        if recorded and recorded != sha256_file(parquet):
            raise SystemExit(f"{version} {signal}: {parquet} does not match its provenance hash.")
        frame = pd.read_parquet(parquet)
        if len(frame) != expected_n or frame["image_id"].duplicated().any():
            raise SystemExit(
                f"{version} {signal}: {len(frame)} rows (expected {expected_n}, unique image_id)."
            )
        frame["image_id"] = frame["image_id"].astype(str)
        frames.append(frame.set_index("image_id"))
    joined = frames[0].join(frames[1:], how="inner")
    if len(joined) != expected_n:
        raise SystemExit(f"{version}: the three artifacts do not share the same image_id set.")
    return joined.reset_index()


def attach_class(frame: pd.DataFrame, candidates: pd.DataFrame, version: str) -> pd.DataFrame:
    """The class of every row is the pool's intended dx; the agreement artifact must say the same."""
    dx = dict(zip(candidates["image_id"].astype(str), candidates["dx"].map(normalize_diagnosis)))
    if set(frame["image_id"]) != set(dx):
        raise SystemExit(f"{version}: image_id set differs from all_candidates.csv.")
    frame = frame.copy()
    frame["dx"] = frame["image_id"].map(dx)
    if (frame["dx"] != frame["agreement_intended_diagnosis"]).any():
        raise SystemExit(f"{version}: agreement_intended_diagnosis disagrees with the pool's dx.")
    return frame


def check_uncertainty_unit(frame: pd.DataFrame, version: str) -> None:
    """Q3's unit check: the column as read must reproduce the band it was written with."""
    mutual = frame["uncertainty_mutual_information"].astype(float)
    bands = np.where(
        mutual <= UNCERTAINTY_LOW_BAND_MAX,
        "low",
        np.where(mutual <= UNCERTAINTY_MODERATE_BAND_MAX, "moderate", "extreme"),
    )
    mismatched = int((bands != frame["uncertainty_band"].astype(str).to_numpy()).sum())
    if mismatched:
        raise SystemExit(
            f"{version}: uncertainty_mutual_information does not reproduce uncertainty_band for "
            f"{mismatched} rows. The unit is not what the criteria assume; stopping."
        )


def load_inputs(input_dir: Path, expected_n: int = EXPECTED_CANDIDATES) -> dict[str, pd.DataFrame]:
    candidates = pd.read_csv(input_dir / "stage2" / "all_candidates.csv")
    loaded = {}
    for version in VERSIONS:
        frame = attach_class(load_version(input_dir / version, version, expected_n), candidates, version)
        check_uncertainty_unit(frame, version)
        loaded[version] = frame
    if set(loaded["v1"]["image_id"]) != set(loaded["v2"]["image_id"]):
        raise SystemExit("v1 and v2 were not computed on the same candidates.")
    return loaded


# ==============================================================================================
# Q1-Q5
# ==============================================================================================


def q1_agreement(frame: pd.DataFrame, real_recall: dict) -> dict:
    per_class, match_rates, recalls = {}, [], []
    for name in CLASSES:
        rows = frame[frame["dx"] == name]
        k = int(rows["agreement_is_argmax_match"].astype(bool).sum())
        correct, support = real_recall[name]
        recall = correct / support
        entry = proportion(k, len(rows))
        entry["real_recall"] = recall
        entry["normalised_match_rate"] = entry["rate"] / recall if recall > 0 else float("nan")
        entry["at_or_below_chance"] = bool(entry["rate"] <= CRITERIA["q1_chance"])
        per_class[name] = entry
        match_rates.append(entry["rate"])
        recalls.append(recall)
    pooled = proportion(int(frame["agreement_is_argmax_match"].astype(bool).sum()), len(frame))
    n_at_chance = sum(entry["at_or_below_chance"] for entry in per_class.values())
    rho = spearman(match_rates, recalls)
    sensible = bool(
        n_at_chance <= CRITERIA["q1_max_classes_at_or_below_chance"]
        and not np.isnan(rho)
        and rho >= CRITERIA["q1_min_spearman_with_real_recall"]
    )
    return {"pooled": pooled, "per_class": per_class, "n_classes_at_or_below_chance": n_at_chance,
            "spearman_match_vs_real_recall": rho, "sensible": sensible}


def q2_mel_to_nv(frame: pd.DataFrame) -> dict:
    mel = frame[frame["dx"] == "mel"]
    mel_to_nv = proportion(int((mel["agreement_predicted_diagnosis"] == "nv").sum()), len(mel))
    mismatched = frame[~frame["agreement_is_argmax_match"].astype(bool)]
    nv_sink = proportion(int((mismatched["agreement_predicted_diagnosis"] == "nv").sum()), len(mismatched))
    predicted_as = mismatched["agreement_predicted_diagnosis"].value_counts().reindex(CLASSES, fill_value=0)
    influential = bool(
        mel_to_nv["rate"] >= CRITERIA["q2_max_mel_to_nv_share"]
        or (nv_sink["n"] > 0 and nv_sink["rate"] >= CRITERIA["q2_max_nv_share_of_mismatches"])
    )
    return {"mel_predicted_nv": mel_to_nv, "nv_share_of_mismatches": nv_sink,
            "mismatches_predicted_as": {k: int(v) for k, v in predicted_as.items()},
            "influential": influential}


def _distribution(values) -> dict:
    values = pd.Series(values, dtype=float)
    return {"median": float(values.median()), "p90": float(values.quantile(0.90)),
            "p99": float(values.quantile(0.99)), "iqr": iqr(values), "max": float(values.max())}


def q3_uncertainty(frame: pd.DataFrame) -> dict:
    mutual = _distribution(frame["uncertainty_mutual_information"])
    per_class = {name: _distribution(frame.loc[frame["dx"] == name, "uncertainty_mutual_information"])
                 for name in CLASSES}
    degenerate = bool(
        mutual["p90"] < CRITERIA["q3_degenerate_p90_below"]
        and mutual["iqr"] < CRITERIA["q3_degenerate_iqr_below"]
    )
    return {"normalised_mutual_information": mutual,
            "mean_std": _distribution(frame["uncertainty_mean_std"]),
            "bands": frame["uncertainty_band"].astype(str).value_counts().to_dict(),
            "per_class": per_class, "degenerate": degenerate}


def tie_share(values) -> float:
    """Fraction of rows whose value is shared, exactly, with at least one other row."""
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return float("nan")
    return float(values.duplicated(keep=False).mean())


def q4_explainability(frame: pd.DataFrame) -> dict:
    per_class = {}
    for name in CLASSES:
        rows = frame[frame["dx"] == name]
        spread = iqr(rows["explainability_calibrated_typicality"])
        ties = tie_share(rows["explainability_peripheral_mass"])
        per_class[name] = {
            "n": int(len(rows)),
            "iqr_calibrated_typicality": spread,
            "tie_share_peripheral_mass": ties,
            "discriminative": bool(
                not np.isnan(spread) and spread >= CRITERIA["q4_min_iqr"]
                and not np.isnan(ties) and ties < CRITERIA["q4_max_tie_share"]
            ),
            "fragile": name == "df",
        }
    count = sum(entry["discriminative"] for entry in per_class.values())
    return {"per_class": per_class, "n_discriminative_classes": count}


def q5_redundancy(frame: pd.DataFrame) -> dict:
    pairs = {}
    for (name_a, column_a), (name_b, column_b) in combinations(Q5_COLUMNS.items(), 2):
        pooled = spearman(frame[column_a], frame[column_b])
        within = {name: spearman(frame.loc[frame["dx"] == name, column_a],
                                 frame.loc[frame["dx"] == name, column_b]) for name in CLASSES}
        n_strong = sum(1 for rho in within.values()
                       if not np.isnan(rho) and abs(rho) >= CRITERIA["q5_min_abs_spearman"])
        pairs[f"{name_a}~{name_b}"] = {
            "pooled_spearman": pooled,
            "within_class_spearman": within,
            "n_classes_abs_rho_at_least_threshold": n_strong,
            "redundant": bool(not np.isnan(pooled) and abs(pooled) >= CRITERIA["q5_min_abs_spearman"]
                              and n_strong >= CRITERIA["q5_min_redundant_classes"]),
        }
    return {"pairs": pairs, "any_redundant": any(pair["redundant"] for pair in pairs.values())}


# ==============================================================================================
# The decision rule
# ==============================================================================================


def decide(q1: dict, q2: dict, q3: dict, q4: dict, q5: dict) -> dict:
    """The decision table of the criteria document, applied to one classifier's results."""
    notes = []
    if not q1["sensible"] or q2["influential"]:
        verdict = "build_v3"
        reason = "Q1 not sensible" if not q1["sensible"] else "Q2 influential"
        if not q1["sensible"] and q2["influential"]:
            reason = "Q1 not sensible and Q2 influential"
    elif q4["n_discriminative_classes"] >= CRITERIA["q4_min_discriminative_classes"]:
        verdict, reason = "v2_sufficient", "Q1 sensible, Q2 not influential, Q4 discriminative in >= 4 classes"
    else:
        verdict = "v2_sufficient"
        reason = "Q1 sensible, Q2 not influential; Q4 discriminative in < 4 classes"
        notes.append("explainability recorded as a weak signal; its fate is decided in the ranking")
    if q3["degenerate"]:
        notes.append("Q3 degenerate: not on its own a reason for v3; replacing MC Dropout is a "
                     "separate design decision for Walaa, fixed in writing before it is tried")
    if q5["any_redundant"]:
        notes.append("Q5 redundancy recorded as an input to the later ranking; does not change v3")
    return {"verdict": verdict, "reason": reason, "notes": notes}


def analyse_version(frame: pd.DataFrame, version: str) -> dict:
    q1 = q1_agreement(frame, REAL_RECALL_COUNTS[version])
    q2 = q2_mel_to_nv(frame)
    q3 = q3_uncertainty(frame)
    q4 = q4_explainability(frame)
    q5 = q5_redundancy(frame)
    return {"cam_model_id": EXPECTED_MODEL_IDS[version], "q1": q1, "q2": q2, "q3": q3, "q4": q4,
            "q5": q5, "rule_applied": decide(q1, q2, q3, q4, q5)}


def analyse(input_dir: Path, expected_n: int = EXPECTED_CANDIDATES) -> dict:
    loaded = load_inputs(input_dir, expected_n)
    results = {version: analyse_version(loaded[version], version) for version in VERSIONS}
    return {
        "criteria_document": "docs/ham10000_v2_signal_criteria.md",
        "criteria": CRITERIA,
        "n_candidates": expected_n,
        "input_dir": str(input_dir),
        "input_sha256": {f"{version}/{signal}": sha256_file(input_dir / version / f"{signal}_scores.parquet")
                         for version in VERSIONS for signal in SIGNALS},
        "results": results,
        # The decision is about v2; v1 goes through the same rule only as a reference point.
        "decision": results["v2"]["rule_applied"],
    }


# ==============================================================================================
# Report
# ==============================================================================================


def _pct(value: float) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value:.{digits}f}"


def _ci(entry: dict) -> str:
    low, high = entry["ci95"]
    return f"{_pct(entry['rate'])} [{_pct(low)}, {_pct(high)}] ({entry['k']}/{entry['n']})"


def render_markdown(report: dict) -> str:
    r1, r2 = report["results"]["v1"], report["results"]["v2"]
    lines = ["# HAM10000 v2 signal diagnostics", "",
             f"Criteria: `{report['criteria_document']}` (applied as written). "
             f"{report['n_candidates']} candidates.", "",
             f"**Decision (v2): {report['decision']['verdict']}**, {report['decision']['reason']}."]
    lines += [f"- {note}" for note in report["decision"]["notes"]]
    lines += ["", f"(v1 under the same rule, for reference only: {r1['rule_applied']['verdict']}, "
              f"{r1['rule_applied']['reason']}.)", ""]

    lines += ["## Q1: agreement", "", "| Class | v1 match | v2 match | v1 real recall | v2 real recall | "
              "v1 normalised | v2 normalised |", "|---|---|---|---|---|---|---|"]
    for name in CLASSES:
        a, b = r1["q1"]["per_class"][name], r2["q1"]["per_class"][name]
        lines.append(f"| {name} | {_ci(a)} | {_ci(b)} | {_pct(a['real_recall'])} | {_pct(b['real_recall'])} | "
                     f"{_num(a['normalised_match_rate'], 2)} | {_num(b['normalised_match_rate'], 2)} |")
    for label, r in (("v1", r1), ("v2", r2)):
        q = r["q1"]
        lines.append(f"- {label}: pooled {_ci(q['pooled'])}; classes at or below chance "
                     f"{q['n_classes_at_or_below_chance']}; Spearman(match, real recall) "
                     f"{_num(q['spearman_match_vs_real_recall'])}; **sensible = {q['sensible']}**")

    lines += ["", "## Q2: mel to nv", ""]
    for label, r in (("v1", r1), ("v2", r2)):
        q = r["q2"]
        lines.append(f"- {label}: mel predicted nv {_ci(q['mel_predicted_nv'])}; nv share of all "
                     f"mismatches {_ci(q['nv_share_of_mismatches'])}; mismatches predicted as "
                     f"{q['mismatches_predicted_as']}; **influential = {q['influential']}**")

    lines += ["", "## Q3: uncertainty (normalised mutual information)", ""]
    for label, r in (("v1", r1), ("v2", r2)):
        d = r["q3"]["normalised_mutual_information"]
        lines.append(f"- {label}: median {_num(d['median'], 5)}, P90 {_num(d['p90'], 5)}, P99 "
                     f"{_num(d['p99'], 5)}, IQR {_num(d['iqr'], 5)}, max {_num(d['max'], 5)}; bands "
                     f"{r['q3']['bands']}; **degenerate = {r['q3']['degenerate']}**")

    lines += ["", "## Q4: explainability", "", "| Class | v1 IQR | v1 ties | v2 IQR | v2 ties | "
              "v1 disc. | v2 disc. |", "|---|---|---|---|---|---|---|"]
    for name in CLASSES:
        a, b = r1["q4"]["per_class"][name], r2["q4"]["per_class"][name]
        flag = " (fragile)" if b["fragile"] else ""
        lines.append(f"| {name}{flag} | {_num(a['iqr_calibrated_typicality'])} | "
                     f"{_pct(a['tie_share_peripheral_mass'])} | {_num(b['iqr_calibrated_typicality'])} | "
                     f"{_pct(b['tie_share_peripheral_mass'])} | {a['discriminative']} | {b['discriminative']} |")
    lines.append(f"- discriminative classes: v1 {r1['q4']['n_discriminative_classes']}, "
                 f"v2 {r2['q4']['n_discriminative_classes']}")

    lines += ["", "## Q5: redundancy (Spearman)", "", "| Pair | v1 pooled | v1 classes | "
              "v2 pooled | v2 classes | v2 redundant |", "|---|---|---|---|---|---|"]
    for pair in r2["q5"]["pairs"]:
        a, b = r1["q5"]["pairs"][pair], r2["q5"]["pairs"][pair]
        lines.append(f"| {pair} | {_num(a['pooled_spearman'])} | {a['n_classes_abs_rho_at_least_threshold']} | "
                     f"{_num(b['pooled_spearman'])} | {b['n_classes_abs_rho_at_least_threshold']} | {b['redundant']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <input-dir>/analysis")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis"
    report = analyse(args.input_dir)
    write_json(output_dir / "diagnostics.json", report)
    (output_dir / "diagnostics.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

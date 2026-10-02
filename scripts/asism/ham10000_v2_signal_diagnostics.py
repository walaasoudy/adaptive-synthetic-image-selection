#!/usr/bin/env python3
"""The diagnostic of the three classifier-dependent signals, and the pre-registered verdict.

Applies docs/ham10000_v2_signal_criteria.md as written: Q1-Q5 on the uncertainty, agreement and
explainability artifacts of each auxiliary classifier, then the decision rule on the last version
given. With the default `--versions v1,v2` this is exactly the 2026-09-27 analysis (v2 sufficient,
or build v3), and its report is what it was. With `--versions v1,v2,v3` the rule is applied to v3
as Amendment 2 fixes it (v3 sufficient, v3b, or the next change chosen from the evidence); v1 and
v2 are reported beside it as the comparison only.

READ-ONLY and CPU-only. It reads a directory of already-downloaded artifacts:

    <input>/v1/{uncertainty,agreement,explainability}_scores.{parquet,provenance.json}
    <input>/v2/...                                               (same six files)
    <input>/v3/...                                               (only with --versions v1,v2,v3)
    <input>/stage2/all_candidates.csv

and writes diagnostics.json and diagnostics.md to --output-dir. Nothing under outputs/ is opened,
no configuration is read, and no threshold here may be changed after the result is seen: the
constants below are the document's, and tests/test_ham10000_v2_signal_diagnostics.py pins them.

The run stops (SystemExit) rather than reporting on inputs it cannot vouch for: a cam_model_id other
than the expected one, a parquet whose hash differs from its provenance, a candidate count or
image_id set that differs between the versions and the Stage 2 pool, an intended diagnosis that disagrees
with the pool's dx, or a mutual-information column that does not reproduce its own uncertainty
bands (the unit check of Q3, Amendment 1).

Usage (after approval only):
    python scripts/asism/ham10000_v2_signal_diagnostics.py --input-dir <downloaded artifacts>
    python scripts/asism/ham10000_v2_signal_diagnostics.py --input-dir <...> --versions v1,v2,v3
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
VERSIONS = ("v1", "v2")  # the 2026-09-27 analysis, and still the default
EXPECTED_CANDIDATES = 3168

EXPECTED_MODEL_IDS = {
    "v1": "ham10000-classifier:a0bf59b1f448f4528f6e0f1c4b72cc114e29bacbc288f7d5975dca396c012bf0",
    "v2": "ham10000-classifier:db8f5001086c4507825f644134f91d4a62f7c51eb48945f8df769cf19f8f82e6",
    # Amendment 2: the v3a model Walaa froze on 2026-10-02, before any v3 signal existed.
    "v3": "ham10000-classifier:6849c456a9581c598a423b93b9c2cdbdbb79118fd6d31c72fe68eefcbe5f8bfd",
}

# Real recall on classifier_val as (correct, support), recorded before the criteria were written.
# v1: snapshot_cmp/stage2_snapshot_cmp/evaluation/classifier_val_recall.json (balanced acc. 0.478).
# v2: the diagonal and row sums of the v2 acceptance confusion matrix (balanced acc. 0.600).
REAL_RECALL_COUNTS = {
    "v1": {"nv": (898, 955), "mel": (51, 149), "bkl": (74, 149), "bcc": (50, 68),
           "akiec": (15, 45), "vasc": (10, 20), "df": (0, 15)},
    "v2": {"nv": (886, 955), "mel": (69, 149), "bkl": (85, 149), "bcc": (49, 68),
           "akiec": (21, 45), "vasc": (13, 20), "df": (6, 15)},
    # v3: the diagonal and row sums of the v3a acceptance confusion matrix (balanced acc. 0.582),
    # recorded in Amendment 2 before any v3 signal existed.
    "v3": {"nv": (786, 955), "mel": (92, 149), "bkl": (98, 149), "bcc": (50, 68),
           "akiec": (19, 45), "vasc": (11, 20), "df": (4, 15)},
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


def check_versions(versions) -> tuple[str, ...]:
    """v1 and v2 always, in that order, optionally followed by v3. The last one is decided on."""
    versions = tuple(versions)
    if versions not in (("v1", "v2"), ("v1", "v2", "v3")):
        raise SystemExit(f"versions must be v1,v2 or v1,v2,v3; got {','.join(versions)}.")
    return versions


def load_inputs(input_dir: Path, expected_n: int = EXPECTED_CANDIDATES,
                versions=VERSIONS) -> dict[str, pd.DataFrame]:
    versions = check_versions(versions)
    candidates = pd.read_csv(input_dir / "stage2" / "all_candidates.csv")
    loaded = {}
    for version in versions:
        frame = attach_class(load_version(input_dir / version, version, expected_n), candidates, version)
        check_uncertainty_unit(frame, version)
        loaded[version] = frame
    if set(loaded["v1"]["image_id"]) != set(loaded["v2"]["image_id"]):
        raise SystemExit("v1 and v2 were not computed on the same candidates.")
    if "v3" in loaded and set(loaded["v3"]["image_id"]) != set(loaded["v1"]["image_id"]):
        raise SystemExit("v3 was not computed on the same candidates as v1 and v2.")
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


def decide_v3(q1: dict, q2: dict, q3: dict, q4: dict, q5: dict) -> dict:
    """Amendment 2's table: the same questions and thresholds, with v3's own outcomes."""
    notes = []
    if not q1["sensible"] or q2["influential"]:
        verdict = "next_change_from_evidence"
        reason = "Q1 not sensible" if not q1["sensible"] else "Q2 influential"
        if not q1["sensible"] and q2["influential"]:
            reason = "Q1 not sensible and Q2 influential"
        notes.append("training length was not the cause; the next change is chosen from the evidence "
                     "and written in the criteria document before it is tried")
    elif q3["degenerate"]:
        verdict, reason = "v3b", "Q1 sensible and Q2 not influential, but Q3 degenerate"
    elif q4["n_discriminative_classes"] >= CRITERIA["q4_min_discriminative_classes"]:
        verdict, reason = "v3_sufficient", "Q1 sensible, Q2 not influential, Q4 discriminative in >= 4 classes"
    else:
        verdict = "v3_sufficient"
        reason = "Q1 sensible, Q2 not influential; Q4 discriminative in < 4 classes"
        notes.append("explainability recorded as a weak signal; its fate is decided in the ranking")
    if q5["any_redundant"]:
        notes.append("Q5 redundancy recorded as an input to the later ranking")
    return {"verdict": verdict, "reason": reason, "notes": notes}


def mel_reading_aids(frame: pd.DataFrame, q2: dict) -> dict:
    """Amendment 2's caution, as numbers: is a mel match recognition of mel, or a shift towards
    predicting mel? Reported beside the verdict; not a criterion, and it changes no outcome."""
    nv = frame[frame["dx"] == "nv"]
    mismatches = q2["mismatches_predicted_as"]
    return {
        "synthetic_nv_predicted_mel": proportion(int((nv["agreement_predicted_diagnosis"] == "mel").sum()), len(nv)),
        "mel_share_of_mismatches": proportion(int(mismatches.get("mel", 0)), sum(mismatches.values())),
    }


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
    rule = decide_v3 if version == "v3" else decide
    result = {"cam_model_id": EXPECTED_MODEL_IDS[version], "q1": q1, "q2": q2, "q3": q3, "q4": q4,
              "q5": q5, "rule_applied": rule(q1, q2, q3, q4, q5)}
    if version == "v3":
        result["mel_reading_aids"] = mel_reading_aids(frame, q2)
    return result


def analyse(input_dir: Path, expected_n: int = EXPECTED_CANDIDATES, versions=VERSIONS) -> dict:
    versions = check_versions(versions)
    loaded = load_inputs(input_dir, expected_n, versions)
    results = {version: analyse_version(loaded[version], version) for version in versions}
    report = {
        "criteria_document": "docs/ham10000_v2_signal_criteria.md",
        "criteria": CRITERIA,
        "n_candidates": expected_n,
        "input_dir": str(input_dir),
        "input_sha256": {f"{version}/{signal}": sha256_file(input_dir / version / f"{signal}_scores.parquet")
                         for version in versions for signal in SIGNALS},
        "results": results,
        # The decision is about the last version; the earlier ones go through their own rule only
        # as reference points.
        "decision": results[versions[-1]]["rule_applied"],
    }
    if versions[-1] != "v2":
        report["decided_on"] = versions[-1]
    return report


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
    results = report["results"]
    versions = list(results)
    decided = versions[-1]
    final = results[decided]
    lines = [f"# HAM10000 {decided} signal diagnostics", "",
             f"Criteria: `{report['criteria_document']}` (applied as written). "
             f"{report['n_candidates']} candidates.", "",
             f"**Decision ({decided}): {report['decision']['verdict']}**, {report['decision']['reason']}."]
    lines += [f"- {note}" for note in report["decision"]["notes"]]
    lines.append("")
    for version in versions[:-1]:
        rule = results[version]["rule_applied"]
        lines.append(f"({version} under the same rule, for reference only: {rule['verdict']}, "
                     f"{rule['reason']}.)")
    lines.append("")

    def table_head(cells):
        return ["| " + " | ".join(cells) + " |", "|" + "---|" * len(cells)]

    q1_cells = ([f"{v} match" for v in versions] + [f"{v} real recall" for v in versions]
                + [f"{v} normalised" for v in versions])
    lines += ["## Q1: agreement", "", *table_head(["Class", *q1_cells])]
    for name in CLASSES:
        entries = [results[v]["q1"]["per_class"][name] for v in versions]
        cells = ([_ci(e) for e in entries] + [_pct(e["real_recall"]) for e in entries]
                 + [_num(e["normalised_match_rate"], 2) for e in entries])
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for version in versions:
        q = results[version]["q1"]
        lines.append(f"- {version}: pooled {_ci(q['pooled'])}; classes at or below chance "
                     f"{q['n_classes_at_or_below_chance']}; Spearman(match, real recall) "
                     f"{_num(q['spearman_match_vs_real_recall'])}; **sensible = {q['sensible']}**")

    lines += ["", "## Q2: mel to nv", ""]
    for version in versions:
        q = results[version]["q2"]
        lines.append(f"- {version}: mel predicted nv {_ci(q['mel_predicted_nv'])}; nv share of all "
                     f"mismatches {_ci(q['nv_share_of_mismatches'])}; mismatches predicted as "
                     f"{q['mismatches_predicted_as']}; **influential = {q['influential']}**")
    if "mel_reading_aids" in final:
        aids = final["mel_reading_aids"]
        lines += ["", f"Reading aids for {decided} (Amendment 2, not a criterion): synthetic nv predicted "
                  f"mel {_ci(aids['synthetic_nv_predicted_mel'])}; mel share of all mismatches "
                  f"{_ci(aids['mel_share_of_mismatches'])}."]

    lines += ["", "## Q3: uncertainty (normalised mutual information)", ""]
    for version in versions:
        r = results[version]
        d = r["q3"]["normalised_mutual_information"]
        lines.append(f"- {version}: median {_num(d['median'], 5)}, P90 {_num(d['p90'], 5)}, P99 "
                     f"{_num(d['p99'], 5)}, IQR {_num(d['iqr'], 5)}, max {_num(d['max'], 5)}; bands "
                     f"{r['q3']['bands']}; **degenerate = {r['q3']['degenerate']}**")

    q4_cells = [f"{v} {label}" for v in versions for label in ("IQR", "ties")] + [f"{v} disc." for v in versions]
    lines += ["", "## Q4: explainability", "", *table_head(["Class", *q4_cells])]
    for name in CLASSES:
        entries = [results[v]["q4"]["per_class"][name] for v in versions]
        flag = " (fragile)" if entries[-1]["fragile"] else ""
        cells = []
        for e in entries:
            cells += [_num(e["iqr_calibrated_typicality"]), _pct(e["tie_share_peripheral_mass"])]
        cells += [str(e["discriminative"]) for e in entries]
        lines.append(f"| {name}{flag} | " + " | ".join(cells) + " |")
    lines.append("- discriminative classes: " + ", ".join(
        f"{v} {results[v]['q4']['n_discriminative_classes']}" for v in versions))

    q5_cells = [f"{v} {label}" for v in versions for label in ("pooled", "classes")] + [f"{decided} redundant"]
    lines += ["", "## Q5: redundancy (Spearman)", "", *table_head(["Pair", *q5_cells])]
    for pair in final["q5"]["pairs"]:
        entries = [results[v]["q5"]["pairs"][pair] for v in versions]
        cells = []
        for e in entries:
            cells += [_num(e["pooled_spearman"]), str(e["n_classes_abs_rho_at_least_threshold"])]
        cells.append(str(entries[-1]["redundant"]))
        lines.append(f"| {pair} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <input-dir>/analysis")
    parser.add_argument("--versions", default=",".join(VERSIONS),
                        help="v1,v2 (the 2026-09-27 analysis, the default) or v1,v2,v3 (Amendment 2). "
                             "The rule is applied to the last one.")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "analysis"
    report = analyse(args.input_dir, versions=args.versions.split(","))
    write_json(output_dir / "diagnostics.json", report)
    (output_dir / "diagnostics.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

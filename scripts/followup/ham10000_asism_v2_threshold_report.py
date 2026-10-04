#!/usr/bin/env python3
"""ASISM v2: where the stopping rule cut each class, as one table. CPU, seconds. A report only.

Runs after `ham10000_asism_v2_learned_select --phase select`. It reads the selection manifest and
C's trajectory, which the selector wrote, and states per class the cut the stopping rule arrived at:
how many candidates the class had, how many were selected, and the ranking score and the lower 95%
bound of the last image accepted. Nothing here selects, counts or changes anything: the selection
files are read and left as they are, and the report is written once beside them.

The rule offers a class's images in the order of their own lower bound and stops the class when the
lower 95% bound of the marginal utility is no longer above 0 (contract §10), so the bound at the cut
is close to 0 in every class by construction. What differs between classes is the count and the
ranking score at the cut.

Usage:
    python -m scripts.followup.ham10000_asism_v2_threshold_report \
        --candidates <stage2/all_candidates.csv> --scores-dir <four_signal/gonogo_signal_set>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.asism_v2.contracts import write_new_json
from scripts.asism_v2.prereg import load_prereg
from scripts.asism_v2.preserve import sha256
from scripts.asism_v2.selection_files import MANIFEST_NAME, TRAJECTORY_NAME
from scripts.asism_v2.supervision import load_signal_table
from scripts.followup import ham10000_asism_v2_utility as utility

REPORT_JSON, REPORT_CSV = "adaptive_thresholds.json", "adaptive_thresholds.csv"
COLUMNS = ["dx", "candidates", "selected", "fraction_selected", "score_threshold", "min_score_selected",
           "lower_bound_at_threshold", "last_accepted_image_id", "stop_reason"]


class ThresholdReportError(RuntimeError):
    pass


def build_report(out_dir: Path, class_sizes: dict[str, int]) -> dict:
    """`class_sizes` is the safe pool's size per class. Reads the selection in `out_dir`; writes nothing."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ThresholdReportError(f"no selection manifest at {manifest_path}; run the select phase first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["selection_outcome"] != "subset":
        raise ThresholdReportError(f"the selection outcome is {manifest['selection_outcome']!r}: there is no "
                                   "trajectory and so no cut to report")
    trajectory_path = out_dir / TRAJECTORY_NAME
    trajectory_sha = sha256(trajectory_path)
    if trajectory_sha != manifest["files_sha256"][TRAJECTORY_NAME]:
        raise ThresholdReportError(f"{trajectory_path} is not the file the selection manifest recorded")
    if sum(class_sizes.values()) != manifest["safe_pool"] or set(class_sizes) != set(manifest["per_class"]):
        raise ThresholdReportError("the class sizes given are not those of the selection's safe pool")
    trajectory = pd.read_csv(trajectory_path)
    rows = []
    for name in sorted(class_sizes):
        accepted = trajectory[trajectory["dx"] == name].sort_values("rank_in_class")
        if len(accepted) != manifest["per_class"][name]:
            raise ThresholdReportError(f"{name}: the trajectory has {len(accepted)} images, the manifest "
                                       f"{manifest['per_class'][name]}")
        row = {"dx": name, "candidates": int(class_sizes[name]), "selected": int(len(accepted)),
               "fraction_selected": len(accepted) / class_sizes[name] if class_sizes[name] else None,
               "score_threshold": None, "min_score_selected": None, "lower_bound_at_threshold": None,
               "last_accepted_image_id": None, "stop_reason": manifest["stops"][name]["reason"]}
        if len(accepted):
            last = accepted.iloc[-1]
            row.update(score_threshold=float(last["ranking_score"]),
                       min_score_selected=float(accepted["ranking_score"].min()),
                       lower_bound_at_threshold=float(last["own_lower_bound"]),
                       last_accepted_image_id=str(last["image_id"]))
        rows.append(row)
    return {
        "stage": "ham10000_asism_v2_threshold_report", "namespace": manifest["namespace"],
        "what_this_is": "a report read from the selection's own trajectory; it selects nothing and "
                        "changes no selection file",
        "score_threshold": "ranking score of the last image the stopping rule accepted in the class",
        "lower_bound_at_threshold": "that image's own lower 95% bound, the quantity the rule orders by",
        "rule": manifest["rule"], "per_class": rows, "n_selected_c": manifest["n_selected_c"],
        "selection_manifest_sha256": sha256(manifest_path), "trajectory_sha256": trajectory_sha,
        "c_ids_sha256": manifest["c_ids_sha256"], "safe_pool_ids_sha256": manifest.get("safe_pool_ids_sha256"),
        "final_eval_heldout_read": False, "classifier_val_read": False,
    }


def write_report(out_dir: Path, report: dict) -> None:
    out_dir = Path(out_dir)
    for name in (REPORT_JSON, REPORT_CSV):
        if (out_dir / name).exists():
            raise ThresholdReportError(f"{out_dir / name} exists: the report is written once")
    with (out_dir / REPORT_CSV).open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(report["per_class"])[COLUMNS].to_csv(stream, index=False)
    write_new_json(out_dir / REPORT_JSON, {**report, "csv_sha256": sha256(out_dir / REPORT_CSV)})


def run(namespace, candidates, scores_dir, out_dir=None) -> dict:
    prereg = load_prereg()
    out_dir = Path(out_dir or utility.default_out_dir(prereg, namespace))
    frame, checked = load_signal_table(candidates, scores_dir, prereg["safety"])
    manifest_path = out_dir / MANIFEST_NAME
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("safe_pool_ids_sha256")
        if recorded != checked["safe_pool_ids_sha256"]:
            raise ThresholdReportError("the candidates and scores given are not the pool the selection was made on")
    report = build_report(out_dir, {str(k): int(v) for k, v in frame["dx"].value_counts().items()})
    write_report(out_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default="ham-stratified-v1")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run(args.namespace, args.candidates, args.scores_dir, args.out_dir)
    print(pd.DataFrame(report["per_class"])[COLUMNS].to_string(index=False))
    print(f"(report: {REPORT_JSON}, {REPORT_CSV})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

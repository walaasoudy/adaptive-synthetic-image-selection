#!/usr/bin/env python3
"""Build and freeze the six-way LESION-level split for HAM10000, then audit it for leakage.

The audit is not a formality appended to the end: it is the reason this script exists as a separate
step. HAM10000 photographs some lesions more than once, so an image-level split would put two
photographs of ONE lesion on both sides of a boundary and inflate every downstream number. Grouping
by lesion_id removes that. The audit then PROVES it removed it, on the actual written files, and
exits non-zero if it did not.

WHAT THE AUDIT DOES AND DOES NOT COVER
  Covered   two images of the same lesion_id landing in different splits (structural, provable)
  Covered   the same image_id appearing in two splits (should be impossible; asserted anyway)
  NOT covered
            two visually near-identical images filed under DIFFERENT lesion_ids. HAM10000 is known
            to contain some of these. They cannot be detected from metadata at all — only by
            comparing pixels — so they are out of scope here by construction, and a separate
            image-similarity audit is the right place for them. This script never claims to have
            ruled them out; `leakage_audit.json` records that limitation explicitly so a reader of
            the thesis is not left with a false guarantee.

STRATIFICATION
  Lesions are partitioned separately within each diagnosis (every lesion has exactly one), with
  exact largest-remainder counts. An unstratified permutation left df and vasc below the support
  minimum in several evaluation splits on the real data; stratification removes that sampling
  variance, and the feasibility gate proves BEFORE assignment that the configured fractions can
  meet the rule at all.

Usage:
    python scripts/data/ham10000/01_build_splits.py --run-id production-thesis-v1 --check-feasibility
    python scripts/data/ham10000/01_build_splits.py --namespace production --run-id production-thesis-v1 --freeze
    python scripts/data/ham10000/01_build_splits.py --namespace dev --run-id dev-smoke-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.utils.config import load_named_config  # noqa: E402
from scripts.utils.ham10000 import (  # noqa: E402
    DIAGNOSIS_CLASSES,
    check_support_rule,
    group_level_support,
    normalize_diagnosis,
    validate_metadata,
)
from scripts.utils.manifest import get_git_commit_hash, write_json  # noqa: E402

SPLIT_NAMES = [
    "gen_train",
    "gen_val",
    "classifier_train",
    "classifier_val",
    "asism_tuning_heldout",
    "final_eval_heldout",
]


def partition_groups(groups: list[str], fractions: dict[str, float], seed: int) -> dict[str, set[str]]:
    """Deterministically split a LESION list into the six partitions.

    Assignment is by contiguous slices of one seeded permutation, and the final partition absorbs
    the rounding remainder, so the six sets are exactly disjoint and exactly cover the input with
    no lesion dropped or duplicated regardless of how the fractions round.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"split fractions must sum to 1.0, got {total}")

    ordered = sorted(groups)  # sort first so the result depends on the seed, not on file order
    rng = np.random.default_rng(seed)
    permuted = [ordered[i] for i in rng.permutation(len(ordered))]

    assignment: dict[str, set[str]] = {}
    start = 0
    for position, name in enumerate(SPLIT_NAMES):
        if position == len(SPLIT_NAMES) - 1:
            assignment[name] = set(permuted[start:])
            break
        count = int(round(len(permuted) * fractions[name]))
        assignment[name] = set(permuted[start : start + count])
        start += count
    return assignment


def lesion_classes(frame: pd.DataFrame, group_column: str = "lesion_id", diagnosis_column: str = "dx") -> dict[str, str]:
    """lesion_id -> normalised diagnosis. Refuses a lesion carrying two diagnoses, because a lesion
    cannot then be placed in one class stratum without contradicting some of its images."""
    normalized = frame.assign(_dx=frame[diagnosis_column].map(normalize_diagnosis))
    per_lesion = normalized.groupby(normalized[group_column].astype(str))["_dx"].unique()
    conflicting = [lesion for lesion, values in per_lesion.items() if len(values) > 1]
    if conflicting:
        raise SystemExit(f"{len(conflicting)} lesion(s) carry more than one diagnosis, e.g. {conflicting[:5]}")
    return {lesion: str(values[0]) for lesion, values in per_lesion.items()}


def stratified_allocation(lesion_count: int, fractions: dict[str, float]) -> dict[str, int]:
    """Exact integer lesion counts for ONE class stratum, by largest remainder.

    Every split receives at least floor(lesion_count * fraction) lesions and the counts sum exactly
    to lesion_count. Ties in the remainder are broken by SPLIT_NAMES order, so the allocation is a
    pure function of (lesion_count, fractions) — no randomness enters the counts.
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise SystemExit(f"split fractions must sum to 1.0, got {total}")
    ideal = {name: lesion_count * fractions[name] for name in SPLIT_NAMES}
    counts = {name: int(np.floor(ideal[name] + 1e-9)) for name in SPLIT_NAMES}
    leftover = lesion_count - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (-(ideal[name] - counts[name]), SPLIT_NAMES.index(name)))
    for name in order[:leftover]:
        counts[name] += 1
    return counts


def feasibility_report(
    lesions_per_class: dict[str, int],
    fractions: dict[str, float],
    min_positive_groups: int,
    min_negative_groups: int,
    decision_bearing_splits: list[str],
) -> dict:
    """Decide, BEFORE any lesion is assigned, whether these fractions can satisfy the support rule.

    Uses the exact stratified allocation, so a PASS here guarantees the written splits pass the
    support check. Also reports the minimum fraction any decision-bearing split would need
    (min_positive_groups / lesions of the scarcest class): that number comes from the data, and is
    what the fractions in splits_ham10000.yaml are derived from.
    """
    allocation = {label: stratified_allocation(count, fractions) for label, count in lesions_per_class.items()}
    failures = []
    for split in decision_bearing_splits:
        split_total = sum(allocation[label][split] for label in lesions_per_class)
        for label in lesions_per_class:
            positives = allocation[label][split]
            negatives = split_total - positives
            if positives < min_positive_groups:
                failures.append({"split": split, "label": label, "positive_groups": positives, "min_positive_groups": min_positive_groups})
            if negatives < min_negative_groups:
                failures.append({"split": split, "label": label, "negative_groups": negatives, "min_negative_groups": min_negative_groups})
    binding_label = min(lesions_per_class, key=lambda label: lesions_per_class[label])
    return {
        "feasible": not failures,
        "failures": failures,
        "allocation_lesions": allocation,
        "binding_class": binding_label,
        "binding_class_lesions": int(lesions_per_class[binding_label]),
        "minimum_fraction_per_decision_split": min_positive_groups / lesions_per_class[binding_label],
        "decision_bearing_fraction_total": sum(fractions[name] for name in decision_bearing_splits),
    }


def partition_groups_stratified(lesion_to_class: dict[str, str], fractions: dict[str, float], seed: int) -> dict[str, set[str]]:
    """Class-stratified lesion partition: each diagnosis stratum is split separately.

    Within a stratum, lesions are sorted, permuted by an RNG seeded with (seed, class position), and
    sliced by `stratified_allocation`. Sorting first makes the result independent of file order;
    seeding per class makes adding or reordering one class unable to reshuffle another. Every
    lesion lands in exactly one split, and all images of a lesion follow it.
    """
    assignment: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for class_position, label in enumerate(DIAGNOSIS_CLASSES):
        stratum = sorted(lesion for lesion, value in lesion_to_class.items() if value == label)
        rng = np.random.default_rng([int(seed), class_position])
        permuted = [stratum[i] for i in rng.permutation(len(stratum))]
        counts = stratified_allocation(len(stratum), fractions)
        start = 0
        for name in SPLIT_NAMES:
            assignment[name].update(permuted[start : start + counts[name]])
            start += counts[name]
    return assignment


def audit_splits(frames: dict[str, pd.DataFrame], group_column: str, image_id_column: str) -> dict:
    """Prove the written splits are disjoint at both the lesion and image level.

    Returns a report dict whose "status" is PASS only when every pairwise overlap is empty.
    """
    overlaps = {"lesion": {}, "image": {}}
    names = list(frames)

    for column, key in ((group_column, "lesion"), (image_id_column, "image")):
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = set(frames[left][column]) & set(frames[right][column])
                # ASCII " & ", not U+2229: a Windows cp1252 console cannot encode the set symbol
                # and the audit print crashed on it before writing anything.
                overlaps[key][f"{left} & {right}"] = sorted(shared)[:10]
                overlaps[key][f"{left} & {right} count"] = len(shared)

    failed = [
        pair
        for key in overlaps
        for pair, value in overlaps[key].items()
        if pair.endswith("count") and value > 0
    ]

    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_pairs": failed,
        "overlaps": overlaps,
        "images_per_split": {name: int(len(frame)) for name, frame in frames.items()},
        "lesions_per_split": {name: int(frame[group_column].nunique()) for name, frame in frames.items()},
        "class_distribution_per_split": {
            name: {label: int((frame["dx"].map(normalize_diagnosis) == label).sum()) for label in DIAGNOSIS_CLASSES}
            for name, frame in frames.items()
        },
        "not_covered_by_this_audit": (
            "Visually near-duplicate images filed under DIFFERENT lesion_ids cannot be detected "
            "from metadata and are NOT ruled out here. HAM10000 is known to contain some. A "
            "separate pixel-level similarity audit is required before claiming they are absent."
        ),
    }


def print_audit(report: dict) -> None:
    line = "=" * 58
    print(f"\n{line}\nHAM10000 SPLIT AUDIT\n{line}", flush=True)

    print("\nImages / lesions per split:", flush=True)
    for name in SPLIT_NAMES:
        print(
            f"  {name:22s} {report['images_per_split'][name]:6d} images"
            f"   {report['lesions_per_split'][name]:6d} lesions",
            flush=True,
        )

    print("\nOverlaps (must all be 0):", flush=True)
    for key in ("lesion", "image"):
        for pair, value in report["overlaps"][key].items():
            if pair.endswith("count"):
                marker = "OK " if value == 0 else "!!!"
                print(f"  {marker} {key:6s} {pair[:-6]:48s} = {value}", flush=True)

    print("\nClass distribution per split:", flush=True)
    header = "  " + " " * 22 + "".join(f"{label:>7s}" for label in DIAGNOSIS_CLASSES)
    print(header, flush=True)
    for name in SPLIT_NAMES:
        counts = report["class_distribution_per_split"][name]
        print("  " + f"{name:22s}" + "".join(f"{counts[label]:7d}" for label in DIAGNOSIS_CLASSES), flush=True)

    print(f"\nSTATUS: {report['status']}\n{line}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", choices=["production", "dev"], default="production")
    parser.add_argument("--run-id", required=True, help="Immutable split run id, e.g. production-thesis-v1")
    parser.add_argument("--freeze", action="store_true", help="Mark the written manifest frozen")
    parser.add_argument(
        "--check-feasibility",
        action="store_true",
        help="Only compute and print the feasibility gate for the configured fractions; write nothing",
    )
    args = parser.parse_args()

    dataset_cfg = load_named_config("dataset_ham10000.yaml", "ham_dataset")
    splits_cfg = load_named_config("splits_ham10000.yaml", "ham_splits")
    schema = dataset_cfg.schema

    group_column = str(schema.group_column)
    image_id_column = str(schema.image_id_column)
    diagnosis_column = str(schema.diagnosis_column)

    raw_dir = Path(splits_cfg.paths.raw_dir)
    metadata_path = raw_dir / str(dataset_cfg.source.metadata_filename)
    if not metadata_path.is_file():
        raise SystemExit(
            f"Missing metadata: {metadata_path}\n"
            "Run: python scripts/data/ham10000/00_download_dataset.py"
        )

    frame = pd.read_csv(metadata_path)
    validate_metadata(frame, diagnosis_column, group_column)

    namespace_cfg = splits_cfg.namespaces[args.namespace]
    if bool(namespace_cfg.require_full_cohort):
        expected = int(dataset_cfg.source.expected_metadata_rows)
        deviation = abs(len(frame) - expected) / expected * 100.0
        tolerance = float(namespace_cfg.full_cohort_row_tolerance_pct)
        if deviation > tolerance:
            raise SystemExit(
                f"Production splits require the complete HAM10000 cohort: expected ~{expected} "
                f"rows, got {len(frame)} ({deviation:.2f}% deviation, tolerance {tolerance}%)."
            )

    fractions = {name: float(splits_cfg.fractions[name]) for name in SPLIT_NAMES}
    support_rule = splits_cfg.support_rule
    decision_splits = [str(name) for name in support_rule.decision_bearing_splits]
    if "final_eval_heldout" not in decision_splits:
        raise SystemExit("final_eval_heldout must be a decision-bearing split for the support rule")
    if str(splits_cfg.stratify_by) != diagnosis_column:
        raise SystemExit(f"stratify_by must be {diagnosis_column!r}; got {splits_cfg.stratify_by!r}")

    lesion_to_class = lesion_classes(frame, group_column, diagnosis_column)
    lesions_per_class = {label: sum(1 for v in lesion_to_class.values() if v == label) for label in DIAGNOSIS_CLASSES}
    feasibility = feasibility_report(
        lesions_per_class,
        fractions,
        int(support_rule.min_positive_groups),
        int(support_rule.min_negative_groups),
        decision_splits,
    )
    print("\nFEASIBILITY GATE (computed from real lesion counts before assignment)", flush=True)
    print(f"  lesions per class: {lesions_per_class}", flush=True)
    print(
        f"  binding class: {feasibility['binding_class']} ({feasibility['binding_class_lesions']} lesions) -> "
        f"each decision-bearing split needs fraction >= {feasibility['minimum_fraction_per_decision_split']:.4f}",
        flush=True,
    )
    print("  lesions allocated per class (rows) x split (columns):", flush=True)
    print("    " + " " * 7 + "".join(f"{name:>22s}" for name in SPLIT_NAMES), flush=True)
    for label in DIAGNOSIS_CLASSES:
        print("    " + f"{label:7s}" + "".join(f"{feasibility['allocation_lesions'][label][name]:22d}" for name in SPLIT_NAMES), flush=True)
    for failure in feasibility["failures"]:
        print(f"  !!! {failure}", flush=True)
    print(f"  FEASIBLE: {feasibility['feasible']}", flush=True)
    if not feasibility["feasible"]:
        raise SystemExit("Refusing to assign lesions: the configured fractions cannot meet the support rule. Nothing written.")
    if args.check_feasibility:
        return 0

    out_dir = Path(splits_cfg.paths.splits_root) / args.run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"REFUSING to overwrite an existing split run at {out_dir}.\n"
            "A split assignment is immutable once written: choose a new --run-id."
        )

    assignment = partition_groups_stratified(lesion_to_class, fractions, int(splits_cfg.split_seed))

    frame[group_column] = frame[group_column].astype(str)
    frames = {
        name: frame[frame[group_column].isin(lesions)].reset_index(drop=True)
        for name, lesions in assignment.items()
    }

    total_assigned = sum(len(f) for f in frames.values())
    if total_assigned != len(frame):
        raise SystemExit(f"partition lost rows: {total_assigned} assigned vs {len(frame)} in metadata")

    report = audit_splits(frames, group_column, image_id_column)
    print_audit(report)

    # Support check BEFORE writing anything, so a split that cannot support the rare classes never
    # reaches disk and gets used by accident.
    support_report = {}
    support_failed = []
    for name in [str(n) for n in support_rule.decision_bearing_splits]:
        support = group_level_support(frames[name], group_column=group_column, diagnosis_column=diagnosis_column)
        passed, failures = check_support_rule(
            support,
            int(support_rule.min_positive_groups),
            int(support_rule.min_negative_groups),
        )
        support_report[name] = {"passed": passed, "failures": failures, "per_label": support.to_dict("records")}
        if not passed:
            support_failed.append(name)

    if support_failed:
        print("\nSUPPORT CHECK FAILED for: " + ", ".join(support_failed), flush=True)
        for name in support_failed:
            for failure in support_report[name]["failures"]:
                print(
                    f"  {name}: {failure['label']} has {failure['positive_groups']} positive lesion(s), "
                    f"needs {failure['min_positive_groups']}",
                    flush=True,
                )

    if report["status"] == "FAIL" or support_failed:
        raise SystemExit(
            "\nRefusing to write splits: "
            + ("leakage audit FAILED. " if report["status"] == "FAIL" else "")
            + ("support check FAILED. " if support_failed else "")
            + "Nothing has been written."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for name, split_frame in frames.items():
        path = out_dir / f"{name}.csv"
        split_frame.to_csv(path, index=False)
        output_files[name] = {
            "filename": path.name,
            "row_count": int(len(split_frame)),
            "lesion_count": int(split_frame[group_column].nunique()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    write_json(out_dir / "leakage_audit.json", report)

    manifest = {
        "manifest_version": int(splits_cfg.manifest_version),
        "split_namespace": args.run_id,
        "namespace_class": args.namespace,
        "dataset": "ham10000",
        "split_unit": str(splits_cfg.split_unit),
        "stratify_by": str(splits_cfg.stratify_by),
        "partition_method": "class_stratified_lesion_largest_remainder_v1",
        "feasibility": {
            "lesions_per_class": lesions_per_class,
            "allocation_lesions": feasibility["allocation_lesions"],
            "binding_class": feasibility["binding_class"],
            "minimum_fraction_per_decision_split": feasibility["minimum_fraction_per_decision_split"],
        },
        "split_seed": int(splits_cfg.split_seed),
        "fractions": fractions,
        "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "total_images": int(len(frame)),
        "total_lesions": int(frame[group_column].nunique()),
        "output_files": output_files,
        "leakage_audit": {"status": report["status"], "not_covered": report["not_covered_by_this_audit"]},
        "support_check": {"passed": True, "per_split": support_report},
        "frozen": bool(args.freeze),
        "git_commit_hash": get_git_commit_hash(),
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    write_json(out_dir / "split_manifest_v2.json", manifest)

    print(f"Support check: PASSED", flush=True)
    print(f"Frozen:        {bool(args.freeze)}", flush=True)
    print(f"Manifest hash: {manifest['manifest_hash']}", flush=True)
    print(f"Wrote:         {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

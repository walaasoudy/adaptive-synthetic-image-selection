from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import load_dataset_config, load_stage1_config  # noqa: E402
from scripts.utils.artifact_contracts import validate_namespace_run_id  # noqa: E402
from scripts.utils.chexpert_schema import validate_chexpert_frame  # noqa: E402
from scripts.utils.labels import (  # noqa: E402
    CLASSIFIER_TARGET_LABELS,
    PRIMARY_ENDPOINT_LABELS,
    check_support_rule,
    patient_level_support,
    normalize_label,
    validate_label_domain,
)
from scripts.utils.manifest import get_git_commit_hash, get_library_versions, hash_dict, sha256_file, write_json  # noqa: E402
from scripts.utils.splits import (  # noqa: E402
    FINAL_EVAL_SPLIT,
    SPLIT_MANIFEST_FILENAME,
    SPLIT_NAMES,
    assert_final_eval_access_allowed,
    load_splits_config,
    resolve_splits_dir,
)

SUPERSEDED_NOTE_FILENAME = "SUPERSEDED_v1.md"

SUPERSEDED_NOTE = """# Superseded schema-v1 split artifacts

The files in this directory's parent (`gen_train.csv`, `gen_val.csv`, `classifier_heldout.csv`,
`asism_tuning_heldout.csv`, `final_eval_heldout.csv`, `split_manifest.json`) were produced by
`scripts/data/02_build_patient_splits.py` under the schema-v1 three-way split
(gen_train 0.70 / gen_val 0.10 / classifier_heldout 0.20, with the ASISM-tuning and final-eval
splits derived as children of classifier_heldout).

That architecture is superseded by the six-way partition in
`scripts/data/02b_build_sixway_splits.py` (docs/stages2_to_5_plan.md §1), because it left no
patient population for a classifier development split, and deriving one from `gen_train` would
have meant the headline classifiers' early-stopping and threshold decisions were made on patients the
SDXL generator had itself trained on.

These files are RETAINED for provenance and reproducibility of anything already built against
them. They are NOT read by any Stage 2-5 code. Current splits live under
`splits/production/` and `splits/dev/`.
"""


def extract_patient_id(path_value: str, patient_id_regex: str) -> str:
    match = re.search(patient_id_regex, str(path_value))
    if not match:
        raise ValueError(f"Could not extract patient id from path: {path_value!r}")
    return match.group(0)


def partition_patients(
    patient_ids: list[str],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, set[str]]:
    """Six direct partitions from ONE shuffle of ONE deterministically ordered patient list.

    Sorting before shuffling is what makes this reproducible regardless of the row order the source
    CSV happens to arrive in. Remainder patients (from rounding) go to the largest split so the
    partition is exact rather than dropping anyone.
    """
    ordered = sorted(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    total = len(ordered)
    counts = {name: int(total * fraction) for name, fraction in fractions.items()}

    # Assign rounding remainder to the largest split, keeping the partition exact.
    assigned = sum(counts.values())
    if assigned < total:
        largest = max(fractions, key=lambda name: fractions[name])
        counts[largest] += total - assigned

    result: dict[str, set[str]] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        take = counts[name]
        result[name] = set(ordered[cursor : cursor + take])
        cursor += take

    assert cursor == total, f"Partition consumed {cursor} of {total} patients"
    return result


def patient_label_vectors(frame: pd.DataFrame, labels: list[str] = PRIMARY_ENDPOINT_LABELS) -> dict[str, tuple[int, ...]]:
    """Collapse image labels to deterministic patient-level any-positive vectors."""
    vectors = {}
    for patient_id, rows in frame.groupby("patient_id", sort=True):
        vectors[str(patient_id)] = tuple(
            int(any(normalize_label(value) == 1 for value in rows[label])) for label in labels
        )
    return vectors


def patient_label_negative_vectors(
    frame: pd.DataFrame, labels: list[str] = PRIMARY_ENDPOINT_LABELS
) -> dict[str, tuple[int, ...]]:
    """Collapse image labels to deterministic patient-level confident-negative vectors.

    Mirrors patient_level_support's negative definition (scripts/utils/labels.py): a patient is
    negative for a label if none of their studies is a confident 1 and at least one is a confident
    0. Kept separate from patient_label_vectors (positive-only) so multilabel_partition_patients can
    target both sides of the §1.3 support rule without changing prevalence_report's positive-only
    semantics.
    """
    vectors = {}
    for patient_id, rows in frame.groupby("patient_id", sort=True):
        vectors[str(patient_id)] = tuple(
            int(
                not any(normalize_label(value) == 1 for value in rows[label])
                and any(normalize_label(value) == 0 for value in rows[label])
            )
            for label in labels
        )
    return vectors


def multilabel_partition_patients(
    vectors: dict[str, tuple[int, ...]],
    neg_vectors: dict[str, tuple[int, ...]],
    fractions: dict[str, float],
    seed: int,
) -> dict[str, set[str]]:
    """Deterministic greedy iterative multilabel allocation.

    Targets both positive AND confident-negative patient counts per label (the §1.3 support rule
    needs both sides, but only positives drove allocation before this fix). Deficits are normalized
    by each split's own target so a small split (e.g. a 5% heldout) is judged by how much of its own
    quota is unmet, not by raw patient counts — otherwise a large split's much bigger absolute target
    (e.g. gen_train at 60%) always looks more "urgent" and starves rare labels out of the small
    splits entirely. Rarest labels (by either their positive or negative population) are placed
    first; each patient goes to the non-full split with the largest normalized deficit for that
    patient's positive+negative labels, then capacity deficit, then a seeded stable tie rank. This
    uses one recorded seed and never retries seeds.
    """
    names = list(SPLIT_NAMES)
    patients = sorted(vectors)
    total = len(patients)
    n_labels = len(next(iter(vectors.values()), ()))
    capacities = {name: int(total * fractions[name]) for name in names}
    capacities[max(fractions, key=fractions.get)] += total - sum(capacities.values())

    pos_totals = [sum(v[j] for v in vectors.values()) for j in range(n_labels)]
    neg_totals = [sum(v[j] for v in neg_vectors.values()) for j in range(n_labels)]
    pos_targets = {name: [pos_totals[j] * fractions[name] for j in range(n_labels)] for name in names}
    neg_targets = {name: [neg_totals[j] * fractions[name] for j in range(n_labels)] for name in names}

    rng = random.Random(seed)
    tie_order = patients[:]
    rng.shuffle(tie_order)
    tie_rank = {patient: rank for rank, patient in enumerate(tie_order)}

    def rarest_signal(patient: str) -> int:
        counts = [pos_totals[j] for j, value in enumerate(vectors[patient]) if value]
        counts += [neg_totals[j] for j, value in enumerate(neg_vectors[patient]) if value]
        return min(counts, default=total + 1)

    ordered = sorted(
        patients,
        key=lambda p: (
            rarest_signal(p),
            -(sum(vectors[p]) + sum(neg_vectors[p])),
            tie_rank[p],
            p,
        ),
    )
    result = {name: set() for name in names}
    pos_observed = {name: [0] * n_labels for name in names}
    neg_observed = {name: [0] * n_labels for name in names}
    for patient in ordered:
        pos_vector = vectors[patient]
        neg_vector = neg_vectors[patient]
        candidates = [name for name in names if len(result[name]) < capacities[name]]

        def score(name):
            pos_deficit = sum(
                max(0.0, (pos_targets[name][j] - pos_observed[name][j]) / max(pos_targets[name][j], 1e-9))
                for j, value in enumerate(pos_vector) if value
            )
            neg_deficit = sum(
                max(0.0, (neg_targets[name][j] - neg_observed[name][j]) / max(neg_targets[name][j], 1e-9))
                for j, value in enumerate(neg_vector) if value
            )
            capacity_deficit = (capacities[name] - len(result[name])) / max(capacities[name], 1)
            return (pos_deficit + neg_deficit, capacity_deficit, -names.index(name))

        chosen = max(candidates, key=score)
        result[chosen].add(patient)
        pos_observed[chosen] = [a + b for a, b in zip(pos_observed[chosen], pos_vector)]
        neg_observed[chosen] = [a + b for a, b in zip(neg_observed[chosen], neg_vector)]
    return result


def prevalence_report(vectors: dict[str, tuple[int, ...]], groups: dict[str, set[str]]) -> dict:
    labels = list(PRIMARY_ENDPOINT_LABELS)
    def rates(ids):
        n = max(len(ids), 1)
        return {label: sum(vectors[p][j] for p in ids) / n for j, label in enumerate(labels)}
    source = rates(set(vectors))
    per_split = {}
    for name, ids in groups.items():
        split_rates = rates(ids)
        per_split[name] = {label: {
            "prevalence": split_rates[label],
            "absolute_deviation": abs(split_rates[label] - source[label]),
            "relative_deviation": (abs(split_rates[label] - source[label]) / source[label] if source[label] else None),
        } for label in labels}
    return {"source": source, "per_split": per_split, "labels_unable_to_meet_tolerance": []}


def assert_disjoint_and_exact(groups: dict[str, set[str]], all_patients: set[str]) -> None:
    """Pairwise disjointness + exact partition. Hard assertions, not warnings — a leak here
    invalidates every downstream comparison."""
    names = list(groups.keys())
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise SystemExit(
                    f"SPLIT LEAK: {len(overlap)} patient(s) in both {left} and {right}; "
                    f"examples: {sorted(overlap)[:5]}"
                )

    union = set().union(*groups.values())
    if union != all_patients:
        missing = all_patients - union
        extra = union - all_patients
        raise SystemExit(
            f"SPLIT NOT EXACT: {len(missing)} patient(s) unassigned, {len(extra)} unknown. "
            f"missing examples: {sorted(missing)[:5]}"
        )


def verify_full_cohort(frame: pd.DataFrame, dataset_cfg, tolerance_pct: float) -> dict:
    """Production namespace gate (§1.4): the source must be the complete intended CheXpert cohort,
    checked against the same expected-row contract 01_verify_download.py uses."""
    expected = int(dataset_cfg.source.expected_train_rows)
    actual = len(frame)
    deviation_pct = abs(actual - expected) / expected * 100.0
    passed = deviation_pct <= tolerance_pct
    return {
        "expected_train_rows": expected,
        "actual_rows": actual,
        "deviation_pct": round(deviation_pct, 4),
        "tolerance_pct": tolerance_pct,
        "passed": passed,
    }


def build_support_report(
    frames: dict[str, pd.DataFrame],
    decision_bearing_splits: list[str],
    min_positive: int,
    min_negative: int,
) -> dict:
    """Run the frozen §1.3 support rule over every decision-bearing split and return a COMPLETE
    report (all splits, all labels, all failures) — not a first-failure abort, because the caller
    needs the whole picture to propose a fraction adjustment."""
    per_split = {}
    all_passed = True

    for split_name in decision_bearing_splits:
        frame = frames[split_name]
        if split_name == FINAL_EVAL_SPLIT:
            assert_final_eval_access_allowed(
                purpose="support_check", caller="02b_build_sixway_splits"
            )
        support = patient_level_support(frame, PRIMARY_ENDPOINT_LABELS)
        passed, failures = check_support_rule(support, min_positive, min_negative)
        all_passed = all_passed and passed
        per_split[split_name] = {
            "passed": passed,
            "failures": failures,
            "per_label": support.to_dict("records"),
        }

    return {
        "rule": {
            "min_positive_patients": min_positive,
            "min_negative_patients": min_negative,
            "unit": "patients (image counts are supplementary only)",
            "labels": PRIMARY_ENDPOINT_LABELS,
        },
        "passed": all_passed,
        "per_split": per_split,
    }


def propose_fraction_adjustment(
    support_report: dict,
    fractions: dict[str, float],
    total_patients: int,
) -> dict:
    """Rough support-scaling estimate when frozen fractions fail the support rule.

    Computes, per failing split, the scale-up factor its fraction would need for its worst-failing
    label to clear the threshold, then proposes taking the shortfall from gen_train (the largest
    split, and the one whose marginal patient is worth least — Stage 1 LoRA training is far less
    sensitive to a few hundred patients than a support-starved evaluation split is).

    This is a PROPOSAL written to the report. The frozen fractions are never modified automatically.
    """
    proposals = {}
    for split_name, result in support_report["per_split"].items():
        if result["passed"]:
            continue
        current_fraction = fractions[split_name]
        current_patients = max(1, int(total_patients * current_fraction))

        worst_ratio = 1.0
        for failure in result["failures"]:
            for observed, required in (
                (failure["positive_patients"], failure["min_positive_patients"]),
                (failure["negative_patients"], failure["min_negative_patients"]),
            ):
                if observed < required:
                    # Scale needed so this label's count reaches the threshold.
                    ratio = required / max(observed, 1)
                    worst_ratio = max(worst_ratio, ratio)

        needed_fraction = min(1.0, current_fraction * worst_ratio)
        proposals[split_name] = {
            "current_fraction": current_fraction,
            "current_patients": current_patients,
            "limiting_scale_factor": round(worst_ratio, 3),
            "proposed_fraction": round(needed_fraction, 4),
            "additional_patients_needed": int(total_patients * (needed_fraction - current_fraction)),
        }

    total_additional = sum(p["additional_patients_needed"] for p in proposals.values())
    return {
        "estimate_type": "rough_support_scaling_estimate",
        "proposals": proposals,
        "suggested_donor_split": "gen_train",
        "total_additional_patients_needed": total_additional,
        "suggested_gen_train_fraction": round(
            fractions["gen_train"] - total_additional / max(total_patients, 1), 4
        ),
        "note": (
            "PROPOSAL ONLY — frozen fractions were not modified. Applying this requires an explicit "
            "decision to change the frozen split policy in configs/splits.yaml, and re-running with "
            "--freeze."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--namespace",
        choices=["production", "dev"],
        required=True,
        help="production = thesis splits (requires full cohort); dev = smoke-test splits",
    )
    parser.add_argument("--run-id", required=True, help="New immutable split run/version ID (for example production-v1 or dev-smoke-v3)")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Source CSV. Defaults to stage1 config's split.input_csv.",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Mark the manifest frozen. Required before Stage 5. Fails if the support check failed.",
    )
    args = parser.parse_args()

    splits_cfg = load_splits_config()
    validate_namespace_run_id(args.run_id, args.namespace)
    stage1_cfg = load_stage1_config()
    dataset_cfg = load_dataset_config()
    out_dir = resolve_splits_dir(args.run_id, splits_cfg)
    if out_dir.exists():
        raise SystemExit(f"Immutable split run already exists; refusing every in-place write: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    fractions = {name: float(splits_cfg.fractions[name]) for name in SPLIT_NAMES}
    fraction_sum = round(sum(fractions.values()), 9)
    if fraction_sum != 1.0:
        raise SystemExit(f"Split fractions must sum to 1.0; got {fraction_sum} from {fractions}")

    input_csv = Path(args.input_csv) if args.input_csv else Path(stage1_cfg.split.input_csv)
    if not input_csv.is_file():
        raise SystemExit(f"Missing source CSV: {input_csv} (run scripts/data/00_download_dataset.py first)")

    frame = pd.read_csv(input_csv)
    schema = dataset_cfg.schema
    try:
        source_validation = validate_chexpert_frame(
            frame, dataset_cfg, Path(stage1_cfg.paths.raw_dir), require_all_images=args.namespace == "production"
        )
    except ValueError as exc:
        raise SystemExit(f"Source schema validation failed: {exc}") from exc
    frame["patient_id"] = frame[schema.path_column].apply(
        lambda value: extract_patient_id(value, schema.patient_id_regex)
    )

    namespace_cfg = splits_cfg.namespaces[args.namespace]
    cohort_check = None
    if bool(namespace_cfg.require_full_cohort):
        cohort_check = verify_full_cohort(
            frame, dataset_cfg, float(namespace_cfg.full_cohort_row_tolerance_pct)
        )
        if not cohort_check["passed"]:
            raise SystemExit(
                "Production splits require the complete intended CheXpert-v1.0-small training "
                f"cohort.\n  expected ~{cohort_check['expected_train_rows']} rows, got "
                f"{cohort_check['actual_rows']} ({cohort_check['deviation_pct']}% deviation, "
                f"tolerance {cohort_check['tolerance_pct']}%)\n"
                f"  source: {input_csv}\n"
                "Use --namespace dev for the development subset, or point split.input_csv at the "
                "full train.csv on a machine where it is available."
            )

    all_patients = set(frame["patient_id"].unique())
    vectors = patient_label_vectors(frame)
    neg_vectors = patient_label_negative_vectors(frame)
    groups = multilabel_partition_patients(vectors, neg_vectors, fractions, int(splits_cfg.split_seed))
    assert_disjoint_and_exact(groups, all_patients)

    frames = {
        name: frame[frame["patient_id"].isin(ids)].reset_index(drop=True)
        for name, ids in groups.items()
    }

    support_report = build_support_report(
        frames,
        list(splits_cfg.support_rule.decision_bearing_splits),
        int(splits_cfg.support_rule.min_positive_patients),
        int(splits_cfg.support_rule.min_negative_patients),
    )

    adjustment = None
    if not support_report["passed"]:
        adjustment = propose_fraction_adjustment(support_report, fractions, len(all_patients))

    # pathlib mkdir avoids Windows tempfile ACL inheritance that can make the staging directory
    # unreadable to the creating process in managed workspaces; uniqueness and same-filesystem
    # atomic publication are preserved.
    staging = out_dir.parent / f".{args.run_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    output_files = {}
    for name, split_frame in frames.items():
        if name == FINAL_EVAL_SPLIT:
            assert_final_eval_access_allowed(
                purpose="split_construction", caller="02b_build_sixway_splits"
            )
        csv_path = staging / f"{name}.csv"
        split_frame.to_csv(csv_path, index=False)
        reread = pd.read_csv(csv_path)
        if len(reread) != len(split_frame) or set(reread["patient_id"].astype(str)) != groups[name]:
            raise SystemExit(f"Staged split validation failed for {name}")
        output_files[name] = {"filename": csv_path.name, "sha256": sha256_file(csv_path), "row_count": len(reread), "patient_count": len(groups[name])}

    # Retain and mark the superseded v1 artifacts rather than deleting them (§1.5).
    legacy_manifest = Path(splits_cfg.paths.splits_root) / "split_manifest.json"
    if legacy_manifest.is_file():
        note_path = Path(splits_cfg.paths.splits_root) / SUPERSEDED_NOTE_FILENAME
        if not note_path.exists():
            note_path.write_text(SUPERSEDED_NOTE, encoding="utf-8")

    core = {
        "manifest_version": int(splits_cfg.manifest_version),
        "split_namespace": args.run_id,
        "namespace_class": args.namespace,
        "split_unit": str(splits_cfg.split_unit),
        "split_seed": int(splits_cfg.split_seed),
        "fractions": fractions,
        "input_csv_path": str(input_csv),
        "input_csv_rows": len(frame),
        "source_csv_sha256": sha256_file(input_csv),
        "source_schema_hash": source_validation["source_schema_hash"],
        "sorted_patient_set_hash": hash_dict({"patient_ids": sorted(all_patients)}, length=64),
        "output_files": output_files,
        "allocation_algorithm": "deterministic_greedy_iterative_multilabel_v1",
        "allocation_library": "internal",
        "prevalence": prevalence_report(vectors, groups),
        "num_patients_total": len(all_patients),
        "patients_per_split": {name: len(ids) for name, ids in groups.items()},
        "images_per_split": {name: len(f) for name, f in frames.items()},
        "patient_id_hash_per_split": {
            name: hash_dict({"patient_ids": sorted(ids)}, length=16) for name, ids in groups.items()
        },
        "primary_endpoint_labels": PRIMARY_ENDPOINT_LABELS,
    }

    manifest = dict(core)
    manifest["git_commit_hash"] = get_git_commit_hash()
    manifest["full_cohort_check"] = cohort_check
    manifest["support_check"] = support_report
    manifest["support_adjustment_proposal"] = adjustment
    manifest["frozen"] = False
    manifest["supersedes"] = "split_manifest.json (schema v1, three-way + derived children)"
    manifest["environment_versions"] = get_library_versions()

    if args.freeze:
        if not support_report["passed"]:
            shutil.rmtree(staging)
            print("\n" + "=" * 78, flush=True)
            print("SUPPORT CHECK FAILED — REFUSING TO FREEZE", flush=True)
            print("=" * 78, flush=True)
            for split_name, result in support_report["per_split"].items():
                if result["passed"]:
                    continue
                print(f"\n[{split_name}] {len(result['failures'])} label(s) below the frozen rule:", flush=True)
                for failure in result["failures"]:
                    print(
                        f"    {failure['label']:<32} "
                        f"pos={failure['positive_patients']:>5} (need {failure['min_positive_patients']})  "
                        f"neg={failure['negative_patients']:>5} (need {failure['min_negative_patients']})",
                        flush=True,
                    )
            if adjustment:
                print("\nRough support scaling estimate (not an optimized allocation):", flush=True)
                for split_name, proposal in adjustment["proposals"].items():
                    print(
                        f"    {split_name}: {proposal['current_fraction']} -> "
                        f"{proposal['proposed_fraction']} "
                        f"(+{proposal['additional_patients_needed']} patients)",
                        flush=True,
                    )
                print(
                    f"    donor: {adjustment['suggested_donor_split']} -> "
                    f"{adjustment['suggested_gen_train_fraction']}",
                    flush=True,
                )
                print(f"\n    {adjustment['note']}", flush=True)
            print("\nNo split run was published because the frozen support gate failed.", flush=True)
            print(
                f"\nThe frozen split fractions {fractions} were NOT modified. Changing them is an "
                "explicit decision (docs/stages2_to_5_plan.md §1.3).",
                flush=True,
            )
            return 2
        manifest["frozen"] = True

    manifest["manifest_hash_scope"] = "full_manifest_without_manifest_hash_v2"
    manifest["manifest_hash"] = hash_dict(manifest, length=64)
    write_json(staging / SPLIT_MANIFEST_FILENAME, manifest)
    os.replace(staging, out_dir)

    print(f"Namespace:     {args.namespace}", flush=True)
    print(f"Source CSV:    {input_csv} ({len(frame)} rows)", flush=True)
    print(f"Patients:      {len(all_patients)}", flush=True)
    print(f"Seed:          {splits_cfg.split_seed}", flush=True)
    for name in SPLIT_NAMES:
        print(
            f"  {name:<24} {len(groups[name]):>7} patients  {len(frames[name]):>8} images  "
            f"({fractions[name]:.0%})",
            flush=True,
        )
    print(f"Support check: {'PASSED' if support_report['passed'] else 'FAILED'}", flush=True)
    if not support_report["passed"]:
        failing = [
            f"{split_name}({len(result['failures'])})"
            for split_name, result in support_report["per_split"].items()
            if not result["passed"]
        ]
        print(f"  failing splits: {', '.join(failing)}", flush=True)
        if args.namespace == "dev":
            print(
                "  (expected on the dev subset — recorded in the manifest, not fatal; "
                "these are NOT production-frozen thesis splits)",
                flush=True,
            )
    print(f"Frozen:        {manifest['frozen']}", flush=True)
    print(f"Manifest hash: {manifest['manifest_hash']}", flush=True)
    print(f"Wrote:         {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
